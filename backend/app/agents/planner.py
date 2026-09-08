"""Deterministic routing gate, entity resolution, and execution planning.

Stage 1 produces a `RoutingDecision` and an `ExecutionPlan` and stops. Nothing
is executed, no Claude call is made, no network is touched.

ENTITY RESOLUTION reuses the canonical registry in
`services/football_data/teams.py` through its PUBLIC api only
(`canonical_ids`, `get`, `resolve`, `normalize_team_name`). It builds a scan
index from the registry's own alias data rather than duplicating aliases, so
there is exactly one source of team truth. Matching is exact alias lookup over
n-grams - never fuzzy.

FALLBACK SEMANTICS. A failed gate does not automatically mean "ask Claude":

* Ambiguity failures (no clear intent, conflicting intents, missing/unknown
  team) -> `requires_llm_fallback=True`. A cheap classifier can genuinely
  resolve these.
* Terminal failures (the capability simply does not exist yet - scouting,
  what-if) -> `requires_llm_fallback=False`. We already know exactly what was
  asked and that we cannot serve it; spending a Claude call to rediscover that
  would be pure waste.

This keeps the Tier 0 invariant intact: a plan is TIER_0 only when no fallback
is required, so `tier == 0` implies zero Claude calls end to end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.agents.contracts import (
    AMBIGUITY_CODES,
    CapabilityStatus,
    ExecutionPlan,
    ExecutionTier,
    Intent,
    RationaleCode,
    ResolvedEntity,
    RoutingDecision,
    RoutingGateCriteria,
    RoutingResult,
    SpecialistName,
    ToolName,
)
from backend.app.agents.intents import (
    IntentClassification,
    classify_intent,
    normalize_question,
)
from backend.app.services.football_data.errors import UnknownTeam
from backend.app.services.football_data.teams import (
    TeamRegistry,
    default_registry,
    normalize_team_name,
)

# How many teams each intent needs before it can be planned deterministically.
MIN_TEAMS: dict[Intent, int] = {
    Intent.SIMPLE_FACT: 0,
    Intent.HISTORICAL_STATISTICAL: 1,
    Intent.PREDICTION: 2,
    Intent.PREDICTION_EXPLANATION: 2,
    Intent.SCORELINE: 2,
    Intent.LIVE_MATCH: 0,
    Intent.RESEARCH: 1,
    Intent.TACTICAL: 2,
    Intent.COMPLEX_SYNTHESIS: 2,
    Intent.WHAT_IF: 0,
    Intent.SCOUTING: 0,
}

# Intents whose deterministic capability does NOT exist yet. Verified against
# backend/app/tools: there is no what-if engine and no player-similarity layer.
UNAVAILABLE_CAPABILITIES: dict[Intent, RationaleCode] = {
    Intent.WHAT_IF: RationaleCode.WHAT_IF_ENGINE_UNAVAILABLE,
    Intent.SCOUTING: RationaleCode.SCOUTING_DATA_UNAVAILABLE,
}


@dataclass(frozen=True)
class _TeamMatch:
    canonical_id: str
    mentioned_as: str
    start: int
    length: int


def _alias_index(registry: TeamRegistry) -> dict[str, str]:
    """Normalized alias -> canonical_id, built from the registry's public data.

    Not a second registry: this is a scan index derived from the one registry,
    so aliases can never drift apart.
    """
    index: dict[str, str] = {}
    for canonical_id in registry.canonical_ids:
        team = registry.get(canonical_id)
        for spelling in (team.canonical_name, *team.historical_names, *team.aliases):
            index[normalize_team_name(spelling)] = canonical_id
    return index


# "Arsenal's next match" must find Arsenal. The registry's normalizer strips
# apostrophes (correct for "Nott'm Forest" -> "nottm forest"), which would
# otherwise turn "Arsenal's" into "arsenals" and match nothing. Removing the
# possessive BEFORE normalizing fixes that without touching the registry and
# without any fuzzy matching. Only "'s" is stripped, so "Nott'm" is untouched.
_POSSESSIVE = re.compile(r"(\w)['’]s\b", re.IGNORECASE)


def strip_possessives(text: str) -> str:
    return _POSSESSIVE.sub(r"\1", text)


def extract_teams(question: str, registry: TeamRegistry) -> tuple[ResolvedEntity, ...]:
    """Find club mentions by exact alias match over word n-grams.

    Longest match wins and consumed tokens are not reused, so
    "Manchester United" never also matches a shorter alias inside it. No fuzzy
    matching: an unrecognised name yields no entity rather than a wrong club.
    """
    index = _alias_index(registry)
    tokens = normalize_team_name(strip_possessives(question)).split()
    max_span = max((len(alias.split()) for alias in index), default=1)

    matches: list[_TeamMatch] = []
    consumed: set[int] = set()
    for span in range(min(max_span, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - span + 1):
            if any(i in consumed for i in range(start, start + span)):
                continue
            candidate = " ".join(tokens[start : start + span])
            canonical_id = index.get(candidate)
            if canonical_id is None:
                continue
            if any(m.canonical_id == canonical_id for m in matches):
                consumed.update(range(start, start + span))
                continue
            matches.append(_TeamMatch(canonical_id, candidate, start, span))
            consumed.update(range(start, start + span))

    matches.sort(key=lambda m: m.start)  # keep the user's ordering (home, away)
    entities = []
    for match in matches:
        team = registry.get(match.canonical_id)
        entities.append(
            ResolvedEntity(
                canonical_id=team.canonical_id,
                canonical_name=team.canonical_name,
                mentioned_as=match.mentioned_as,
                historical_ml_history_available=team.historical_ml_history_available,
            )
        )
    return tuple(entities)


def _resolve_explicit_teams(
    teams: list[str], registry: TeamRegistry
) -> tuple[ResolvedEntity, ...]:
    """Resolve caller-supplied team names through the registry.

    Preserves `UnknownTeam`: an explicitly named club that does not resolve is
    an error, never a silent omission or a fuzzy guess.
    """
    entities = []
    for raw in teams:
        team = registry.resolve(raw)  # raises UnknownTeam
        entities.append(
            ResolvedEntity(
                canonical_id=team.canonical_id,
                canonical_name=team.canonical_name,
                mentioned_as=raw,
                historical_ml_history_available=team.historical_ml_history_available,
            )
        )
    return tuple(entities)


def _plan_tools_and_specialists(
    intent: Intent, classification: IntentClassification
) -> tuple[tuple[ToolName, ...], tuple[SpecialistName, ...], ExecutionTier]:
    """Map an intent onto REAL tools, the two permitted specialists, and a tier."""
    if intent is Intent.SIMPLE_FACT:
        if RationaleCode.MATCHED_STANDINGS in classification.rationale_codes:
            return (ToolName.GET_CURRENT_STANDINGS,), (), ExecutionTier.TIER_0
        return (ToolName.GET_FIXTURES,), (), ExecutionTier.TIER_0

    if intent is Intent.HISTORICAL_STATISTICAL:
        # Stats stays deterministic: fixtures/results plus the table cover
        # form and comparison. No stats specialist, by locked decision.
        return (ToolName.GET_FIXTURES, ToolName.GET_CURRENT_STANDINGS), (), ExecutionTier.TIER_1

    if intent is Intent.PREDICTION:
        return (
            (ToolName.RUN_OUTCOME_PREDICTION, ToolName.GET_SCORELINE_PREDICTION),
            (),
            ExecutionTier.TIER_1,
        )

    if intent is Intent.PREDICTION_EXPLANATION:
        return (
            (ToolName.RUN_OUTCOME_PREDICTION, ToolName.EXPLAIN_OUTCOME_PREDICTION),
            (),
            ExecutionTier.TIER_1,
        )

    if intent is Intent.SCORELINE:
        return (ToolName.GET_SCORELINE_PREDICTION,), (), ExecutionTier.TIER_1

    if intent is Intent.LIVE_MATCH:
        if classification.live_reasoning_requested:
            # "Why is X struggling right now?" is reasoning over live state.
            return (
                (ToolName.GET_LIVE_MATCHES, ToolName.RUN_OUTCOME_PREDICTION),
                (SpecialistName.TACTICAL,),
                ExecutionTier.TIER_2,
            )
        return (ToolName.GET_LIVE_MATCHES,), (), ExecutionTier.TIER_0

    if intent is Intent.RESEARCH:
        return (ToolName.GET_FIXTURES,), (SpecialistName.RESEARCH,), ExecutionTier.TIER_2

    if intent is Intent.TACTICAL:
        return (
            (ToolName.RUN_OUTCOME_PREDICTION, ToolName.EXPLAIN_OUTCOME_PREDICTION),
            (SpecialistName.TACTICAL,),
            ExecutionTier.TIER_2,
        )

    if intent is Intent.COMPLEX_SYNTHESIS:
        return (
            (
                ToolName.GET_CURRENT_STANDINGS,
                ToolName.GET_FIXTURES,
                ToolName.RUN_OUTCOME_PREDICTION,
                ToolName.EXPLAIN_OUTCOME_PREDICTION,
                ToolName.GET_SCORELINE_PREDICTION,
            ),
            (SpecialistName.RESEARCH, SpecialistName.TACTICAL),
            ExecutionTier.TIER_3,
        )

    # WHAT_IF / SCOUTING never reach here - handled as unavailable upstream.
    return (), (), ExecutionTier.TIER_0


def route_and_plan(
    question: str,
    *,
    teams: list[str] | None = None,
    registry: TeamRegistry | None = None,
) -> RoutingResult:
    """Deterministically route one question and produce an execution plan.

    `teams` lets a caller (e.g. a UI with an explicit match selection) supply
    club names directly; those go through `TeamRegistry.resolve` and raise
    `UnknownTeam` if unrecognised. Without it, clubs are extracted from the
    question text by exact alias match.
    """
    registry = registry if registry is not None else default_registry()
    classification = classify_intent(question)

    entities = (
        _resolve_explicit_teams(teams, registry)
        if teams
        else extract_teams(question, registry)
    )

    intent = classification.intent
    rationale: list[RationaleCode] = list(classification.rationale_codes)
    failures: list[RationaleCode] = []

    single_clear_intent = intent is not None
    no_conflicting_intents = classification.failure_code is not RationaleCode.MULTIPLE_INTENTS
    if classification.failure_code is not None:
        failures.append(classification.failure_code)

    # Entity requirement, only meaningful once an intent is known.
    required_entities_resolved = True
    if intent is not None:
        needed = MIN_TEAMS[intent]
        if len(entities) < needed:
            required_entities_resolved = False
            failures.append(
                RationaleCode.MISSING_MATCHUP if needed >= 2 else RationaleCode.MISSING_REQUIRED_TEAM
            )

    # Whether the intent resolved to a KNOWN deterministic routing outcome -
    # NOT whether that outcome is currently executable. For every intent in
    # `Intent` today there is always a defined outcome (a real tool/specialist
    # plan, or the deliberate "capability unavailable" plan for WHAT_IF /
    # SCOUTING), so this is True whenever an intent was identified at all.
    # Executability is tracked separately below and surfaces only on the
    # ExecutionPlan, never as a routing-ambiguity signal.
    deterministic_capability_known = intent is not None

    unavailable_reason = UNAVAILABLE_CAPABILITIES.get(intent) if intent is not None else None
    if unavailable_reason is not None:
        # Informational only: recorded in rationale_codes for transparency,
        # but deliberately NOT an AMBIGUITY_CODE, so it can never trigger the
        # fallback classifier - the request was understood perfectly well.
        failures.append(unavailable_reason)

    gate = RoutingGateCriteria(
        single_clear_intent=single_clear_intent,
        required_entities_resolved=required_entities_resolved,
        no_conflicting_intents=no_conflicting_intents,
        deterministic_capability_known=deterministic_capability_known,
    )

    # Only AMBIGUITY failures justify spending a classifier call.
    ambiguity_failures = tuple(code for code in failures if code in AMBIGUITY_CODES)
    requires_llm_fallback = bool(ambiguity_failures)

    decision = RoutingDecision(
        intent=intent,
        entities=entities,
        gate=gate,
        rationale_codes=tuple(dict.fromkeys(rationale + failures)),
        requires_llm_fallback=requires_llm_fallback,
        used_llm_fallback=False,  # Stage 1 never calls Claude
        fallback_reason_codes=tuple(dict.fromkeys(ambiguity_failures)),
    )

    # ---- Plan -------------------------------------------------------------
    if unavailable_reason is not None:
        # Terminal and deterministic: no tools, no specialists, no fabrication.
        plan = ExecutionPlan(
            intent=intent,
            tier=ExecutionTier.TIER_0,
            capability_status=CapabilityStatus.UNAVAILABLE,
            unavailable_reason=unavailable_reason,
            entities=entities,
            requires_llm_fallback=False,
        )
    elif requires_llm_fallback:
        # Ambiguous: Tier 0 is unreachable, and Stage 4 will re-plan after one
        # classifier call. Plan no tools now - we do not know what is needed.
        plan = ExecutionPlan(
            intent=intent,
            tier=ExecutionTier.TIER_1,
            entities=entities,
            requires_llm_fallback=True,
        )
    else:
        tools, specialists, tier = _plan_tools_and_specialists(intent, classification)
        plan = ExecutionPlan(
            intent=intent,
            tier=tier,
            planned_tools=tools,
            planned_specialists=specialists,
            entities=entities,
            requires_llm_fallback=False,
        )

    return RoutingResult(question=question, decision=decision, plan=plan)


__all__ = [
    "MIN_TEAMS",
    "UNAVAILABLE_CAPABILITIES",
    "extract_teams",
    "route_and_plan",
    "UnknownTeam",
]

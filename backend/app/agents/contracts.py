"""Structured contracts for PitchMind's agent layer.

Stage 1 scope: everything needed to *describe* a routing decision and an
execution plan. Nothing here executes anything.

Two deliberate absences:

* **No `confidence: float`.** PitchMind has no calibrated confidence model, so a
  hand-authored float would imply a precision that does not exist. The routing
  gate is four explicit booleans plus machine-readable rationale codes.
* **No free-text reasoning field.** Every explanatory field is a constrained
  enum, so a model's chain-of-thought can never be smuggled into a contract that
  is later serialized to a user.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    """The eleven agreed query families. Deliberately not collapsed: two
    intents that need different execution paths stay separate even when their
    wording overlaps."""

    SIMPLE_FACT = "SIMPLE_FACT"
    HISTORICAL_STATISTICAL = "HISTORICAL_STATISTICAL"
    PREDICTION = "PREDICTION"
    PREDICTION_EXPLANATION = "PREDICTION_EXPLANATION"
    SCORELINE = "SCORELINE"
    LIVE_MATCH = "LIVE_MATCH"
    RESEARCH = "RESEARCH"
    TACTICAL = "TACTICAL"
    COMPLEX_SYNTHESIS = "COMPLEX_SYNTHESIS"
    WHAT_IF = "WHAT_IF"
    SCOUTING = "SCOUTING"


class ExecutionTier(int, Enum):
    """Cost ceiling for a request.

    TIER_0 is zero Claude calls END TO END - routing included. A plan may only
    be TIER_0 when the deterministic gate passed, so a fallback classifier call
    can never hide inside a "free" tier.
    """

    TIER_0 = 0  # deterministic tools + deterministic response
    TIER_1 = 1  # deterministic tools + exactly one bounded synthesis call
    TIER_2 = 2  # SDK orchestrator + at most one specialist
    TIER_3 = 3  # SDK orchestrator + Research/Tactical + conditional critic


class ToolName(str, Enum):
    """Deterministic capabilities that ACTUALLY EXIST in `backend.app.tools`.

    Every member must be a real exported callable - a test asserts this, so a
    planner can never schedule a function that does not exist.
    """

    GET_CURRENT_STANDINGS = "get_current_standings"
    GET_FIXTURES = "get_fixtures"
    GET_LIVE_MATCHES = "get_live_matches"
    GET_LIVE_MATCH_STATE = "get_live_match_state"
    RUN_OUTCOME_PREDICTION = "run_outcome_prediction"
    EXPLAIN_OUTCOME_PREDICTION = "explain_outcome_prediction"
    GET_SCORELINE_PREDICTION = "get_scoreline_prediction"


class SpecialistName(str, Enum):
    """The ONLY genuine Claude specialists.

    Stats, ML, What-If and Scouting are deliberately absent: they are
    deterministic capabilities, and giving them a Claude agent would add cost
    and a fabrication surface for work Python does exactly.
    """

    RESEARCH = "research"
    TACTICAL = "tactical"


class CapabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class RationaleCode(str, Enum):
    """Machine-readable reasons. Constrained on purpose - no free text."""

    # Successful matches
    MATCHED_STANDINGS = "MATCHED_STANDINGS"
    MATCHED_FIXTURES = "MATCHED_FIXTURES"
    MATCHED_LIVE = "MATCHED_LIVE"
    MATCHED_PREDICTION = "MATCHED_PREDICTION"
    MATCHED_EXPLANATION = "MATCHED_EXPLANATION"
    MATCHED_SCORELINE = "MATCHED_SCORELINE"
    MATCHED_RESEARCH = "MATCHED_RESEARCH"
    MATCHED_TACTICAL = "MATCHED_TACTICAL"
    MATCHED_HISTORICAL = "MATCHED_HISTORICAL"
    MATCHED_WHAT_IF = "MATCHED_WHAT_IF"
    MATCHED_SCOUTING = "MATCHED_SCOUTING"
    MATCHED_COMPLEX_SYNTHESIS = "MATCHED_COMPLEX_SYNTHESIS"
    LIVE_REASONING_REQUESTED = "LIVE_REASONING_REQUESTED"

    # Ambiguity failures -> these DO require the Claude fallback classifier
    NO_CLEAR_INTENT = "NO_CLEAR_INTENT"
    MULTIPLE_INTENTS = "MULTIPLE_INTENTS"
    MISSING_REQUIRED_TEAM = "MISSING_REQUIRED_TEAM"
    MISSING_MATCHUP = "MISSING_MATCHUP"
    UNKNOWN_TEAM = "UNKNOWN_TEAM"

    # Terminal failures -> deterministic, NOT ambiguous, so no fallback
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    SCOUTING_DATA_UNAVAILABLE = "SCOUTING_DATA_UNAVAILABLE"
    WHAT_IF_ENGINE_UNAVAILABLE = "WHAT_IF_ENGINE_UNAVAILABLE"


# Failures that mean "the question is ambiguous, ask a cheap classifier".
# Everything else is a confident, terminal answer and must NOT spend a call.
AMBIGUITY_CODES: frozenset[RationaleCode] = frozenset(
    {
        RationaleCode.NO_CLEAR_INTENT,
        RationaleCode.MULTIPLE_INTENTS,
        RationaleCode.MISSING_REQUIRED_TEAM,
        RationaleCode.MISSING_MATCHUP,
        RationaleCode.UNKNOWN_TEAM,
    }
)


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ResolvedEntity(_Base):
    """One club mentioned in the question, resolved through the canonical
    registry. `mentioned_as` is the user's own wording, kept for provenance."""

    canonical_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    mentioned_as: str = Field(min_length=1)
    historical_ml_history_available: bool


class RoutingGateCriteria(_Base):
    """The four boolean criteria. All must hold for deterministic ROUTING.

    Deliberately booleans, not a score: an uncalibrated float would look like a
    measurement when it is really a guess.

    The gate answers one question: "do we understand the user's request well
    enough to route it deterministically?" - NOT "can PitchMind currently
    execute the resulting capability?". Those are different questions with
    different consequences: an unresolved question needs a Claude fallback
    call to disambiguate; a resolved-but-unsupported capability (WHAT_IF,
    SCOUTING today) needs no such call - we already know exactly what was
    asked and that we cannot serve it yet. That distinction lives on
    `ExecutionPlan.capability_status`/`unavailable_reason`, not here.

    `deterministic_capability_known` is therefore True whenever the intent
    itself resolved to a defined routing outcome - which, for every intent in
    `Intent` today, it always does: either a real tool/specialist plan, or the
    deliberate "capability unavailable" plan. It is False only when there is no
    intent to route at all (i.e. `single_clear_intent` is already False), so it
    exists as a distinct, independently-testable criterion for future intents
    that might not yet have ANY defined routing outcome - not as a proxy for
    "is the capability currently executable".
    """

    single_clear_intent: bool
    required_entities_resolved: bool
    no_conflicting_intents: bool
    deterministic_capability_known: bool

    @property
    def passed(self) -> bool:
        return (
            self.single_clear_intent
            and self.required_entities_resolved
            and self.no_conflicting_intents
            and self.deterministic_capability_known
        )


class RoutingDecision(_Base):
    """What the deterministic router concluded.

    `requires_llm_fallback` vs `used_llm_fallback` are deliberately distinct:

    * `requires_llm_fallback` - Stage 4 SHOULD spend one cheap classifier call.
    * `used_llm_fallback`     - a call HAS been made. Always False in Stage 1,
                                because Stage 1 never calls Claude.
    """

    intent: Intent | None
    entities: tuple[ResolvedEntity, ...] = ()
    gate: RoutingGateCriteria
    rationale_codes: tuple[RationaleCode, ...] = ()
    requires_llm_fallback: bool
    used_llm_fallback: bool = False
    fallback_reason_codes: tuple[RationaleCode, ...] = ()


class ExecutionPlan(_Base):
    """The deterministic plan. Nothing in Stage 1 executes it."""

    intent: Intent | None
    tier: ExecutionTier
    planned_tools: tuple[ToolName, ...] = ()
    planned_specialists: tuple[SpecialistName, ...] = ()
    capability_status: CapabilityStatus = CapabilityStatus.AVAILABLE
    unavailable_reason: RationaleCode | None = None
    entities: tuple[ResolvedEntity, ...] = ()
    requires_llm_fallback: bool = False

    @property
    def is_available(self) -> bool:
        return self.capability_status is CapabilityStatus.AVAILABLE


class RoutingResult(_Base):
    """What Stage 1 returns: the decision and the plan it produced."""

    question: str
    decision: RoutingDecision
    plan: ExecutionPlan


__all__ = [
    "AMBIGUITY_CODES",
    "CapabilityStatus",
    "ExecutionPlan",
    "ExecutionTier",
    "Intent",
    "RationaleCode",
    "ResolvedEntity",
    "RoutingDecision",
    "RoutingGateCriteria",
    "RoutingResult",
    "SpecialistName",
    "ToolName",
]

"""Stage 1 agent-layer tests: deterministic routing and execution planning.

Fully offline. No LLM, no network, no credentials. Nothing is executed - these
tests only assert what the router DECIDES and what the planner PLANS.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.agents.contracts import (  # noqa: E402
    AMBIGUITY_CODES,
    CapabilityStatus,
    ExecutionTier,
    Intent,
    RationaleCode,
    SpecialistName,
    ToolName,
)
from backend.app.agents.intents import classify_intent, normalize_question  # noqa: E402
from backend.app.agents.planner import extract_teams, route_and_plan  # noqa: E402
from backend.app.services.football_data.errors import UnknownTeam  # noqa: E402
from backend.app.services.football_data.teams import default_registry  # noqa: E402

AGENTS_DIR = REPO_ROOT / "backend" / "app" / "agents"


def plan_for(question: str, **kwargs):
    return route_and_plan(question, **kwargs).plan


def decision_for(question: str, **kwargs):
    return route_and_plan(question, **kwargs).decision


# --------------------------------------------------------------------------
# 1. All eleven intent families classify
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,expected",
    [
        ("Who is top of the Premier League?", Intent.SIMPLE_FACT),
        ("Arsenal's last five league matches", Intent.HISTORICAL_STATISTICAL),
        ("Predict Arsenal vs Liverpool.", Intent.PREDICTION),
        ("Why does the model favour Arsenal against Liverpool?", Intent.PREDICTION_EXPLANATION),
        ("What's the most likely score for Arsenal vs Liverpool?", Intent.SCORELINE),
        ("What's happening in this match?", Intent.LIVE_MATCH),
        ("What injuries could affect Arsenal's next match?", Intent.RESEARCH),
        ("Why might Arsenal struggle tactically against Liverpool?", Intent.TACTICAL),
        (
            "Analyse Arsenal vs Liverpool considering current injuries, recent reports, "
            "model expectations and tactical matchup.",
            Intent.COMPLEX_SYNTHESIS,
        ),
        ("What if Arsenal's recent PPG signal improved by +0.5?", Intent.WHAT_IF),
        ("Find players stylistically similar to Saka.", Intent.SCOUTING),
    ],
)
def test_each_intent_family_classifies(question, expected):
    assert decision_for(question).intent is expected


def test_all_eleven_intents_are_covered_by_the_taxonomy():
    assert len(Intent) == 11


# --------------------------------------------------------------------------
# 2-6. Named acceptance cases from the specification
# --------------------------------------------------------------------------
def test_simple_fact_standings_is_tier_zero_with_no_specialists():
    result = route_and_plan("Who is top of the Premier League?")
    assert result.plan.tier is ExecutionTier.TIER_0
    assert result.plan.planned_tools == (ToolName.GET_CURRENT_STANDINGS,)
    assert result.plan.planned_specialists == ()
    assert result.decision.requires_llm_fallback is False


def test_team_fixture_resolves_through_registry_and_plans_fixtures():
    result = route_and_plan("Who does Arsenal play next?")
    assert [e.canonical_id for e in result.decision.entities] == ["arsenal"]
    assert ToolName.GET_FIXTURES in result.plan.planned_tools
    assert result.plan.tier is ExecutionTier.TIER_0


def test_prediction_plans_model_tools_and_no_specialist():
    result = route_and_plan("Predict Arsenal vs Liverpool.")
    assert [e.canonical_id for e in result.decision.entities] == ["arsenal", "liverpool"]
    assert ToolName.RUN_OUTCOME_PREDICTION in result.plan.planned_tools
    assert result.plan.planned_specialists == ()
    assert result.plan.tier is ExecutionTier.TIER_1


def test_prediction_explanation_plans_prediction_and_explanation():
    plan = plan_for("Why does the model favour Arsenal against Liverpool?")
    assert ToolName.RUN_OUTCOME_PREDICTION in plan.planned_tools
    assert ToolName.EXPLAIN_OUTCOME_PREDICTION in plan.planned_tools
    assert plan.planned_specialists == ()


def test_scoreline_plans_the_dixon_coles_tool():
    plan = plan_for("What's the most likely score for Arsenal vs Liverpool?")
    assert plan.planned_tools == (ToolName.GET_SCORELINE_PREDICTION,)
    assert plan.planned_specialists == ()


def test_tactical_plans_tactical_specialist_without_research():
    plan = plan_for("Why might Arsenal struggle tactically against Liverpool?")
    assert plan.planned_specialists == (SpecialistName.TACTICAL,)
    assert SpecialistName.RESEARCH not in plan.planned_specialists
    assert plan.tier is ExecutionTier.TIER_2


def test_research_plans_research_specialist():
    plan = plan_for("What injuries could affect Arsenal's next match?")
    assert plan.planned_specialists == (SpecialistName.RESEARCH,)
    assert plan.tier is ExecutionTier.TIER_2


def test_complex_synthesis_plans_both_specialists_at_tier_three():
    plan = plan_for(
        "Analyse Arsenal vs Liverpool considering current injuries, recent reports, "
        "model expectations and tactical matchup."
    )
    assert plan.tier is ExecutionTier.TIER_3
    assert set(plan.planned_specialists) == {SpecialistName.RESEARCH, SpecialistName.TACTICAL}
    assert len(plan.planned_tools) >= 3


def test_live_match_simple_retrieval_is_tier_zero():
    plan = plan_for("What's happening in this match?")
    assert plan.tier is ExecutionTier.TIER_0
    assert plan.planned_tools == (ToolName.GET_LIVE_MATCHES,)
    assert plan.planned_specialists == ()


def test_live_match_reasoning_escalates_to_tier_two_with_tactical():
    plan = plan_for("Why is Liverpool struggling right now?")
    assert plan.tier is ExecutionTier.TIER_2
    assert plan.planned_specialists == (SpecialistName.TACTICAL,)


# --------------------------------------------------------------------------
# 10-11. Unsupported capabilities: no fabrication
# --------------------------------------------------------------------------
def test_what_if_is_routed_but_marked_capability_unavailable():
    """The what-if engine does not exist yet. Route it correctly, then say so -
    do not pretend a tool exists and do not invent a WhatIf agent."""
    result = route_and_plan("What if Arsenal's recent PPG signal improved by +0.5?")
    assert result.decision.intent is Intent.WHAT_IF
    assert result.plan.capability_status is CapabilityStatus.UNAVAILABLE
    assert result.plan.unavailable_reason is RationaleCode.WHAT_IF_ENGINE_UNAVAILABLE
    assert result.plan.planned_tools == ()
    assert result.plan.planned_specialists == ()


def test_scouting_is_capability_unavailable_with_no_specialist_or_tool():
    result = route_and_plan("Find players stylistically similar to Saka.")
    assert result.decision.intent is Intent.SCOUTING
    assert result.plan.capability_status is CapabilityStatus.UNAVAILABLE
    assert result.plan.unavailable_reason is RationaleCode.SCOUTING_DATA_UNAVAILABLE
    assert result.plan.planned_tools == ()
    assert result.plan.planned_specialists == ()


def test_unavailable_capabilities_do_not_spend_a_fallback_call():
    """We know exactly what was asked and that we cannot serve it. Spending a
    classifier call to rediscover that would be pure waste.

    Crucially, the ROUTING gate passes fully here: the request was understood
    perfectly well. Unavailability lives on the ExecutionPlan, never as a
    routing-ambiguity signal - so `gate.passed` is True even though the
    capability itself cannot execute.
    """
    for question in (
        "What if Arsenal's recent PPG signal improved by +0.5?",
        "Find players stylistically similar to Saka.",
    ):
        result = route_and_plan(question)
        assert result.decision.requires_llm_fallback is False
        assert result.decision.gate.deterministic_capability_known is True
        assert result.decision.gate.passed is True
        assert result.plan.capability_status is CapabilityStatus.UNAVAILABLE


# --------------------------------------------------------------------------
# 12-13. Entity resolution
# --------------------------------------------------------------------------
def test_current_only_club_resolves_and_reports_no_ml_history():
    entities = extract_teams("Coventry City vs Arsenal", default_registry())
    coventry = next(e for e in entities if e.canonical_id == "coventry_city")
    assert coventry.historical_ml_history_available is False
    arsenal = next(e for e in entities if e.canonical_id == "arsenal")
    assert arsenal.historical_ml_history_available is True


def test_explicitly_supplied_unknown_team_preserves_unknown_team_error():
    with pytest.raises(UnknownTeam):
        route_and_plan("Predict this match", teams=["Wrexham", "Arsenal"])


def test_unrecognised_club_in_free_text_fails_the_entity_gate_rather_than_guessing():
    """Free text cannot distinguish 'unknown club' from 'no club mentioned', so
    it fails the entity gate. It must never fuzzy-match onto a real club."""
    decision = decision_for("Predict Wrexham vs Arsenal")
    assert [e.canonical_id for e in decision.entities] == ["arsenal"]
    assert decision.gate.required_entities_resolved is False
    assert decision.requires_llm_fallback is True


def test_alias_and_possessive_forms_resolve():
    registry = default_registry()
    assert [e.canonical_id for e in extract_teams("Arsenal's next match", registry)] == ["arsenal"]
    assert [e.canonical_id for e in extract_teams("Man Utd vs Spurs", registry)] == [
        "man_united",
        "tottenham",
    ]
    assert [e.canonical_id for e in extract_teams("Nott'm Forest at home", registry)] == [
        "nottingham_forest"
    ]


def test_longest_alias_wins_and_tokens_are_not_reused():
    registry = default_registry()
    assert [e.canonical_id for e in extract_teams("Manchester United vs Manchester City", registry)] == [
        "man_united",
        "man_city",
    ]


def test_entity_order_follows_the_question():
    registry = default_registry()
    assert [e.canonical_id for e in extract_teams("Liverpool vs Arsenal", registry)] == [
        "liverpool",
        "arsenal",
    ]


def test_missing_matchup_fails_rather_than_inventing_teams():
    decision = decision_for("What's the most likely score?")
    assert decision.intent is Intent.SCORELINE
    assert decision.gate.required_entities_resolved is False
    assert RationaleCode.MISSING_MATCHUP in decision.fallback_reason_codes
    assert decision.entities == ()


def test_alias_index_is_derived_from_the_registry_not_hardcoded():
    """Behavioural proof that there is no second alias table: a club added to a
    custom registry must immediately be extractable, which is only possible if
    the scan index is built from the registry at runtime.

    Stronger than searching the source for club names, which would also trip on
    a comment that merely mentions one.
    """
    from backend.app.services.football_data.teams import CanonicalTeam, TeamRegistry

    registry = TeamRegistry()
    assert extract_teams("Testburgh United at home", registry) == ()

    registry.register(
        CanonicalTeam(
            canonical_id="testburgh",
            canonical_name="Testburgh United",
            aliases=("Testburgh",),
            historical_ml_history_available=False,
        )
    )
    entities = extract_teams("Testburgh United at home", registry)
    assert [e.canonical_id for e in entities] == ["testburgh"]
    assert entities[0].historical_ml_history_available is False


# --------------------------------------------------------------------------
# 14-16. Gate behaviour
# --------------------------------------------------------------------------
def test_ambiguous_query_requires_fallback_but_has_not_used_one():
    decision = decision_for("Tell me about football.")
    assert decision.intent is None
    assert decision.requires_llm_fallback is True
    assert decision.used_llm_fallback is False
    assert RationaleCode.NO_CLEAR_INTENT in decision.fallback_reason_codes


def test_conflicting_intents_fail_rather_than_arbitrarily_choosing():
    decision = decision_for("Predict Arsenal vs Liverpool and what injuries do they have?")
    assert decision.intent is None
    assert decision.gate.no_conflicting_intents is False
    assert RationaleCode.MULTIPLE_INTENTS in decision.fallback_reason_codes


def test_three_ambiguity_gate_criteria_can_each_fail_independently():
    """Criteria 1-3 answer 'do we understand this well enough to route it',
    and each has a real question that fails only that one criterion."""
    # (1) single_clear_intent
    assert decision_for("Tell me about football.").gate.single_clear_intent is False
    # (2) required_entities_resolved
    assert decision_for("What's the most likely score?").gate.required_entities_resolved is False
    # (3) no_conflicting_intents
    conflict = decision_for("Predict Arsenal vs Liverpool and what injuries do they have?")
    assert conflict.gate.no_conflicting_intents is False


def test_capability_known_reflects_routing_understanding_not_executability():
    """`deterministic_capability_known` answers "do we know the deterministic
    ROUTE for this intent", not "can PitchMind currently execute it".

    Under Stage 1's current Intent taxonomy every resolvable intent - including
    WHAT_IF and SCOUTING - has a defined routing outcome (a real plan, or the
    deliberate "capability unavailable" plan), so this criterion is currently
    True whenever `single_clear_intent` is True: it has no real-question
    scenario where it fails independently of criterion 1 TODAY. That is
    expected, not a gap - it exists as a distinct, separately-named criterion
    for a future intent that might have no defined route at all, and unit-level
    independence is verified directly on the schema below.
    """
    for question in (
        "Find players stylistically similar to Saka.",  # SCOUTING: unavailable, but routed
        "What if Arsenal's recent PPG signal improved by +0.5?",  # WHAT_IF: unavailable, but routed
        "Predict Arsenal vs Liverpool.",  # ordinary available intent
    ):
        gate = decision_for(question).gate
        assert gate.single_clear_intent is True
        assert gate.deterministic_capability_known is True

    # Schema-level independence: the field is genuinely load-bearing in
    # `passed`, not decorative - constructing it False (with everything else
    # True) must fail the gate, proving it is checked, not merely carried.
    from backend.app.agents.contracts import RoutingGateCriteria

    hypothetically_unrouted = RoutingGateCriteria(
        single_clear_intent=True,
        required_entities_resolved=True,
        no_conflicting_intents=True,
        deterministic_capability_known=False,
    )
    assert hypothetically_unrouted.passed is False


def test_gate_passed_property_requires_all_four():
    passing = decision_for("Who is top of the Premier League?").gate
    assert passing.passed is True
    failing = decision_for("Tell me about football.").gate
    assert failing.passed is False


def test_subsumed_fact_reference_is_not_treated_as_a_conflict():
    """'Arsenal's next match' inside a research question is scoping context,
    not a rival intent."""
    decision = decision_for("What injuries could affect Arsenal's next match?")
    assert decision.intent is Intent.RESEARCH
    assert decision.gate.no_conflicting_intents is True


# --------------------------------------------------------------------------
# 17-18. Locked invariants
# --------------------------------------------------------------------------
TIER_0_QUESTIONS = [
    "Who is top of the Premier League?",
    "Who does Arsenal play next?",
    "What's happening in this match?",
    "What if Arsenal's recent PPG signal improved by +0.5?",
    "Find players stylistically similar to Saka.",
]


@pytest.mark.parametrize("question", TIER_0_QUESTIONS)
def test_tier_zero_invariant(question):
    """tier == 0 implies zero Claude calls END TO END - routing included."""
    result = route_and_plan(question)
    if result.plan.tier is ExecutionTier.TIER_0:
        assert result.decision.requires_llm_fallback is False
        assert result.decision.used_llm_fallback is False
        assert result.plan.requires_llm_fallback is False
        assert result.plan.planned_specialists == ()


def test_a_plan_requiring_fallback_is_never_tier_zero():
    for question in (
        "Tell me about football.",
        "What's the most likely score?",
        "Predict Arsenal vs Liverpool and what injuries do they have?",
    ):
        plan = plan_for(question)
        assert plan.requires_llm_fallback is True
        assert plan.tier is not ExecutionTier.TIER_0


ALL_QUESTIONS = TIER_0_QUESTIONS + [
    "Arsenal's last five league matches",
    "Predict Arsenal vs Liverpool.",
    "Why does the model favour Arsenal against Liverpool?",
    "What's the most likely score for Arsenal vs Liverpool?",
    "Why is Liverpool struggling right now?",
    "What injuries could affect Arsenal's next match?",
    "Why might Arsenal struggle tactically against Liverpool?",
    "Analyse Arsenal vs Liverpool considering injuries, reports, model and tactical matchup.",
    "Tell me about football.",
]


def test_only_research_and_tactical_specialists_exist():
    assert {s.value for s in SpecialistName} == {"research", "tactical"}
    for forbidden in ("stats", "ml", "what_if", "whatif", "scouting"):
        assert forbidden not in {s.value for s in SpecialistName}


@pytest.mark.parametrize("question", ALL_QUESTIONS)
def test_no_forbidden_specialist_is_ever_planned(question):
    for specialist in plan_for(question).planned_specialists:
        assert specialist in (SpecialistName.RESEARCH, SpecialistName.TACTICAL)


def test_every_planned_tool_is_a_real_callable():
    """A planner must never schedule a function that does not exist."""
    from backend.app import tools

    for tool_name in ToolName:
        assert hasattr(tools, tool_name.value), f"{tool_name.value} is not exported by tools"
        assert callable(getattr(tools, tool_name.value))


@pytest.mark.parametrize("question", ALL_QUESTIONS)
def test_planned_tools_for_every_question_are_real(question):
    from backend.app import tools

    for tool_name in plan_for(question).planned_tools:
        assert callable(getattr(tools, tool_name.value))


# --------------------------------------------------------------------------
# 19-20. Determinism and serialization
# --------------------------------------------------------------------------
@pytest.mark.parametrize("question", ALL_QUESTIONS)
def test_routing_is_deterministic(question):
    first = route_and_plan(question)
    second = route_and_plan(question)
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize("question", ALL_QUESTIONS)
def test_contracts_serialize_cleanly_to_json(question):
    payload = json.loads(route_and_plan(question).model_dump_json())
    assert set(payload) == {"question", "decision", "plan"}
    assert "confidence" not in json.dumps(payload), "no uncalibrated confidence score"


def test_contracts_forbid_extra_fields():
    from pydantic import ValidationError

    from backend.app.agents.contracts import RoutingGateCriteria

    with pytest.raises(ValidationError):
        RoutingGateCriteria(
            single_clear_intent=True,
            required_entities_resolved=True,
            no_conflicting_intents=True,
            deterministic_capability_known=True,
            sneaky_reasoning="chain of thought",
        )


def test_no_contract_exposes_a_free_text_reasoning_field():
    """Rationale is constrained enums, so model reasoning cannot leak."""
    source = (AGENTS_DIR / "contracts.py").read_text()
    for banned in ("reasoning:", "thoughts:", "chain_of_thought", "explanation_text"):
        assert banned not in source


def test_fallback_reason_codes_are_always_ambiguity_codes():
    for question in ALL_QUESTIONS:
        for code in decision_for(question).fallback_reason_codes:
            assert code in AMBIGUITY_CODES


def test_normalize_question_is_stable():
    assert normalize_question("  Who is TOP of the Premier League??  ") == (
        "who is top of the premier league"
    )


def test_classifier_returns_no_intent_rather_than_guessing():
    classification = classify_intent("qwertyuiop")
    assert classification.intent is None
    assert classification.failure_code is RationaleCode.NO_CLEAR_INTENT


# --------------------------------------------------------------------------
# 21-23. Structural safety
# --------------------------------------------------------------------------
def _stage1_modules() -> list[Path]:
    return sorted(AGENTS_DIR.glob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_stage1_imports_no_ml_training_or_artifact_machinery():
    forbidden_prefixes = (
        "backend.app.ml.training",
        "backend.app.ml.datasets",
        "backend.app.ml.feature_engineering",
        "backend.app.ml.score_models",
        "backend.app.ml.score_model_artifact",
        "backend.app.ml.explanations",
        "backend.app.ml.baselines",
        "backend.app.ml.calibration",
    )
    for module_path in _stage1_modules():
        for imported in _imported_modules(module_path):
            assert not imported.startswith(forbidden_prefixes), f"{module_path.name}: {imported}"


def test_stage1_imports_no_llm_or_network_libraries():
    """Direct imports only; a transitive package __init__ is out of scope."""
    forbidden = ("anthropic", "claude", "claude_agent_sdk", "httpx", "requests", "urllib", "socket",
                 "openai", "langchain", "langgraph", "crewai", "autogen")
    for module_path in _stage1_modules():
        for imported in _imported_modules(module_path):
            root = imported.split(".")[0].lower()
            assert root not in forbidden, f"{module_path.name} imports {imported}"


def test_stage1_never_references_the_sealed_season():
    for module_path in _stage1_modules():
        assert "2025_26" not in module_path.read_text(), module_path.name


def test_stage1_executes_nothing():
    """Stage 1 plans; it must not call any tool or execute a model."""
    forbidden_calls = {
        "run_outcome_prediction", "explain_outcome_prediction", "get_scoreline_prediction",
        "get_current_standings", "get_fixtures", "get_live_matches", "get_live_match_state",
        "predict_proba", "fit", "query",
    }
    for module_path in _stage1_modules():
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                assert name not in forbidden_calls, f"{module_path.name} calls {name}"


def test_stage1_makes_no_network_calls(monkeypatch):
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError("Stage 1 routing must not touch the network")

    monkeypatch.setattr(socket, "socket", _forbidden)
    for question in ALL_QUESTIONS:
        route_and_plan(question)


def test_stage1_package_contains_only_the_approved_modules():
    assert {p.name for p in _stage1_modules()} == {
        "__init__.py",
        "contracts.py",
        "intents.py",
        "planner.py",
    }

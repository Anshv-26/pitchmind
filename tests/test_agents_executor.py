"""Stage 2 tests: ToolExecutor and EvidenceLedger.

Fully offline. Providers are mocked httpx transports or the deterministic
replay provider; models are the real local frozen artifacts. No network, no
credentials, no LLM.
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.agents.contracts import (  # noqa: E402
    EvidenceKind,
    ToolErrorCode,
    ToolName,
)
from backend.app.agents.executor import (  # noqa: E402
    TOOL_REGISTRY,
    EvidenceLedger,
    ToolExecutor,
)
from backend.app.core.config import FootballDataSettings  # noqa: E402
from backend.app.services.football_data.football_data_org import (  # noqa: E402
    FootballDataOrgProvider,
)
from backend.app.services.football_data.replay import ReplayProvider  # noqa: E402
from backend.app.services.football_data.service import FootballDataService  # noqa: E402

AGENTS_DIR = REPO_ROOT / "backend" / "app" / "agents"
T0 = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)

PREDICTION_ARGS = {"elo_diff": 85.0, "diff_ewma_ppg": 0.42, "diff_ewma_sot_diff": 1.3}
SCORELINE_ARGS = {"home_team": "Arsenal", "away_team": "Liverpool"}

FD_ORG_MATCHES = {
    "matches": [
        {
            "id": 501,
            "utcDate": "2026-09-05T14:00:00Z",
            "status": "FINISHED",
            "matchday": 4,
            "homeTeam": {"id": 57, "name": "Arsenal FC"},
            "awayTeam": {"id": 64, "name": "Liverpool FC"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        }
    ]
}

FD_ORG_STANDINGS = {
    "standings": [
        {
            "type": "TOTAL",
            "table": [
                {
                    "position": 1,
                    "team": {"id": 57, "name": "Arsenal FC"},
                    "playedGames": 4, "won": 3, "draw": 1, "lost": 0,
                    "goalsFor": 9, "goalsAgainst": 2, "goalDifference": 7, "points": 10,
                }
            ],
        }
    ]
}


def make_settings(**overrides) -> FootballDataSettings:
    base = dict(
        football_data_org_key="test-key",
        api_football_key=None,
        live_ttl_seconds=300,
        standings_ttl_seconds=1800,
        fixtures_ttl_seconds=1800,
        metadata_ttl_seconds=21600,
        http_connect_timeout=5.0,
        http_read_timeout=10.0,
    )
    base.update(overrides)
    return FootballDataSettings(**base)


def counting_provider(payload, status_code: int = 200, headers: dict | None = None):
    """Provider over a mocked transport that records every HTTP call."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(status_code, json=payload, headers=headers or {})

    provider = FootballDataOrgProvider(
        make_settings(), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return provider, calls


def make_executor(payload=FD_ORG_MATCHES, *, status_code=200, headers=None, live_step=3, now=T0):
    provider, calls = counting_provider(payload, status_code=status_code, headers=headers)
    service = FootballDataService(
        current_provider=provider,
        live_provider=ReplayProvider(step=live_step),
        settings=make_settings(),
    )
    return ToolExecutor(service=service, now=now), calls


# --------------------------------------------------------------------------
# 1-2. Dispatch reaches the real application tools
# --------------------------------------------------------------------------
def test_every_registry_entry_maps_to_an_approved_tool_name():
    assert set(TOOL_REGISTRY) == set(ToolName)


def test_all_service_backed_tools_dispatch():
    for tool_name, args, payload in [
        (ToolName.GET_CURRENT_STANDINGS, {}, FD_ORG_STANDINGS),
        (ToolName.GET_FIXTURES, {"team": "Arsenal"}, FD_ORG_MATCHES),
        (ToolName.GET_LIVE_MATCHES, {}, FD_ORG_MATCHES),
        (ToolName.GET_LIVE_MATCH_STATE, {"provider_fixture_id": "replay-1"}, FD_ORG_MATCHES),
    ]:
        executor, _ = make_executor(payload)
        result = executor.execute(tool_name, args)
        assert result.ok is True, f"{tool_name} failed: {result.failure}"
        assert result.evidence is not None


def test_all_model_backed_tools_dispatch():
    executor = ToolExecutor()
    for tool_name, args in [
        (ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS),
        (ToolName.EXPLAIN_OUTCOME_PREDICTION, PREDICTION_ARGS),
        (ToolName.GET_SCORELINE_PREDICTION, SCORELINE_ARGS),
    ]:
        result = executor.execute(tool_name, args)
        assert result.ok is True, f"{tool_name} failed: {result.failure}"


def test_executor_calls_the_real_tools_rather_than_reimplementing_them(monkeypatch):
    """Business logic must live in backend/app/tools, not be duplicated here."""
    from backend.app.tools import prediction_tools

    called: list[str] = []
    original = prediction_tools.run_outcome_prediction

    def spy(request, **kwargs):
        called.append("run_outcome_prediction")
        return original(request, **kwargs)

    monkeypatch.setattr(prediction_tools, "run_outcome_prediction", spy)
    ToolExecutor().execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS)
    assert called == ["run_outcome_prediction"]


def test_no_dynamic_attribute_dispatch_from_strings():
    """Dispatch must go through the explicit registry, never dynamic lookup on
    a caller-supplied name.

    Checked at AST level over EVERY call site rather than by searching the
    source text: the module's own docstring legitimately describes the
    `getattr(module, name)` pattern it refuses to use, so a prose grep would
    flag the very documentation of the guarantee.
    """
    source = (AGENTS_DIR / "executor.py").read_text()
    tree = ast.parse(source)

    # No arbitrary code execution or dynamic imports anywhere.
    assert "importlib" not in {m.split(".")[0] for m in _imported_modules(AGENTS_DIR / "executor.py")}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert getattr(node.func, "id", None) not in {"eval", "exec", "__import__"}

    # Every getattr must name its attribute either as a string literal or via
    # our OWN registry spec - never from caller-supplied input.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr"):
            continue
        attribute_arg = node.args[1]
        if isinstance(attribute_arg, ast.Constant) and isinstance(attribute_arg.value, str):
            continue  # literal attribute name: safe
        assert isinstance(attribute_arg, ast.Attribute), (
            f"line {node.lineno}: getattr attribute name must be a literal or a "
            f"registry-spec field, got {ast.unparse(attribute_arg)}"
        )
        # e.g. `spec.provenance_field` - sourced from the hand-written registry.
        assert ast.unparse(attribute_arg).startswith("spec."), (
            f"line {node.lineno}: unexpected dynamic attribute source "
            f"{ast.unparse(attribute_arg)}"
        )

    # Tool lookup genuinely goes through the explicit registry.
    assert "TOOL_REGISTRY.get(" in source


# --------------------------------------------------------------------------
# 3-6. Evidence ids and request-scoped deduplication
# --------------------------------------------------------------------------
def test_evidence_ids_increment_deterministically():
    executor = ToolExecutor()
    first = executor.execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS)
    second = executor.execute(ToolName.GET_SCORELINE_PREDICTION, SCORELINE_ARGS)
    assert first.evidence.evidence_id == "E1"
    assert second.evidence.evidence_id == "E2"
    assert [item.evidence_id for item in executor.ledger.items] == ["E1", "E2"]


def test_identical_repeat_reuses_the_same_evidence_and_runs_the_tool_once():
    """The central Stage 2 economics claim: two specialists asking the same
    question cost one execution."""
    executor, calls = make_executor(FD_ORG_MATCHES)
    first = executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal"})
    second = executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal"})

    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.evidence.evidence_id == first.evidence.evidence_id
    assert len(executor.ledger) == 1
    assert len(calls) == 1, "underlying provider must be hit exactly once"


def test_different_arguments_produce_distinct_evidence():
    executor, _ = make_executor(FD_ORG_MATCHES)
    first = executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal"})
    second = executor.execute(ToolName.GET_FIXTURES, {"team": "Liverpool"})
    assert first.evidence.evidence_id != second.evidence.evidence_id
    assert len(executor.ledger) == 2


def test_argument_ordering_does_not_defeat_deduplication():
    executor = ToolExecutor()
    first = executor.execute(
        ToolName.RUN_OUTCOME_PREDICTION,
        {"elo_diff": 85.0, "diff_ewma_ppg": 0.42, "diff_ewma_sot_diff": 1.3},
    )
    second = executor.execute(
        ToolName.RUN_OUTCOME_PREDICTION,
        {"diff_ewma_sot_diff": 1.3, "elo_diff": 85.0, "diff_ewma_ppg": 0.42},
    )
    assert second.deduplicated is True
    assert second.evidence.evidence_id == first.evidence.evidence_id


def test_team_aliases_collapse_to_the_same_dedup_key():
    """'Arsenal' and 'Arsenal FC' are the same question, so they must not cost
    two executions."""
    executor, calls = make_executor(FD_ORG_MATCHES)
    executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal"})
    repeat = executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal FC"})
    assert repeat.deduplicated is True
    assert len(calls) == 1


def test_dedup_key_is_stable_and_not_process_hash_based():
    executor = ToolExecutor()
    result = executor.execute(ToolName.GET_SCORELINE_PREDICTION, SCORELINE_ARGS)
    key = result.evidence.dedup_key
    assert key.startswith("get_scoreline_prediction:")
    assert "arsenal" in key and "liverpool" in key  # canonical ids, human-readable
    # A fresh executor derives the identical key - no per-process salt.
    assert ToolExecutor().execute(ToolName.GET_SCORELINE_PREDICTION, SCORELINE_ARGS).evidence.dedup_key == key


def test_deduplication_is_request_scoped_not_global():
    """A second executor is a second request and must execute again."""
    first_executor, first_calls = make_executor(FD_ORG_MATCHES)
    first_executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal"})
    second_executor, second_calls = make_executor(FD_ORG_MATCHES)
    second_executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal"})
    assert len(first_calls) == 1 and len(second_calls) == 1


# --------------------------------------------------------------------------
# 7-8. Refusal before unsafe execution
# --------------------------------------------------------------------------
def test_unknown_tool_name_fails_deterministically():
    result = ToolExecutor().execute("definitely_not_a_tool")
    assert result.ok is False
    assert result.failure.error_code is ToolErrorCode.UNKNOWN_TOOL
    assert result.evidence is None


def test_invalid_arguments_fail_before_execution(monkeypatch):
    from backend.app.tools import prediction_tools

    def _forbidden(*args, **kwargs):
        raise AssertionError("tool must not run with invalid arguments")

    monkeypatch.setattr(prediction_tools, "run_outcome_prediction", _forbidden)
    result = ToolExecutor().execute(ToolName.RUN_OUTCOME_PREDICTION, {"bogus_feature": 1.0})
    assert result.ok is False
    assert result.failure.error_code is ToolErrorCode.INVALID_ARGUMENTS


def test_extra_arguments_are_rejected():
    result = ToolExecutor().execute(
        ToolName.RUN_OUTCOME_PREDICTION, {**PREDICTION_ARGS, "possession": 60.0}
    )
    assert result.failure.error_code is ToolErrorCode.INVALID_ARGUMENTS


def test_service_backed_tool_without_service_fails_typed():
    result = ToolExecutor().execute(ToolName.GET_CURRENT_STANDINGS)
    assert result.ok is False
    assert result.failure.error_code is ToolErrorCode.SERVICE_NOT_CONFIGURED


# --------------------------------------------------------------------------
# 9-12. Numeric fidelity and provenance preservation
# --------------------------------------------------------------------------
def test_prediction_evidence_preserves_exact_probabilities():
    """Stage 3's grounding validator depends on these being byte-identical to
    the tool's own output - no re-derivation, no rounding."""
    from backend.app.tools import prediction_tools
    from backend.app.tools.schemas import OutcomePredictionRequest

    expected = prediction_tools.run_outcome_prediction(OutcomePredictionRequest(**PREDICTION_ARGS))
    evidence = ToolExecutor().execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS).evidence

    assert evidence.payload["home_win_probability"] == expected.home_win_probability
    assert evidence.payload["draw_probability"] == expected.draw_probability
    assert evidence.payload["away_win_probability"] == expected.away_win_probability
    assert evidence.payload["predicted_class"] == expected.predicted_class
    total = sum(
        evidence.payload[k] for k in ("home_win_probability", "draw_probability", "away_win_probability")
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_scoreline_evidence_preserves_expected_goals_and_score_probabilities():
    from backend.app.tools import scoreline_tools
    from backend.app.tools.schemas import ScorelinePredictionRequest

    expected = scoreline_tools.get_scoreline_prediction(ScorelinePredictionRequest(**SCORELINE_ARGS))
    evidence = ToolExecutor().execute(ToolName.GET_SCORELINE_PREDICTION, SCORELINE_ARGS).evidence

    assert evidence.payload["expected_home_goals"] == expected.expected_home_goals
    assert evidence.payload["expected_away_goals"] == expected.expected_away_goals
    assert evidence.payload["most_likely_scoreline"] == expected.most_likely_scoreline
    assert len(evidence.payload["top_scorelines"]) == len(expected.top_scorelines)
    for got, want in zip(evidence.payload["top_scorelines"], expected.top_scorelines):
        assert got["probability"] == want.probability


def test_explanation_evidence_preserves_structured_contributions():
    evidence = ToolExecutor().execute(ToolName.EXPLAIN_OUTCOME_PREDICTION, PREDICTION_ARGS).evidence
    features = evidence.payload["features"]
    assert [f["name"] for f in features] == ["elo_diff", "diff_ewma_ppg", "diff_ewma_sot_diff"]
    # Structured, not flattened into prose.
    assert isinstance(features[0]["grouped_logit_contribution"], dict)
    assert isinstance(features[0]["probability_sensitivity"], dict)
    assert evidence.evidence_kind is EvidenceKind.MODEL_EXPLANATION


def test_model_provenance_is_preserved_verbatim():
    evidence = ToolExecutor().execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS).evidence
    provenance = evidence.payload["model_provenance"]
    assert provenance["model_id"] == "baseline_strength_trio"
    assert provenance["training_cutoff_season"] == "2024_25"
    assert provenance["sealed_final_test_completed"] is False
    assert evidence.source_kind == "LOCAL_MODEL"


def test_score_model_provenance_is_preserved():
    evidence = ToolExecutor().execute(ToolName.GET_SCORELINE_PREDICTION, SCORELINE_ARGS).evidence
    provenance = evidence.payload["model_provenance"]
    assert provenance["model_id"] == "dixon_coles_l2_decay"
    assert provenance["training_cutoff_season"] == "2024_25"
    assert provenance["sealed_final_test_completed"] is False


def test_standings_evidence_preserves_table_values_and_provenance():
    executor, _ = make_executor(FD_ORG_STANDINGS)
    evidence = executor.execute(ToolName.GET_CURRENT_STANDINGS).evidence
    row = evidence.payload["standings"][0]
    assert row["position"] == 1 and row["points"] == 10
    assert row["team"]["canonical_id"] == "arsenal"
    assert evidence.payload["provenance"]["provider"] == "football_data_org"
    assert evidence.source_kind == "REAL_PROVIDER"


def test_fixtures_evidence_preserves_provenance():
    executor, _ = make_executor(FD_ORG_MATCHES)
    evidence = executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal"}).evidence
    assert evidence.payload["provenance"]["provider"] == "football_data_org"
    assert evidence.payload["fixtures"][0]["home_team"]["canonical_id"] == "arsenal"
    assert evidence.evidence_kind is EvidenceKind.CURRENT_DATA


def test_no_raw_provider_json_shape_leaks_into_evidence():
    """Evidence carries normalized application schemas, not vendor payloads."""
    executor, _ = make_executor(FD_ORG_MATCHES)
    evidence = executor.execute(ToolName.GET_FIXTURES, {"team": "Arsenal"}).evidence
    serialized = json.dumps(evidence.payload)
    for vendor_key in ("utcDate", "homeTeam", "awayTeam", "playedGames", "fullTime"):
        assert vendor_key not in serialized


# --------------------------------------------------------------------------
# 13-14. Replay and staleness provenance
# --------------------------------------------------------------------------
def test_replay_live_evidence_is_clearly_marked_replay():
    """No later stage may mistake simulated live data for real football."""
    executor, _ = make_executor(FD_ORG_MATCHES)
    evidence = executor.execute(ToolName.GET_LIVE_MATCHES).evidence
    assert evidence.source_kind == "REPLAY"
    assert evidence.is_replay is True
    assert evidence.payload["provenance"]["provider"] == "replay"
    assert evidence.payload["matches"][0]["provider"] == "replay"


def test_live_match_state_evidence_keeps_replay_provenance():
    executor, _ = make_executor(FD_ORG_MATCHES)
    evidence = executor.execute(
        ToolName.GET_LIVE_MATCH_STATE, {"provider_fixture_id": "replay-1"}
    ).evidence
    assert evidence.is_replay is True
    assert evidence.payload["match"]["freshness"]["provider"] == "replay"


def test_real_provider_evidence_is_not_marked_replay():
    executor, _ = make_executor(FD_ORG_STANDINGS)
    evidence = executor.execute(ToolName.GET_CURRENT_STANDINGS).evidence
    assert evidence.is_replay is False


def test_stale_provider_evidence_stays_marked_stale():
    """A refresh failure serves the last good value flagged stale; the flag
    must survive into evidence."""
    from backend.app.services.football_data.errors import ProviderUnavailable

    provider, _ = counting_provider(FD_ORG_STANDINGS)
    service = FootballDataService(
        current_provider=provider,
        live_provider=ReplayProvider(step=3),
        settings=make_settings(),
    )
    executor = ToolExecutor(service=service, now=T0)
    fresh = executor.execute(ToolName.GET_CURRENT_STANDINGS).evidence
    assert fresh.is_stale is False

    def _broken(*args, **kwargs):
        raise ProviderUnavailable("simulated outage")

    provider.get_standings = _broken
    later = ToolExecutor(service=service, now=T0 + timedelta(seconds=3600))
    stale = later.execute(ToolName.GET_CURRENT_STANDINGS).evidence
    assert stale.is_stale is True
    assert stale.payload["provenance"]["is_stale"] is True


# --------------------------------------------------------------------------
# 15-17. Typed failure behaviour
# --------------------------------------------------------------------------
def test_provider_error_creates_no_successful_evidence():
    executor, _ = make_executor({}, status_code=503)
    result = executor.execute(ToolName.GET_CURRENT_STANDINGS)
    assert result.ok is False
    assert result.evidence is None
    assert len(executor.ledger) == 0, "a failure must never enter the evidence ledger"
    assert len(executor.ledger.failures) == 1


@pytest.mark.parametrize(
    "status_code,headers,expected",
    [
        (503, None, ToolErrorCode.PROVIDER_UNAVAILABLE),
        (429, {"Retry-After": "42"}, ToolErrorCode.PROVIDER_RATE_LIMITED),
    ],
)
def test_provider_failures_keep_distinct_typed_codes(status_code, headers, expected):
    """Rate limiting must not collapse into a generic outage - it subclasses
    ProviderUnavailable, so ordering in the mapping matters."""
    executor, _ = make_executor({}, status_code=status_code, headers=headers)
    result = executor.execute(ToolName.GET_CURRENT_STANDINGS)
    assert result.failure.error_code is expected


def test_malformed_provider_payload_is_typed():
    executor, _ = make_executor({"unexpected": True})
    result = executor.execute(ToolName.GET_FIXTURES, {})
    assert result.failure.error_code is ToolErrorCode.MALFORMED_PROVIDER_PAYLOAD


def test_unknown_team_remains_typed_and_distinguishable():
    result = ToolExecutor().execute(
        ToolName.GET_SCORELINE_PREDICTION, {"home_team": "Wrexham", "away_team": "Arsenal"}
    )
    assert result.ok is False
    assert result.failure.error_code is ToolErrorCode.UNKNOWN_TEAM


def test_team_not_in_score_model_is_distinct_from_unknown_team():
    """Coventry is a real known club with no fitted Dixon-Coles parameters -
    a different failure from an unrecognised club."""
    result = ToolExecutor().execute(
        ToolName.GET_SCORELINE_PREDICTION, {"home_team": "Coventry City", "away_team": "Arsenal"}
    )
    assert result.failure.error_code is ToolErrorCode.TEAM_NOT_IN_SCORE_MODEL


def test_missing_model_artifact_is_typed(monkeypatch):
    from backend.app.tools import prediction_tools

    def _missing(*args, **kwargs):
        raise prediction_tools.ModelArtifactUnavailable("artifact missing")

    monkeypatch.setattr(prediction_tools, "get_artifact", _missing)
    result = ToolExecutor().execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS)
    assert result.failure.error_code is ToolErrorCode.MODEL_ARTIFACT_UNAVAILABLE


def test_unsupported_capability_is_typed():
    """football-data.org genuinely cannot serve live data."""
    provider, _ = counting_provider(FD_ORG_MATCHES)
    service = FootballDataService(
        current_provider=provider, live_provider=provider, settings=make_settings()
    )
    result = ToolExecutor(service=service).execute(ToolName.GET_LIVE_MATCHES)
    assert result.failure.error_code is ToolErrorCode.UNSUPPORTED_CAPABILITY


def test_unexpected_exceptions_propagate_rather_than_being_masked(monkeypatch):
    """A programming error must fail loudly, not become a tidy ToolFailure."""
    from backend.app.tools import prediction_tools

    def _bug(*args, **kwargs):
        raise ZeroDivisionError("a real bug")

    monkeypatch.setattr(prediction_tools, "run_outcome_prediction", _bug)
    with pytest.raises(ZeroDivisionError):
        ToolExecutor().execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS)


def test_failure_messages_expose_no_paths_or_stack_traces():
    executor, _ = make_executor({}, status_code=503)
    message = executor.execute(ToolName.GET_CURRENT_STANDINGS).failure.message
    assert "Traceback" not in message
    assert str(REPO_ROOT) not in message


# --------------------------------------------------------------------------
# 18-19. Secrets and serialization
# --------------------------------------------------------------------------
def test_no_credentials_enter_the_evidence_ledger(monkeypatch):
    secret = "super-secret-token-value"
    monkeypatch.setenv("PITCHMIND_FOOTBALL_DATA_ORG_KEY", secret)

    provider, _ = counting_provider(FD_ORG_STANDINGS)
    service = FootballDataService(
        current_provider=provider,
        live_provider=ReplayProvider(step=3),
        settings=make_settings(football_data_org_key=secret),
    )
    executor = ToolExecutor(service=service, now=T0)
    executor.execute(ToolName.GET_CURRENT_STANDINGS)
    executor.execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS)
    assert secret not in executor.ledger.to_json()


def test_credentials_are_scrubbed_from_failure_messages(monkeypatch):
    secret = "leaky-token-abc123"
    monkeypatch.setenv("PITCHMIND_FOOTBALL_DATA_ORG_KEY", secret)

    provider, _ = counting_provider(FD_ORG_STANDINGS)

    def _leaky(*args, **kwargs):
        from backend.app.services.football_data.errors import ProviderUnavailable

        raise ProviderUnavailable(f"auth failed using {secret}")

    provider.get_standings = _leaky
    service = FootballDataService(
        current_provider=provider, live_provider=ReplayProvider(), settings=make_settings()
    )
    result = ToolExecutor(service=service).execute(ToolName.GET_CURRENT_STANDINGS)
    assert secret not in result.failure.message
    assert "[REDACTED]" in result.failure.message


def test_ledger_serializes_cleanly_to_json():
    executor, _ = make_executor(FD_ORG_STANDINGS)
    executor.execute(ToolName.GET_CURRENT_STANDINGS)
    executor.execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS)
    executor.execute(ToolName.GET_SCORELINE_PREDICTION, SCORELINE_ARGS)
    executor.execute("nope")  # a failure too

    payload = json.loads(executor.ledger.to_json())
    assert len(payload["evidence"]) == 3
    assert len(payload["failures"]) == 1
    assert [e["evidence_id"] for e in payload["evidence"]] == ["E1", "E2", "E3"]


def test_evidence_ids_are_safe_for_a_public_trace():
    executor = ToolExecutor()
    executor.execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS)
    for item in executor.ledger.items:
        assert item.evidence_id.startswith("E")
        assert item.evidence_id[1:].isdigit()


# --------------------------------------------------------------------------
# 21-24. Structural safety
# --------------------------------------------------------------------------
def _agent_modules() -> list[Path]:
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


def test_no_anthropic_or_claude_imports():
    forbidden = ("anthropic", "claude", "claude_agent_sdk", "openai", "langchain", "langgraph")
    for module_path in _agent_modules():
        for imported in _imported_modules(module_path):
            assert imported.split(".")[0].lower() not in forbidden, module_path.name


def test_no_training_or_fitting_imports():
    forbidden_prefixes = (
        "backend.app.ml.training",
        "backend.app.ml.datasets",
        "backend.app.ml.feature_engineering",
        "backend.app.ml.baselines",
        "backend.app.ml.calibration",
    )
    for module_path in _agent_modules():
        for imported in _imported_modules(module_path):
            assert not imported.startswith(forbidden_prefixes), f"{module_path.name}: {imported}"


def test_agent_layer_performs_no_fitting_or_persistence():
    forbidden_calls = {
        "fit", "fit_score_model", "build_strength_trio_artifact", "save_strength_trio_artifact",
        "build_score_model_artifact", "save_score_model_artifact", "to_parquet", "to_csv",
        "write_text", "write_bytes",
    }
    for module_path in _agent_modules():
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                assert name not in forbidden_calls, f"{module_path.name} calls {name}"


def test_agent_layer_never_references_the_sealed_season():
    for module_path in _agent_modules():
        assert "2025_26" not in module_path.read_text(), module_path.name


def test_inference_does_not_mutate_models_or_processed_data():
    models_dir = REPO_ROOT / "models"
    processed = REPO_ROOT / "data" / "processed"
    before_models = {p.name: p.stat().st_mtime for p in models_dir.iterdir()}
    before_processed = {p.name: p.stat().st_mtime for p in processed.iterdir()}

    executor, _ = make_executor(FD_ORG_STANDINGS)
    executor.execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS)
    executor.execute(ToolName.EXPLAIN_OUTCOME_PREDICTION, PREDICTION_ARGS)
    executor.execute(ToolName.GET_SCORELINE_PREDICTION, SCORELINE_ARGS)
    executor.execute(ToolName.GET_CURRENT_STANDINGS)

    assert {p.name: p.stat().st_mtime for p in models_dir.iterdir()} == before_models
    assert {p.name: p.stat().st_mtime for p in processed.iterdir()} == before_processed


def test_no_real_network_during_execution(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("Stage 2 must not perform real HTTP requests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _forbidden)
    executor, _ = make_executor(FD_ORG_STANDINGS)
    assert executor.execute(ToolName.GET_CURRENT_STANDINGS).ok is True
    assert executor.execute(ToolName.RUN_OUTCOME_PREDICTION, PREDICTION_ARGS).ok is True


def test_stage2_package_contains_only_approved_modules():
    assert {p.name for p in _agent_modules()} == {
        "__init__.py",
        "contracts.py",
        "executor.py",
        "intents.py",
        "planner.py",
    }

"""Tests for the deterministic tool layer and the FastAPI serving layer.

Fully offline: no network, no credentials, no LLM. Providers are fakes or
mocked httpx transports; the frozen model artifact is the real local one.
"""

from __future__ import annotations

import ast
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import dependencies  # noqa: E402
from backend.app.api.main import app  # noqa: E402
from backend.app.core.config import FootballDataSettings  # noqa: E402
from backend.app.services.football_data.errors import (  # noqa: E402
    MalformedProviderPayload,
    ProviderRateLimited,
    ProviderUnavailable,
    UnknownTeam,
    UnsupportedCapability,
)
from backend.app.services.football_data.football_data_org import FootballDataOrgProvider  # noqa: E402
from backend.app.services.football_data.models import FixtureStatus  # noqa: E402
from backend.app.services.football_data.replay import ReplayProvider  # noqa: E402
from backend.app.services.football_data.service import FootballDataService  # noqa: E402
from backend.app.tools import football_data_tools, prediction_tools  # noqa: E402
from backend.app.tools.schemas import OutcomePredictionRequest, SourceKind  # noqa: E402

T0 = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)


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
        },
        {
            "id": 502,
            "utcDate": "2026-09-19T13:00:00Z",
            "status": "SCHEDULED",
            "matchday": 5,
            "homeTeam": {"id": 65, "name": "Manchester City FC"},
            "awayTeam": {"id": 57, "name": "Arsenal FC"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
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


def mock_provider(payload, status_code: int = 200, headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload, headers=headers or {})

    return FootballDataOrgProvider(
        make_settings(), client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def make_service(current_provider=None, live_provider=None, **settings_overrides) -> FootballDataService:
    return FootballDataService(
        current_provider=current_provider,
        live_provider=live_provider if live_provider is not None else ReplayProvider(step=3),
        settings=make_settings(**settings_overrides),
    )


@pytest.fixture()
def client():
    """TestClient with the shared service reset between tests."""
    dependencies.reset_service()
    with TestClient(app) as test_client:
        yield test_client
    dependencies.reset_service()


@pytest.fixture()
def wired_client(client):
    """Client whose service uses mocked current data + replay live data."""

    def _wire(current_payload=FD_ORG_MATCHES, standings_payload=None, **kwargs):
        provider = mock_provider(standings_payload if standings_payload else current_payload)
        dependencies.set_service(make_service(current_provider=provider, **kwargs))
        return client

    return _wire


# --------------------------------------------------------------------------
# 1. Health - no credentials, no network, no model
# --------------------------------------------------------------------------
def test_health_works_with_no_credentials_or_network(client, monkeypatch):
    monkeypatch.delenv("PITCHMIND_FOOTBALL_DATA_ORG_KEY", raising=False)
    monkeypatch.delenv("PITCHMIND_API_FOOTBALL_KEY", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "pitchmind"}


def test_app_imports_without_credentials():
    """Importing the app must not require keys or touch the network."""
    import importlib

    module = importlib.import_module("backend.app.api.main")
    assert module.app is not None


# --------------------------------------------------------------------------
# 2. Standings
# --------------------------------------------------------------------------
def test_standings_returns_normalized_rows(wired_client):
    client = wired_client(standings_payload=FD_ORG_STANDINGS)
    response = client.get("/api/v1/standings")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    row = body["standings"][0]
    assert row["team"]["canonical_id"] == "arsenal"
    assert row["points"] == 10
    assert body["provenance"]["provider"] == "football_data_org"
    assert body["provenance"]["source_kind"] == SourceKind.REAL_PROVIDER.value


def test_standings_maps_provider_unavailable_to_503(wired_client):
    client = wired_client(standings_payload=None)
    dependencies.set_service(make_service(current_provider=mock_provider({}, status_code=503)))
    response = client.get("/api/v1/standings")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_unavailable"


def test_standings_without_configured_key_is_503_not_501(client):
    """A missing credential is a configuration/availability problem, not
    'this feature does not exist'."""
    dependencies.set_service(
        make_service(current_provider=FootballDataOrgProvider(make_settings(football_data_org_key=None)),
                    football_data_org_key=None)
    )
    response = client.get("/api/v1/standings")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_unavailable"
    assert "PITCHMIND_FOOTBALL_DATA_ORG_KEY" in response.json()["error"]["message"]


# --------------------------------------------------------------------------
# 3. Fixtures
# --------------------------------------------------------------------------
def test_fixtures_returns_normalized_fixtures(wired_client):
    client = wired_client()
    response = client.get("/api/v1/fixtures")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["fixtures"][0]["home_team"]["canonical_id"] == "arsenal"
    assert body["provenance"]["provider"] == "football_data_org"


def test_fixtures_team_filter_uses_canonical_identity(wired_client):
    """'Man City' (historical spelling) must resolve to the same club as the
    provider's 'Manchester City FC'."""
    client = wired_client()
    response = client.get("/api/v1/fixtures", params={"team": "Man City"})
    assert response.status_code == 200
    body = response.json()
    assert body["filters_applied"]["team"] == "man_city"
    assert body["count"] == 1
    assert body["fixtures"][0]["home_team"]["canonical_id"] == "man_city"


def test_fixtures_unknown_team_is_404_and_not_fuzzy_matched(wired_client):
    client = wired_client()
    response = client.get("/api/v1/fixtures", params={"team": "Sheffield Wednesday"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_team"


def test_fixtures_status_filter(wired_client):
    client = wired_client()
    response = client.get("/api/v1/fixtures", params={"status": "FINISHED"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["fixtures"][0]["status"] == "FINISHED"


def test_fixtures_invalid_status_is_422(wired_client):
    client = wired_client()
    assert client.get("/api/v1/fixtures", params={"status": "NOT_A_STATUS"}).status_code == 422


def test_fixtures_rate_limited_maps_to_429_with_retry_after(client):
    dependencies.set_service(
        make_service(
            current_provider=mock_provider({}, status_code=429, headers={"Retry-After": "42"})
        )
    )
    response = client.get("/api/v1/fixtures")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "provider_rate_limited"
    assert response.headers.get("Retry-After") == "42"


def test_fixtures_malformed_payload_maps_to_502(client):
    dependencies.set_service(make_service(current_provider=mock_provider({"nonsense": True})))
    response = client.get("/api/v1/fixtures")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "malformed_provider_payload"


# --------------------------------------------------------------------------
# 4. Live (replay-backed)
# --------------------------------------------------------------------------
def test_live_matches_declares_replay_provenance(wired_client):
    """Replay data must never be presentable as real live football."""
    client = wired_client()
    response = client.get("/api/v1/live/matches")
    assert response.status_code == 200
    body = response.json()
    assert body["provenance"]["source_kind"] == SourceKind.REPLAY.value
    assert body["provenance"]["provider"] == "replay"
    assert body["matches"][0]["provider"] == "replay"


def test_live_match_state_returns_live_match_state_shape(wired_client):
    client = wired_client()
    body = client.get("/api/v1/live/matches/replay-1").json()
    match = body["match"]
    assert match is not None
    assert match["fixture"]["home_team"]["canonical_id"] == "arsenal"
    assert match["minute"] == 60
    assert match["home_score"] == 1 and match["away_score"] == 1
    # Unreported statistic stays null - never coerced to 0.
    assert match["home_stats"]["corners"] is None
    assert match["home_stats"]["shots"] == 9
    assert match["freshness"]["provider"] == "replay"


def test_live_match_state_for_unknown_fixture_is_null_not_error(wired_client):
    client = wired_client()
    response = client.get("/api/v1/live/matches/does-not-exist")
    assert response.status_code == 200
    assert response.json()["match"] is None


def test_live_endpoint_reuses_one_snapshot_across_many_requests(client):
    """The 300s lazy cache must survive across HTTP requests - otherwise every
    user question would cost a provider call."""

    class CountingReplay(ReplayProvider):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def get_live_matches(self, season_label=None):
            self.calls += 1
            return super().get_live_matches(season_label)

    provider = CountingReplay(step=3)
    dependencies.set_service(make_service(live_provider=provider))
    for _ in range(10):
        assert client.get("/api/v1/live/matches").status_code == 200
    assert provider.calls == 1


def test_live_provider_that_cannot_do_live_maps_to_501(client):
    """A provider genuinely lacking the capability is 501 - distinct from an
    unconfigured deployment (503)."""
    dependencies.set_service(
        make_service(live_provider=FootballDataOrgProvider(make_settings()))
    )
    response = client.get("/api/v1/live/matches")
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "unsupported_capability"


# --------------------------------------------------------------------------
# 5. Prediction
# --------------------------------------------------------------------------
PREDICT_BODY = {"elo_diff": 85.0, "diff_ewma_ppg": 0.42, "diff_ewma_sot_diff": 1.3}


def test_prediction_matches_direct_frozen_model_path(client):
    """The endpoint must return exactly what the persisted pipeline's own
    predict_proba returns - no re-derivation, no drift."""
    import pandas as pd

    from backend.app.ml.baselines import STRENGTH_TRIO_COLUMNS

    artifact = prediction_tools.get_artifact()
    frame = pd.DataFrame([PREDICT_BODY])[list(STRENGTH_TRIO_COLUMNS)]
    expected = artifact.pipeline.predict_proba(frame)[0]

    body = client.post("/api/v1/predict", json=PREDICT_BODY).json()
    assert body["home_win_probability"] == pytest.approx(expected[0], abs=1e-12)
    assert body["draw_probability"] == pytest.approx(expected[1], abs=1e-12)
    assert body["away_win_probability"] == pytest.approx(expected[2], abs=1e-12)


def test_prediction_probabilities_sum_to_one(client):
    body = client.post("/api/v1/predict", json=PREDICT_BODY).json()
    total = body["home_win_probability"] + body["draw_probability"] + body["away_win_probability"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_prediction_reports_frozen_model_provenance(client):
    body = client.post("/api/v1/predict", json=PREDICT_BODY).json()
    provenance = body["model_provenance"]
    assert provenance["model_id"] == "baseline_strength_trio"
    assert provenance["training_cutoff_season"] == "2024_25"
    assert provenance["sealed_final_test_completed"] is False
    assert provenance["source_kind"] == SourceKind.LOCAL_MODEL.value


def test_prediction_rejects_extra_features(client):
    """The frozen model takes exactly three features; anything else is a
    caller bug, not something to silently ignore."""
    response = client.post("/api/v1/predict", json={**PREDICT_BODY, "possession": 61.0})
    assert response.status_code == 422


def test_prediction_accepts_missing_optional_features_and_flags_imputation(client):
    body = client.post("/api/v1/predict", json={"elo_diff": 20.0}).json()
    assert body["features_used"]["diff_ewma_ppg"] is None
    explanation = client.post("/api/v1/predict/explain", json={"elo_diff": 20.0}).json()
    imputed = {f["name"]: f["was_imputed"] for f in explanation["features"]}
    assert imputed["diff_ewma_ppg"] is True
    assert imputed["elo_diff"] is False


def test_prediction_does_not_retrain(client, monkeypatch):
    """Serving must never fit anything."""
    from sklearn.pipeline import Pipeline

    def _forbidden_fit(self, *args, **kwargs):
        raise AssertionError("serving must never call Pipeline.fit")

    monkeypatch.setattr(Pipeline, "fit", _forbidden_fit)
    assert client.post("/api/v1/predict", json=PREDICT_BODY).status_code == 200
    assert client.post("/api/v1/predict/explain", json=PREDICT_BODY).status_code == 200


def test_missing_model_artifact_maps_to_503(client, monkeypatch):
    def _missing(*args, **kwargs):
        raise prediction_tools.ModelArtifactUnavailable("artifact missing")

    monkeypatch.setattr(prediction_tools, "get_artifact", _missing)
    response = client.post("/api/v1/predict", json=PREDICT_BODY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_artifact_unavailable"


# --------------------------------------------------------------------------
# 6. Explanation
# --------------------------------------------------------------------------
def test_explanation_probabilities_match_prediction_endpoint(client):
    prediction = client.post("/api/v1/predict", json=PREDICT_BODY).json()
    explanation = client.post("/api/v1/predict/explain", json=PREDICT_BODY).json()
    assert explanation["probabilities"]["home"] == pytest.approx(
        prediction["home_win_probability"], abs=1e-12
    )
    assert explanation["predicted_class"] == prediction["predicted_class"]


def test_explanation_reconstruction_identity_holds(client):
    """intercept + sum(grouped contributions) == logit, per class - the exact
    guarantee the underlying decomposition provides."""
    explanation = client.post("/api/v1/predict/explain", json=PREDICT_BODY).json()
    for cls in ("home", "draw", "away"):
        total = explanation["intercepts"][cls] + sum(
            f["grouped_logit_contribution"][cls] for f in explanation["features"]
        )
        assert total == pytest.approx(explanation["logits"][cls], abs=1e-9)
    assert explanation["reconstruction_verified"] is True


def test_explanation_returns_three_semantic_features_and_rankings(client):
    explanation = client.post("/api/v1/predict/explain", json=PREDICT_BODY).json()
    assert [f["name"] for f in explanation["features"]] == [
        "elo_diff", "diff_ewma_ppg", "diff_ewma_sot_diff",
    ]
    assert explanation["class_order"] == ["H", "D", "A"]
    logit_values = [d["value"] for d in explanation["drivers_by_logit_contribution"]]
    assert logit_values == sorted(logit_values, reverse=True)
    assert len(explanation["drivers_by_probability_sensitivity"]) == 3


def test_explanation_separates_logit_contribution_from_probability_sensitivity(client):
    """Two distinct, separately-named quantities - a logit contribution is
    never presented as a percentage-point effect."""
    explanation = client.post("/api/v1/predict/explain", json=PREDICT_BODY).json()
    feature = explanation["features"][0]
    assert "grouped_logit_contribution" in feature
    assert "probability_sensitivity" in feature
    assert feature["grouped_logit_contribution"] != feature["probability_sensitivity"]


def test_explanation_is_deterministic(client):
    a = client.post("/api/v1/predict/explain", json=PREDICT_BODY).json()
    b = client.post("/api/v1/predict/explain", json=PREDICT_BODY).json()
    assert a == b


# --------------------------------------------------------------------------
# 7. Tool layer directly (agents will call these, not HTTP)
# --------------------------------------------------------------------------
def test_tools_return_typed_responses_without_http():
    service = make_service(current_provider=mock_provider(FD_ORG_MATCHES))
    fixtures = football_data_tools.get_fixtures(service, now=T0)
    assert fixtures.count == 2
    assert fixtures.fixtures[0].home_team.canonical_id == "arsenal"

    live = football_data_tools.get_live_matches(service, now=T0)
    assert live.provenance.source_kind is SourceKind.REPLAY

    prediction = prediction_tools.run_outcome_prediction(
        OutcomePredictionRequest(elo_diff=85.0, diff_ewma_ppg=0.42, diff_ewma_sot_diff=1.3)
    )
    assert prediction.predicted_class in {"H", "D", "A"}


def test_tool_unknown_team_raises_rather_than_guessing():
    service = make_service(current_provider=mock_provider(FD_ORG_MATCHES))
    with pytest.raises(UnknownTeam):
        football_data_tools.get_fixtures(service, team="Sheffield Wednesday", now=T0)


# --------------------------------------------------------------------------
# 8. No network, no LLM, sealed-season safety
# --------------------------------------------------------------------------
SERVING_PACKAGES = [REPO_ROOT / "backend" / "app" / "tools", REPO_ROOT / "backend" / "app" / "api"]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _serving_modules() -> list[Path]:
    return [p for package in SERVING_PACKAGES for p in package.glob("*.py")]


def test_no_llm_or_agent_framework_imported_anywhere_in_serving_layer():
    forbidden = ("anthropic", "claude", "openai", "langchain", "langgraph", "crewai", "autogen")
    for module_path in _serving_modules():
        modules = _imported_modules(module_path)
        for name in modules:
            assert not any(bad in name.lower() for bad in forbidden), f"{module_path.name}: {name}"


def test_no_real_http_calls_during_tests(client, monkeypatch):
    """Any un-mocked outbound HTTP would be a bug: assert the real transport
    is never used by an endpoint the test suite exercises."""

    def _forbidden_send(*args, **kwargs):
        raise AssertionError("tests must not perform real HTTP requests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _forbidden_send)
    assert client.get("/health").status_code == 200
    assert client.post("/api/v1/predict", json=PREDICT_BODY).status_code == 200
    assert client.get("/api/v1/live/matches").status_code == 200


def test_serving_layer_never_retrains_or_persists():
    """Structural guard: no fit/save call, no write, anywhere in tools/ or api/."""
    forbidden_calls = {
        "fit", "build_strength_trio_artifact", "save_strength_trio_artifact",
        "fit_score_model", "_fit_logistic_pipeline", "to_parquet", "to_csv",
        "dump", "write_text", "write_bytes",
    }
    offenders: dict[str, set[str]] = {}
    for module_path in _serving_modules():
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in forbidden_calls:
                offenders.setdefault(module_path.name, set()).add(name)
    assert not offenders, f"serving layer must not train or persist: {offenders}"


def test_serving_layer_does_not_reference_the_sealed_season_as_data():
    """The sealed season may be NAMED in provenance text (to say it has not
    been evaluated), but must never be loaded or scored here."""
    forbidden_imports = {
        "backend.app.ml.training",
        "backend.app.ml.datasets",
        "backend.app.ml.score_models",
    }
    for module_path in _serving_modules():
        modules = _imported_modules(module_path)
        assert not (modules & forbidden_imports), f"{module_path.name} imports {modules & forbidden_imports}"


def test_models_directory_is_untouched_by_serving(client):
    """Serving must not add, remove or rewrite anything under models/."""
    models_dir = REPO_ROOT / "models"
    before = {p.name: p.stat().st_mtime for p in models_dir.iterdir()}
    client.post("/api/v1/predict", json=PREDICT_BODY)
    client.post("/api/v1/predict/explain", json=PREDICT_BODY)
    after = {p.name: p.stat().st_mtime for p in models_dir.iterdir()}
    assert before == after


def test_processed_data_is_untouched_by_serving(client):
    processed = REPO_ROOT / "data" / "processed"
    if not processed.exists():
        pytest.skip("data/processed not present")
    before = {p.name: p.stat().st_mtime for p in processed.iterdir()}
    client.get("/api/v1/live/matches")
    client.post("/api/v1/predict", json=PREDICT_BODY)
    after = {p.name: p.stat().st_mtime for p in processed.iterdir()}
    assert before == after


def test_scoreline_tool_is_not_exposed():
    """Dixon-Coles serving is deliberately DEFERRED - no artifact is
    persisted, so exposing it would mean fitting at request time (which would
    read the sealed 2025/26 season). Asserted so it cannot appear by accident."""
    from backend.app import tools

    assert not any("scoreline" in name.lower() for name in dir(tools))
    assert not any("dixon" in name.lower() for name in dir(tools))
    routes = {route.path for route in app.routes}
    assert not any("scoreline" in path or "dixon" in path for path in routes)

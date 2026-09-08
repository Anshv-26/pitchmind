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


def _imported_names(path: Path) -> set[str]:
    """Names pulled in via `from X import name`, not just module paths."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def test_serving_layer_cannot_reach_training_machinery():
    """Serving may import `score_models` for INFERENCE (`predict_match`), but
    must never import the fold/training machinery, and must never import
    `fit_score_model` - the one function that would read the sealed season.

    This guard deliberately targets the training ENTRY POINTS rather than
    banning the module outright: inference and fitting live in the same
    frozen module, and only fitting is dangerous.
    """
    forbidden_modules = {"backend.app.ml.training", "backend.app.ml.datasets"}
    forbidden_names = {"fit_score_model", "build_features", "load_fold", "run_stage1"}
    for module_path in _serving_modules():
        modules = _imported_modules(module_path)
        assert not (modules & forbidden_modules), (
            f"{module_path.name} imports {modules & forbidden_modules}"
        )
        names = _imported_names(module_path)
        assert not (names & forbidden_names), (
            f"{module_path.name} imports training entry point(s) {names & forbidden_names}"
        )


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


def test_scoreline_serving_is_now_exposed():
    """Dixon-Coles scoreline serving is no longer deferred: it is backed by a
    persisted artifact and reachable through both the tool layer and the API."""
    from backend.app import tools

    assert hasattr(tools, "get_scoreline_prediction")
    routes = {route.path for route in app.routes}
    assert "/api/v1/predict/scoreline" in routes


# --------------------------------------------------------------------------
# 9. Dixon-Coles scoreline serving (frozen artifact, JSON-persisted)
# --------------------------------------------------------------------------
SCORELINE_BODY = {"home_team": "Arsenal", "away_team": "Liverpool"}


def test_score_model_artifact_excludes_the_sealed_season():
    """The artifact's own metadata must prove 2025/26 was never trained on."""
    from backend.app.ml.feature_engineering import SEALED_SEASON
    from backend.app.tools.scoreline_tools import get_score_model_artifact

    meta = get_score_model_artifact().metadata
    assert SEALED_SEASON not in meta.training_seasons
    assert meta.training_seasons[-1] == "2024_25"
    assert meta.training_cutoff_season == "2024_25"
    assert meta.sealed_final_test_scored is False


def test_score_model_artifact_training_source_has_no_sealed_rows():
    """Data-level assertion, not just metadata: rebuilding the training frame
    the same way the builder does must contain zero sealed-season matches."""
    import pandas as pd

    from backend.app.ml.feature_engineering import SEALED_SEASON
    from backend.app.ml.score_model_artifact import MATCHES_PATH

    frame = pd.read_parquet(MATCHES_PATH, columns=["Season"])
    train = frame.loc[frame["Season"] != SEALED_SEASON]
    assert SEALED_SEASON not in set(train["Season"].unique())
    assert len(train) == 3800
    assert len(frame) > len(train), "source genuinely contains sealed rows that must be filtered"


def test_score_model_artifact_metadata_records_the_frozen_configuration():
    from backend.app.tools.scoreline_tools import get_score_model_artifact

    meta = get_score_model_artifact().metadata
    assert meta.model_id == "dixon_coles_l2_decay"
    assert meta.model_type == "dixon_coles"
    assert meta.use_dixon_coles is True
    assert meta.l2_sigma == pytest.approx(0.25)
    assert meta.decay_half_life_days == pytest.approx(365.0)
    assert meta.training_match_count == 3800


def test_scoreline_serving_never_calls_fit_score_model(client, monkeypatch):
    """The serving path must load, never fit."""
    from backend.app.ml import score_models

    def _forbidden_fit(*args, **kwargs):
        raise AssertionError("serving must never call fit_score_model")

    monkeypatch.setattr(score_models, "fit_score_model", _forbidden_fit)
    assert client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).status_code == 200


def test_scoreline_values_match_direct_frozen_inference(client):
    """Endpoint output must equal `predict_match` on the frozen params -
    no re-derivation anywhere in the serving path."""
    from backend.app.ml.score_models import predict_match
    from backend.app.tools.scoreline_tools import get_score_model_artifact

    artifact = get_score_model_artifact()
    expected = predict_match(artifact.params, artifact.config, "Arsenal", "Liverpool")

    body = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).json()
    assert body["expected_home_goals"] == pytest.approx(expected.expected_home_goals, abs=1e-12)
    assert body["expected_away_goals"] == pytest.approx(expected.expected_away_goals, abs=1e-12)
    assert body["lambda_home"] == pytest.approx(expected.lambda_home, abs=1e-12)
    assert body["most_likely_scoreline"] == expected.most_likely_scoreline


def test_scoreline_inference_is_deterministic(client):
    a = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).json()
    b = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).json()
    assert a == b


def test_scoreline_top_scorelines_are_internally_consistent(client):
    body = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).json()
    top = body["top_scorelines"]
    assert len(top) == 3
    probabilities = [s["probability"] for s in top]
    assert probabilities == sorted(probabilities, reverse=True), "must be ranked descending"
    assert all(0.0 < p < 1.0 for p in probabilities)
    # The named scoreline must agree with its own goal fields, and with the
    # separately-reported most likely scoreline.
    for entry in top:
        assert entry["scoreline"] == f"{entry['home_goals']}-{entry['away_goals']}"
    assert body["most_likely_scoreline"] == top[0]["scoreline"]


def test_scoreline_secondary_outcome_probabilities_sum_to_one(client):
    """The model renormalises after its truncation check, so these sum to 1."""
    body = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).json()
    probabilities = body["score_model_outcome_probabilities"]
    total = probabilities["home"] + probabilities["draw"] + probabilities["away"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_scoreline_expected_goals_are_positive_and_plausible(client):
    body = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).json()
    assert 0.0 < body["expected_home_goals"] < 6.0
    assert 0.0 < body["expected_away_goals"] < 6.0


def test_scoreline_model_provenance_is_correct(client):
    body = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).json()
    provenance = body["model_provenance"]
    assert provenance["model_id"] == "dixon_coles_l2_decay"
    assert provenance["model_type"] == "dixon_coles"
    assert provenance["training_cutoff_season"] == "2024_25"
    assert provenance["sealed_final_test_completed"] is False
    assert provenance["l2_sigma"] == pytest.approx(0.25)
    assert provenance["decay_half_life_days"] == pytest.approx(365.0)
    # Must not be mistakable for the primary H/D/A model.
    assert "strength-trio" in provenance["usage_note"]


def test_scoreline_resolves_team_aliases_to_fitted_model_keys(client):
    """Canonical identity maps provider/modern spellings onto the historical
    keys the fitted model is actually keyed on."""
    body = client.post(
        "/api/v1/predict/scoreline",
        json={"home_team": "Manchester City", "away_team": "Nottingham Forest"},
    ).json()
    assert body["home_team_model_key"] == "Man City"
    assert body["away_team_model_key"] == "Nott'm Forest"
    assert body["home_team"] == "Manchester City"


def test_scoreline_refuses_current_only_club_rather_than_fabricating(client):
    """Coventry City is a real current PL club with NO fitted parameters. It
    must error, not receive a promoted-prior-derived fabricated prediction."""
    response = client.post(
        "/api/v1/predict/scoreline", json={"home_team": "Coventry City", "away_team": "Arsenal"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "team_not_in_score_model"


def test_scoreline_unknown_club_is_404_distinct_from_unfitted_club(client):
    """Two different failures must stay distinguishable."""
    response = client.post(
        "/api/v1/predict/scoreline", json={"home_team": "Wrexham", "away_team": "Arsenal"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_team"


def test_scoreline_tool_does_not_use_the_promoted_prior():
    """Direct tool-level proof: the promoted-prior fallback that exists in the
    frozen model is deliberately not reachable through serving."""
    from backend.app.tools.scoreline_tools import (
        TeamNotInScoreModel,
        get_score_model_artifact,
        get_scoreline_prediction,
    )
    from backend.app.tools.schemas import ScorelinePredictionRequest

    artifact = get_score_model_artifact()
    # The frozen params WOULD happily return a promoted-prior value...
    assert artifact.params.team_attack("Coventry City") == pytest.approx(
        artifact.params.promoted_attack_offset
    )
    # ...but serving refuses instead.
    with pytest.raises(TeamNotInScoreModel):
        get_scoreline_prediction(
            ScorelinePredictionRequest(home_team="Coventry City", away_team="Arsenal")
        )


def test_scoreline_rejects_extra_fields(client):
    response = client.post(
        "/api/v1/predict/scoreline", json={**SCORELINE_BODY, "possession": 60.0}
    )
    assert response.status_code == 422


def test_scoreline_top_n_is_honoured_within_the_models_real_ceiling(client):
    body = client.post(
        "/api/v1/predict/scoreline", json={**SCORELINE_BODY, "top_n": 2}
    ).json()
    assert len(body["top_scorelines"]) == 2
    # Above the frozen model's real ceiling of 3 is rejected, not silently truncated.
    assert client.post(
        "/api/v1/predict/scoreline", json={**SCORELINE_BODY, "top_n": 9}
    ).status_code == 422


def test_missing_score_model_artifact_maps_to_503(client, monkeypatch):
    from backend.app.tools import scoreline_tools

    def _missing(*args, **kwargs):
        raise prediction_tools.ModelArtifactUnavailable("score artifact missing")

    monkeypatch.setattr(scoreline_tools, "get_score_model_artifact", _missing)
    response = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_artifact_unavailable"


def test_scoreline_does_not_affect_primary_prediction_endpoint(client):
    """The two models are independent; calling one must not perturb the other."""
    before = client.post("/api/v1/predict", json=PREDICT_BODY).json()
    client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY)
    after = client.post("/api/v1/predict", json=PREDICT_BODY).json()
    assert before == after


def test_scoreline_primary_and_secondary_probabilities_are_separately_labelled(client):
    """The score model's H/D/A must never be presented as the primary one."""
    primary = client.post("/api/v1/predict", json=PREDICT_BODY).json()
    scoreline = client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY).json()
    assert "home_win_probability" in primary
    assert "home_win_probability" not in scoreline
    assert "score_model_outcome_probabilities" in scoreline
    assert "score_model_outcome_probabilities" not in primary


def test_scoreline_inference_does_not_modify_processed_data_or_models(client):
    processed = REPO_ROOT / "data" / "processed"
    models_dir = REPO_ROOT / "models"
    before_processed = {p.name: p.stat().st_mtime for p in processed.iterdir()}
    before_models = {p.name: p.stat().st_mtime for p in models_dir.iterdir()}
    client.post("/api/v1/predict/scoreline", json=SCORELINE_BODY)
    assert {p.name: p.stat().st_mtime for p in processed.iterdir()} == before_processed
    assert {p.name: p.stat().st_mtime for p in models_dir.iterdir()} == before_models


def test_score_model_artifact_is_plain_json_not_a_pickle():
    """Transparent, inspectable, no arbitrary-code-execution on load."""
    import json

    from backend.app.ml.score_model_artifact import SCORE_MODEL_ARTIFACT_PATH

    payload = json.loads(SCORE_MODEL_ARTIFACT_PATH.read_text())
    assert set(payload) == {"metadata", "config", "params"}
    assert payload["params"]["config_id"] == "dixon_coles_l2_decay"
    assert isinstance(payload["params"]["attack"], dict)


def test_score_model_artifact_round_trips_exactly(tmp_path):
    from backend.app.ml.score_model_artifact import (
        load_score_model_artifact,
        save_score_model_artifact,
    )
    from backend.app.tools.scoreline_tools import get_score_model_artifact

    original = get_score_model_artifact()
    path = tmp_path / "artifact.json"
    save_score_model_artifact(original, path=path)
    reloaded = load_score_model_artifact(path=path)
    assert reloaded.params == original.params
    assert reloaded.config == original.config
    assert reloaded.metadata == original.metadata


def test_score_model_artifact_with_sealed_season_in_metadata_fails_loudly(tmp_path):
    import dataclasses
    import json

    from backend.app.ml.feature_engineering import SEALED_SEASON
    from backend.app.ml.score_model_artifact import load_score_model_artifact
    from backend.app.tools.scoreline_tools import get_score_model_artifact

    original = get_score_model_artifact()
    bad_metadata = dataclasses.replace(
        original.metadata, training_seasons=[*original.metadata.training_seasons, SEALED_SEASON]
    )
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "metadata": dataclasses.asdict(bad_metadata),
                "config": {
                    "config_id": original.config.config_id,
                    "use_dixon_coles": original.config.use_dixon_coles,
                    "l2_sigma": original.config.l2_sigma,
                    "half_life_days": original.config.half_life_days,
                },
                "params": {
                    "config_id": original.params.config_id,
                    "teams": list(original.params.teams),
                    "intercept": original.params.intercept,
                    "home_advantage": original.params.home_advantage,
                    "attack": original.params.attack,
                    "defence": original.params.defence,
                    "rho": original.params.rho,
                    "promoted_attack_offset": original.params.promoted_attack_offset,
                    "promoted_defence_offset": original.params.promoted_defence_offset,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="sealed season"):
        load_score_model_artifact(path=path)


def test_build_script_takes_no_training_data_arguments():
    """No parameter through which a caller could widen the training window."""
    import inspect

    from backend.app.ml.score_model_artifact import build_score_model_artifact

    assert list(inspect.signature(build_score_model_artifact).parameters) == []

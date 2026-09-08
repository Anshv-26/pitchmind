"""PitchMind HTTP API - thin routes over the deterministic tool layer.

    HTTP route -> deterministic tool -> football-data service / frozen model

There is deliberately no LLM anywhere in this path. Every route is a thin
wrapper: routing, validation and error mapping live here, all football and
model logic lives in `backend.app.tools`.

Run locally:

    uvicorn backend.app.api.main:app --reload

Example requests (no secrets involved):

    curl localhost:8000/health
    curl localhost:8000/api/v1/standings
    curl "localhost:8000/api/v1/fixtures?team=Arsenal&status=FINISHED&limit=5"
    curl localhost:8000/api/v1/live/matches
    curl -X POST localhost:8000/api/v1/predict \
         -H 'content-type: application/json' \
         -d '{"elo_diff": 85.0, "diff_ewma_ppg": 0.42, "diff_ewma_sot_diff": 1.3}'
    curl -X POST localhost:8000/api/v1/predict/explain \
         -H 'content-type: application/json' \
         -d '{"elo_diff": 85.0}'

`/api/v1/standings` and `/api/v1/fixtures` need PITCHMIND_FOOTBALL_DATA_ORG_KEY
and return 503 without it. `/health`, `/api/v1/live/*` (replay-backed) and both
predict routes work with no credentials at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from backend.app.core.config import load_settings
from backend.app.services.football_data.errors import (
    CurrentSeasonNotAvailable,
    FootballDataError,
    MalformedProviderPayload,
    ProviderRateLimited,
    ProviderUnavailable,
    UnknownTeam,
    UnsupportedCapability,
)
from backend.app.services.football_data.models import FixtureStatus
from backend.app.services.football_data.service import FootballDataService
from backend.app.tools import football_data_tools as football_tools
from backend.app.tools import prediction_tools, scoreline_tools
from backend.app.tools.prediction_tools import ModelArtifactUnavailable
from backend.app.tools.scoreline_tools import TeamNotInScoreModel
from backend.app.tools.schemas import (
    FixturesResponse,
    LiveMatchesResponse,
    LiveMatchStateResponse,
    OutcomeExplanationResponse,
    OutcomePredictionRequest,
    OutcomePredictionResponse,
    ScorelinePredictionRequest,
    ScorelinePredictionResponse,
    StandingsResponse,
)

SERVICE_NAME = "pitchmind"
API_V1 = "/api/v1"


def _scrub(message: str) -> str:
    """Remove any configured credential from an outbound message.

    The adapters never put a key in an exception message, so this is defence
    in depth rather than a fix - but an API response is exactly the wrong
    place to discover that assumption was wrong.
    """
    text = str(message)
    settings = load_settings()
    for secret in (settings.football_data_org_key, settings.api_football_key):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def _error(status_code: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    """One error shape everywhere. Never a stack trace, never a raw payload."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": _scrub(message)}},
        headers=headers,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="PitchMind API",
        version="1.0.0",
        description=(
            "Deterministic Premier League data and frozen-model predictions. "
            "No LLM is involved in any endpoint."
        ),
    )

    # ---- Error mapping ------------------------------------------------
    # Registered most-specific first. ProviderRateLimited subclasses
    # ProviderUnavailable, so it needs its own handler to reach 429.
    @app.exception_handler(UnknownTeam)
    async def _unknown_team(request: Request, exc: UnknownTeam) -> JSONResponse:
        # 404: the requested team resource is not one we know. Consistent for
        # every unknown-team case; we never fuzzy-match to something close.
        return _error(404, "unknown_team", str(exc))

    @app.exception_handler(ProviderRateLimited)
    async def _rate_limited(request: Request, exc: ProviderRateLimited) -> JSONResponse:
        headers = None
        if exc.retry_after_seconds is not None:
            headers = {"Retry-After": str(int(exc.retry_after_seconds))}
        return _error(429, "provider_rate_limited", str(exc), headers=headers)

    @app.exception_handler(UnsupportedCapability)
    async def _unsupported(request: Request, exc: UnsupportedCapability) -> JSONResponse:
        return _error(501, "unsupported_capability", str(exc))

    @app.exception_handler(MalformedProviderPayload)
    async def _malformed(request: Request, exc: MalformedProviderPayload) -> JSONResponse:
        # 502: we are the gateway and upstream sent something unusable.
        return _error(502, "malformed_provider_payload", str(exc))

    @app.exception_handler(CurrentSeasonNotAvailable)
    async def _season_unavailable(request: Request, exc: CurrentSeasonNotAvailable) -> JSONResponse:
        return _error(503, "current_season_unavailable", str(exc))

    @app.exception_handler(ProviderUnavailable)
    async def _unavailable(request: Request, exc: ProviderUnavailable) -> JSONResponse:
        return _error(503, "provider_unavailable", str(exc))

    @app.exception_handler(ModelArtifactUnavailable)
    async def _artifact_unavailable(request: Request, exc: ModelArtifactUnavailable) -> JSONResponse:
        return _error(503, "model_artifact_unavailable", str(exc))

    @app.exception_handler(TeamNotInScoreModel)
    async def _team_not_in_score_model(request: Request, exc: TeamNotInScoreModel) -> JSONResponse:
        # 422, not 404: the club genuinely exists and is known to PitchMind -
        # this specific model simply has no fitted parameters for it. A 404
        # would wrongly imply the team is unrecognised. The distinct `code`
        # separates it from Pydantic's own 422s.
        return _error(422, "team_not_in_score_model", str(exc))

    @app.exception_handler(FootballDataError)
    async def _football_data_error(request: Request, exc: FootballDataError) -> JSONResponse:
        return _error(500, "football_data_error", str(exc))

    # ---- Health --------------------------------------------------------
    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        """Network-free liveness check. Touches no provider and no model."""
        return {"status": "ok", "service": SERVICE_NAME}

    # ---- Current data --------------------------------------------------
    @app.get(f"{API_V1}/standings", response_model=StandingsResponse, tags=["current"])
    async def standings(
        service: FootballDataService = Depends(_service_dependency),
    ) -> StandingsResponse:
        return football_tools.get_current_standings(service)

    @app.get(f"{API_V1}/fixtures", response_model=FixturesResponse, tags=["current"])
    async def fixtures(
        service: FootballDataService = Depends(_service_dependency),
        team: str | None = Query(default=None, description="Any known club spelling."),
        status: FixtureStatus | None = Query(default=None),
        date_from: datetime | None = Query(default=None),
        date_to: datetime | None = Query(default=None),
        limit: int | None = Query(default=None, ge=1, le=500),
    ) -> FixturesResponse:
        return football_tools.get_fixtures(
            service,
            team=team,
            status=status,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

    # ---- Live (replay-backed today) -------------------------------------
    @app.get(f"{API_V1}/live/matches", response_model=LiveMatchesResponse, tags=["live"])
    async def live_matches(
        service: FootballDataService = Depends(_service_dependency),
    ) -> LiveMatchesResponse:
        """Every in-play match. `provenance.source_kind` is REPLAY while no
        verified real live provider is available."""
        return football_tools.get_live_matches(service)

    @app.get(
        f"{API_V1}/live/matches/{{provider_fixture_id}}",
        response_model=LiveMatchStateResponse,
        tags=["live"],
    )
    async def live_match_state(
        provider_fixture_id: str,
        service: FootballDataService = Depends(_service_dependency),
    ) -> LiveMatchStateResponse:
        return football_tools.get_live_match_state(service, provider_fixture_id)

    # ---- Frozen-model prediction ----------------------------------------
    @app.post(f"{API_V1}/predict", response_model=OutcomePredictionResponse, tags=["model"])
    async def predict(request: OutcomePredictionRequest) -> OutcomePredictionResponse:
        """H/D/A probabilities from the frozen strength-trio artifact.

        Takes a caller-supplied pre-kickoff snapshot; this endpoint never
        builds current-season features and never retrains.
        """
        return prediction_tools.run_outcome_prediction(request)

    @app.post(
        f"{API_V1}/predict/explain", response_model=OutcomeExplanationResponse, tags=["model"]
    )
    async def predict_explain(request: OutcomePredictionRequest) -> OutcomeExplanationResponse:
        """Exact linear decomposition of the same prediction. No LLM, no SHAP."""
        return prediction_tools.explain_outcome_prediction(request)

    @app.post(
        f"{API_V1}/predict/scoreline",
        response_model=ScorelinePredictionResponse,
        tags=["model"],
    )
    async def predict_scoreline(request: ScorelinePredictionRequest) -> ScorelinePredictionResponse:
        """Expected goals and scoreline distribution from the frozen
        Dixon-Coles model.

        This is PitchMind's SCORELINE model, separate from the primary
        strength-trio H/D/A predictor at `/api/v1/predict`. The H/D/A numbers
        it returns are labelled `score_model_outcome_probabilities` and are
        secondary - they must not replace the primary prediction.
        """
        return scoreline_tools.get_scoreline_prediction(request)

    return app


def _service_dependency() -> FootballDataService:
    # Indirection so tests can swap the shared service via
    # `dependencies.set_service(...)` without re-creating the app.
    from backend.app.api.dependencies import get_service

    return get_service()


app = create_app()

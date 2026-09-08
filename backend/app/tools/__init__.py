"""PitchMind's deterministic tool layer.

The boundary a future Claude Agent SDK will call. Every function here is
ordinary Python: it reads normalized provider data or the frozen local model
and returns typed Pydantic responses. Nothing in this package calls an LLM,
and nothing here fits, retrains, or persists a model.

Scoreline / Dixon-Coles is deliberately absent - see the stage report: no
Dixon-Coles artifact is persisted, so exposing it would require fitting at
request time (which would read the sealed 2025/26 season). Deferred.
"""

from backend.app.tools.football_data_tools import (
    get_current_standings,
    get_fixtures,
    get_live_match_state,
    get_live_matches,
)
from backend.app.tools.prediction_tools import (
    ModelArtifactUnavailable,
    explain_outcome_prediction,
    get_artifact,
    run_outcome_prediction,
)
from backend.app.tools.schemas import (
    ClassValues,
    DataProvenance,
    DriverRanking,
    FeatureContribution,
    FixturesResponse,
    LiveMatchesResponse,
    LiveMatchStateResponse,
    ModelProvenance,
    OutcomeExplanationResponse,
    OutcomePredictionRequest,
    OutcomePredictionResponse,
    SourceKind,
    StandingsResponse,
)

__all__ = [
    "get_current_standings",
    "get_fixtures",
    "get_live_matches",
    "get_live_match_state",
    "run_outcome_prediction",
    "explain_outcome_prediction",
    "get_artifact",
    "ModelArtifactUnavailable",
    "ClassValues",
    "DataProvenance",
    "DriverRanking",
    "FeatureContribution",
    "FixturesResponse",
    "LiveMatchesResponse",
    "LiveMatchStateResponse",
    "ModelProvenance",
    "OutcomeExplanationResponse",
    "OutcomePredictionRequest",
    "OutcomePredictionResponse",
    "SourceKind",
    "StandingsResponse",
]

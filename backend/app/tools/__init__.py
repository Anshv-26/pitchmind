"""PitchMind's deterministic tool layer.

The boundary a future Claude Agent SDK will call. Every function here is
ordinary Python: it reads normalized provider data or the frozen local model
and returns typed Pydantic responses. Nothing in this package calls an LLM,
and nothing here fits, retrains, or persists a model.

Scoreline / Dixon-Coles is served from its own frozen artifact
(`models/dixon_coles_l2_decay_through_2024_25.json`), loaded read-only. It
provides expected goals and a scoreline distribution; PitchMind's PRIMARY
H/D/A prediction remains the strength-trio classifier.
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
from backend.app.tools.scoreline_tools import (
    TeamNotInScoreModel,
    get_score_model_artifact,
    get_scoreline_prediction,
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
    ScoreModelProvenance,
    ScorelinePredictionRequest,
    ScorelinePredictionResponse,
    ScorelineProbability,
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
    "get_scoreline_prediction",
    "get_score_model_artifact",
    "TeamNotInScoreModel",
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
    "ScoreModelProvenance",
    "ScorelinePredictionRequest",
    "ScorelinePredictionResponse",
    "ScorelineProbability",
    "SourceKind",
    "StandingsResponse",
]

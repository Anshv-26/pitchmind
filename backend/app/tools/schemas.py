"""Typed request/response schemas for PitchMind's deterministic tool layer.

These are the shapes a future Claude Agent SDK tool will hand back. Two
properties matter most:

1. **Provider-neutral.** Nothing here exposes a vendor's JSON. The current-data
   schemas reuse the already-normalized `services.football_data.models` types
   directly rather than re-declaring near-duplicates.

2. **Self-describing provenance.** Every response says where its numbers came
   from and how old they are, so a downstream agent cannot accidentally
   present replay data as live, stale data as current, or a pre-final-test
   model's output as a production prediction.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.football_data.models import (
    Fixture,
    FixtureStatus,
    LiveMatchState,
    StandingRow,
)


class SourceKind(str, Enum):
    """How literally a consumer may take this data.

    `REPLAY` exists so a demo snapshot can never be mistaken for real live
    football - the distinction is carried in the payload, not just in a
    docstring.
    """

    REAL_PROVIDER = "REAL_PROVIDER"
    REPLAY = "REPLAY"
    LOCAL_MODEL = "LOCAL_MODEL"


class DataProvenance(BaseModel):
    """Where a current/live payload came from and how fresh it is."""

    model_config = ConfigDict(frozen=True)

    provider: str
    source_kind: SourceKind
    fetched_at: datetime | None = None
    cache_age_seconds: float | None = None
    ttl_seconds: int | None = None
    is_stale: bool = False
    last_successful_refresh: datetime | None = None


class ModelProvenance(BaseModel):
    """Which frozen artifact produced a prediction, and how far it may be trusted."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_id: str
    training_cutoff_season: str
    artifact_format_version: str
    source_kind: SourceKind = SourceKind.LOCAL_MODEL
    sealed_final_test_completed: bool = False
    usage_note: str = (
        "Frozen pre-final-test model trained through 2024/25. The sealed 2025/26 "
        "evaluation has NOT been run, so outputs are not production-validated "
        "predictions for the current season."
    )


# --------------------------------------------------------------------------
# Current data
# --------------------------------------------------------------------------
class StandingsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    season_label: str
    count: int
    provenance: DataProvenance
    standings: list[StandingRow]


class FixturesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    season_label: str
    count: int
    provenance: DataProvenance
    filters_applied: dict[str, str | None]
    fixtures: list[Fixture]


class LiveMatchesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    provenance: DataProvenance
    matches: list[LiveMatchState]


class LiveMatchStateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provenance: DataProvenance
    match: LiveMatchState | None = None
    """`None` means that fixture is not currently in play - a genuine answer,
    distinct from the provider being unavailable (which raises)."""


# --------------------------------------------------------------------------
# Outcome prediction (frozen strength trio)
# --------------------------------------------------------------------------
class OutcomePredictionRequest(BaseModel):
    """The exact three semantic strength-trio inputs, and nothing else.

    `extra="forbid"` is deliberate: the frozen model was fitted on precisely
    these three features, so an unexpected field is a caller bug and must be a
    422, never silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    elo_diff: float = Field(description="Home Elo minus away Elo (0 = equal strength).")
    diff_ewma_ppg: float | None = Field(
        default=None,
        description="Home minus away EWMA points-per-game. None = unknown (model imputes).",
    )
    diff_ewma_sot_diff: float | None = Field(
        default=None,
        description="Home minus away EWMA shots-on-target dominance. None = unknown (model imputes).",
    )


class OutcomePredictionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_class: str = Field(description="H, D or A.")
    features_used: dict[str, float | None]
    model_provenance: ModelProvenance


# --------------------------------------------------------------------------
# Explanation (exact linear decomposition - no LLM, no SHAP)
# --------------------------------------------------------------------------
class ClassValues(BaseModel):
    """A value per outcome class, in the fixed H/D/A order."""

    model_config = ConfigDict(frozen=True)

    home: float
    draw: float
    away: float


class FeatureContribution(BaseModel):
    """One semantic feature's exact contribution.

    `grouped_logit_contribution` is additive in LOGIT space and is never a
    percentage-point effect; `probability_sensitivity` is the separate
    counterfactual quantity that genuinely is in probability points.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    human_label: str
    description: str
    raw_value: float | None
    effective_model_value: float
    was_imputed: bool
    value_logit_contribution: ClassValues
    missingness_logit_contribution: ClassValues | None
    grouped_logit_contribution: ClassValues
    probability_sensitivity: ClassValues
    counterfactual_changed_missingness: bool


class DriverRanking(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    value: float
    direction: str


class OutcomeExplanationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    probabilities: ClassValues
    predicted_class: str
    class_order: list[str]
    logits: ClassValues
    intercepts: ClassValues
    features: list[FeatureContribution]
    drivers_by_logit_contribution: list[DriverRanking]
    drivers_by_probability_sensitivity: list[DriverRanking]
    model_provenance: ModelProvenance
    reconstruction_verified: bool = Field(
        default=True,
        description=(
            "True because the underlying implementation asserts its reconstructed "
            "logits/probabilities match the fitted model's decision_function/"
            "predict_proba within 1e-9 before returning."
        ),
    )


# --------------------------------------------------------------------------
# Scoreline prediction (frozen Dixon-Coles) - SEPARATE from the primary
# strength-trio H/D/A model above.
# --------------------------------------------------------------------------
class ScoreModelProvenance(BaseModel):
    """Which frozen scoreline artifact produced this, and its configuration."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_id: str
    model_type: str
    training_cutoff_season: str
    artifact_format_version: str
    l2_sigma: float | None
    decay_half_life_days: float | None
    training_match_count: int
    source_kind: SourceKind = SourceKind.LOCAL_MODEL
    sealed_final_test_completed: bool = False
    usage_note: str = (
        "Frozen Dixon-Coles scoreline model trained through 2024/25. This is "
        "PitchMind's expected-goals / scoreline model, NOT its primary H/D/A "
        "predictor - that remains the strength-trio classifier at /api/v1/predict."
    )


class ScorelinePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    home_team: str = Field(min_length=1, description="Any known club spelling.")
    away_team: str = Field(min_length=1, description="Any known club spelling.")
    top_n: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description=(
            "How many most-likely scorelines to return (1-3, default 3). Capped at 3 "
            "because the frozen score model's predict_match returns exactly three; "
            "advertising more would promise what it cannot deliver."
        ),
    )


class ScorelineProbability(BaseModel):
    model_config = ConfigDict(frozen=True)

    home_goals: int
    away_goals: int
    scoreline: str
    probability: float


class ScorelinePredictionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    home_team: str
    away_team: str
    home_team_model_key: str
    away_team_model_key: str
    expected_home_goals: float
    expected_away_goals: float
    lambda_home: float
    lambda_away: float
    most_likely_scoreline: str
    top_scorelines: list[ScorelineProbability]
    score_model_outcome_probabilities: ClassValues = Field(
        description=(
            "SECONDARY H/D/A probabilities implied by the scoreline matrix. NOT "
            "PitchMind's primary outcome prediction - use /api/v1/predict for that."
        )
    )
    scoreline_grid_max_goals: int = Field(
        description=(
            "Goals per side covered by the distribution. The underlying model "
            "extends this grid until truncated tail mass is below 1e-5, then "
            "renormalises, so the returned probabilities sum consistently."
        )
    )
    model_provenance: ScoreModelProvenance


__all__ = [
    "SourceKind",
    "DataProvenance",
    "ModelProvenance",
    "ScoreModelProvenance",
    "ScorelinePredictionRequest",
    "ScorelinePredictionResponse",
    "ScorelineProbability",
    "StandingsResponse",
    "FixturesResponse",
    "LiveMatchesResponse",
    "LiveMatchStateResponse",
    "OutcomePredictionRequest",
    "OutcomePredictionResponse",
    "ClassValues",
    "FeatureContribution",
    "DriverRanking",
    "OutcomeExplanationResponse",
    "Fixture",
    "FixtureStatus",
    "LiveMatchState",
    "StandingRow",
]

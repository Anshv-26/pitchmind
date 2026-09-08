"""Deterministic Dixon-Coles scoreline tool.

Serves the FROZEN `dixon_coles_l2_decay` artifact persisted at
`models/dixon_coles_l2_decay_through_2024_25.json`. The artifact is loaded
read-only and cached per process - `fit_score_model` is NEVER called on the
serving path.

SECONDARY OUTCOME PROBABILITIES
-------------------------------
The scoreline matrix implies H/D/A probabilities, and they are returned - but
under `score_model_outcome_probabilities`, explicitly labelled secondary.
PitchMind's PRIMARY H/D/A prediction remains the strength-trio classifier
served by `/api/v1/predict`. These two must never be conflated: the trio won
model selection on log loss (0.9587 vs Dixon-Coles' 0.9921).

UNSUPPORTED TEAMS
-----------------
`ScoreModelParams` has a promoted-team prior that substitutes league-derived
offsets for a club absent from training. That prior exists for the ORIGINAL
rolling-origin evaluation, where a promoted club genuinely appears in a
validation season. At the serving boundary it is deliberately NOT used: a
caller asking about a club the model never saw should get an explicit error,
not a plausible-looking number derived from a generic offset. The frozen
mathematics in `ml.score_models` is unchanged - serving is simply stricter.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from backend.app.ml.score_model_artifact import (
    SCORE_MODEL_ARTIFACT_PATH,
    ScoreModelArtifact,
    load_score_model_artifact,
)
from backend.app.ml.score_models import TOP_N_SCORELINES, predict_match
from backend.app.services.football_data.teams import TeamRegistry, default_registry
from backend.app.tools.prediction_tools import ModelArtifactUnavailable
from backend.app.tools.schemas import (
    ClassValues,
    ScoreModelProvenance,
    ScorelinePredictionRequest,
    ScorelinePredictionResponse,
    ScorelineProbability,
)

# The frozen `predict_match` returns exactly TOP_N_SCORELINES (3) entries, so
# this is a real ceiling, not an arbitrary policy choice.
MAX_TOP_SCORELINES = TOP_N_SCORELINES


class TeamNotInScoreModel(ValueError):
    """The club is known to PitchMind but has no fitted Dixon-Coles parameters.

    Raised instead of falling back to the promoted-team prior, so a
    current-only club (e.g. one promoted after the 2024/25 training cutoff)
    can never receive a fabricated scoreline prediction.
    """

    def __init__(self, team: str, canonical_id: str) -> None:
        super().__init__(
            f"team {team!r} (canonical id {canonical_id!r}) has no fitted parameters in the "
            f"frozen Dixon-Coles model, which was trained through 2024/25. No scoreline "
            f"prediction is available for this club."
        )
        self.team = team
        self.canonical_id = canonical_id


@lru_cache(maxsize=1)
def _load_artifact_cached(path_str: str) -> ScoreModelArtifact:
    """Load once per process. Read-only; never triggers a fit."""
    return load_score_model_artifact(path=Path(path_str))


def get_score_model_artifact(path: Path = SCORE_MODEL_ARTIFACT_PATH) -> ScoreModelArtifact:
    """The frozen scoreline artifact, cached in-process.

    Raises `ModelArtifactUnavailable` (shared with the strength-trio path, so
    the API has one thing to map to 503) rather than a raw filesystem error.
    """
    try:
        return _load_artifact_cached(str(path))
    except FileNotFoundError as exc:
        raise ModelArtifactUnavailable(
            "frozen Dixon-Coles scoreline artifact is not available; "
            "run scripts/build_score_model_artifact.py to create it"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ModelArtifactUnavailable(
            f"frozen Dixon-Coles scoreline artifact failed its contract check: {exc}"
        ) from exc


def _resolve_to_fitted_team(
    raw_name: str, artifact: ScoreModelArtifact, registry: TeamRegistry
) -> str:
    """Canonical-resolve a club, then map it to the training-name key the
    fitted model is actually keyed on.

    Two distinct failures, deliberately kept apart:
      * `UnknownTeam` (from the registry) - PitchMind does not know this club.
      * `TeamNotInScoreModel` - we know the club, but this model never saw it.
    """
    team = registry.resolve(raw_name)  # raises UnknownTeam; never fuzzy-matches

    # The fitted model is keyed on historical Football-Data spellings.
    candidates = [*team.historical_names, team.canonical_name, *team.aliases]
    for candidate in candidates:
        if candidate in artifact.params.attack and candidate in artifact.params.defence:
            return candidate

    raise TeamNotInScoreModel(raw_name, team.canonical_id)


def get_scoreline_prediction(
    request: ScorelinePredictionRequest,
    *,
    artifact: ScoreModelArtifact | None = None,
    registry: TeamRegistry | None = None,
) -> ScorelinePredictionResponse:
    """Expected goals and a scoreline distribution for one fixture.

    Every number comes from `ml.score_models.predict_match` on the frozen
    fitted parameters - this function computes no football quantities itself.
    """
    artifact = artifact if artifact is not None else get_score_model_artifact()
    registry = registry if registry is not None else default_registry()

    home_key = _resolve_to_fitted_team(request.home_team, artifact, registry)
    away_key = _resolve_to_fitted_team(request.away_team, artifact, registry)

    prediction = predict_match(artifact.params, artifact.config, home_key, away_key)

    top_n = min(request.top_n or TOP_N_SCORELINES, MAX_TOP_SCORELINES)
    top_scorelines = [
        ScorelineProbability(
            home_goals=home_goals, away_goals=away_goals,
            scoreline=f"{home_goals}-{away_goals}", probability=probability,
        )
        for (home_goals, away_goals), probability in prediction.top_scorelines[:top_n]
    ]

    return ScorelinePredictionResponse(
        home_team=registry.resolve(request.home_team).canonical_name,
        away_team=registry.resolve(request.away_team).canonical_name,
        home_team_model_key=home_key,
        away_team_model_key=away_key,
        expected_home_goals=prediction.expected_home_goals,
        expected_away_goals=prediction.expected_away_goals,
        lambda_home=prediction.lambda_home,
        lambda_away=prediction.lambda_away,
        most_likely_scoreline=prediction.most_likely_scoreline,
        top_scorelines=top_scorelines,
        score_model_outcome_probabilities=ClassValues(
            home=prediction.p_home, draw=prediction.p_draw, away=prediction.p_away
        ),
        scoreline_grid_max_goals=int(prediction.matrix.shape[0] - 1),
        model_provenance=ScoreModelProvenance(
            model_id=artifact.metadata.model_id,
            model_type=artifact.metadata.model_type,
            training_cutoff_season=artifact.metadata.training_cutoff_season,
            artifact_format_version=artifact.metadata.artifact_format_version,
            l2_sigma=artifact.metadata.l2_sigma,
            decay_half_life_days=artifact.metadata.decay_half_life_days,
            training_match_count=artifact.metadata.training_match_count,
            sealed_final_test_completed=False,
        ),
    )


__all__ = [
    "TeamNotInScoreModel",
    "get_score_model_artifact",
    "get_scoreline_prediction",
    "MAX_TOP_SCORELINES",
]

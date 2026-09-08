"""Deterministic outcome-prediction and explanation tools.

Both tools serve the FROZEN `baseline_strength_trio` artifact persisted at
`models/baseline_strength_trio_through_2024_25.joblib`. Neither fits, refits,
nor persists anything: the artifact is loaded read-only and cached in-process.

NO LLM is involved. Probabilities come from the fitted scikit-learn pipeline's
own `predict_proba`; contributions come from the exact linear decomposition in
`ml.explanations` (which asserts its reconstruction matches the model within
1e-9 before returning). No SHAP, no invented numbers.

SEALED-SEASON BOUNDARY
----------------------
The artifact was trained through 2024/25 and the sealed 2025/26 final test has
NOT been run. These tools therefore accept a caller-supplied pre-kickoff
feature snapshot; they never build current-season features (which would
require 2025/26 outcomes), never load the sealed season, and never retrain.
Every response says so via `ModelProvenance.sealed_final_test_completed=False`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.app.ml.baselines import STRENGTH_TRIO_COLUMNS
from backend.app.ml.explanations import (
    STRENGTH_TRIO_ARTIFACT_PATH,
    StrengthTrioArtifact,
    explain_match,
    load_strength_trio_artifact,
)
from backend.app.tools.schemas import (
    ClassValues,
    DriverRanking,
    FeatureContribution,
    ModelProvenance,
    OutcomeExplanationResponse,
    OutcomePredictionRequest,
    OutcomePredictionResponse,
)


class ModelArtifactUnavailable(RuntimeError):
    """The frozen model artifact could not be loaded.

    Distinct from a bad request: the caller did nothing wrong, the deployment
    is missing its artifact. The API maps this to 503, never a traceback.
    """


@lru_cache(maxsize=1)
def _load_artifact_cached(path_str: str) -> StrengthTrioArtifact:
    """Load once per process. Loading is read-only and never triggers a fit."""
    return load_strength_trio_artifact(path=Path(path_str))


def get_artifact(path: Path = STRENGTH_TRIO_ARTIFACT_PATH) -> StrengthTrioArtifact:
    """The frozen artifact, cached in-process.

    Raises `ModelArtifactUnavailable` (never a raw FileNotFoundError/ValueError)
    so the serving layer has one thing to catch.
    """
    try:
        return _load_artifact_cached(str(path))
    except FileNotFoundError as exc:
        raise ModelArtifactUnavailable(
            "frozen strength-trio model artifact is not available; "
            "run scripts/build_model_artifacts.py to create it"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ModelArtifactUnavailable(
            f"frozen strength-trio model artifact failed its contract check: {exc}"
        ) from exc


def _model_provenance(artifact: StrengthTrioArtifact) -> ModelProvenance:
    return ModelProvenance(
        model_id=artifact.metadata.model_id,
        training_cutoff_season=artifact.metadata.training_cutoff_season,
        artifact_format_version=artifact.metadata.artifact_format_version,
        sealed_final_test_completed=False,
    )


def _feature_frame(request: OutcomePredictionRequest) -> pd.DataFrame:
    """Build the single-row frame the frozen pipeline expects.

    Column order comes from `STRENGTH_TRIO_COLUMNS` (imported, not retyped) so
    it cannot drift from what the model was fitted on. `None` becomes NaN,
    which the pipeline's fitted imputer handles - and which the explanation
    layer reports as `was_imputed=True`.
    """
    row = {
        "elo_diff": request.elo_diff,
        "diff_ewma_ppg": request.diff_ewma_ppg,
        "diff_ewma_sot_diff": request.diff_ewma_sot_diff,
    }
    return pd.DataFrame([row])[list(STRENGTH_TRIO_COLUMNS)]


def run_outcome_prediction(
    request: OutcomePredictionRequest, *, artifact: StrengthTrioArtifact | None = None
) -> OutcomePredictionResponse:
    """H/D/A probabilities from the frozen model, for one pre-kickoff snapshot."""
    artifact = artifact if artifact is not None else get_artifact()
    frame = _feature_frame(request)

    proba = artifact.pipeline.predict_proba(frame)[0]
    class_names = ("H", "D", "A")
    predicted_class = class_names[int(proba.argmax())]

    return OutcomePredictionResponse(
        home_win_probability=float(proba[0]),
        draw_probability=float(proba[1]),
        away_win_probability=float(proba[2]),
        predicted_class=predicted_class,
        features_used={
            "elo_diff": request.elo_diff,
            "diff_ewma_ppg": request.diff_ewma_ppg,
            "diff_ewma_sot_diff": request.diff_ewma_sot_diff,
        },
        model_provenance=_model_provenance(artifact),
    )


def _class_values(raw: dict) -> ClassValues:
    return ClassValues(home=raw["home"], draw=raw["draw"], away=raw["away"])


def explain_outcome_prediction(
    request: OutcomePredictionRequest, *, artifact: StrengthTrioArtifact | None = None
) -> OutcomeExplanationResponse:
    """The exact model-grounded explanation for one pre-kickoff snapshot.

    Delegates entirely to `ml.explanations.explain_match` - this function only
    reshapes that already-verified output into typed response models. It adds
    no numbers of its own.
    """
    artifact = artifact if artifact is not None else get_artifact()
    explanation = explain_match(artifact, _feature_frame(request))

    features = [
        FeatureContribution(
            name=feature["name"],
            human_label=feature["human_label"],
            description=feature["description"],
            raw_value=feature["raw_value"],
            effective_model_value=feature["effective_model_value"],
            was_imputed=feature["was_imputed"],
            value_logit_contribution=_class_values(feature["value_logit_contribution"]),
            missingness_logit_contribution=(
                _class_values(feature["missingness_logit_contribution"])
                if feature["missingness_logit_contribution"] is not None
                else None
            ),
            grouped_logit_contribution=_class_values(feature["grouped_logit_contribution"]),
            probability_sensitivity=_class_values(feature["probability_sensitivity"]),
            counterfactual_changed_missingness=feature["counterfactual_changed_missingness"],
        )
        for feature in explanation["features"]
    ]

    return OutcomeExplanationResponse(
        probabilities=_class_values(explanation["probabilities"]),
        predicted_class=explanation["predicted_class"],
        class_order=list(explanation["class_order"]),
        logits=_class_values(explanation["logits"]),
        intercepts=_class_values(explanation["intercepts"]),
        features=features,
        drivers_by_logit_contribution=[
            DriverRanking(
                feature=driver["feature"],
                value=driver["grouped_logit_contribution_predicted_class"],
                direction=driver["direction"],
            )
            for driver in explanation["drivers_by_logit_contribution"]
        ],
        drivers_by_probability_sensitivity=[
            DriverRanking(
                feature=driver["feature"],
                value=driver["probability_sensitivity_predicted_class"],
                direction=driver["direction"],
            )
            for driver in explanation["drivers_by_probability_sensitivity"]
        ],
        model_provenance=_model_provenance(artifact),
    )


__all__ = [
    "ModelArtifactUnavailable",
    "get_artifact",
    "run_outcome_prediction",
    "explain_outcome_prediction",
]

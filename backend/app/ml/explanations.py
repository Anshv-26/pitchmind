"""Model-grounded explanation layer for PitchMind's frozen champions.

This module turns the ALREADY-FITTED `baseline_strength_trio` model into a
deterministic, numerically-verified explanation of one match: which of the
three semantic features pushed the prediction toward Home, Draw, or Away, and
by how much. Every number in the output either comes directly from the fitted
`sklearn` objects or from re-running the fitted pipeline's own
`predict_proba` on a counterfactual row - nothing here approximates or
re-derives model behaviour independently. Claude's later role is to phrase
these numbers in prose; it must never originate them.

WHY EXACT DECOMPOSITION, NOT SHAP
----------------------------------
`baseline_strength_trio` is `SimpleImputer(add_indicator=True) ->
StandardScaler() -> LogisticRegression(...)` - a single linear layer plus a
softmax. A closed-form reconstruction of its logits and probabilities exists
and has been verified bit-identical (max abs diff 0.0) against
`decision_function`/`predict_proba` on real fold data. SHAP's LinearExplainer
would compute the same `coef * (x - reference)` quantity at the cost of a new
dependency, a background-data choice, and an API whose output shape has
changed across versions - all to reproduce a sum this module already computes
exactly. SHAP is unnecessary for this production champion and is not used
anywhere in this module; `shap` is not installed and is not added.

THE 3-VS-5 COLUMN FACT THAT DRIVES THIS MODULE'S STRUCTURE
------------------------------------------------------------
The fitted pipeline's `SimpleImputer(add_indicator=True)` adds a missingness
indicator column for every feature that had a NaN anywhere in training -
verified to be `diff_ewma_ppg` and `diff_ewma_sot_diff` only (`elo_diff` is
never missing), so `model.coef_.shape == (3, 5)`, not `(3, 3)`. Every
contribution computed here is reported at THREE levels so this fact is never
hidden nor silently lost:

    value_logit_contribution        - the feature's own (imputed) numeric value
    missingness_logit_contribution  - its indicator column, or None if that
                                       feature structurally has no indicator
    grouped_logit_contribution      - the sum of the two above; this is the
                                       number a user-facing "3 features" story
                                       actually needs, and it is exact:
                                       intercept + sum(grouped over the 3
                                       semantic features) == the class logit,
                                       for every class, on every row.

LOGIT CONTRIBUTION VS PROBABILITY SENSITIVITY - NEVER CONFLATED
------------------------------------------------------------------
`grouped_logit_contribution` is additive and exact, but it lives in LOGIT
space - softmax is nonlinear, so it is never a percentage-point effect and is
never labelled as one anywhere in this module or its output schema.
`probability_sensitivity` is a genuine, separately-computed counterfactual:
rerun the FULL fitted pipeline's `predict_proba` with one semantic feature set
to 0 (its natural "teams are equal on this signal" reference - all three
features are home-minus-away differences, so 0 is not an approximation), and
report the difference from the original probability. This is a real
percentage-point quantity because it is the difference of two real
`predict_proba` calls, not a linearisation of the logit contribution.

If the original raw value was NaN (imputed), zeroing it changes not only the
imputed numeric value but also the missingness indicator (0.0 is observed,
not missing) - `counterfactual_changed_missingness` exposes this explicitly so
that case is never described as isolating "the numeric effect alone".
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from backend.app.ml.baselines import STRENGTH_TRIO_COLUMNS, _fit_logistic_pipeline
from backend.app.ml.datasets import DEVELOPMENT_FOLD_DEFINITIONS, artifact_path_for_fold
from backend.app.ml.evaluation import CLASS_NAMES, EXPECTED_CLASSES

# JSON-facing long-form class names, positionally aligned with CLASS_NAMES /
# EXPECTED_CLASSES (both "H","D","A" and 0,1,2 order) reused from
# evaluation.py. `predicted_class` and `class_order` use the short H/D/A code
# (matching evaluation.py / TARGET_MAPPING); every probability/logit/
# contribution dict and every "direction" string uses these long-form names,
# per the approved explanation JSON contract.
CLASS_LONG_NAMES: tuple[str, ...] = ("home", "draw", "away")
from backend.app.ml.feature_engineering import (
    SEALED_SEASON,
    TARGET_COLUMN,
    assert_artifact_valid_for,
)
from backend.app.ml.score_models import (
    ScoreModelConfig,
    ScoreModelParams,
    predict_match as predict_score_model_match,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------
# Human-facing feature semantics - deterministic labels, no tactical prose.
# --------------------------------------------------------------------------
FEATURE_HUMAN_LABELS: dict[str, str] = {
    "elo_diff": "Overall team strength",
    "diff_ewma_ppg": "Recent form",
    "diff_ewma_sot_diff": "Shots-on-target dominance",
}
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "elo_diff": "Long-term relative team strength from Elo.",
    "diff_ewma_ppg": "Difference in exponentially weighted recent points per game.",
    "diff_ewma_sot_diff": "Difference in recent exponentially weighted shots-on-target dominance.",
}

# Zero-reference counterfactual: all three STRENGTH_TRIO_COLUMNS are signed
# home-minus-away differences, so 0.0 means "teams equal on this signal" by
# construction, not by convention - and it uses no training/validation data.
COUNTERFACTUAL_REFERENCE_VALUE = 0.0

# How tightly this module's own reconstructed logits/probabilities must agree
# with the fitted model's decision_function/predict_proba. Verified bit-
# identical (0.0 diff) on real data; this tolerance exists only to survive
# ordinary floating-point noise, never to paper over a real discrepancy.
RECONSTRUCTION_TOLERANCE = 1e-9

STRENGTH_TRIO_ARTIFACT_FORMAT_VERSION = "1.0"

# The frozen model is fit on every match through this season inclusive.
TRAINING_CUTOFF_SEASON = "2024_25"

# Source feature artifact: deliberately the SAME parquet file Stage 1's
# development fold 3 reads (datasets.DEVELOPMENT_FOLD_DEFINITIONS[3]) - NOT
# because this is a fold-validation exercise, but because that file's row cap
# already stops at 2024_25. It is structurally INCAPABLE of containing
# 2025/26 rows, unlike `features_causal_through_2024_25.parquet` (whose row
# cap extends into 2025/26 specifically so that file can later be used to
# build 2025/26 FEATURES for scoring - reading target/outcome rows from it
# here would put a sealed-season row one `Season != SEALED_SEASON` filter
# away from entering training; reading from the 2023_24-cutoff file instead
# removes that failure mode entirely, since there is nothing to filter).
_SOURCE_FOLD_NUMBER = 3
assert DEVELOPMENT_FOLD_DEFINITIONS[_SOURCE_FOLD_NUMBER]["validation_season"] == TRAINING_CUTOFF_SEASON, (
    "the source fold's row-cap season must match TRAINING_CUTOFF_SEASON"
)
SOURCE_FEATURE_ARTIFACT_PATH = artifact_path_for_fold(_SOURCE_FOLD_NUMBER)

STRENGTH_TRIO_ARTIFACT_PATH = REPO_ROOT / "models" / "baseline_strength_trio_through_2024_25.joblib"


# --------------------------------------------------------------------------
# Model artifact: build once, load read-only, never retrain per request.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ArtifactMetadata:
    """Everything needed to verify a loaded artifact's contract without
    touching the fitted pipeline object itself."""

    model_id: str
    artifact_format_version: str
    training_cutoff_season: str
    training_seasons: list[str]
    feature_columns: list[str]
    transformed_feature_names: list[str]
    class_mapping: dict[str, int]
    source_feature_artifact: str
    n_training_rows: int
    library_versions: dict[str, str]
    # This artifact is FROZEN and ready-to-use: it is the exact model that
    # will eventually be applied to `intended_final_evaluation_season`
    # (2025/26), and `requires_retraining_before_final_evaluation` is always
    # False for a valid artifact - there is no retrain-then-replace step
    # between now and that evaluation (see `StrengthTrioArtifact`'s docstring
    # for the full lifecycle).
    intended_final_evaluation_season: str
    requires_retraining_before_final_evaluation: bool


@dataclass(frozen=True)
class StrengthTrioArtifact:
    """The FROZEN `baseline_strength_trio` inference model plus its provenance.

    Model-selection and calibration/ensemble decisions are complete (the raw
    strength trio was retained). This artifact is the direct result: fit ONCE
    on every match through 2024/25 inclusive, with no development season held
    out, since there is no longer a decision left to validate against a
    held-out season.

    This IS the model that will eventually be applied EXACTLY ONCE to the
    sealed 2025/26 season, once the rest of the project is frozen. It is not
    a temporary artifact to be retrained or replaced before that evaluation:

        development / model-selection completed
                    v
        fit this frozen model once, through 2024/25   <- this artifact
                    v
        serialize artifact
                    v
        build the product around this frozen contract
                    v
        when everything else is finally frozen: apply THIS artifact to
        2025/26, exactly once

    A separate, later, POST-EVALUATION production model may eventually be
    built for deployment - but only after the sealed-test result is recorded,
    and it is not part of the current evaluation protocol or this artifact's
    lifecycle. Building or loading this artifact never opens or scores
    2025/26: `build_strength_trio_artifact` reads a feature file whose row
    cap stops at 2024/25 and cannot structurally contain sealed-season rows.
    """

    metadata: ArtifactMetadata
    pipeline: Pipeline


def library_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def build_strength_trio_artifact() -> StrengthTrioArtifact:
    """Fit the FROZEN `baseline_strength_trio` model on every match through
    2024/25 inclusive - the one-time fit described in `StrengthTrioArtifact`.

    Takes NO arguments. There is exactly one approved training source for
    this artifact - `SOURCE_FEATURE_ARTIFACT_PATH`
    (`features_causal_through_2023_24.parquet`) - and it is used internally,
    not accepted as a parameter. An arbitrary `source_path` argument would
    itself be an unsafe "train on arbitrary seasons" path: nothing would then
    stop a caller from pointing this function at
    `features_causal_through_2024_25.parquet` (whose row cap extends into
    2025/26) or any other file. Removing the parameter removes that
    possibility structurally rather than relying on callers to pass the
    right default.

    Safety, in order:
      1. Reads `SOURCE_FEATURE_ARTIFACT_PATH`, the SAME parquet file Stage 1's
         development fold 3 reads. Its row cap stops at 2024/25, so it is
         structurally INCAPABLE of containing 2025/26 rows - unlike
         `features_causal_through_2024_25.parquet` (whose row cap extends
         into 2025/26 to serve a different purpose: building 2025/26
         FEATURES for later scoring), which this function never reads.
      2. `assert_artifact_valid_for(SOURCE_FEATURE_ARTIFACT_PATH,
         [TRAINING_CUTOFF_SEASON])` - the mandatory provenance gate already
         used everywhere else in this codebase before any fit. This call
         reads and validates only the artifact's JSON provenance sidecar
         (row counts, season contents, training-cutoff/evaluation-season
         pairing); it confirms this is the artifact correctly built with
         training cutoff paired to `TRAINING_CUTOFF_SEASON` (2024/25).
      3. Assert the sealed season is genuinely absent from the loaded frame
         (defence in depth - even though step 1's file choice already makes
         this structurally impossible, this is never merely trusted).
      4. Assert the training seasons end exactly at `TRAINING_CUTOFF_SEASON`
         - catches a source-artifact mismatch loudly rather than silently
         training on the wrong window.
      5. Fit on EVERY row in the file (2015/16 through 2024/25 inclusive) -
         no train/validation split, since model-selection is already frozen
         and there is no longer a decision left to validate against a
         held-out season.
    """
    assert_artifact_valid_for(SOURCE_FEATURE_ARTIFACT_PATH, [TRAINING_CUTOFF_SEASON])

    train_frame = pd.read_parquet(SOURCE_FEATURE_ARTIFACT_PATH)

    # Defence in depth: this file's row cap already makes the sealed season
    # structurally absent (see step 1 in the docstring above), so there is
    # nothing to filter out - but that structural guarantee is never merely
    # trusted without also being checked here.
    present_seasons = set(train_frame["Season"].unique())
    if SEALED_SEASON in present_seasons:
        raise RuntimeError(
            f"sealed season {SEALED_SEASON!r} is present in "
            f"{SOURCE_FEATURE_ARTIFACT_PATH}; this artifact-building path must "
            f"never read a feature artifact whose rows can include the sealed "
            f"season. Refusing to fit."
        )

    training_seasons = sorted(train_frame["Season"].unique().tolist())
    if not training_seasons or training_seasons[-1] != TRAINING_CUTOFF_SEASON:
        raise ValueError(
            f"expected training seasons to end at {TRAINING_CUTOFF_SEASON!r}, "
            f"got {training_seasons[-1] if training_seasons else 'no rows'} "
            f"(full list: {training_seasons})"
        )

    X = train_frame[STRENGTH_TRIO_COLUMNS]
    y = train_frame[TARGET_COLUMN]
    pipeline = _fit_logistic_pipeline(X, y, with_imputer=True)

    imputer = pipeline.named_steps["imputer"]
    transformed_names = list(imputer.get_feature_names_out(STRENGTH_TRIO_COLUMNS))

    metadata = ArtifactMetadata(
        model_id="baseline_strength_trio",
        artifact_format_version=STRENGTH_TRIO_ARTIFACT_FORMAT_VERSION,
        training_cutoff_season=TRAINING_CUTOFF_SEASON,
        training_seasons=training_seasons,
        feature_columns=list(STRENGTH_TRIO_COLUMNS),
        transformed_feature_names=transformed_names,
        class_mapping={name: cls for name, cls in zip(CLASS_NAMES, EXPECTED_CLASSES)},
        source_feature_artifact=str(SOURCE_FEATURE_ARTIFACT_PATH),
        n_training_rows=len(train_frame),
        library_versions=library_versions(),
        intended_final_evaluation_season=SEALED_SEASON,
        requires_retraining_before_final_evaluation=False,
    )
    return StrengthTrioArtifact(metadata=metadata, pipeline=pipeline)


def _validate_artifact_contract(artifact: StrengthTrioArtifact) -> None:
    """Fail loudly on any contract mismatch. Called on every load."""
    meta = artifact.metadata
    if meta.model_id != "baseline_strength_trio":
        raise ValueError(f"unexpected model_id {meta.model_id!r} in loaded artifact")
    if meta.artifact_format_version != STRENGTH_TRIO_ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"artifact format version {meta.artifact_format_version!r} does not match "
            f"the version this code expects ({STRENGTH_TRIO_ARTIFACT_FORMAT_VERSION!r})"
        )
    if list(meta.feature_columns) != list(STRENGTH_TRIO_COLUMNS):
        raise ValueError(
            f"artifact feature columns {meta.feature_columns} do not match the current "
            f"STRENGTH_TRIO_COLUMNS {STRENGTH_TRIO_COLUMNS}"
        )
    if meta.training_cutoff_season != TRAINING_CUTOFF_SEASON:
        raise ValueError(
            f"artifact training cutoff {meta.training_cutoff_season!r} does not match "
            f"the expected {TRAINING_CUTOFF_SEASON!r}"
        )
    if SEALED_SEASON in meta.training_seasons:
        raise ValueError(f"artifact metadata lists the sealed season {SEALED_SEASON!r} as a training season")
    if meta.class_mapping != {name: cls for name, cls in zip(CLASS_NAMES, EXPECTED_CLASSES)}:
        raise ValueError(f"artifact class_mapping {meta.class_mapping} does not match the expected H/D/A order")
    if meta.intended_final_evaluation_season != SEALED_SEASON:
        raise ValueError(
            f"artifact intended_final_evaluation_season {meta.intended_final_evaluation_season!r} "
            f"does not match the sealed season {SEALED_SEASON!r}"
        )
    if meta.requires_retraining_before_final_evaluation:
        raise ValueError(
            "loaded artifact claims retraining is required before its final evaluation - this "
            "contract requires a frozen, ready-to-use model with nothing left to refit"
        )
    if not isinstance(artifact.pipeline, Pipeline) or "model" not in artifact.pipeline.named_steps:
        raise ValueError("artifact pipeline is missing the expected 'model' step")


def save_strength_trio_artifact(
    artifact: StrengthTrioArtifact, path: Path = STRENGTH_TRIO_ARTIFACT_PATH
) -> None:
    """Persist the artifact via joblib, plus a small JSON metadata sidecar
    (metadata only - never the fitted pipeline binary) for inspectability."""
    _validate_artifact_contract(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar_path.write_text(json.dumps(asdict(artifact.metadata), indent=2, sort_keys=True) + "\n")


def load_strength_trio_artifact(path: Path = STRENGTH_TRIO_ARTIFACT_PATH) -> StrengthTrioArtifact:
    """Load a previously-built artifact. Never fits or refits anything."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_model_artifacts.py first")
    artifact = joblib.load(path)
    if not isinstance(artifact, StrengthTrioArtifact):
        raise TypeError(f"{path} does not contain a StrengthTrioArtifact (got {type(artifact)!r})")
    _validate_artifact_contract(artifact)
    return artifact


# --------------------------------------------------------------------------
# Transformed-column <-> semantic-feature grouping
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _ColumnGroup:
    value_index: int
    indicator_index: int | None


def _build_column_groups(pipeline: Pipeline, semantic_features: list[str]) -> dict[str, _ColumnGroup]:
    """Map each semantic feature to its transformed-column index (and its
    missingness-indicator column index, if the fitted imputer created one).

    Read directly from `imputer.get_feature_names_out`, never re-derived by
    assuming column order or by independently recomputing which features had
    missing values - so this can never silently drift from what the fitted
    imputer actually did.
    """
    imputer = pipeline.named_steps.get("imputer")
    if imputer is None:
        return {feature: _ColumnGroup(value_index=i, indicator_index=None) for i, feature in enumerate(semantic_features)}

    output_names = list(imputer.get_feature_names_out(semantic_features))
    groups: dict[str, _ColumnGroup] = {}
    for feature in semantic_features:
        value_index = output_names.index(feature)
        indicator_name = f"missingindicator_{feature}"
        indicator_index = output_names.index(indicator_name) if indicator_name in output_names else None
        groups[feature] = _ColumnGroup(value_index=value_index, indicator_index=indicator_index)
    return groups


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _class_dict(values: np.ndarray) -> dict[str, float]:
    """Long-form {"home", "draw", "away"} dict, positionally aligned with
    CLASS_NAMES/EXPECTED_CLASSES - this is the JSON-facing key convention for
    every probability/logit/contribution field in the explanation contract."""
    return {name: float(v) for name, v in zip(CLASS_LONG_NAMES, values)}


# --------------------------------------------------------------------------
# Core explanation computation
# --------------------------------------------------------------------------
def explain_match(artifact: StrengthTrioArtifact, raw_row: pd.DataFrame) -> dict[str, Any]:
    """Produce a deterministic, numerically-verified explanation for one match.

    `raw_row` must be a single-row DataFrame containing exactly
    `STRENGTH_TRIO_COLUMNS`. No fitting happens here - only reads of the
    already-fitted `artifact.pipeline`.
    """
    if len(raw_row) != 1:
        raise ValueError(f"explain_match expects exactly one row, got {len(raw_row)}")
    if set(raw_row.columns) != set(STRENGTH_TRIO_COLUMNS):
        raise ValueError(
            f"raw_row columns {list(raw_row.columns)} do not match "
            f"STRENGTH_TRIO_COLUMNS {STRENGTH_TRIO_COLUMNS}"
        )
    raw_row = raw_row[STRENGTH_TRIO_COLUMNS]

    pipeline = artifact.pipeline
    imputer = pipeline.named_steps.get("imputer")
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    groups = _build_column_groups(pipeline, STRENGTH_TRIO_COLUMNS)

    def transform_pre_scale(row: pd.DataFrame) -> np.ndarray:
        return imputer.transform(row) if imputer is not None else row.to_numpy(dtype=float)

    original_transformed = transform_pre_scale(raw_row)
    original_scaled = scaler.transform(original_transformed)
    original_proba_sklearn = pipeline.predict_proba(raw_row)[0]

    # Reconstruct logits and probabilities independently, and verify they
    # match the fitted model's own decision_function/predict_proba exactly
    # (within floating-point tolerance) before trusting the decomposition.
    coef = model.coef_  # shape (3, n_transformed)
    intercept = model.intercept_  # shape (3,)
    contributions = coef * original_scaled[0][None, :]  # shape (3, n_transformed)
    reconstructed_logits = intercept + contributions.sum(axis=1)

    decision_logits = model.decision_function(original_scaled)[0]
    if not np.allclose(reconstructed_logits, decision_logits, atol=RECONSTRUCTION_TOLERANCE):
        raise RuntimeError(
            f"reconstructed logits {reconstructed_logits} do not match "
            f"decision_function {decision_logits} within {RECONSTRUCTION_TOLERANCE}"
        )
    reconstructed_proba = _softmax(reconstructed_logits)
    if not np.allclose(reconstructed_proba, original_proba_sklearn, atol=RECONSTRUCTION_TOLERANCE):
        raise RuntimeError(
            f"reconstructed probabilities {reconstructed_proba} do not match "
            f"predict_proba {original_proba_sklearn} within {RECONSTRUCTION_TOLERANCE}"
        )

    predicted_index = int(np.argmax(reconstructed_proba))
    predicted_class_name = CLASS_NAMES[predicted_index]  # short code, e.g. "H"
    predicted_class_long_name = CLASS_LONG_NAMES[predicted_index]  # e.g. "home"

    feature_entries: list[dict[str, Any]] = []
    for feature in STRENGTH_TRIO_COLUMNS:
        group = groups[feature]
        value_component = contributions[:, group.value_index]

        if group.indicator_index is not None:
            missingness_component = contributions[:, group.indicator_index]
            was_imputed = bool(original_transformed[0, group.indicator_index] == 1.0)
            missingness_contribution_dict = _class_dict(missingness_component)
        else:
            missingness_component = np.zeros(3)
            was_imputed = False
            missingness_contribution_dict = None

        grouped_component = value_component + missingness_component

        raw_value = raw_row[feature].iloc[0]
        effective_model_value = float(original_transformed[0, group.value_index])

        # Zero-reference counterfactual, through the REAL fitted pipeline -
        # never by editing the transformed/scaled array by hand.
        counterfactual_row = raw_row.copy()
        counterfactual_row[feature] = COUNTERFACTUAL_REFERENCE_VALUE
        counterfactual_proba = pipeline.predict_proba(counterfactual_row)[0]
        sensitivity = original_proba_sklearn - counterfactual_proba

        counterfactual_changed_missingness = False
        if group.indicator_index is not None:
            counterfactual_transformed = transform_pre_scale(counterfactual_row)
            counterfactual_changed_missingness = bool(
                original_transformed[0, group.indicator_index]
                != counterfactual_transformed[0, group.indicator_index]
            )

        feature_entries.append(
            {
                "name": feature,
                "human_label": FEATURE_HUMAN_LABELS[feature],
                "description": FEATURE_DESCRIPTIONS[feature],
                "raw_value": (None if pd.isna(raw_value) else float(raw_value)),
                "effective_model_value": effective_model_value,
                "was_imputed": was_imputed,
                "value_logit_contribution": _class_dict(value_component),
                "missingness_logit_contribution": missingness_contribution_dict,
                "grouped_logit_contribution": _class_dict(grouped_component),
                "probability_sensitivity": _class_dict(sensitivity),
                "counterfactual_changed_missingness": counterfactual_changed_missingness,
                "_grouped_predicted_class_value": float(grouped_component[predicted_index]),
                "_sensitivity_predicted_class_value": float(sensitivity[predicted_index]),
            }
        )

    def _direction(value: float) -> str:
        if value > 0:
            return f"supports_{predicted_class_long_name}"
        if value < 0:
            return f"opposes_{predicted_class_long_name}"
        return "neutral"

    drivers_by_logit_contribution = sorted(
        (
            {
                "feature": entry["name"],
                "grouped_logit_contribution_predicted_class": entry["_grouped_predicted_class_value"],
                "direction": _direction(entry["_grouped_predicted_class_value"]),
            }
            for entry in feature_entries
        ),
        key=lambda d: d["grouped_logit_contribution_predicted_class"],
        reverse=True,
    )
    drivers_by_probability_sensitivity = sorted(
        (
            {
                "feature": entry["name"],
                "probability_sensitivity_predicted_class": entry["_sensitivity_predicted_class_value"],
                "direction": _direction(entry["_sensitivity_predicted_class_value"]),
            }
            for entry in feature_entries
        ),
        key=lambda d: d["probability_sensitivity_predicted_class"],
        reverse=True,
    )

    for entry in feature_entries:
        del entry["_grouped_predicted_class_value"]
        del entry["_sensitivity_predicted_class_value"]

    return {
        "model_id": artifact.metadata.model_id,
        "artifact_provenance": {
            "training_cutoff_season": artifact.metadata.training_cutoff_season,
            "artifact_format_version": artifact.metadata.artifact_format_version,
        },
        "class_order": list(CLASS_NAMES),
        "probabilities": _class_dict(reconstructed_proba),
        "predicted_class": predicted_class_name,
        "logits": _class_dict(reconstructed_logits),
        "intercepts": _class_dict(intercept),
        "features": feature_entries,
        "drivers_by_logit_contribution": drivers_by_logit_contribution,
        "drivers_by_probability_sensitivity": drivers_by_probability_sensitivity,
    }


# --------------------------------------------------------------------------
# Dixon-Coles explanation - minimal, deterministic, NOT the logistic schema
# --------------------------------------------------------------------------
def explain_score_model_match(
    params: ScoreModelParams, config: ScoreModelConfig, home_team: str, away_team: str
) -> dict[str, Any]:
    """A minimal deterministic explanation for one Dixon-Coles/Poisson fixture.

    Exposes only quantities the score model already computed
    (`score_models.predict_match`) - expected goals, home advantage, fitted
    attack/defence strengths, rho, and top scorelines. No feature-attribution
    framework, no SHAP, and no tactical narrative: this is a direct read of
    already-fitted model state, not a new explanation method.
    """
    prediction = predict_score_model_match(params, config, home_team, away_team)
    return {
        "model_id": config.config_id,
        "home_team": home_team,
        "away_team": away_team,
        "expected_home_goals": prediction.expected_home_goals,
        "expected_away_goals": prediction.expected_away_goals,
        "home_advantage": params.home_advantage,
        "intercept": params.intercept,
        "attack": {
            "home_team": params.team_attack(home_team),
            "away_team": params.team_attack(away_team),
        },
        "defence": {
            "home_team": params.team_defence(home_team),
            "away_team": params.team_defence(away_team),
        },
        "rho": params.rho,
        "probabilities": {
            "home": prediction.p_home,
            "draw": prediction.p_draw,
            "away": prediction.p_away,
        },
        "top_scorelines": [
            {"score": f"{home_goals}-{away_goals}", "probability": probability}
            for (home_goals, away_goals), probability in prediction.top_scorelines
        ],
        "most_likely_scoreline": prediction.most_likely_scoreline,
    }

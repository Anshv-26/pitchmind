"""The Stage 1 rolling-origin training/evaluation loop.

Fits each model family on each development fold's TRAINING rows only,
predicts on that fold's validation rows only, and aggregates metrics across
the three approved folds (validating on 2022/23, 2023/24, 2024/25). 2025/26
never appears here - `datasets.load_fold` has no path to it, and nothing in
this module accepts a raw path or fold number outside `datasets.DEVELOPMENT_FOLDS`.

Model-selection rule (pre-committed, Stage 1 plan section 10):

1. Lowest mean development log loss.
2. Compare every other candidate against the current best via a PAIRED
   per-row log-loss comparison over the pooled development rows
   (`evaluation.paired_log_loss_comparison`); treat as a tie when
   `abs(mean_diff) < 2 * standard_error`.
3. Among tied candidates, prefer lower worst-season log loss.
4. Then prefer the simpler model family (logreg < random_forest <
   xgboost ~ catboost).
5. Then lower across-fold log-loss variance.

Accuracy never enters model selection.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import catboost
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.app.ml.baselines import (
    class_frequency_baseline,
    elo_only_baseline,
    strength_trio_baseline,
)
from backend.app.ml.datasets import DEVELOPMENT_FOLDS, FoldData, load_all_development_folds
from backend.app.ml.evaluation import (
    FoldMetrics,
    align_proba_to_expected_classes,
    compute_fold_metrics,
    paired_log_loss_comparison,
)
from backend.app.ml.feature_engineering import FEATURE_COLUMNS, LINEAR_SAFE_FEATURE_COLUMNS

RANDOM_SEED = 42

# V1 default: no class weighting anywhere. Optimising minority-class (draw)
# recall via class_weight actively damages probability calibration and log
# loss, which are the primary objective here - see Stage 1 plan section 7.
# A weighted variant may exist later only as an explicit, separately-labelled
# experimental configuration, never the default.

# Complexity ranking for the model-selection tie-break (lower = simpler).
# Baseline names are absent (default rank 99), so a baseline can still win
# outright on log loss, but never wins a tie against a real model family.
MODEL_COMPLEXITY_RANK: dict[str, int] = {
    "logreg": 0,
    "random_forest": 1,
    "xgboost": 2,
    "catboost": 2,
}


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """One model family: which features it receives, how to build it, and
    the compact hyperparameter grid to sweep across development folds."""

    name: str
    feature_columns: tuple[str, ...]
    build: Callable[[dict[str, Any]], Any]
    param_grid: list[dict[str, Any]]


def build_logistic_regression(params: dict[str, Any]) -> Pipeline:
    """LogReg on LINEAR_SAFE_FEATURE_COLUMNS: median-impute (+missingness
    indicators) -> scale -> L2-regularised multinomial logistic regression."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=params["C"],
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


LOGISTIC_REGRESSION_GRID: list[dict[str, Any]] = [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]]


def build_random_forest(params: dict[str, Any]) -> Pipeline:
    """Random Forest on the full FEATURE_COLUMNS: median-impute (+indicators,
    since sklearn's RandomForestClassifier has no native NaN support) ->
    forest. No scaler - trees are scale-invariant."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=RANDOM_SEED,
                    n_jobs=1,  # conservative: stable, reproducible local runs
                ),
            ),
        ]
    )


RANDOM_FOREST_GRID: list[dict[str, Any]] = [
    {"max_depth": depth, "min_samples_leaf": leaf}
    for depth in [6, 10, None]
    for leaf in [5, 20]
]


def build_xgboost(params: dict[str, Any]) -> XGBClassifier:
    """XGBoost on the full FEATURE_COLUMNS, native NaN handling (learned
    per-split default direction) - no imputer. multi:softprob for calibrated
    3-class probabilities; CPU-only, single-threaded for reproducibility."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


XGBOOST_GRID: list[dict[str, Any]] = [
    {"max_depth": depth, "learning_rate": lr, "n_estimators": n}
    for depth in [2, 3, 4]
    for lr in [0.03, 0.1]
    for n in [200, 500]
]


def build_catboost(params: dict[str, Any]) -> CatBoostClassifier:
    """CatBoost on the full FEATURE_COLUMNS, native NaN handling - no
    imputer, no categorical (team-identity) features in V1. Fixed iteration
    count (not tuned via early stopping on the validation season)."""
    return CatBoostClassifier(
        loss_function="MultiClass",
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        iterations=500,
        random_seed=RANDOM_SEED,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )


CATBOOST_GRID: list[dict[str, Any]] = [
    {"depth": depth, "learning_rate": lr, "l2_leaf_reg": reg}
    for depth in [4, 6]
    for lr in [0.03, 0.1]
    for reg in [3, 10]
]

MODEL_SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        name="logreg",
        feature_columns=tuple(LINEAR_SAFE_FEATURE_COLUMNS),
        build=build_logistic_regression,
        param_grid=LOGISTIC_REGRESSION_GRID,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        feature_columns=tuple(FEATURE_COLUMNS),
        build=build_random_forest,
        param_grid=RANDOM_FOREST_GRID,
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        feature_columns=tuple(FEATURE_COLUMNS),
        build=build_xgboost,
        param_grid=XGBOOST_GRID,
    ),
    "catboost": ModelSpec(
        name="catboost",
        feature_columns=tuple(FEATURE_COLUMNS),
        build=build_catboost,
        param_grid=CATBOOST_GRID,
    ),
}


def fit_predict(spec: ModelSpec, params: dict[str, Any], fold: FoldData) -> np.ndarray:
    """Fit on `fold`'s training rows only; predict on its validation rows only.

    Always verifies the fitted estimator's `classes_` and aligns the output
    to the fixed [H, D, A] = [0, 1, 2] column order.
    """
    columns = list(spec.feature_columns)
    X_train = fold.X_train[columns]
    X_val = fold.X_val[columns]

    model = spec.build(params)
    model.fit(X_train, fold.y_train)

    classes = model.named_steps["model"].classes_ if hasattr(model, "named_steps") else model.classes_
    proba = model.predict_proba(X_val)
    return align_proba_to_expected_classes(proba, classes)


# --------------------------------------------------------------------------
# Experiment results and the rolling-fold loop
# --------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    """One model configuration's aggregated performance across every
    development fold. `fold_proba`/`fold_y_true` carry the raw per-fold
    predictions needed for paired comparison in `select_best_configuration`;
    they are excluded from `to_dict()` (use the prediction rows for that)."""

    model: str
    config_id: str
    params: dict[str, Any]
    fold_metrics: list[FoldMetrics]
    fold_proba: list[np.ndarray] = field(repr=False)
    fold_y_true: list[np.ndarray] = field(repr=False)
    mean_log_loss: float
    worst_log_loss: float
    log_loss_std: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "config_id": self.config_id,
            "params": self.params,
            "fold_metrics": [m.to_dict() for m in self.fold_metrics],
            "mean_log_loss": self.mean_log_loss,
            "worst_log_loss": self.worst_log_loss,
            "log_loss_std": self.log_loss_std,
        }


def _config_id(model_name: str, params: dict[str, Any]) -> str:
    if not params:
        return model_name
    rendered = "_".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{model_name}[{rendered}]"


def _fold_predictions_to_rows(
    fold: FoldData, proba: np.ndarray, *, model: str, config_id: str
) -> list[dict[str, Any]]:
    """One row per validation match: identity, actual result, and predicted
    probabilities. This is the schema the Stage 4 calibration pool needs
    (Stage 1 plan section 16): fold, Season, Date, HomeTeam, AwayTeam, actual
    target, p_home/p_draw/p_away, model/config id."""
    meta = fold.metadata_val
    rows = []
    for i in range(len(meta)):
        rows.append(
            {
                "fold": fold.fold,
                "Season": meta.loc[i, "Season"],
                "Date": pd.Timestamp(meta.loc[i, "Date"]).date().isoformat(),
                "HomeTeam": meta.loc[i, "HomeTeam"],
                "AwayTeam": meta.loc[i, "AwayTeam"],
                "actual_target": int(fold.y_val.iloc[i]),
                "actual_ftr": meta.loc[i, "FTR"],
                "p_home": float(proba[i, 0]),
                "p_draw": float(proba[i, 1]),
                "p_away": float(proba[i, 2]),
                "model": model,
                "config_id": config_id,
            }
        )
    return rows


def _evaluate_predictor(
    name: str,
    config_id: str,
    params: dict[str, Any],
    predict_fn: Callable[[FoldData], np.ndarray],
    folds: list[FoldData],
) -> tuple[ExperimentResult, list[dict[str, Any]]]:
    fold_metrics: list[FoldMetrics] = []
    fold_proba: list[np.ndarray] = []
    fold_y_true: list[np.ndarray] = []
    predictions_rows: list[dict[str, Any]] = []

    for fold in folds:
        proba = predict_fn(fold)
        metrics = compute_fold_metrics(
            fold=fold.fold, validation_season=fold.validation_season, y_true=fold.y_val, proba=proba
        )
        fold_metrics.append(metrics)
        fold_proba.append(proba)
        fold_y_true.append(fold.y_val.to_numpy())
        predictions_rows.extend(_fold_predictions_to_rows(fold, proba, model=name, config_id=config_id))

    log_losses = np.array([m.log_loss for m in fold_metrics])
    result = ExperimentResult(
        model=name,
        config_id=config_id,
        params=params,
        fold_metrics=fold_metrics,
        fold_proba=fold_proba,
        fold_y_true=fold_y_true,
        mean_log_loss=float(log_losses.mean()),
        worst_log_loss=float(log_losses.max()),
        log_loss_std=float(log_losses.std(ddof=0)),
    )
    return result, predictions_rows


def run_baseline_experiments(
    folds: list[FoldData],
) -> tuple[list[ExperimentResult], list[dict[str, Any]]]:
    """Run all three baselines (class-frequency, Elo-only, strength-trio)
    across every development fold."""
    predictors: dict[str, Callable[[FoldData], np.ndarray]] = {
        "baseline_class_frequency": lambda fold: class_frequency_baseline(fold.y_train, len(fold.y_val)),
        "baseline_elo_only": lambda fold: elo_only_baseline(fold.X_train, fold.y_train, fold.X_val),
        "baseline_strength_trio": lambda fold: strength_trio_baseline(fold.X_train, fold.y_train, fold.X_val),
    }
    results: list[ExperimentResult] = []
    predictions: list[dict[str, Any]] = []
    for name, predict_fn in predictors.items():
        result, rows = _evaluate_predictor(name, name, {}, predict_fn, folds)
        results.append(result)
        predictions.extend(rows)
    return results, predictions


def run_grid_experiments(
    spec: ModelSpec, folds: list[FoldData]
) -> tuple[list[ExperimentResult], list[dict[str, Any]]]:
    """Run every configuration in `spec`'s hyperparameter grid across every
    development fold. Never uses the validation season for early stopping -
    each configuration uses a fixed iteration/estimator count from the grid."""
    results: list[ExperimentResult] = []
    predictions: list[dict[str, Any]] = []
    for params in spec.param_grid:
        config_id = _config_id(spec.name, params)
        result, rows = _evaluate_predictor(
            spec.name, config_id, params, lambda fold, p=params: fit_predict(spec, p, fold), folds
        )
        results.append(result)
        predictions.extend(rows)
    return results, predictions


def select_best_configuration(results: list[ExperimentResult]) -> ExperimentResult:
    """Apply the pre-committed selection rule (module docstring) to a pool
    of experiment results (baselines and/or model-grid configurations)."""
    if not results:
        raise ValueError("no experiment results to select from")

    ranked = sorted(results, key=lambda r: r.mean_log_loss)
    best = ranked[0]

    tied = [best]
    for candidate in ranked[1:]:
        comparison = paired_log_loss_comparison(best.fold_y_true, best.fold_proba, candidate.fold_proba)
        if comparison["is_tie"]:
            tied.append(candidate)

    if len(tied) == 1:
        return best

    def complexity(result: ExperimentResult) -> int:
        return MODEL_COMPLEXITY_RANK.get(result.model, 99)

    tied.sort(key=lambda r: (r.worst_log_loss, complexity(r), r.log_loss_std))
    return tied[0]


def run_stage1(folds: list[FoldData] | None = None) -> dict[str, Any]:
    """Run every Stage 1 baseline and model-grid experiment across the three
    development folds, and select the best configuration overall.

    Never touches 2025/26: `folds` defaults to `load_all_development_folds()`,
    which only ever returns the three approved development folds.
    """
    folds = folds if folds is not None else load_all_development_folds()

    all_results: list[ExperimentResult] = []
    all_predictions: list[dict[str, Any]] = []

    baseline_results, baseline_predictions = run_baseline_experiments(folds)
    all_results.extend(baseline_results)
    all_predictions.extend(baseline_predictions)

    for spec in MODEL_SPECS.values():
        grid_results, grid_predictions = run_grid_experiments(spec, folds)
        all_results.extend(grid_results)
        all_predictions.extend(grid_predictions)

    best = select_best_configuration(all_results)

    return {
        "development_folds": list(DEVELOPMENT_FOLDS),
        "results": all_results,
        "predictions": all_predictions,
        "best": best,
        "library_versions": library_versions(),
    }


def library_versions() -> dict[str, str]:
    """Library/runtime versions, recorded with every experiment (Stage 1
    plan section 6)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
    }

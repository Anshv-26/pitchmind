"""Sanity baselines for PitchMind's H/D/A classifiers.

Each baseline is a plain function: fit on training rows only, predict
probabilities for validation rows only, output shape (n_val, 3) in the fixed
[H, D, A] = [0, 1, 2] class order. No baseline here ever sees a validation
label before producing its predictions - `class_frequency_baseline` does not
even accept one as an argument.

Bookmaker odds are never used as an input here or anywhere else in PitchMind
(see CLAUDE.md's Betting Data rules).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.ml.evaluation import EXPECTED_CLASSES, align_proba_to_expected_classes

RANDOM_SEED = 42

ELO_ONLY_COLUMNS = ["elo_diff"]
STRENGTH_TRIO_COLUMNS = ["elo_diff", "diff_ewma_ppg", "diff_ewma_sot_diff"]


def class_frequency_baseline(y_train: pd.Series, n_val: int) -> np.ndarray:
    """Constant [P(H), P(D), P(A)] from TRAINING frequencies only.

    Broadcast to every validation row. `n_val` is a row COUNT, not a
    validation label - there is no parameter through which this function
    could see validation targets even if a caller wanted it to.
    """
    counts = y_train.value_counts()
    frequencies = np.array([counts.get(cls, 0) for cls in EXPECTED_CLASSES], dtype=float)
    total = frequencies.sum()
    if total <= 0:
        raise ValueError("y_train must contain at least one row")
    probabilities = frequencies / total
    return np.tile(probabilities, (n_val, 1))


def _fit_logistic_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series, *, with_imputer: bool
) -> Pipeline:
    """A multinomial logistic regression pipeline, train-fit only.

    No `multi_class` argument is passed: with the `lbfgs` solver and more
    than two classes, scikit-learn already fits a genuine multinomial
    (softmax) model, which is what "do not invent draw probabilities
    manually" requires - the draw band is learned from data, not hand-tuned.
    """
    steps = []
    if with_imputer:
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
        )
    )
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    return pipeline


def elo_only_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> np.ndarray:
    """Multinomial logistic regression on `elo_diff` alone.

    `elo_diff` is a non-nullable feature (always populated), so no imputer is
    used; a `StandardScaler` is kept for numerically stable optimisation.
    """
    pipeline = _fit_logistic_pipeline(X_train[ELO_ONLY_COLUMNS], y_train, with_imputer=False)
    proba = pipeline.predict_proba(X_val[ELO_ONLY_COLUMNS])
    return align_proba_to_expected_classes(proba, pipeline.named_steps["model"].classes_)


def strength_trio_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> np.ndarray:
    """Multinomial logistic regression on three interpretable strength features.

    `diff_ewma_ppg` / `diff_ewma_sot_diff` can be NaN during cold-start rows,
    so this pipeline imputes (median, fit on training rows only) before
    scaling. This is the interpretable benchmark the larger model stack must
    beat to justify its extra complexity.
    """
    pipeline = _fit_logistic_pipeline(X_train[STRENGTH_TRIO_COLUMNS], y_train, with_imputer=True)
    proba = pipeline.predict_proba(X_val[STRENGTH_TRIO_COLUMNS])
    return align_proba_to_expected_classes(proba, pipeline.named_steps["model"].classes_)

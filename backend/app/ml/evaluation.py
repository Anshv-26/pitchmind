"""Evaluation metrics and the probability contract for PitchMind classifiers.

Every model in this project must emit probabilities, not just labels. This
module defines the one probability contract every prediction array must
satisfy, the metric suite used to compare models, and the paired-comparison
building block for the pre-committed model-selection rule (see
`training.select_best_configuration`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

# Fixed class order for every probability array in this project: H=0, D=1, A=2.
CLASS_NAMES = ("H", "D", "A")
EXPECTED_CLASSES = (0, 1, 2)


def align_proba_to_expected_classes(proba: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """Reorder (and zero-fill any absent) predict_proba columns to [H, D, A].

    scikit-learn's predict_proba columns follow `estimator.classes_`, which is
    only already `[0, 1, 2]` if all three classes were present in the
    training fold. This makes the fixed H=0/D=1/A=2 order an explicit,
    verified property of every array leaving this module, rather than an
    assumption baked silently into downstream code.

    After aligning, each row is renormalized to sum to exactly 1 in float64.
    Some backends (e.g. XGBoost's native float32 `multi:softprob`) return row
    sums that deviate from 1.0 by ~1e-7 - correct to float32 precision, but
    upcasting to float64 here (`np.asarray(proba, dtype=float)`) preserves
    that noise as float64 *values* without reducing it, which is enough to
    trip sklearn's `log_loss` self-consistency check (its tolerance is
    derived from the array's dtype, so it tightens by ~4 orders of magnitude
    on the upcast even though nothing about the underlying precision
    improved). Dividing each row by its own sum is purely a numerical
    correction back onto the probability simplex: it cannot change which
    class a row favours, and it moves any single probability by less than
    the precision it was already computed to. It is not a calibration or
    model change.
    """
    proba = np.asarray(proba, dtype=float)
    classes = list(classes)
    if classes == list(EXPECTED_CLASSES):
        aligned = proba
    else:
        aligned = np.zeros((proba.shape[0], len(EXPECTED_CLASSES)), dtype=float)
        for source_index, cls in enumerate(classes):
            if cls in EXPECTED_CLASSES:
                aligned[:, EXPECTED_CLASSES.index(cls)] = proba[:, source_index]

    row_sum = aligned.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sum).all():
        bad = int((~np.isfinite(row_sum)).sum())
        raise ValueError(f"cannot renormalize probabilities: {bad} row(s) have a non-finite sum")
    if (row_sum <= 0).any():
        bad = int((row_sum <= 0).sum())
        raise ValueError(f"cannot renormalize probabilities: {bad} row(s) have a non-positive sum")
    return aligned / row_sum


def validate_probabilities(proba: np.ndarray, *, n_rows: int | None = None) -> list[str]:
    """Return a list of probability-contract violations (empty means sound).

    Contract: shape (n, 3); all finite; non-negative; each row sums to ~1.
    The fixed [H, D, A] = [0, 1, 2] column order is a construction guarantee
    of `align_proba_to_expected_classes`, not something re-derivable from the
    array's values alone, so it is not (and cannot be) re-checked here.
    """
    problems: list[str] = []
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != 3:
        problems.append(f"expected shape (n, 3), got {proba.shape}")
        return problems
    if n_rows is not None and proba.shape[0] != n_rows:
        problems.append(f"expected {n_rows} rows, got {proba.shape[0]}")
    if not np.isfinite(proba).all():
        problems.append("probabilities contain non-finite values")
        return problems
    if (proba < -1e-9).any():
        problems.append("probabilities contain negative values")
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = int((~np.isclose(row_sums, 1.0, atol=1e-6)).sum())
        problems.append(f"{bad} row(s) do not sum to 1")
    return problems


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between one-hot targets and predicted probabilities.

    The standard multiclass generalisation of the (binary) Brier score:
    mean over rows of sum_c (p_c - 1{y=c})^2.
    """
    y_true = np.asarray(y_true)
    one_hot = np.zeros((len(y_true), len(EXPECTED_CLASSES)))
    for row, cls in enumerate(y_true):
        one_hot[row, EXPECTED_CLASSES.index(int(cls))] = 1.0
    return float(np.mean(np.sum((np.asarray(proba) - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = 10) -> float:
    """ECE over the predicted top-class probability.

    Bins predictions by their MAX predicted-class probability (the model's
    confidence in whichever class it favours) and compares that confidence to
    the empirical accuracy within each bin - the standard top-label ECE.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    confidences = proba.max(axis=1)
    predictions = np.array(EXPECTED_CLASSES)[proba.argmax(axis=1)]
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if lo > 0 else (
            (confidences >= lo) & (confidences <= hi)
        )
        if not in_bin.any():
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = correct[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_confidence - bin_accuracy)
    return float(ece)


@dataclass
class FoldMetrics:
    """The full metric suite for one model's predictions on one validation fold."""

    fold: int
    validation_season: str
    n_rows: int
    log_loss: float
    brier_score: float
    accuracy: float
    macro_f1: float
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    f1_per_class: dict[str, float]
    confusion_matrix: list[list[int]]
    mean_predicted_probability: dict[str, float]
    realised_frequency: dict[str, float]
    expected_calibration_error: float

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "validation_season": self.validation_season,
            "n_rows": self.n_rows,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class,
            "confusion_matrix": self.confusion_matrix,
            "mean_predicted_probability": self.mean_predicted_probability,
            "realised_frequency": self.realised_frequency,
            "expected_calibration_error": self.expected_calibration_error,
        }


def compute_fold_metrics(
    *,
    fold: int,
    validation_season: str,
    y_true: "pd.Series | np.ndarray",
    proba: np.ndarray,
) -> FoldMetrics:
    """Compute the full metric suite for one model's predictions on one fold.

    Raises if `proba` violates the probability contract - metrics are never
    computed on an unsound prediction array.
    """
    problems = validate_probabilities(proba, n_rows=len(y_true))
    if problems:
        raise ValueError(f"invalid probabilities for fold {fold}: {problems}")

    y_true_arr = np.asarray(y_true)
    predictions = np.array(EXPECTED_CLASSES)[np.asarray(proba).argmax(axis=1)]

    ll = float(log_loss(y_true_arr, proba, labels=list(EXPECTED_CLASSES)))
    brier = multiclass_brier_score(y_true_arr, proba)
    accuracy = float((predictions == y_true_arr).mean())
    ece = expected_calibration_error(y_true_arr, proba)

    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true_arr, predictions, labels=list(EXPECTED_CLASSES), zero_division=0
    )
    macro_f1 = float(
        f1_score(y_true_arr, predictions, labels=list(EXPECTED_CLASSES), average="macro", zero_division=0)
    )
    confusion = confusion_matrix(y_true_arr, predictions, labels=list(EXPECTED_CLASSES)).tolist()

    proba_arr = np.asarray(proba)
    mean_predicted = {name: float(proba_arr[:, i].mean()) for i, name in enumerate(CLASS_NAMES)}
    realised = {
        name: float((y_true_arr == cls).mean()) for name, cls in zip(CLASS_NAMES, EXPECTED_CLASSES)
    }

    return FoldMetrics(
        fold=fold,
        validation_season=validation_season,
        n_rows=len(y_true_arr),
        log_loss=ll,
        brier_score=brier,
        accuracy=accuracy,
        macro_f1=macro_f1,
        precision_per_class=dict(zip(CLASS_NAMES, map(float, precision))),
        recall_per_class=dict(zip(CLASS_NAMES, map(float, recall))),
        f1_per_class=dict(zip(CLASS_NAMES, map(float, f1))),
        confusion_matrix=confusion,
        mean_predicted_probability=mean_predicted,
        realised_frequency=realised,
        expected_calibration_error=ece,
    )


def paired_log_loss_comparison(
    y_true_by_fold: Sequence[np.ndarray],
    proba_a_by_fold: Sequence[np.ndarray],
    proba_b_by_fold: Sequence[np.ndarray],
) -> dict:
    """Paired per-row log-loss comparison between two candidates across folds.

    Pools the per-row log-loss DIFFERENCES (a - b) across every development
    row (not per-fold means, which would understate the sample size and
    overstate the standard error), and reports the mean difference plus its
    standard error. Implements the pre-committed tie rule (Stage 1 plan,
    section 10): candidates are a statistical tie when
    `abs(mean_diff) < 2 * standard_error`.
    """
    diffs = []
    for y_true, proba_a, proba_b in zip(y_true_by_fold, proba_a_by_fold, proba_b_by_fold):
        y_true = np.asarray(y_true)
        eps = 1e-15
        row_ll_a = -np.log(np.clip(np.asarray(proba_a)[np.arange(len(y_true)), y_true], eps, 1.0))
        row_ll_b = -np.log(np.clip(np.asarray(proba_b)[np.arange(len(y_true)), y_true], eps, 1.0))
        diffs.append(row_ll_a - row_ll_b)
    all_diffs = np.concatenate(diffs)
    mean_diff = float(all_diffs.mean())
    se = float(all_diffs.std(ddof=1) / np.sqrt(len(all_diffs))) if len(all_diffs) > 1 else 0.0
    return {
        "mean_diff": mean_diff,
        "standard_error": se,
        "n": int(len(all_diffs)),
        "is_tie": abs(mean_diff) < 2 * se if se > 0 else mean_diff == 0.0,
    }

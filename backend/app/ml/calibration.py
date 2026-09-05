"""Calibration and ensemble meta-layer over the two frozen PitchMind models.

This module asks one narrow question: can a *tiny* meta-layer - one temperature
scalar and/or one convex pooling weight - beat the raw strength-trio outcome
champion on genuinely held-out development rows?

Inputs are the existing out-of-fold (OOF) prediction CSVs only. Nothing here
refits feature artifacts, Stage 1 models, or Dixon-Coles models; `evaluation.py`
and `datasets.py` are imported read-only.

DEVELOPMENT META-VALIDATION, NOT A PRISTINE FINAL TEST
------------------------------------------------------
The design pass that produced this module already inspected development labels
across all 1,140 OOF rows (per-season optimal temperature, in-sample pooling
weight, paired trio-vs-DC comparison). The strict chronological protocol below
is therefore honest *development meta-validation*: it is held out from each
meta-parameter fit, but it is not a never-inspected test set. The genuinely
sealed final test remains 2025/26, which this module can never reach.

STRICT CHRONOLOGICAL PROTOCOL
-----------------------------
    M1: meta-train 2022_23              -> meta-evaluate 2023_24
    M2: meta-train 2022_23 + 2023_24    -> meta-evaluate 2024_25

2022_23 is the calibration seed and is never itself evaluated. The primary
held-out development pool is therefore 760 matches (2023_24 + 2024_25).

Meta-training seasons always strictly precede the evaluated season, so no
future information - directly, or indirectly via a base model that was itself
trained on the evaluated season - can reach a meta-parameter. The anti-causal
leave-one-season-out variant is deliberately NOT implemented.

FAIR COMPARISON
---------------
The historical `baseline_strength_trio` figure of 0.9587 was computed over all
three development validation seasons. It is NOT the comparison target here and
appears in reports only as historical three-season context. Every challenger is
compared against the raw trio recomputed on the exact same 760 evaluation rows
(`INCUMBENT_CANDIDATE`), including the paired per-match comparison.

SCORELINE CONSISTENCY
---------------------
Temperature scaling is applied ONLY to the trio, never to the Dixon-Coles
probabilities, so the DC scoreline matrix and its derived H/D/A can never
diverge. The DC probability columns pass through this module untouched.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import log_loss

from backend.app.ml.datasets import DEVELOPMENT_FOLD_DEFINITIONS
from backend.app.ml.evaluation import (
    CLASS_NAMES,
    EXPECTED_CLASSES,
    FoldMetrics,
    compute_fold_metrics,
    paired_log_loss_comparison,
    validate_probabilities,
)
from backend.app.ml.feature_engineering import SEALED_SEASON

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "reports" / "modeling"
STAGE1_PREDICTIONS_PATH = REPORTS_DIR / "stage1_predictions.csv"
SCORE_MODEL_PREDICTIONS_PATH = REPORTS_DIR / "score_models_predictions.csv"

# The two frozen configurations. Hard-coded so the other 32 Stage 1 and 4
# score-model configurations in those files can never be selected.
FROZEN_OUTCOME_CONFIG = "baseline_strength_trio"
FROZEN_SCORELINE_CONFIG = "dixon_coles_l2_decay"

MERGE_KEY = ["fold", "Season", "Date", "HomeTeam", "AwayTeam"]
EXPECTED_ROWS = 1140

TRIO_PROBA_COLUMNS = ["p_home_trio", "p_draw_trio", "p_away_trio"]
DC_PROBA_COLUMNS = ["p_home_dc", "p_draw_dc", "p_away_dc"]

# Historical three-season figure. Reporting context ONLY - never a selection
# target, because it is computed over 1,140 rows while selection happens on 760.
HISTORICAL_TRIO_THREE_SEASON_LOG_LOSS = 0.9587

# --------------------------------------------------------------------------
# Strict chronological meta-folds
# --------------------------------------------------------------------------
CALIBRATION_SEED_SEASON = "2022_23"

META_FOLD_DEFINITIONS: dict[int, dict[str, Any]] = {
    1: {"meta_train_seasons": ["2022_23"], "evaluation_season": "2023_24"},
    2: {"meta_train_seasons": ["2022_23", "2023_24"], "evaluation_season": "2024_25"},
}
META_FOLDS: tuple[int, ...] = tuple(sorted(META_FOLD_DEFINITIONS))
EVALUATION_SEASONS: tuple[str, ...] = tuple(
    META_FOLD_DEFINITIONS[f]["evaluation_season"] for f in META_FOLDS
)
EXPECTED_EVALUATION_ROWS = 760

# --------------------------------------------------------------------------
# Optimizer configuration
# --------------------------------------------------------------------------
LOG_T_BOUNDS = (float(np.log(0.2)), float(np.log(5.0)))
W_BOUNDS = (0.0, 1.0)

# Probability floor before taking logs. The minimum probability actually
# observed in either frozen model's OOF output is ~1.4e-2, so this floor is
# defensive only and never binds in practice.
LOG_PROBA_FLOOR = 1e-15

# A fitted objective may exceed the identity objective by at most this much
# (pure floating-point noise). Anything larger is treated as an optimizer
# failure and raises, rather than being silently accepted.
IDENTITY_TOLERANCE = 1e-9

RELIABILITY_BINS = 5


# --------------------------------------------------------------------------
# Safe OOF prediction loading and 1:1 merge
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetaPredictions:
    """The 1,140 aligned development OOF rows for the two frozen models."""

    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def y(self) -> np.ndarray:
        return self.frame["actual_target"].to_numpy()

    @property
    def proba_trio(self) -> np.ndarray:
        return self.frame[TRIO_PROBA_COLUMNS].to_numpy(dtype=float)

    @property
    def proba_dc(self) -> np.ndarray:
        return self.frame[DC_PROBA_COLUMNS].to_numpy(dtype=float)

    @property
    def dates(self) -> pd.Series:
        return pd.to_datetime(self.frame["Date"])

    @property
    def seasons(self) -> list[str]:
        return sorted(self.frame["Season"].unique().tolist())

    def season_mask(self, seasons: list[str] | tuple[str, ...]) -> np.ndarray:
        return self.frame["Season"].isin(list(seasons)).to_numpy()

    def subset(self, seasons: list[str] | tuple[str, ...]) -> "MetaPredictions":
        mask = self.season_mask(seasons)
        return MetaPredictions(self.frame.loc[mask].reset_index(drop=True))


def _load_frozen_config(path: Path, config_id: str, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run the {label} training script first to "
            f"regenerate the OOF predictions."
        )
    frame = pd.read_csv(path)
    available = set(frame["config_id"].unique())
    if config_id not in available:
        raise ValueError(
            f"frozen {label} configuration {config_id!r} not present in {path}; "
            f"found {sorted(available)}"
        )
    # Filter to the frozen config ONLY - the other configurations in this file
    # are not part of this stage and must never enter the meta-layer.
    selected = frame.loc[frame["config_id"] == config_id].reset_index(drop=True)
    if len(selected) != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} rows for {config_id!r} in {path}, got {len(selected)}"
        )
    duplicates = int(selected.duplicated(MERGE_KEY).sum())
    if duplicates:
        raise ValueError(f"{config_id!r} in {path} has {duplicates} duplicate {MERGE_KEY} row(s)")
    return selected


def load_meta_predictions(
    *,
    stage1_path: Path | None = None,
    score_model_path: Path | None = None,
) -> MetaPredictions:
    """Load and 1:1-merge the two frozen models' development OOF predictions.

    Every guard raises with a specific message; nothing is silently dropped,
    coerced, or de-duplicated.
    """
    stage1_path = stage1_path if stage1_path is not None else STAGE1_PREDICTIONS_PATH
    score_model_path = (
        score_model_path if score_model_path is not None else SCORE_MODEL_PREDICTIONS_PATH
    )

    trio = _load_frozen_config(stage1_path, FROZEN_OUTCOME_CONFIG, "Stage 1")
    dc = _load_frozen_config(score_model_path, FROZEN_SCORELINE_CONFIG, "score model")

    merged = trio.merge(
        dc,
        on=MERGE_KEY,
        how="outer",
        suffixes=("_trio", "_dc"),
        validate="1:1",
        indicator=True,
    )

    unmatched = merged.loc[merged["_merge"] != "both"]
    if len(unmatched):
        raise ValueError(
            f"OOF prediction merge is not one-to-one: {len(unmatched)} row(s) did not "
            f"match on {MERGE_KEY} "
            f"({merged['_merge'].value_counts().to_dict()})."
        )
    if len(merged) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} merged rows, got {len(merged)}")

    target_mismatch = int((merged["actual_target_trio"] != merged["actual_target_dc"]).sum())
    if target_mismatch:
        raise ValueError(f"{target_mismatch} row(s) have mismatched actual_target between models")
    ftr_mismatch = int((merged["actual_ftr_trio"] != merged["actual_ftr_dc"]).sum())
    if ftr_mismatch:
        raise ValueError(f"{ftr_mismatch} row(s) have mismatched actual_ftr between models")

    present_seasons = set(merged["Season"].unique())
    if SEALED_SEASON in present_seasons:
        raise ValueError(
            f"OOF predictions contain sealed-season ({SEALED_SEASON!r}) rows; "
            f"refusing to use them for development meta-validation."
        )

    expected_pairs = {
        (fold, definition["validation_season"])
        for fold, definition in DEVELOPMENT_FOLD_DEFINITIONS.items()
    }
    actual_pairs = set(map(tuple, merged[["fold", "Season"]].drop_duplicates().to_numpy()))
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"fold/season metadata {sorted(actual_pairs)} does not match the approved "
            f"development fold table {sorted(expected_pairs)}"
        )

    merged = merged.rename(columns={"actual_target_trio": "actual_target", "actual_ftr_trio": "actual_ftr"})
    merged = merged.drop(columns=["_merge", "actual_target_dc", "actual_ftr_dc"])
    merged = merged.sort_values(MERGE_KEY).reset_index(drop=True)

    for label, columns in [("strength trio", TRIO_PROBA_COLUMNS), ("dixon-coles", DC_PROBA_COLUMNS)]:
        problems = validate_probabilities(merged[columns].to_numpy(dtype=float), n_rows=len(merged))
        if problems:
            raise ValueError(f"{label} OOF probabilities violate the probability contract: {problems}")

    return MetaPredictions(merged)


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------
def temperature_scale(proba: np.ndarray, temperature: float) -> np.ndarray:
    """q(T) = softmax(log(p) / T), computed stably.

    Because softmax is shift-invariant and each input row sums to 1,
    `temperature_scale(p, 1.0)` returns `p` exactly (up to floating point):
    softmax(log p) = p / sum(p) = p. Class ordering within a row is preserved
    for any T > 0; only confidence changes.
    """
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")
    proba = np.asarray(proba, dtype=float)
    logits = np.log(np.maximum(proba, LOG_PROBA_FLOOR)) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def convex_pool(proba_a: np.ndarray, proba_b: np.ndarray, weight: float) -> np.ndarray:
    """w * proba_a + (1 - w) * proba_b.

    Both inputs are row-stochastic, so the result is too by convexity - no
    clipping and no renormalisation is applied or needed.
    """
    if not np.isfinite(weight) or not (W_BOUNDS[0] <= weight <= W_BOUNDS[1]):
        raise ValueError(f"pool weight must be finite and within {W_BOUNDS}, got {weight!r}")
    return weight * np.asarray(proba_a, dtype=float) + (1.0 - weight) * np.asarray(proba_b, dtype=float)


def _log_loss(proba: np.ndarray, y: np.ndarray) -> float:
    return float(log_loss(y, proba, labels=list(EXPECTED_CLASSES)))


# --------------------------------------------------------------------------
# Meta-parameter fitting
# --------------------------------------------------------------------------
def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Fit one temperature by minimising multiclass log loss.

    `minimize_scalar(method="bounded")` takes no starting point, so the
    identity point T=1 is evaluated EXPLICITLY and used as the reference:
    the optimizer's solution is accepted only if it is no worse than the
    identity beyond `IDENTITY_TOLERANCE`. A genuinely worse result indicates an
    optimizer failure and raises rather than being silently accepted.
    """
    identity_objective = _log_loss(temperature_scale(proba, 1.0), y)

    result = minimize_scalar(
        lambda log_t: _log_loss(temperature_scale(proba, float(np.exp(log_t))), y),
        bounds=LOG_T_BOUNDS,
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"temperature fit did not converge: {getattr(result, 'message', result)}")
    if not np.isfinite(result.x) or not np.isfinite(result.fun):
        raise RuntimeError(f"temperature fit produced a non-finite result (x={result.x!r}, fun={result.fun!r})")

    temperature = float(np.exp(result.x))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError(f"temperature fit produced an invalid temperature {temperature!r}")

    fitted_objective = _log_loss(temperature_scale(proba, temperature), y)
    if fitted_objective > identity_objective + IDENTITY_TOLERANCE:
        raise RuntimeError(
            f"temperature fit returned T={temperature!r} with objective "
            f"{fitted_objective!r}, which is worse than the identity T=1 objective "
            f"{identity_objective!r} by more than {IDENTITY_TOLERANCE}. Refusing to "
            f"accept an optimizer failure silently."
        )
    return temperature


def fit_pool_weight(proba_trio: np.ndarray, proba_dc: np.ndarray, y: np.ndarray) -> float:
    """Fit the convex pooling weight w.

    Bounded scalar optimization is NOT relied upon to land exactly on an
    endpoint, so w=0 (pure Dixon-Coles) and w=1 (pure trio - "retain the trio
    only", a valid and meaningful solution) are evaluated explicitly and
    compared against the optimizer's interior solution. The lowest-objective
    valid candidate wins.
    """
    candidates: list[tuple[float, float]] = []

    for endpoint in (W_BOUNDS[0], W_BOUNDS[1]):
        candidates.append((endpoint, _log_loss(convex_pool(proba_trio, proba_dc, endpoint), y)))

    result = minimize_scalar(
        lambda w: _log_loss(convex_pool(proba_trio, proba_dc, float(np.clip(w, *W_BOUNDS))), y),
        bounds=W_BOUNDS,
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"pool weight fit did not converge: {getattr(result, 'message', result)}")
    if np.isfinite(result.x) and np.isfinite(result.fun):
        interior = float(np.clip(result.x, *W_BOUNDS))
        candidates.append((interior, _log_loss(convex_pool(proba_trio, proba_dc, interior), y)))

    weight, _objective = min(candidates, key=lambda item: item[1])
    return float(weight)


def fit_joint_temperature_and_weight(
    proba_trio: np.ndarray, proba_dc: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Jointly fit (log T, w) for the calibrated-trio + Dixon-Coles pool.

    The identity point is exactly (log T = 0, w = 1) - i.e. the raw trio - and
    is evaluated explicitly. L-BFGS-B is started there, and its result is
    accepted only if it is no worse than the identity beyond
    `IDENTITY_TOLERANCE`.
    """
    def objective(theta: np.ndarray) -> float:
        log_t, w = float(theta[0]), float(np.clip(theta[1], *W_BOUNDS))
        return _log_loss(convex_pool(temperature_scale(proba_trio, float(np.exp(log_t))), proba_dc, w), y)

    identity_theta = np.array([0.0, 1.0])
    identity_objective = objective(identity_theta)

    result = minimize(
        objective,
        identity_theta,
        method="L-BFGS-B",
        bounds=[LOG_T_BOUNDS, W_BOUNDS],
    )
    if not result.success:
        raise RuntimeError(f"joint (T, w) fit did not converge: {result.message}")
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f"joint (T, w) fit produced a non-finite result (x={result.x!r}, fun={result.fun!r})")

    temperature = float(np.exp(result.x[0]))
    weight = float(np.clip(result.x[1], *W_BOUNDS))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError(f"joint fit produced an invalid temperature {temperature!r}")

    fitted_objective = objective(np.array([np.log(temperature), weight]))
    if fitted_objective > identity_objective + IDENTITY_TOLERANCE:
        raise RuntimeError(
            f"joint fit returned (T={temperature!r}, w={weight!r}) with objective "
            f"{fitted_objective!r}, which is worse than the identity "
            f"(T=1, w=1) objective {identity_objective!r} by more than "
            f"{IDENTITY_TOLERANCE}. Refusing to accept an optimizer failure silently."
        )
    return temperature, weight


# --------------------------------------------------------------------------
# Candidate definitions (pre-committed before any held-out number is seen)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetaCandidate:
    """One meta-layer candidate: how to fit its parameters and how to apply them."""

    name: str
    n_params: int
    fit: Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]
    apply: Callable[[np.ndarray, np.ndarray, dict[str, float]], np.ndarray]
    description: str


def _fit_identity(proba_trio: np.ndarray, proba_dc: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {}


def _apply_identity(proba_trio: np.ndarray, proba_dc: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return np.asarray(proba_trio, dtype=float)


def _fit_temperature_only(proba_trio: np.ndarray, proba_dc: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"T": fit_temperature(proba_trio, y)}


def _apply_temperature_only(
    proba_trio: np.ndarray, proba_dc: np.ndarray, params: dict[str, float]
) -> np.ndarray:
    return temperature_scale(proba_trio, params["T"])


def _fit_pool_only(proba_trio: np.ndarray, proba_dc: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"w": fit_pool_weight(proba_trio, proba_dc, y)}


def _apply_pool_only(proba_trio: np.ndarray, proba_dc: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return convex_pool(proba_trio, proba_dc, params["w"])


def _fit_joint(proba_trio: np.ndarray, proba_dc: np.ndarray, y: np.ndarray) -> dict[str, float]:
    temperature, weight = fit_joint_temperature_and_weight(proba_trio, proba_dc, y)
    return {"T": temperature, "w": weight}


def _apply_joint(proba_trio: np.ndarray, proba_dc: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return convex_pool(temperature_scale(proba_trio, params["T"]), proba_dc, params["w"])


INCUMBENT_CANDIDATE = "raw_trio"

CANDIDATES: dict[str, MetaCandidate] = {
    "raw_trio": MetaCandidate(
        name="raw_trio",
        n_params=0,
        fit=_fit_identity,
        apply=_apply_identity,
        description="Incumbent: raw baseline_strength_trio probabilities, unchanged.",
    ),
    "temperature_trio": MetaCandidate(
        name="temperature_trio",
        n_params=1,
        fit=_fit_temperature_only,
        apply=_apply_temperature_only,
        description="Temperature-scaled trio: softmax(log(p_trio) / T).",
    ),
    "pool_raw_trio": MetaCandidate(
        name="pool_raw_trio",
        n_params=1,
        fit=_fit_pool_only,
        apply=_apply_pool_only,
        description="Convex pool: w * p_trio + (1 - w) * p_dixon_coles.",
    ),
    "pool_calibrated_trio": MetaCandidate(
        name="pool_calibrated_trio",
        n_params=2,
        fit=_fit_joint,
        apply=_apply_joint,
        description="Convex pool of the temperature-scaled trio with Dixon-Coles; T and w fitted jointly.",
    ),
}


# --------------------------------------------------------------------------
# Strict chronological meta-validation
# --------------------------------------------------------------------------
@dataclass
class MetaFoldResult:
    """One candidate's held-out result on one chronological meta-fold."""

    meta_fold: int
    meta_train_seasons: list[str]
    evaluation_season: str
    n_meta_train_rows: int
    n_evaluation_rows: int
    fitted_params: dict[str, float]
    metrics: FoldMetrics
    proba: np.ndarray = field(repr=False)
    y_true: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "meta_fold": self.meta_fold,
            "meta_train_seasons": self.meta_train_seasons,
            "evaluation_season": self.evaluation_season,
            "n_meta_train_rows": self.n_meta_train_rows,
            "n_evaluation_rows": self.n_evaluation_rows,
            "fitted_params": self.fitted_params,
            **self.metrics.to_dict(),
        }


@dataclass
class MetaCandidateResult:
    """One candidate's aggregated performance across the chronological meta-folds."""

    name: str
    n_params: int
    fold_results: list[MetaFoldResult]
    mean_log_loss: float
    worst_log_loss: float

    @property
    def fold_proba(self) -> list[np.ndarray]:
        return [r.proba for r in self.fold_results]

    @property
    def fold_y_true(self) -> list[np.ndarray]:
        return [r.y_true for r in self.fold_results]

    @property
    def per_season_log_loss(self) -> dict[str, float]:
        return {r.evaluation_season: r.metrics.log_loss for r in self.fold_results}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_params": self.n_params,
            "mean_log_loss": self.mean_log_loss,
            "worst_log_loss": self.worst_log_loss,
            "per_season_log_loss": self.per_season_log_loss,
            "fitted_params_by_fold": {
                r.meta_fold: r.fitted_params for r in self.fold_results
            },
        }


def run_meta_validation(
    candidate: MetaCandidate, predictions: MetaPredictions
) -> MetaCandidateResult:
    """Fit `candidate` on each meta-fold's meta-training seasons ONLY, and
    evaluate on that fold's held-out evaluation season."""
    fold_results: list[MetaFoldResult] = []

    for meta_fold in META_FOLDS:
        definition = META_FOLD_DEFINITIONS[meta_fold]
        meta_train_seasons = list(definition["meta_train_seasons"])
        evaluation_season = definition["evaluation_season"]

        if evaluation_season in meta_train_seasons:
            raise ValueError(
                f"meta-fold {meta_fold}: evaluation season {evaluation_season!r} "
                f"appears in its own meta-training seasons {meta_train_seasons}"
            )

        train = predictions.subset(meta_train_seasons)
        evaluate = predictions.subset([evaluation_season])
        if len(train) == 0 or len(evaluate) == 0:
            raise ValueError(f"meta-fold {meta_fold} has an empty meta-train or evaluation split")
        if train.dates.max() >= evaluate.dates.min():
            raise ValueError(
                f"meta-fold {meta_fold}: meta-training rows are not strictly earlier than "
                f"evaluation rows ({train.dates.max()} >= {evaluate.dates.min()})"
            )

        params = candidate.fit(train.proba_trio, train.proba_dc, train.y)
        proba = candidate.apply(evaluate.proba_trio, evaluate.proba_dc, params)

        problems = validate_probabilities(proba, n_rows=len(evaluate))
        if problems:
            raise ValueError(
                f"candidate {candidate.name!r} produced invalid probabilities on "
                f"meta-fold {meta_fold}: {problems}"
            )

        metrics = compute_fold_metrics(
            fold=meta_fold, validation_season=evaluation_season, y_true=evaluate.y, proba=proba
        )
        fold_results.append(
            MetaFoldResult(
                meta_fold=meta_fold,
                meta_train_seasons=meta_train_seasons,
                evaluation_season=evaluation_season,
                n_meta_train_rows=len(train),
                n_evaluation_rows=len(evaluate),
                fitted_params=params,
                metrics=metrics,
                proba=proba,
                y_true=evaluate.y,
            )
        )

    log_losses = np.array([r.metrics.log_loss for r in fold_results])
    return MetaCandidateResult(
        name=candidate.name,
        n_params=candidate.n_params,
        fold_results=fold_results,
        mean_log_loss=float(log_losses.mean()),
        worst_log_loss=float(log_losses.max()),
    )


# --------------------------------------------------------------------------
# Selection rule
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetaSelection:
    """The FROZEN outcome of development meta-validation.

    Only `select_best_candidate` constructs this, and `fit_final_meta_parameters`
    requires one - so a final all-1,140-row meta-parameter fit is structurally
    impossible before the method has been selected on held-out data.
    """

    selected: str
    incumbent_retained: bool
    reasons: dict[str, dict[str, Any]]
    incumbent_mean_log_loss: float
    n_evaluation_rows: int


def _challenger_verdict(
    challenger: MetaCandidateResult, incumbent: MetaCandidateResult
) -> dict[str, Any]:
    """Apply the four pre-committed conditions. All must hold to replace the incumbent."""
    lower_mean = challenger.mean_log_loss < incumbent.mean_log_loss

    comparison = paired_log_loss_comparison(
        incumbent.fold_y_true, challenger.fold_proba, incumbent.fold_proba
    )
    # mean_diff = challenger - incumbent, so a genuine improvement is negative
    # AND not a tie under evaluation.py's existing |mean_diff| < 2*SE rule.
    not_a_tie_in_favour = (not comparison["is_tie"]) and comparison["mean_diff"] < 0.0

    challenger_seasons = challenger.per_season_log_loss
    incumbent_seasons = incumbent.per_season_log_loss
    improves_every_season = all(
        challenger_seasons[season] < incumbent_seasons[season] for season in incumbent_seasons
    )

    no_worst_season_regression = challenger.worst_log_loss <= incumbent.worst_log_loss

    passed = bool(lower_mean and not_a_tie_in_favour and improves_every_season and no_worst_season_regression)
    return {
        "lower_mean_log_loss": bool(lower_mean),
        "paired_improvement_not_a_tie": bool(not_a_tie_in_favour),
        "improves_every_evaluation_season": bool(improves_every_season),
        "no_worst_season_regression": bool(no_worst_season_regression),
        "paired_comparison": comparison,
        "mean_log_loss": challenger.mean_log_loss,
        "per_season_log_loss": challenger_seasons,
        "passes_all_conditions": passed,
    }


def select_best_candidate(results: list[MetaCandidateResult]) -> MetaSelection:
    """Conservative selection: the incumbent raw trio is retained unless a
    challenger satisfies EVERY pre-committed condition.

    Ties, near-ties and mixed-sign season results all resolve to the incumbent.
    Accuracy, Brier, macro-F1 and ECE are reported but never decide.
    """
    by_name = {r.name: r for r in results}
    if INCUMBENT_CANDIDATE not in by_name:
        raise ValueError(f"incumbent {INCUMBENT_CANDIDATE!r} missing from meta-validation results")
    incumbent = by_name[INCUMBENT_CANDIDATE]

    reasons: dict[str, dict[str, Any]] = {}
    qualifying: list[MetaCandidateResult] = []
    for result in results:
        if result.name == INCUMBENT_CANDIDATE:
            continue
        verdict = _challenger_verdict(result, incumbent)
        reasons[result.name] = verdict
        if verdict["passes_all_conditions"]:
            qualifying.append(result)

    if not qualifying:
        selected = INCUMBENT_CANDIDATE
    else:
        # Among qualifying challengers: lowest mean log loss, then fewest
        # parameters (simplicity) as the tie-break.
        qualifying.sort(key=lambda r: (r.mean_log_loss, r.n_params))
        selected = qualifying[0].name

    n_rows = sum(r.n_evaluation_rows for r in incumbent.fold_results)
    return MetaSelection(
        selected=selected,
        incumbent_retained=(selected == INCUMBENT_CANDIDATE),
        reasons=reasons,
        incumbent_mean_log_loss=incumbent.mean_log_loss,
        n_evaluation_rows=n_rows,
    )


# --------------------------------------------------------------------------
# Final meta-parameter fit (only after selection is frozen)
# --------------------------------------------------------------------------
def fit_final_meta_parameters(
    selection: MetaSelection, predictions: MetaPredictions
) -> dict[str, float]:
    """Refit the SELECTED method's parameters once on all 1,140 development rows.

    This is deliberately separate from meta-validation. It is legitimate rather
    than leakage because, at deployment time for the sealed season, all three
    development seasons are in the past - it is M2's expanding window extended
    one step. The result is recorded for later application and is NEVER applied
    to the sealed season here.

    Requires a `MetaSelection`, which only `select_best_candidate` can produce,
    so this cannot run before the method has been chosen on held-out data.
    """
    if not isinstance(selection, MetaSelection):
        raise TypeError(
            "fit_final_meta_parameters requires a MetaSelection produced by "
            "select_best_candidate; the final fit must never precede method selection."
        )
    if len(predictions) != EXPECTED_ROWS:
        raise ValueError(
            f"final meta-parameter fit expects all {EXPECTED_ROWS} development rows, "
            f"got {len(predictions)}"
        )

    if selection.incumbent_retained:
        # Nothing to fit - the identity is the frozen meta-layer.
        return {"T": 1.0, "w": 1.0}

    candidate = CANDIDATES[selection.selected]
    return candidate.fit(predictions.proba_trio, predictions.proba_dc, predictions.y)


# --------------------------------------------------------------------------
# Reliability reporting (per-class; evaluation.py's ECE is top-label only)
# --------------------------------------------------------------------------
def per_class_reliability(
    y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = RELIABILITY_BINS
) -> dict[str, list[dict[str, float]]]:
    """Equal-width reliability bins per class: mean predicted vs observed rate."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    table: dict[str, list[dict[str, float]]] = {}
    for index, class_name in enumerate(CLASS_NAMES):
        predicted = proba[:, index]
        observed = (y_true == EXPECTED_CLASSES[index]).astype(float)
        rows: list[dict[str, float]] = []
        for low, high in zip(edges[:-1], edges[1:]):
            in_bin = (predicted > low) & (predicted <= high) if low > 0 else (predicted >= low) & (predicted <= high)
            count = int(in_bin.sum())
            if count == 0:
                continue
            rows.append(
                {
                    "bin_low": float(low),
                    "bin_high": float(high),
                    "n": count,
                    "mean_predicted": float(predicted[in_bin].mean()),
                    "observed_rate": float(observed[in_bin].mean()),
                }
            )
        table[class_name] = rows
    return table


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_calibration_experiments(predictions: MetaPredictions | None = None) -> dict[str, Any]:
    """Run every pre-committed candidate through strict chronological
    meta-validation and apply the selection rule."""
    predictions = predictions if predictions is not None else load_meta_predictions()

    results = [run_meta_validation(candidate, predictions) for candidate in CANDIDATES.values()]
    selection = select_best_candidate(results)
    final_params = fit_final_meta_parameters(selection, predictions)

    return {
        "predictions": predictions,
        "results": results,
        "selection": selection,
        "final_meta_parameters": final_params,
        "library_versions": library_versions(),
    }


def library_versions() -> dict[str, str]:
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }

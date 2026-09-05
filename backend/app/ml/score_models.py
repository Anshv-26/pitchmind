"""Independent Poisson and Dixon-Coles football score models.

Unlike the Stage 1 classifiers, these models predict a full scoreline
distribution per fixture (expected goals per side, a scoreline probability
matrix, most-likely scorelines) from which H/D/A probabilities are *derived*
and evaluated with the same `evaluation.py` metrics and the same three
approved rolling-origin development folds used in Stage 1.

Design summary (see the approved plan for the full derivation):

* `log(lambda_home) = c + gamma + attack[home] + defence[away]`
  `log(lambda_away) = c +         attack[away] + defence[home]`
  with `attack`/`defence` sum-to-zero identified (0 = league average), which
  is also exactly the right L2 shrinkage target.
* Dixon-Coles adds a low-score correction tau(x,y; lambda_H, lambda_A, rho)
  for the (0,0)/(0,1)/(1,0)/(1,1) cells only. tau feasibility (all four
  cells strictly positive and finite) is enforced by REJECTING infeasible
  parameter points with a large, violation-scaled objective penalty - never
  by flooring or clipping an invalid tau. The same rejection-not-flooring
  rule applies to a violated attack/defence bound after sum-to-zero
  reconstruction (see `_infeasibility_penalty`).
* Fitting reads only `data/processed/matches.parquet`, restricted to
  `Season, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR`, via `load_score_fold`
  - which reuses the same fold table as Stage 1
  (`datasets.DEVELOPMENT_FOLD_DEFINITIONS`) read-only, so there is no path to
  fold 4 / the sealed 2025/26 season through this module either.
* Model-selection logic here is intentionally LOCAL to this module (a small
  complexity ranking specific to these five configs, and a local tie-break
  function) rather than reusing/extending
  `training.MODEL_COMPLEXITY_RANK`/`training.select_best_configuration`,
  which know nothing about score-model config names. `evaluation.py`'s
  metric and probability-contract functions ARE reused directly, unchanged.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from backend.app.ml.datasets import DEVELOPMENT_FOLD_DEFINITIONS, DEVELOPMENT_FOLDS
from backend.app.ml.evaluation import (
    FoldMetrics,
    compute_fold_metrics,
    paired_log_loss_comparison,
    validate_probabilities,
)
from backend.app.ml.feature_engineering import SEALED_SEASON, TARGET_MAPPING

REPO_ROOT = Path(__file__).resolve().parents[3]
MATCHES_PATH = REPO_ROOT / "data" / "processed" / "matches.parquet"

# The only columns these models ever read. Deliberately excludes every other
# column in matches.parquet (half-time score, shots, cards, referee, ...) -
# these models use nothing but final-score goal counts and match identity.
MATCH_COLUMNS = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# --------------------------------------------------------------------------
# Parameter bounds (a-priori, not data-estimated)
# --------------------------------------------------------------------------
BOUND_INTERCEPT = (-3.0, 3.0)
BOUND_HOME_ADVANTAGE = (-1.0, 1.0)
BOUND_TEAM_PARAM = (-1.5, 1.5)  # applies to the FULL reconstructed atk/def vectors
BOUND_RHO = (-0.2, 0.2)  # a search-range convenience only - carries NO feasibility guarantee (see module docstring)

# Rejection thresholds. Both are used only to REJECT an infeasible parameter
# point (large objective penalty); neither is ever substituted into a
# likelihood or a returned probability.
INFEASIBLE_PENALTY = 1.0e8
TAU_FEASIBILITY_EPS = 1.0e-10  # tau must be >= this to be accepted; 0 or negative is rejected

# Scoreline grid: start here, extend in steps up to the ceiling until the
# truncated tail mass is below the threshold. Never renormalise before that
# check passes - see `_build_scoreline_matrix`.
SCORELINE_GRID_START = 15
SCORELINE_GRID_CEILING = 30
SCORELINE_GRID_STEP = 5
TRUNCATION_MASS_THRESHOLD = 1.0e-5

TOP_N_SCORELINES = 3


# --------------------------------------------------------------------------
# Fold loading (safe, read-only reuse of the Stage 1 fold table)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreFoldData:
    """One development fold's match history, for score models only.

    Unlike `datasets.FoldData` (causal 35-feature artifacts, which
    deliberately exclude current-match goals), this carries raw
    `Season, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR` - everything a score
    model needs and nothing else.
    """

    fold: int
    train_seasons: list[str]
    validation_season: str
    train: pd.DataFrame
    validation: pd.DataFrame


def load_score_fold(fold_number: int) -> ScoreFoldData:
    """Load one development fold's match history, safely.

    Reuses the exact same fold table as Stage 1
    (`datasets.DEVELOPMENT_FOLD_DEFINITIONS`), read-only, so fold numbers
    outside {1, 2, 3} are refused here exactly as they are in
    `datasets.load_fold` - there is no path to fold 4 / the sealed 2025/26
    season through this function.
    """
    if fold_number not in DEVELOPMENT_FOLD_DEFINITIONS:
        raise ValueError(
            f"unknown development fold {fold_number!r}; only "
            f"{DEVELOPMENT_FOLDS} are supported here. The final fold (train "
            f"through 2024/25, test 2025/26) is intentionally not available "
            f"through this loader - the sealed season must never be touched "
            f"during development."
        )
    definition = DEVELOPMENT_FOLD_DEFINITIONS[fold_number]
    training_cutoff = definition["training_cutoff"]
    validation_season = definition["validation_season"]

    if not MATCHES_PATH.exists():
        raise FileNotFoundError(f"{MATCHES_PATH} not found; run scripts/process_data.py first")

    frame = pd.read_parquet(MATCHES_PATH, columns=MATCH_COLUMNS)
    frame["Date"] = pd.to_datetime(frame["Date"])

    order = list(frame.groupby("Season")["Date"].min().sort_values().index)
    if validation_season not in order:
        raise ValueError(f"validation season {validation_season!r} not present in {MATCHES_PATH}")

    seasons_through_validation = set(order[: order.index(validation_season) + 1])
    frame = frame[frame["Season"].isin(seasons_through_validation)].reset_index(drop=True)

    present_seasons = set(frame["Season"].unique())
    if SEALED_SEASON in present_seasons:
        raise ValueError(
            f"fold {fold_number} score-model frame contains sealed-season "
            f"({SEALED_SEASON!r}) rows; refusing to use it for development."
        )

    validation_mask = frame["Season"] == validation_season
    if int(validation_mask.sum()) == 0:
        raise ValueError(
            f"fold {fold_number} has no rows for validation season {validation_season!r}."
        )
    validation_seasons_present = set(frame.loc[validation_mask, "Season"].unique())
    if validation_seasons_present != {validation_season}:
        raise ValueError(
            f"expected exactly one validation season {validation_season!r}, "
            f"found {validation_seasons_present}."
        )

    train_mask = ~validation_mask
    if validation_season in set(frame.loc[train_mask, "Season"].unique()):
        raise ValueError(
            f"validation season {validation_season!r} rows leaked into the "
            f"training split for fold {fold_number}."
        )

    train_frame = frame.loc[train_mask].reset_index(drop=True)
    val_frame = frame.loc[validation_mask].reset_index(drop=True)

    train_seasons = sorted(train_frame["Season"].unique().tolist(), key=order.index)
    if train_seasons[-1] != training_cutoff:
        raise ValueError(
            f"fold {fold_number}: training seasons end at {train_seasons[-1]!r}, "
            f"expected {training_cutoff!r}."
        )

    if train_frame["Date"].max() >= val_frame["Date"].min():
        raise ValueError(
            f"fold {fold_number}: training rows are not strictly earlier than "
            f"validation rows (train max date {train_frame['Date'].max()} >= "
            f"validation min date {val_frame['Date'].min()})."
        )

    return ScoreFoldData(
        fold=fold_number,
        train_seasons=train_seasons,
        validation_season=validation_season,
        train=train_frame,
        validation=val_frame,
    )


def load_all_development_score_folds() -> list[ScoreFoldData]:
    """Convenience: every development fold, in order."""
    return [load_score_fold(fold_number) for fold_number in DEVELOPMENT_FOLDS]


# --------------------------------------------------------------------------
# Config grid (five pre-committed configurations)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreModelConfig:
    config_id: str
    use_dixon_coles: bool
    l2_sigma: float | None  # None = no regularisation; else Gaussian prior SD on attack/defence
    half_life_days: float | None  # None = no time decay


CONFIGS: dict[str, ScoreModelConfig] = {
    "poisson": ScoreModelConfig("poisson", use_dixon_coles=False, l2_sigma=None, half_life_days=None),
    "poisson_l2": ScoreModelConfig("poisson_l2", use_dixon_coles=False, l2_sigma=0.25, half_life_days=None),
    "dixon_coles": ScoreModelConfig("dixon_coles", use_dixon_coles=True, l2_sigma=None, half_life_days=None),
    "dixon_coles_l2": ScoreModelConfig("dixon_coles_l2", use_dixon_coles=True, l2_sigma=0.25, half_life_days=None),
    "dixon_coles_l2_decay": ScoreModelConfig(
        "dixon_coles_l2_decay", use_dixon_coles=True, l2_sigma=0.25, half_life_days=365.0
    ),
}

# Local complexity ranking for these five configs only (lower = simpler).
# Deliberately NOT added to training.MODEL_COMPLEXITY_RANK, which is keyed to
# Stage 1's classifier family names and has no notion of these config ids.
SCORE_MODEL_COMPLEXITY_RANK: dict[str, int] = {
    "poisson": 0,
    "poisson_l2": 1,
    "dixon_coles": 2,
    "dixon_coles_l2": 3,
    "dixon_coles_l2_decay": 4,
}


# --------------------------------------------------------------------------
# Fitted model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreModelParams:
    """A fitted Independent Poisson or Dixon-Coles model.

    `attack`/`defence` are the FULL reconstructed, sum-to-zero vectors (one
    entry per team seen anywhere in the training window) - 0 means exactly
    league average. `attack[i]` higher = scores more; `defence[j]` higher =
    concedes more (leakiness, not quality - documented here to avoid a sign
    trap). `rho` is `None` for the independent Poisson configs.
    """

    config_id: str
    teams: tuple[str, ...]
    intercept: float
    home_advantage: float
    attack: dict[str, float]
    defence: dict[str, float]
    rho: float | None
    promoted_attack_offset: float
    promoted_defence_offset: float

    def team_attack(self, team: str) -> float:
        """Fitted attack if seen in training (continuing or returning);
        otherwise the training-only promoted-team prior (unseen team)."""
        return self.attack.get(team, self.promoted_attack_offset)

    def team_defence(self, team: str) -> float:
        return self.defence.get(team, self.promoted_defence_offset)


def _team_universe(train: pd.DataFrame) -> list[str]:
    return sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))


def _season_order(frame: pd.DataFrame) -> list[str]:
    return list(frame.groupby("Season")["Date"].min().sort_values().index)


def _promoted_team_prior(train: pd.DataFrame) -> tuple[float, float]:
    """Method-of-moments attack/defence offset for a team never seen in
    training, estimated from TRAINING ROWS ONLY.

    For each team-season in the training window that is that team's first
    appearance (excluding the window's opening season, which has no prior
    baseline to compare against), compute that team's mean goals for/against
    and the league's mean goals per team-match in that same season, then
    average the log ratios across every such newcomer team-season. Falls
    back to (0.0, 0.0) - i.e. league average - if the window has no
    newcomers to learn from.
    """
    order = _season_order(train)
    if len(order) < 2:
        return 0.0, 0.0

    attack_offsets: list[float] = []
    defence_offsets: list[float] = []
    seen: set[str] = set()

    for index, season in enumerate(order):
        season_df = train[train["Season"] == season]
        season_teams = set(season_df["HomeTeam"]) | set(season_df["AwayTeam"])
        if index == 0:
            seen |= season_teams
            continue

        newcomers = season_teams - seen
        if newcomers:
            n_matches = len(season_df)
            league_gpg = (season_df["FTHG"].sum() + season_df["FTAG"].sum()) / (2 * n_matches)
            if league_gpg > 0:
                for team in newcomers:
                    goals_for = pd.concat(
                        [
                            season_df.loc[season_df["HomeTeam"] == team, "FTHG"],
                            season_df.loc[season_df["AwayTeam"] == team, "FTAG"],
                        ]
                    )
                    goals_against = pd.concat(
                        [
                            season_df.loc[season_df["HomeTeam"] == team, "FTAG"],
                            season_df.loc[season_df["AwayTeam"] == team, "FTHG"],
                        ]
                    )
                    gf_mean = goals_for.mean()
                    ga_mean = goals_against.mean()
                    if len(goals_for) > 0 and gf_mean > 0:
                        attack_offsets.append(float(np.log(gf_mean / league_gpg)))
                    if len(goals_against) > 0 and ga_mean > 0:
                        defence_offsets.append(float(np.log(ga_mean / league_gpg)))
        seen |= season_teams

    attack_offset = float(np.mean(attack_offsets)) if attack_offsets else 0.0
    defence_offset = float(np.mean(defence_offsets)) if defence_offsets else 0.0
    return attack_offset, defence_offset


def _time_decay_weights(dates: pd.Series, half_life_days: float | None) -> np.ndarray:
    """`w_k = exp(-xi * delta_k)`, `delta_k` in days from the LATEST TRAINING
    date (never a validation date - zero leakage surface), `xi = ln(2) /
    half_life_days`. Returns all-ones if `half_life_days` is None."""
    if half_life_days is None:
        return np.ones(len(dates), dtype=float)
    t_ref = dates.max()
    delta_days = (t_ref - dates).dt.days.to_numpy(dtype=float)
    xi = np.log(2.0) / half_life_days
    return np.exp(-xi * delta_days)


def _unpack_theta(theta: np.ndarray, n_teams: int, has_rho: bool):
    """Split the flat optimiser vector and reconstruct the sum-to-zero
    attack/defence vectors: `atk_n = -sum(atk_free)`, `def_n = -sum(def_free)`.

    L-BFGS-B's `bounds=` only constrains the n-1 explicit free variables it
    is given; it has no way to bound the reconstructed nth value. That
    reconstructed value is therefore re-checked explicitly wherever this
    function's output is used (see `_infeasibility_penalty`, and the
    post-fit assertions in `fit_score_model`).
    """
    index = 0
    c = theta[index]
    index += 1
    gamma = theta[index]
    index += 1
    atk_free = theta[index : index + n_teams - 1]
    index += n_teams - 1
    def_free = theta[index : index + n_teams - 1]
    index += n_teams - 1
    rho = theta[index] if has_rho else None

    atk_full = np.append(atk_free, -atk_free.sum())
    def_full = np.append(def_free, -def_free.sum())
    return c, gamma, atk_full, def_full, rho


def _bound_violation(values: np.ndarray, bounds: tuple[float, float]) -> float:
    """Sum of how far `values` exceed `bounds`, in the direction of the
    violation only (0 if fully within bounds). Used to build an
    infeasibility penalty that still points the optimiser back toward the
    feasible region, rather than a flat wall."""
    lo, hi = bounds
    return float(np.sum(np.maximum(0.0, values - hi) + np.maximum(0.0, lo - values)))


def _dixon_coles_tau(lam_h: np.ndarray, lam_a: np.ndarray, rho: float):
    """The four Dixon-Coles low-score correction values, per match.
    `tau11` does not depend on lambda but is broadcast to match shape so all
    four can be feasibility-checked uniformly."""
    tau00 = 1.0 - lam_h * lam_a * rho
    tau01 = 1.0 + lam_h * rho
    tau10 = 1.0 + lam_a * rho
    tau11 = np.full_like(lam_h, 1.0 - rho)
    return tau00, tau01, tau10, tau11


def _negative_log_likelihood(
    theta: np.ndarray,
    *,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    has_rho: bool,
    l2_sigma: float | None,
) -> float:
    """Weighted negative log-likelihood (+ optional L2 penalty on attack/defence).

    Two independent infeasibility gates, checked cheapest-first, BEFORE any
    logarithm is taken: (1) the full reconstructed attack/defence vectors
    must lie within BOUND_TEAM_PARAM; (2) for Dixon-Coles, all four tau cells
    must be finite and >= TAU_FEASIBILITY_EPS for every training match. A
    violation returns `INFEASIBLE_PENALTY + violation_magnitude` - never a
    flat wall, and NEVER a floored/clipped value fed into the likelihood.
    """
    c, gamma, atk_full, def_full, rho = _unpack_theta(theta, n_teams, has_rho)

    if not (np.isfinite(atk_full).all() and np.isfinite(def_full).all()):
        return INFEASIBLE_PENALTY
    violation = _bound_violation(atk_full, BOUND_TEAM_PARAM) + _bound_violation(def_full, BOUND_TEAM_PARAM)
    if violation > 0.0:
        return INFEASIBLE_PENALTY + violation

    log_lam_h = c + gamma + atk_full[home_idx] + def_full[away_idx]
    log_lam_a = c + atk_full[away_idx] + def_full[home_idx]
    if not (np.isfinite(log_lam_h).all() and np.isfinite(log_lam_a).all()):
        return INFEASIBLE_PENALTY

    lam_h = np.exp(log_lam_h)
    lam_a = np.exp(log_lam_a)
    if not (np.isfinite(lam_h).all() and np.isfinite(lam_a).all()):
        return INFEASIBLE_PENALTY

    poisson_ll = (
        home_goals * log_lam_h - lam_h - gammaln(home_goals + 1.0)
        + away_goals * log_lam_a - lam_a - gammaln(away_goals + 1.0)
    )

    if has_rho:
        tau00, tau01, tau10, tau11 = _dixon_coles_tau(lam_h, lam_a, rho)
        taus = np.stack([tau00, tau01, tau10, tau11])
        if not np.isfinite(taus).all():
            return INFEASIBLE_PENALTY
        tau_violation = float(np.sum(np.maximum(0.0, TAU_FEASIBILITY_EPS - taus)))
        if tau_violation > 0.0:
            return INFEASIBLE_PENALTY + tau_violation

        log_tau = np.zeros_like(home_goals, dtype=float)
        is_00 = (home_goals == 0) & (away_goals == 0)
        is_01 = (home_goals == 0) & (away_goals == 1)
        is_10 = (home_goals == 1) & (away_goals == 0)
        is_11 = (home_goals == 1) & (away_goals == 1)
        log_tau[is_00] = np.log(tau00[is_00])
        log_tau[is_01] = np.log(tau01[is_01])
        log_tau[is_10] = np.log(tau10[is_10])
        log_tau[is_11] = np.log(tau11[is_11])
        log_likelihood = poisson_ll + log_tau
    else:
        log_likelihood = poisson_ll

    weighted_nll = -float(np.sum(weights * log_likelihood))

    if l2_sigma is not None:
        weighted_nll += (1.0 / (2.0 * l2_sigma**2)) * float(np.sum(atk_full**2) + np.sum(def_full**2))

    if not np.isfinite(weighted_nll):
        return INFEASIBLE_PENALTY

    return weighted_nll


def fit_score_model(train: pd.DataFrame, config: ScoreModelConfig) -> ScoreModelParams:
    """Fit an Independent Poisson or Dixon-Coles model on `train` ONLY.

    There is no parameter through which validation data could enter - the
    same structural guarantee used by Stage 1's `class_frequency_baseline`.
    Determinism: no RNG anywhere; a fixed initial point plus a deterministic
    optimiser (L-BFGS-B) means two calls on the same `train` frame produce
    bit-identical fitted parameters.
    """
    teams = _team_universe(train)
    n_teams = len(teams)
    if n_teams < 2:
        raise ValueError(f"need at least 2 teams to fit a score model, got {n_teams}")
    team_to_idx = {team: index for index, team in enumerate(teams)}

    home_idx = train["HomeTeam"].map(team_to_idx).to_numpy()
    away_idx = train["AwayTeam"].map(team_to_idx).to_numpy()
    home_goals = train["FTHG"].to_numpy(dtype=float)
    away_goals = train["FTAG"].to_numpy(dtype=float)
    weights = _time_decay_weights(train["Date"], config.half_life_days)

    weighted_mean_home = float(np.sum(weights * home_goals) / np.sum(weights))
    weighted_mean_away = float(np.sum(weights * away_goals) / np.sum(weights))
    c0 = float(np.log(weighted_mean_away))
    gamma0 = float(np.log(weighted_mean_home / weighted_mean_away))

    theta0 = np.concatenate(
        [
            [c0, gamma0],
            np.zeros(n_teams - 1),  # attack, free
            np.zeros(n_teams - 1),  # defence, free
        ]
    )
    bounds = [BOUND_INTERCEPT, BOUND_HOME_ADVANTAGE]
    bounds += [BOUND_TEAM_PARAM] * (n_teams - 1)  # attack, free
    bounds += [BOUND_TEAM_PARAM] * (n_teams - 1)  # defence, free
    if config.use_dixon_coles:
        theta0 = np.concatenate([theta0, [-0.05]])
        bounds += [BOUND_RHO]

    objective = partial(
        _negative_log_likelihood,
        home_idx=home_idx,
        away_idx=away_idx,
        home_goals=home_goals,
        away_goals=away_goals,
        weights=weights,
        n_teams=n_teams,
        has_rho=config.use_dixon_coles,
        l2_sigma=config.l2_sigma,
    )
    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8},
    )

    if not result.success:
        raise RuntimeError(f"score model fit ({config.config_id}) did not converge: {result.message}")
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"score model fit ({config.config_id}) produced non-finite parameters")
    if not np.isfinite(result.fun) or result.fun >= INFEASIBLE_PENALTY / 2.0:
        raise RuntimeError(
            f"score model fit ({config.config_id}) terminated in an infeasible "
            f"region (objective={result.fun!r}); this indicates the optimiser "
            f"could not find a feasible point, not merely a poor one."
        )

    c, gamma, atk_full, def_full, rho = _unpack_theta(result.x, n_teams, config.use_dixon_coles)

    if abs(float(atk_full.sum())) > 1e-6:
        raise RuntimeError(f"attack identifiability failed: sum={atk_full.sum()!r}")
    if abs(float(def_full.sum())) > 1e-6:
        raise RuntimeError(f"defence identifiability failed: sum={def_full.sum()!r}")
    if _bound_violation(atk_full, BOUND_TEAM_PARAM) > 0.0 or _bound_violation(def_full, BOUND_TEAM_PARAM) > 0.0:
        raise RuntimeError(
            f"fitted attack/defence vectors violate BOUND_TEAM_PARAM={BOUND_TEAM_PARAM} "
            f"after sum-to-zero reconstruction (config={config.config_id})"
        )

    log_lam_h = c + gamma + atk_full[home_idx] + def_full[away_idx]
    log_lam_a = c + atk_full[away_idx] + def_full[home_idx]
    lam_h = np.exp(log_lam_h)
    lam_a = np.exp(log_lam_a)
    if not (np.isfinite(lam_h).all() and np.isfinite(lam_a).all() and (lam_h > 0).all() and (lam_a > 0).all()):
        raise RuntimeError(f"fitted model ({config.config_id}) produces non-positive/non-finite lambda on training data")

    if config.use_dixon_coles:
        tau00, tau01, tau10, tau11 = _dixon_coles_tau(lam_h, lam_a, rho)
        taus = np.stack([tau00, tau01, tau10, tau11])
        if not np.isfinite(taus).all() or (taus <= 0.0).any():
            raise RuntimeError(
                f"fitted Dixon-Coles model ({config.config_id}) has a non-positive or "
                f"non-finite tau on training data (rho={rho!r}); this must never happen "
                f"post-fit given the in-objective feasibility rejection."
            )

    attack_offset, defence_offset = _promoted_team_prior(train)

    return ScoreModelParams(
        config_id=config.config_id,
        teams=tuple(teams),
        intercept=float(c),
        home_advantage=float(gamma),
        attack=dict(zip(teams, map(float, atk_full))),
        defence=dict(zip(teams, map(float, def_full))),
        rho=(float(rho) if config.use_dixon_coles else None),
        promoted_attack_offset=attack_offset,
        promoted_defence_offset=defence_offset,
    )


# --------------------------------------------------------------------------
# Prediction: expected goals, scoreline matrix, derived H/D/A
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScorelinePrediction:
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray  # shape (max_goals+1, max_goals+1); rows=home goals, cols=away goals; sums to 1
    p_home: float
    p_draw: float
    p_away: float
    expected_home_goals: float
    expected_away_goals: float
    top_scorelines: list[tuple[tuple[int, int], float]]
    most_likely_scoreline: str


def _build_scoreline_matrix(lam_h: float, lam_a: float, rho: float | None, *, use_dixon_coles: bool) -> np.ndarray:
    """Build a scoreline probability matrix, renormalising only after every
    validity check has passed.

    Order of operations (never reversed): (1) if Dixon-Coles, compute and
    validate the four tau cells for THIS fixture's (lambda_home, lambda_away,
    rho) - raise if any is <= 0 or non-finite, never floor/clip; (2) build
    the Poisson outer-product grid, starting at `SCORELINE_GRID_START` goals
    per side and extending in `SCORELINE_GRID_STEP` increments up to
    `SCORELINE_GRID_CEILING` until the truncated tail mass is below
    `TRUNCATION_MASS_THRESHOLD`, raising loudly if the ceiling is reached
    first; (3) assert non-negativity; (4) only then renormalise so the
    matrix sums to exactly 1.
    """
    if use_dixon_coles:
        tau00, tau01, tau10, tau11 = _dixon_coles_tau(np.array(lam_h), np.array(lam_a), rho)
        taus = {"tau(0,0)": float(tau00), "tau(0,1)": float(tau01), "tau(1,0)": float(tau10), "tau(1,1)": float(tau11)}
        if not all(np.isfinite(v) for v in taus.values()) or min(taus.values()) < TAU_FEASIBILITY_EPS:
            raise RuntimeError(
                f"Dixon-Coles correction is infeasible for lambda_home={lam_h!r}, "
                f"lambda_away={lam_a!r}, rho={rho!r}: {taus}. Refusing to floor or "
                f"clip - this fixture cannot be scored with this fitted model."
            )
    else:
        taus = None

    max_goals = SCORELINE_GRID_START
    while True:
        goals = np.arange(0, max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
        if use_dixon_coles:
            matrix[0, 0] *= taus["tau(0,0)"]
            matrix[0, 1] *= taus["tau(0,1)"]
            matrix[1, 0] *= taus["tau(1,0)"]
            matrix[1, 1] *= taus["tau(1,1)"]

        truncated_mass = 1.0 - float(matrix.sum())
        if truncated_mass < TRUNCATION_MASS_THRESHOLD:
            break
        if max_goals >= SCORELINE_GRID_CEILING:
            raise RuntimeError(
                f"could not reduce truncated scoreline mass below "
                f"{TRUNCATION_MASS_THRESHOLD} even at the {SCORELINE_GRID_CEILING}-goal "
                f"ceiling (lambda_home={lam_h!r}, lambda_away={lam_a!r}, "
                f"truncated_mass={truncated_mass!r}); this likely indicates an "
                f"implausible fitted lambda rather than a grid that is merely too small."
            )
        max_goals += SCORELINE_GRID_STEP

    if (matrix < -1e-9).any():
        raise RuntimeError(
            f"scoreline matrix contains a materially negative probability "
            f"(min={matrix.min()!r}) for lambda_home={lam_h!r}, lambda_away={lam_a!r}, "
            f"rho={rho!r}; this should be impossible given the tau feasibility check above."
        )

    return matrix / matrix.sum()


def _summarize_matrix(matrix: np.ndarray, lam_h: float, lam_a: float, *, top_n: int = TOP_N_SCORELINES) -> ScorelinePrediction:
    size = matrix.shape[0]
    row_index, col_index = np.indices((size, size))
    p_home = float(matrix[row_index > col_index].sum())
    p_draw = float(matrix[row_index == col_index].sum())
    p_away = float(matrix[row_index < col_index].sum())

    goal_values = np.arange(size)
    expected_home = float((matrix.sum(axis=1) * goal_values).sum())
    expected_away = float((matrix.sum(axis=0) * goal_values).sum())

    flat_order = np.argsort(matrix, axis=None)[::-1][:top_n]
    top_scorelines: list[tuple[tuple[int, int], float]] = []
    for flat_index in flat_order:
        home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
        top_scorelines.append(((int(home_goals), int(away_goals)), float(matrix[home_goals, away_goals])))
    most_likely = f"{top_scorelines[0][0][0]}-{top_scorelines[0][0][1]}"

    return ScorelinePrediction(
        lambda_home=lam_h,
        lambda_away=lam_a,
        matrix=matrix,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        expected_home_goals=expected_home,
        expected_away_goals=expected_away,
        top_scorelines=top_scorelines,
        most_likely_scoreline=most_likely,
    )


def predict_match(params: ScoreModelParams, config: ScoreModelConfig, home_team: str, away_team: str) -> ScorelinePrediction:
    """Predict one fixture's scoreline distribution.

    `home_team`/`away_team` may be any team, whether or not it appeared in
    training: `ScoreModelParams.team_attack`/`team_defence` fall back to the
    training-only promoted-team prior for a team absent from the fitted
    dictionaries (see `_promoted_team_prior`).
    """
    log_lam_h = params.intercept + params.home_advantage + params.team_attack(home_team) + params.team_defence(away_team)
    log_lam_a = params.intercept + params.team_attack(away_team) + params.team_defence(home_team)
    lam_h = float(np.exp(log_lam_h))
    lam_a = float(np.exp(log_lam_a))
    if not (np.isfinite(lam_h) and np.isfinite(lam_a) and lam_h > 0.0 and lam_a > 0.0):
        raise RuntimeError(
            f"non-positive/non-finite lambda for {home_team!r} vs {away_team!r}: "
            f"lambda_home={lam_h!r}, lambda_away={lam_a!r}"
        )

    matrix = _build_scoreline_matrix(lam_h, lam_a, params.rho, use_dixon_coles=config.use_dixon_coles)
    return _summarize_matrix(matrix, lam_h, lam_a)


def predict_fold(
    params: ScoreModelParams, config: ScoreModelConfig, validation: pd.DataFrame
) -> tuple[np.ndarray, list[ScorelinePrediction]]:
    """Predict every fixture in `validation`. Returns the (n, 3) [H, D, A]
    probability array (already in the fixed [0, 1, 2] order by construction
    - these columns are built directly, not routed through an estimator's
    `classes_`) plus the full per-match `ScorelinePrediction` objects."""
    predictions = [
        predict_match(params, config, row.HomeTeam, row.AwayTeam) for row in validation.itertuples()
    ]
    proba = np.array([[p.p_home, p.p_draw, p.p_away] for p in predictions], dtype=float)
    problems = validate_probabilities(proba, n_rows=len(validation))
    if problems:
        raise ValueError(f"score-model predictions violate the probability contract: {problems}")
    return proba, predictions


# --------------------------------------------------------------------------
# Experiment loop and LOCAL model selection
# --------------------------------------------------------------------------
@dataclass
class ScoreModelResult:
    """One config's aggregated performance across every development fold.
    Mirrors the shape of `training.ExperimentResult` for reporting
    consistency, but is defined locally - it is never passed to
    `training.select_best_configuration`."""

    config_id: str
    fold_metrics: list[FoldMetrics]
    fold_proba: list[np.ndarray] = field(repr=False)
    fold_y_true: list[np.ndarray] = field(repr=False)
    mean_log_loss: float
    worst_log_loss: float
    log_loss_std: float

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "fold_metrics": [m.to_dict() for m in self.fold_metrics],
            "mean_log_loss": self.mean_log_loss,
            "worst_log_loss": self.worst_log_loss,
            "log_loss_std": self.log_loss_std,
        }


def _score_predictions_to_rows(
    fold: ScoreFoldData,
    proba: np.ndarray,
    predictions: list[ScorelinePrediction],
    y_true: np.ndarray,
    config_id: str,
) -> list[dict]:
    validation = fold.validation
    rows = []
    for i in range(len(validation)):
        prediction = predictions[i]
        rows.append(
            {
                "fold": fold.fold,
                "Season": validation.iloc[i]["Season"],
                "Date": pd.Timestamp(validation.iloc[i]["Date"]).date().isoformat(),
                "HomeTeam": validation.iloc[i]["HomeTeam"],
                "AwayTeam": validation.iloc[i]["AwayTeam"],
                "actual_target": int(y_true[i]),
                "actual_ftr": validation.iloc[i]["FTR"],
                "expected_home_goals": prediction.expected_home_goals,
                "expected_away_goals": prediction.expected_away_goals,
                "p_home": float(proba[i, 0]),
                "p_draw": float(proba[i, 1]),
                "p_away": float(proba[i, 2]),
                "most_likely_scoreline": prediction.most_likely_scoreline,
                "model": "score_model",
                "config_id": config_id,
            }
        )
    return rows


def run_score_model_experiment(
    config: ScoreModelConfig, folds: list[ScoreFoldData]
) -> tuple[ScoreModelResult, list[dict]]:
    """Fit+predict one config across every development fold."""
    fold_metrics: list[FoldMetrics] = []
    fold_proba: list[np.ndarray] = []
    fold_y_true: list[np.ndarray] = []
    predictions_rows: list[dict] = []

    for fold in folds:
        params = fit_score_model(fold.train, config)
        proba, predictions = predict_fold(params, config, fold.validation)
        y_true = fold.validation["FTR"].map(TARGET_MAPPING).to_numpy()

        metrics = compute_fold_metrics(
            fold=fold.fold, validation_season=fold.validation_season, y_true=y_true, proba=proba
        )
        fold_metrics.append(metrics)
        fold_proba.append(proba)
        fold_y_true.append(y_true)
        predictions_rows.extend(_score_predictions_to_rows(fold, proba, predictions, y_true, config.config_id))

    log_losses = np.array([m.log_loss for m in fold_metrics])
    result = ScoreModelResult(
        config_id=config.config_id,
        fold_metrics=fold_metrics,
        fold_proba=fold_proba,
        fold_y_true=fold_y_true,
        mean_log_loss=float(log_losses.mean()),
        worst_log_loss=float(log_losses.max()),
        log_loss_std=float(log_losses.std(ddof=0)),
    )
    return result, predictions_rows


def select_best_score_model(results: list[ScoreModelResult]) -> ScoreModelResult:
    """LOCAL selection rule, mirroring Stage 1's tie-break structure
    (`training.select_best_configuration`) without depending on it:

    1. Lowest mean development log loss.
    2. Compare every other candidate against the current best via
       `evaluation.paired_log_loss_comparison`; a tie is
       `abs(mean_diff) < 2 * standard_error`.
    3. Among tied candidates, prefer lower worst-season log loss.
    4. Then prefer the simpler config (`SCORE_MODEL_COMPLEXITY_RANK`).
    5. Then lower across-fold log-loss variance.
    """
    if not results:
        raise ValueError("no score-model results to select from")

    ranked = sorted(results, key=lambda r: r.mean_log_loss)
    best = ranked[0]

    tied = [best]
    for candidate in ranked[1:]:
        comparison = paired_log_loss_comparison(best.fold_y_true, best.fold_proba, candidate.fold_proba)
        if comparison["is_tie"]:
            tied.append(candidate)

    if len(tied) == 1:
        return best

    tied.sort(
        key=lambda r: (r.worst_log_loss, SCORE_MODEL_COMPLEXITY_RANK.get(r.config_id, 99), r.log_loss_std)
    )
    return tied[0]


def run_score_model_experiments(folds: list[ScoreFoldData] | None = None) -> dict:
    """Run all five pre-committed configs across the three development
    folds, and select the best. Never touches 2025/26: `folds` defaults to
    `load_all_development_score_folds()`, which has no path to it."""
    folds = folds if folds is not None else load_all_development_score_folds()

    results: list[ScoreModelResult] = []
    predictions: list[dict] = []
    for config in CONFIGS.values():
        result, rows = run_score_model_experiment(config, folds)
        results.append(result)
        predictions.extend(rows)

    best = select_best_score_model(results)

    return {
        "development_folds": list(DEVELOPMENT_FOLDS),
        "results": results,
        "predictions": predictions,
        "best": best,
        "library_versions": library_versions(),
    }


def library_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

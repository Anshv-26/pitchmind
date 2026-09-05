"""Leakage-safe pre-match feature engineering for PitchMind.

Every feature emitted for a match is a function of matches that finished
strictly *before* that match's date — the current match never contributes to
its own feature row.

Four rules drive the design:

1. **Date batching.** Matches are processed grouped by date. All feature rows
   for date D are emitted from state as of the end of dates < D; only then are
   D's results applied to state. Same-date alphabetical order therefore carries
   no temporal information, and one fixture's result can never leak into
   another fixture played the same day.

2. **Causal Elo parameters, not fold-wide ones.** `home_advantage`,
   `season_shrink` and `promoted_prior_delta` are all estimated FROM MATCH
   OUTCOMES (`FTR`) — they are supervised statistics, not unsupervised
   properties of the inputs. Estimating one fixed `EloParams` object from an
   entire multi-season training window (still supported, see below, as a
   diagnostic mode) would let an early training row's features depend on a
   parameter estimate that is partly computed from that row's own match
   result — target leakage into the predictors, closer to target encoding
   without cross-fitting than to fitting a `StandardScaler`.

   The safe, production mechanism is `EloParamSchedule`: an immutable
   season -> `EloParams` mapping in which every season's parameters are
   estimated ONLY from strictly earlier seasons (see
   `build_causal_elo_schedule`). Pass a schedule to `build_features` and every
   row's Elo features are pre-match causal at the *parameter* level, not only
   at the state-machine level. A single `EloParams` object remains accepted
   by `build_features` for backward-compatible/diagnostic use, but it must
   never be used to build an artifact intended for model evaluation.

3. **Explicit parameters, validated artifacts.** Parameters (fixed or
   scheduled) are passed in explicitly; no module-level constant holds a
   data-estimated value. `assert_artifact_valid_for` must be called before any
   fit or scoring — it rejects the canonical a-priori artifact, rejects the
   wrong fold/evaluation pairing, and (for causal artifacts) re-validates the
   entire schedule's causal ordering from its provenance sidecar.

4. **Stable Elo scale.** After each season's rollover the active clubs are
   re-centred on `initial_rating`, so `home_elo`/`away_elo` mean the same thing
   in 2015/16 as in 2025/26. Re-centring is a deterministic affine shift; it
   preserves every rating difference and estimates nothing from data.

The functions in the "Artifact validation" section touch the filesystem by
design (they read an artifact's provenance sidecar). Everything else in this
module is pure.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

# --------------------------------------------------------------------------
# A-priori constants: chosen by convention, estimated from no data, and
# therefore fold-safe. Frozen for V1.
# --------------------------------------------------------------------------
DEFAULT_EWMA_HALFLIFE = 10.0
DEFAULT_MIN_PERIODS = 3
DEFAULT_EFFICIENCY_WINDOW = 10
DEFAULT_K_FACTOR = 20.0
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_REST_DAYS_CAP = 21
HISTORY_DEPTH_CAP = 38

# The sealed final test season. It must never be used to estimate any
# parameter, for Elo or anything else.
SEALED_SEASON = "2025_26"

# The dataset's historical start. `estimate_elo_params` refuses a frame that
# does not begin here, because the burn-in drop and the newcomer flags are both
# defined relative to the first season present: a truncated frame would
# silently reclassify incumbents as newcomers. Centralised so it can be changed
# in one place if the data source ever expands backward.
EXPECTED_DATASET_START_SEASON = "2015_16"

# The approved rolling-origin evaluation scheme. Each evaluation season may
# only be scored with parameters estimated through exactly the paired cutoff —
# chronological precedence alone is not enough, or a stale earlier fold would
# qualify for a later evaluation season (e.g. the sealed test).
INTENDED_TRAINING_CUTOFF: dict[str, str] = {
    "2022_23": "2021_22",
    "2023_24": "2022_23",
    "2024_25": "2023_24",
    "2025_26": "2024_25",
}

# Conservative V1 policy for the causal Elo schedule: a season is assigned
# a-priori parameters unless at least this many full seasons of strictly
# earlier history exist. See build_causal_elo_schedule's docstring for why
# this is a policy choice, not a mathematical necessity, and how it relates to
# estimate_elo_params's own internal minimum-window guard.
MIN_PRIOR_SEASONS_FOR_ESTIMATION = 3

# A-priori placeholders used for the canonical artifact and as the causal
# schedule's early-season fallback. These are conventional values, not
# estimates, so they carry empty provenance and are deliberately rejected for
# model evaluation wherever they appear.
CANONICAL_HOME_ADVANTAGE = 45.0
CANONICAL_SEASON_SHRINK = 0.80
CANONICAL_PROMOTED_PRIOR_DELTA = -100.0

CANONICAL_PURPOSE = (
    "Canonical convenience artifact: schema inspection, debugging, leakage "
    "audits and development only."
)
CANONICAL_NOT_VALID_NOTE = (
    "NOT VALID FOR MODEL EVALUATION. Built with a-priori placeholder Elo "
    "parameters that were estimated from no data. Any fold evaluation must "
    "rebuild features with a causal EloParamSchedule "
    "(see build_causal_elo_schedule)."
)

# Required columns on the input clean-match frame.
REQUIRED_INPUT_COLUMNS = (
    "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST",
)

# Team-level metrics tracked as EWMA state.
EWMA_METRICS = (
    "gf", "ga", "sot_for", "sot_against", "shots_for", "shots_against", "ppg",
)

RESULT_COLUMN = "FTR"
TARGET_COLUMN = "target"
TARGET_MAPPING = {"H": 0, "D": 1, "A": 2}
MATCH_KEY_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]

METADATA_COLUMNS = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "home_matches_played_season",
    "away_matches_played_season",
    "home_prior_matches_all_time",
    "away_prior_matches_all_time",
    "home_rest_days",
    "away_rest_days",
    "home_is_returning",
    "away_is_returning",
]

FEATURE_COLUMNS = [
    # Elo (3)
    "home_elo",
    "away_elo",
    "elo_diff",
    # EWMA team strength, home (7)
    "home_ewma_gf",
    "home_ewma_ga",
    "home_ewma_sot_for",
    "home_ewma_sot_against",
    "home_ewma_shots_for",
    "home_ewma_shots_against",
    "home_ewma_ppg",
    # EWMA team strength, away (7)
    "away_ewma_gf",
    "away_ewma_ga",
    "away_ewma_sot_for",
    "away_ewma_sot_against",
    "away_ewma_shots_for",
    "away_ewma_shots_against",
    "away_ewma_ppg",
    # Matchup differences (6)
    "diff_ewma_ppg",
    "diff_ewma_sot_diff",
    "diff_ewma_gd",
    "home_attack_vs_away_defence_goals",
    "away_attack_vs_home_defence_goals",
    "home_attack_vs_away_defence_sot",
    # Season-to-date (4)
    "home_std_ppg",
    "away_std_ppg",
    "diff_std_ppg",
    "diff_std_gd",
    # Trailing-10 efficiency (4)
    "home_sot_ratio",
    "away_sot_ratio",
    "home_conversion",
    "away_conversion",
    # History / cold start (4)
    "home_history_depth",
    "away_history_depth",
    "home_is_cold_start",
    "away_is_cold_start",
]

# Features that are exact deterministic reconstructions of other features:
# eight exact linear combinations, plus the two cold-start indicators, which
# are a threshold function of history depth (and identically zero on every
# complete case). Tree models are unaffected and keep the full 35.
DERIVED_FEATURE_COLUMNS = [
    "elo_diff",
    "diff_ewma_ppg",
    "diff_ewma_sot_diff",
    "diff_ewma_gd",
    "home_attack_vs_away_defence_goals",
    "away_attack_vs_home_defence_goals",
    "home_attack_vs_away_defence_sot",
    "diff_std_ppg",
    "home_is_cold_start",
    "away_is_cold_start",
]

# Full-rank subset for linear and neural experiments (Logistic Regression, MLP,
# TabPFN), where perfect multicollinearity leaves coefficients unidentified.
# `diff_std_gd` is retained: home/away season-to-date goal difference are not
# themselves emitted, so it is not reconstructible from the other columns.
LINEAR_SAFE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in DERIVED_FEATURE_COLUMNS]

# Features that may legitimately be NaN (historical information genuinely
# unavailable). Everything else must always be populated.
NULLABLE_FEATURE_COLUMNS = frozenset(
    [c for c in FEATURE_COLUMNS if c.endswith(tuple(f"ewma_{m}" for m in EWMA_METRICS))]
    + [
        "diff_ewma_ppg", "diff_ewma_sot_diff", "diff_ewma_gd",
        "home_attack_vs_away_defence_goals", "away_attack_vs_home_defence_goals",
        "home_attack_vs_away_defence_sot",
        "home_std_ppg", "away_std_ppg", "diff_std_ppg", "diff_std_gd",
        "home_sot_ratio", "away_sot_ratio", "home_conversion", "away_conversion",
    ]
)
NON_NULLABLE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in NULLABLE_FEATURE_COLUMNS]

OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + [RESULT_COLUMN, TARGET_COLUMN]

# Current-match outcome/statistic columns. These are outcomes of the match
# being predicted and may only ever enter as history for *prior* matches.
INADMISSIBLE_COLUMNS = (
    "FTHG", "FTAG", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "Referee",
)

# Deliberately wide: a sanity band, not a fit to today's exact output.
PLAUSIBLE_ELO_MIN = 800.0
PLAUSIBLE_ELO_MAX = 2200.0

NAN = float("nan")


# --------------------------------------------------------------------------
# Elo parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EloParams:
    """Elo configuration.

    `home_advantage`, `season_shrink` and `promoted_prior_delta` are
    data-estimated and must be produced by `estimate_elo_params` from a
    specific training window; `estimated_from_seasons` records that window.
    An empty `estimated_from_seasons` means the values are a-priori
    placeholders, which are deliberately not valid for model evaluation.
    """

    home_advantage: float
    season_shrink: float
    promoted_prior_delta: float
    k_factor: float = DEFAULT_K_FACTOR
    initial_rating: float = DEFAULT_INITIAL_RATING
    estimated_from_seasons: tuple[str, ...] = ()

    @property
    def is_estimated(self) -> bool:
        return len(self.estimated_from_seasons) > 0

    @property
    def training_cutoff(self) -> str | None:
        """The latest season used to estimate these parameters."""
        if not self.estimated_from_seasons:
            return None
        return max(self.estimated_from_seasons, key=_season_sort_key)


def canonical_elo_params() -> EloParams:
    """A-priori placeholder parameters for the canonical artifact.

    Estimated from no data (empty provenance), so the evaluation guards reject
    them.
    """
    return EloParams(
        home_advantage=CANONICAL_HOME_ADVANTAGE,
        season_shrink=CANONICAL_SEASON_SHRINK,
        promoted_prior_delta=CANONICAL_PROMOTED_PRIOR_DELTA,
        estimated_from_seasons=(),
    )


def elo_params_to_dict(params: EloParams) -> dict[str, Any]:
    return {
        "home_advantage": float(params.home_advantage),
        "season_shrink": float(params.season_shrink),
        "promoted_prior_delta": float(params.promoted_prior_delta),
        "k_factor": float(params.k_factor),
        "initial_rating": float(params.initial_rating),
        "estimated_from_seasons": list(params.estimated_from_seasons),
    }


def elo_params_from_dict(payload: dict[str, Any]) -> EloParams:
    try:
        return EloParams(
            home_advantage=float(payload["home_advantage"]),
            season_shrink=float(payload["season_shrink"]),
            promoted_prior_delta=float(payload["promoted_prior_delta"]),
            k_factor=float(payload.get("k_factor", DEFAULT_K_FACTOR)),
            initial_rating=float(payload.get("initial_rating", DEFAULT_INITIAL_RATING)),
            estimated_from_seasons=tuple(payload.get("estimated_from_seasons", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed EloParams payload: {exc}") from exc


def _season_sort_key(season: str) -> tuple[int, int, str]:
    """Order season labels of the form 'YYYY_YY'.

    Labels that do not parse fall into a second bucket ordered by string, so
    the key type stays comparable either way.
    """
    head = str(season).split("_", 1)[0]
    try:
        return (0, int(head), "")
    except ValueError:
        return (1, 0, str(season))


def assert_params_precede(
    elo_params: EloParams, evaluation_seasons: Iterable[str]
) -> None:
    """Raise unless `elo_params` may legitimately be used to evaluate those seasons.

    This is the *temporal* half of the guard: parameters must be estimated
    strictly before the evaluation window. `assert_intended_fold_pairing` adds
    the stricter requirement that the fold pairing is the intended one.
    """
    evaluation = list(evaluation_seasons)
    if not evaluation:
        raise ValueError("evaluation_seasons must not be empty")

    if not elo_params.is_estimated:
        raise ValueError(
            "EloParams carry no estimation provenance (a-priori placeholders). "
            "These are valid for schema inspection, debugging and leakage "
            "audits only, never for model evaluation. Build fold features with "
            "estimate_elo_params(matches, through_season=<fold train cutoff>)."
        )

    latest_estimated = elo_params.training_cutoff
    earliest_evaluated = min(evaluation, key=_season_sort_key)
    if _season_sort_key(latest_estimated) >= _season_sort_key(earliest_evaluated):
        raise ValueError(
            f"Elo parameters were estimated through {latest_estimated!r}, which is "
            f"not strictly earlier than the earliest evaluation season "
            f"{earliest_evaluated!r}. Using them would leak the evaluation window "
            f"into the feature values. Re-estimate with "
            f"through_season < {earliest_evaluated!r}."
        )


def assert_intended_fold_pairing(
    elo_params: EloParams,
    evaluation_seasons: Iterable[str],
    *,
    pairing: dict[str, str] | None = None,
) -> None:
    """Raise unless the parameters are the *intended* ones for these seasons.

    Chronological precedence is necessary but not sufficient: parameters
    estimated through 2021/22 precede 2025/26, yet the sealed test must be
    scored with parameters estimated through 2024/25. This enforces the exact
    approved pairing.
    """
    table = INTENDED_TRAINING_CUTOFF if pairing is None else pairing
    cutoff = elo_params.training_cutoff
    for season in evaluation_seasons:
        required = table.get(season)
        if required is None:
            raise ValueError(
                f"evaluation season {season!r} has no approved training cutoff in "
                f"the rolling-origin scheme (known: {sorted(table)}). Refusing to "
                f"guess which parameters are intended."
            )
        if cutoff != required:
            raise ValueError(
                f"evaluation season {season!r} must be scored with parameters "
                f"estimated through {required!r}, but these were estimated "
                f"through {cutoff!r}. Chronological precedence is not enough — "
                f"rebuild features with through_season={required!r}."
            )


# --------------------------------------------------------------------------
# Causal Elo parameter schedule
# --------------------------------------------------------------------------
def _causal_violations(entries: Mapping[str, EloParams]) -> list[str]:
    """Human-readable list of causal-ordering violations (empty = causal).

    A schedule entry for season S violates causality if any season in its
    `estimated_from_seasons` is not strictly earlier than S, or is the sealed
    season (or later) at all — regardless of which season key it is attached
    to, since the sealed season must never inform ANY parameter estimate, not
    even (hypothetically) its own.
    """
    violations: list[str] = []
    for season, params in entries.items():
        for source_season in params.estimated_from_seasons:
            if _season_sort_key(source_season) >= _season_sort_key(SEALED_SEASON):
                violations.append(
                    f"entry for {season!r} was estimated using {source_season!r}, "
                    f"which is at or after the sealed season {SEALED_SEASON!r}"
                )
            elif _season_sort_key(source_season) >= _season_sort_key(season):
                violations.append(
                    f"entry for {season!r} is not causal: estimated_from_seasons "
                    f"includes {source_season!r}, which is not strictly earlier "
                    f"than {season!r}"
                )
    return violations


@dataclass(frozen=True)
class EloParamSchedule:
    """Immutable season -> EloParams mapping, causal by construction.

    Every entry's EloParams must either be a-priori (empty estimation
    provenance) or estimated ONLY from seasons strictly before the season it
    is attached to. This is what makes a row's Elo features pre-match causal
    at the PARAMETER level, not only at the state-machine level: applying one
    EloParams object across a whole multi-season training window (the "fixed"
    mode `build_features` still accepts for backward-compatible/diagnostic
    use) would let an early row's features depend on a home-advantage /
    season-shrink / promoted-prior estimate computed partly from that row's
    own match result — since all three quantities are estimated from match
    outcomes (FTR), not from unsupervised statistics of the inputs alone.

    Construction validates the invariant immediately and raises ValueError on
    any violation, so a schedule object can never exist in a non-causal state.
    """

    entries: Mapping[str, EloParams]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("EloParamSchedule must contain at least one season")
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

        initial_ratings = {params.initial_rating for params in self.entries.values()}
        if len(initial_ratings) > 1:
            raise ValueError(
                f"initial_rating must be constant across the schedule, "
                f"got {sorted(initial_ratings)}"
            )
        k_factors = {params.k_factor for params in self.entries.values()}
        if len(k_factors) > 1:
            raise ValueError(
                f"k_factor must be constant across the schedule, got {sorted(k_factors)}"
            )

        violations = _causal_violations(self.entries)
        if violations:
            raise ValueError("EloParamSchedule is not causal: " + "; ".join(violations))

    def params_for(self, season: str) -> EloParams:
        """The EloParams to use when generating feature rows for `season`."""
        try:
            return self.entries[season]
        except KeyError:
            raise KeyError(
                f"no Elo parameters scheduled for season {season!r}; "
                f"available: {self.seasons}"
            ) from None

    def is_causal(self) -> bool:
        """True iff every entry is causal (always True for a constructed schedule).

        An explicit, independently-testable check rather than relying solely
        on construction having raised — useful after deserializing a schedule
        from a provenance sidecar that could in principle have been edited.
        """
        return not _causal_violations(self.entries)

    @property
    def seasons(self) -> list[str]:
        return sorted(self.entries, key=_season_sort_key)

    @property
    def initial_rating(self) -> float:
        return next(iter(self.entries.values())).initial_rating

    @property
    def k_factor(self) -> float:
        return next(iter(self.entries.values())).k_factor


def elo_schedule_to_list(schedule: EloParamSchedule) -> list[dict[str, Any]]:
    """Serialize a schedule to the provenance list representation."""
    return [
        {
            "season": season,
            "source": "estimated" if schedule.params_for(season).is_estimated else "apriori",
            "estimated_from_seasons": list(schedule.params_for(season).estimated_from_seasons),
            "params": elo_params_to_dict(schedule.params_for(season)),
        }
        for season in schedule.seasons
    ]


def elo_schedule_from_list(payload: Sequence[dict[str, Any]]) -> EloParamSchedule:
    """Reconstruct a schedule from its provenance list representation.

    Reconstruction re-validates causal ordering via `EloParamSchedule`'s own
    constructor, so a tampered or corrupted schedule in a sidecar cannot
    silently pass as valid.
    """
    entries: dict[str, EloParams] = {}
    for entry in payload:
        try:
            season = entry["season"]
            params_payload = entry["params"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed elo_schedule entry: {exc}") from exc
        entries[season] = elo_params_from_dict(params_payload)
    return EloParamSchedule(entries)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def build_provenance(
    *,
    elo_params: EloParams | EloParamSchedule,
    ewma_halflife: float,
    min_periods: int,
    efficiency_window: int,
    rest_days_cap: int,
    source_path: str,
    source_sha256: str,
    n_rows: int,
    purpose: str,
    valid_for_evaluation: bool,
    earliest_valid_evaluation_season: str | None = None,
    intended_evaluation_seasons: Sequence[str] = (),
    artifact_row_cap_season: str | None = None,
    min_prior_seasons_for_estimation: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Assemble the provenance record for a feature artifact (pure; no I/O).

    Accepts either a single `EloParams` — the diagnostic/backward-compatible
    mode, where the WHOLE artifact was built with one fixed parameter set —
    or an `EloParamSchedule` — the causal production mode, where every
    season's rows were generated using parameters estimated only from
    strictly earlier seasons. Only the schedule case adds an `elo_schedule`
    list and `schedule_is_causal: true`; the single-`EloParams` record shape
    is unchanged from before schedule support existed.
    """
    if isinstance(elo_params, EloParamSchedule):
        schedule = elo_params
        # The backward-compatible single-cutoff fields describe the LAST
        # (highest-season) schedule entry, which is the one governing the
        # artifact's declared row cap / evaluation target.
        governing_season = schedule.seasons[-1]
        governing_params = schedule.params_for(governing_season)
        return {
            "artifact_purpose": purpose,
            "valid_for_model_evaluation": bool(valid_for_evaluation),
            "earliest_valid_evaluation_season": earliest_valid_evaluation_season,
            "intended_evaluation_seasons": list(intended_evaluation_seasons),
            "artifact_row_cap_season": artifact_row_cap_season,
            "training_cutoff": governing_params.training_cutoff,
            "notes": notes,
            "sealed_season": SEALED_SEASON,
            "elo_params": elo_params_to_dict(governing_params),
            "estimated_from_seasons": list(governing_params.estimated_from_seasons),
            "schedule_is_causal": schedule.is_causal(),
            "min_prior_seasons_for_estimation": (
                MIN_PRIOR_SEASONS_FOR_ESTIMATION
                if min_prior_seasons_for_estimation is None
                else min_prior_seasons_for_estimation
            ),
            "elo_schedule": elo_schedule_to_list(schedule),
            "apriori_parameters": {
                "ewma_halflife": float(ewma_halflife),
                "min_periods": int(min_periods),
                "efficiency_window": int(efficiency_window),
                "rest_days_cap": int(rest_days_cap),
                "history_depth_cap": HISTORY_DEPTH_CAP,
                "k_factor": float(schedule.k_factor),
                "initial_rating": float(schedule.initial_rating),
                "elo_recentred_each_season": True,
                "home_advantage": CANONICAL_HOME_ADVANTAGE,
                "season_shrink": CANONICAL_SEASON_SHRINK,
                "promoted_prior_delta": CANONICAL_PROMOTED_PRIOR_DELTA,
            },
            "source": {"path": source_path, "sha256": source_sha256},
            "n_rows": int(n_rows),
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "linear_safe_feature_columns": list(LINEAR_SAFE_FEATURE_COLUMNS),
            "metadata_columns": list(METADATA_COLUMNS),
            "target_column": TARGET_COLUMN,
            "target_mapping": dict(TARGET_MAPPING),
        }

    # ---- Diagnostic / backward-compatible single-EloParams mode. Shape is
    # unchanged from before EloParamSchedule support was added.
    return {
        "artifact_purpose": purpose,
        "valid_for_model_evaluation": bool(valid_for_evaluation),
        "earliest_valid_evaluation_season": earliest_valid_evaluation_season,
        "intended_evaluation_seasons": list(intended_evaluation_seasons),
        "artifact_row_cap_season": artifact_row_cap_season,
        "training_cutoff": elo_params.training_cutoff,
        "notes": notes,
        "sealed_season": SEALED_SEASON,
        "elo_params": elo_params_to_dict(elo_params),
        "estimated_from_seasons": list(elo_params.estimated_from_seasons),
        "schedule_is_causal": None,
        "apriori_parameters": {
            "ewma_halflife": float(ewma_halflife),
            "min_periods": int(min_periods),
            "efficiency_window": int(efficiency_window),
            "rest_days_cap": int(rest_days_cap),
            "history_depth_cap": HISTORY_DEPTH_CAP,
            "k_factor": float(elo_params.k_factor),
            "initial_rating": float(elo_params.initial_rating),
            "elo_recentred_each_season": True,
        },
        "source": {"path": source_path, "sha256": source_sha256},
        "n_rows": int(n_rows),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "linear_safe_feature_columns": list(LINEAR_SAFE_FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target_mapping": dict(TARGET_MAPPING),
    }


def intended_evaluation_seasons_for(training_cutoff: str) -> list[str]:
    """Which evaluation seasons this training cutoff is the approved choice for."""
    return [
        season
        for season, cutoff in INTENDED_TRAINING_CUTOFF.items()
        if cutoff == training_cutoff
    ]


# --------------------------------------------------------------------------
# Artifact validation (reads the provenance sidecar; the only I/O here)
# --------------------------------------------------------------------------
def sidecar_path_for(parquet_path: str | Path) -> Path:
    """Provenance sidecar paired with a feature artifact."""
    return Path(parquet_path).with_suffix(".params.json")


def load_provenance(parquet_path: str | Path) -> dict[str, Any]:
    """Read and shallow-validate an artifact's provenance sidecar."""
    sidecar = sidecar_path_for(parquet_path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no provenance sidecar at {sidecar}. A feature artifact without "
            f"provenance cannot be validated for evaluation; rebuild it with "
            f"scripts/build_features.py."
        )
    try:
        payload = json.loads(sidecar.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"provenance sidecar {sidecar} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"provenance sidecar {sidecar} must contain a JSON object")
    for key in ("elo_params", "valid_for_model_evaluation", "feature_columns", "n_rows"):
        if key not in payload:
            raise ValueError(f"provenance sidecar {sidecar} is missing required key {key!r}")
    return payload


def load_artifact_params(parquet_path: str | Path) -> EloParams:
    """Reconstruct the (single, backward-compatible) EloParams for an artifact.

    For a causal artifact this returns the LAST schedule entry (the one
    governing the artifact's declared row cap), matching `build_provenance`'s
    backward-compatible top-level `elo_params` field. Use `load_elo_schedule`
    to get the full per-season schedule.
    """
    return elo_params_from_dict(load_provenance(parquet_path)["elo_params"])


def load_elo_schedule(parquet_path: str | Path) -> EloParamSchedule:
    """Reconstruct the causal Elo parameter schedule that produced an artifact.

    Only valid for artifacts built in causal-schedule mode; raises ValueError
    if the sidecar has no `elo_schedule` (a diagnostic single-EloParams
    artifact — use `load_artifact_params` for those).
    """
    payload = load_provenance(parquet_path)
    if not payload.get("elo_schedule"):
        raise ValueError(
            f"{parquet_path} has no elo_schedule in its provenance; it was "
            f"built in single-EloParams diagnostic mode. Use "
            f"load_artifact_params() instead."
        )
    return elo_schedule_from_list(payload["elo_schedule"])


def _assert_shared_artifact_checks(
    path: Path,
    payload: dict[str, Any],
    seasons: list[str],
    *,
    expected_source_sha256: str | None,
    check_contents: bool,
) -> pd.DataFrame | None:
    """Schema, source-hash, and row-count/season-presence checks shared by
    both the causal-schedule and single-EloParams validation paths."""
    stored_features = list(payload.get("feature_columns", []))
    if stored_features != FEATURE_COLUMNS:
        raise ValueError(
            f"artifact {path} was built with a different feature schema "
            f"({len(stored_features)} columns) than this module defines "
            f"({len(FEATURE_COLUMNS)}). Rebuild it before evaluating."
        )

    if expected_source_sha256 is not None:
        actual = payload.get("source", {}).get("sha256")
        if actual != expected_source_sha256:
            raise ValueError(
                f"artifact {path} was built from source sha256 {actual!r}, "
                f"expected {expected_source_sha256!r}."
            )

    if not check_contents:
        return None

    if not path.exists():
        raise FileNotFoundError(f"feature artifact {path} does not exist")
    frame = pd.read_parquet(path, columns=["Season"])
    if len(frame) != int(payload["n_rows"]):
        raise ValueError(
            f"artifact {path} holds {len(frame)} rows but its sidecar records "
            f"{payload['n_rows']}; the pair is out of sync."
        )
    present = set(frame["Season"].unique())
    missing = [s for s in seasons if s not in present]
    if missing:
        raise ValueError(
            f"artifact {path} does not contain evaluation season(s) {missing}."
        )
    return frame


def _assert_causal_artifact_valid_for(
    path: Path,
    payload: dict[str, Any],
    seasons: list[str],
    *,
    require_intended_pairing: bool,
    expected_source_sha256: str | None,
    check_contents: bool,
) -> EloParamSchedule:
    """The causal-artifact branch of `assert_artifact_valid_for`.

    Checks, in order: the whole schedule is reconstructable and causal (item
    1 — enforced by `EloParamSchedule` itself, so this covers every season in
    the schedule, not only the ones being evaluated); the artifact is marked
    evaluable; each requested evaluation season has a schedule entry whose
    parameters were estimated strictly before it (item 3) and — unless
    disabled — are the intended pairing (items 3, 6); shared schema/hash/row
    checks (item 7); and the artifact's declared row cap is honoured, with
    development artifacts (cap != sealed season) never containing sealed-season
    rows (items 4, 5).
    """
    schedule = elo_schedule_from_list(payload["elo_schedule"])  # raises if not causal

    if not payload.get("valid_for_model_evaluation", False):
        raise ValueError(
            f"artifact {path} is marked not valid for model evaluation "
            f"({payload.get('artifact_purpose', 'no purpose recorded')})."
        )

    for season in seasons:
        if season not in schedule.entries:
            raise ValueError(
                f"artifact {path}'s Elo schedule has no entry for evaluation "
                f"season {season!r}; available: {schedule.seasons}"
            )
        params = schedule.params_for(season)
        assert_params_precede(params, [season])
        if require_intended_pairing:
            assert_intended_fold_pairing(params, [season])

    frame = _assert_shared_artifact_checks(
        path, payload, seasons,
        expected_source_sha256=expected_source_sha256,
        check_contents=check_contents,
    )

    cap_season = payload.get("artifact_row_cap_season")
    if frame is not None and cap_season is not None:
        present_seasons = set(frame["Season"].unique())
        beyond_cap = [
            s for s in present_seasons if _season_sort_key(s) > _season_sort_key(cap_season)
        ]
        if beyond_cap:
            raise ValueError(
                f"artifact {path} declares a row cap of {cap_season!r} but "
                f"contains rows from season(s) beyond it: {sorted(beyond_cap)}."
            )
        if cap_season != SEALED_SEASON and SEALED_SEASON in present_seasons:
            raise ValueError(
                f"artifact {path} is a development artifact (row cap "
                f"{cap_season!r}) but contains sealed-season "
                f"({SEALED_SEASON!r}) rows."
            )

    return schedule


def assert_artifact_valid_for(
    parquet_path: str | Path,
    evaluation_seasons: Iterable[str],
    *,
    require_intended_pairing: bool = True,
    expected_source_sha256: str | None = None,
    check_contents: bool = True,
) -> EloParams | EloParamSchedule:
    """Assert a feature artifact may be used to evaluate `evaluation_seasons`.

    Intended to be called by the training/experiment entry point before any fit
    or score, so evaluation integrity does not depend on the modeller
    remembering the rules. Raises on every failure mode.

    Dispatches on the artifact's provenance shape: if it has an `elo_schedule`
    (built via `build_causal_elo_schedule`), returns the reconstructed
    `EloParamSchedule` after verifying it is entirely causal, evaluable, and
    correctly paired per season. Otherwise treats it as a diagnostic
    single-`EloParams` artifact (unchanged behavior) and returns that
    `EloParams`.
    """
    seasons = list(evaluation_seasons)
    if not seasons:
        raise ValueError("evaluation_seasons must not be empty")

    path = Path(parquet_path)
    payload = load_provenance(path)

    if payload.get("elo_schedule"):
        return _assert_causal_artifact_valid_for(
            path, payload, seasons,
            require_intended_pairing=require_intended_pairing,
            expected_source_sha256=expected_source_sha256,
            check_contents=check_contents,
        )

    # ---- Single-EloParams (diagnostic/backward-compatible) path, unchanged.
    params = elo_params_from_dict(payload["elo_params"])

    if not payload.get("valid_for_model_evaluation", False):
        raise ValueError(
            f"artifact {path} is marked not valid for model evaluation "
            f"({payload.get('artifact_purpose', 'no purpose recorded')}). "
            f"Build a fold artifact with --estimate-through."
        )

    assert_params_precede(params, seasons)
    if require_intended_pairing:
        assert_intended_fold_pairing(params, seasons)

    _assert_shared_artifact_checks(
        path, payload, seasons,
        expected_source_sha256=expected_source_sha256,
        check_contents=check_contents,
    )

    return params


# --------------------------------------------------------------------------
# Parameter estimation (training window only)
# --------------------------------------------------------------------------
def _season_order(matches: pd.DataFrame) -> list[str]:
    """League season order, derived from each season's first match date."""
    first_dates = matches.groupby("Season")["Date"].min().sort_values()
    return list(first_dates.index)


def _team_season_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Per (Season, Team) mean match score (W=1/D=0.5/L=0) and points per game."""
    home = matches.assign(
        Team=matches["HomeTeam"],
        score=matches["FTR"].map({"H": 1.0, "D": 0.5, "A": 0.0}),
        pts=matches["FTR"].map({"H": 3.0, "D": 1.0, "A": 0.0}),
    )[["Season", "Team", "score", "pts"]]
    away = matches.assign(
        Team=matches["AwayTeam"],
        score=matches["FTR"].map({"A": 1.0, "D": 0.5, "H": 0.0}),
        pts=matches["FTR"].map({"A": 3.0, "D": 1.0, "H": 0.0}),
    )[["Season", "Team", "score", "pts"]]
    stacked = pd.concat([home, away], ignore_index=True)
    return stacked.groupby(["Season", "Team"], as_index=False)[["score", "pts"]].mean()


def _elo_gap_from_score(mean_score: float) -> float:
    """Invert the Elo expectancy formula: expected score -> rating gap."""
    if not (0.0 < mean_score < 1.0):
        raise ValueError(f"mean score {mean_score!r} outside (0, 1); cannot invert")
    return -400.0 * math.log10(1.0 / mean_score - 1.0)


def estimate_elo_params(
    matches: pd.DataFrame,
    through_season: str,
    *,
    k_factor: float = DEFAULT_K_FACTOR,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    expected_start_season: str | None = EXPECTED_DATASET_START_SEASON,
) -> EloParams:
    """Estimate Elo parameters using only seasons up to and including `through_season`.

    Estimates three quantities:

    * `home_advantage` — inverted from the league-wide mean home match score.
    * `promoted_prior_delta` — inverted from the mean match score of teams in
      their first season of a spell, i.e. how far below the field a
      newly-promoted or returning side actually performs.
    * `season_shrink` — the slope of next-season points per game regressed on
      previous-season points per game, for teams present in both.

    The first season of the window is a burn-in (every team looks new), so it
    is excluded from the newcomer and persistence estimates. Because both of
    those estimates are defined relative to the first season present, the frame
    must begin at `expected_start_season`; otherwise incumbents would be
    silently reclassified as newcomers. Pass `expected_start_season=None` to
    disable the check (synthetic fixtures) or a different label if the dataset
    ever extends backward.

    This function requires at least 3 total seasons in its estimation window
    (`through_season` and everything before it) — an explicit internal guard
    below, not merely a consequence of the shrink regression (which is already
    computable from 2 seasons, i.e. one transition). Called directly, this is
    the earliest window it will accept; `build_causal_elo_schedule` layers its
    own, more conservative burn-in policy on top of this floor (see its
    docstring).

    Raises if `through_season` is the sealed season or later: the sealed season
    must never participate in parameter estimation.
    """
    _validate_input_columns(matches)

    if _season_sort_key(through_season) >= _season_sort_key(SEALED_SEASON):
        raise ValueError(
            f"through_season={through_season!r} is the sealed season "
            f"({SEALED_SEASON}) or later. The sealed season must never "
            f"participate in parameter estimation."
        )

    order = _season_order(matches)

    if expected_start_season is not None and order[0] != expected_start_season:
        raise ValueError(
            f"matches frame begins at {order[0]!r} but estimation assumes history "
            f"begins at {expected_start_season!r}. A truncated frame would "
            f"silently reclassify incumbents as newcomers and shift the burn-in "
            f"season. Pass the full history, or set expected_start_season "
            f"explicitly if this is intentional."
        )

    if through_season not in order:
        raise ValueError(
            f"through_season={through_season!r} not present in matches; "
            f"available: {order}"
        )

    used = order[: order.index(through_season) + 1]
    if len(used) < 3:
        raise ValueError(
            f"need at least 3 seasons through {through_season!r} to estimate "
            f"parameters, got {len(used)}"
        )

    window = matches[matches["Season"].isin(used)]

    # Home advantage, from the league-wide mean home score.
    home_score = window["FTR"].map({"H": 1.0, "D": 0.5, "A": 0.0}).mean()
    home_advantage = _elo_gap_from_score(float(home_score))

    # Team-season table, with the burn-in season dropped for the estimates
    # that depend on knowing who is genuinely new.
    table = _team_season_table(window)
    newcomer_flags = []
    previous_teams: set[str] = set()
    for season in used:
        season_teams = set(table.loc[table["Season"] == season, "Team"])
        for team in season_teams:
            newcomer_flags.append(
                {"Season": season, "Team": team, "is_newcomer": team not in previous_teams}
            )
        previous_teams = season_teams
    table = table.merge(pd.DataFrame(newcomer_flags), on=["Season", "Team"], how="left")
    scored = table[table["Season"] != used[0]]

    newcomers = scored[scored["is_newcomer"]]
    if newcomers.empty:
        raise ValueError(
            f"no newcomer team-seasons found through {through_season!r}; "
            f"cannot estimate the promoted/returning prior"
        )
    promoted_prior_delta = _elo_gap_from_score(float(newcomers["score"].mean()))

    # Season-over-season persistence -> shrink factor.
    pivot = scored.pivot_table(index="Team", columns="Season", values="pts")
    pairs: list[pd.DataFrame] = []
    for earlier, later in zip(used, used[1:]):
        if earlier in pivot.columns and later in pivot.columns:
            pair = pivot[[earlier, later]].dropna()
            if not pair.empty:
                pairs.append(pair.rename(columns={earlier: "prev", later: "next"}))
    if not pairs:
        raise ValueError(
            f"no team-season transitions available through {through_season!r}; "
            f"cannot estimate the season shrink factor"
        )
    transitions = pd.concat(pairs, ignore_index=True)
    slope, _intercept = _linear_fit(transitions["prev"].to_numpy(), transitions["next"].to_numpy())

    return EloParams(
        home_advantage=float(home_advantage),
        season_shrink=float(slope),
        promoted_prior_delta=float(promoted_prior_delta),
        k_factor=float(k_factor),
        initial_rating=float(initial_rating),
        estimated_from_seasons=tuple(used),
    )


def _linear_fit(x, y) -> tuple[float, float]:
    """Ordinary least squares slope/intercept, without importing numpy.polyfit."""
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 points for a linear fit")
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx == 0:
        raise ValueError("cannot fit: zero variance in x")
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = sxy / sxx
    return float(slope), float(mean_y - slope * mean_x)


def build_causal_elo_schedule(
    matches: pd.DataFrame,
    *,
    through_season: str | None = None,
    min_prior_seasons: int = MIN_PRIOR_SEASONS_FOR_ESTIMATION,
    apriori: EloParams | None = None,
    k_factor: float = DEFAULT_K_FACTOR,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    expected_start_season: str | None = EXPECTED_DATASET_START_SEASON,
) -> EloParamSchedule:
    """Build a causal, per-season Elo parameter schedule.

    For every season S present in `matches` (up to and including
    `through_season`, or all of them if omitted):

    * if S has fewer than `min_prior_seasons` strictly earlier seasons in the
      full history, S is assigned the a-priori parameters (empty provenance);
    * otherwise S is assigned `estimate_elo_params(matches, through_season=P)`,
      where P is the season immediately before S — i.e. parameters estimated
      ONLY from seasons strictly earlier than S.

    This reuses `estimate_elo_params`'s formulas verbatim; nothing here
    recomputes home advantage, season shrink, or the promoted-prior delta.

    Because every entry's parameters come only from strictly earlier seasons,
    the resulting schedule is fold-independent: the SAME schedule is correct
    for generating training rows for any fold and for the final evaluation
    build, since no entry ever depends on outcomes from a fold's own
    validation/test season or later. In practice this means one canonical
    causal feature matrix can be built once (`through_season=None`) and every
    fold/final artifact is simply that matrix's rows filtered to a row cap.

    On `min_prior_seasons=3` (the V1 default): this is a conservative POLICY
    choice, not a hard mathematical requirement — two prior seasons already
    produce one season-to-season transition, which is technically enough for
    the shrink regression to run. Two things justify the higher threshold
    instead. First, `estimate_elo_params` itself refuses to run on fewer than
    3 total seasons in its estimation window (an explicit guard inside that
    function, independent of this policy), which in this dataset happens to
    make 2017/18 the earliest technically-computable cutoff. Second, that
    earliest technically-computable estimate is visibly unstable: on the real
    dataset, parameters estimated through 2017/18 alone give
    home_advantage=56.6 and season_shrink=0.907, well outside the ~41-45 /
    ~0.76-0.83 range every later cutoff converges to. V1 therefore prefers the
    documented a-priori constants for 2015/16-2017/18 rather than trusting
    that single noisy early estimate — a burn-in policy that happens to
    coincide with, but is conceptually independent of, the estimator's own
    minimum-window floor.

    `expected_start_season` is forwarded to every internal `estimate_elo_params`
    call (see that function's docstring); pass `None` to disable the check for
    synthetic fixtures whose history does not begin at 2015/16.

    Raises if `apriori` carries estimation provenance (a-priori parameters
    must be constants, not derived from data) or if `through_season` is not
    present in `matches`.
    """
    apriori_params = apriori if apriori is not None else canonical_elo_params()
    if apriori_params.is_estimated:
        raise ValueError("apriori params must carry empty estimation provenance")
    if apriori_params.k_factor != k_factor or apriori_params.initial_rating != initial_rating:
        apriori_params = replace(apriori_params, k_factor=k_factor, initial_rating=initial_rating)

    _validate_input_columns(matches)
    order = _season_order(matches)
    if through_season is not None:
        if through_season not in order:
            raise ValueError(
                f"through_season={through_season!r} not present in matches; "
                f"available: {order}"
            )
        order = order[: order.index(through_season) + 1]

    entries: dict[str, EloParams] = {}
    for index, season in enumerate(order):
        prior_seasons = order[:index]
        if len(prior_seasons) < min_prior_seasons:
            entries[season] = apriori_params
        else:
            cutoff = prior_seasons[-1]
            entries[season] = estimate_elo_params(
                matches, cutoff, k_factor=k_factor, initial_rating=initial_rating,
                expected_start_season=expected_start_season,
            )

    return EloParamSchedule(entries)


# --------------------------------------------------------------------------
# Rolling state
# --------------------------------------------------------------------------
class _Ewma:
    """Adjusted EWMA, matching pandas `ewm(halflife=h, adjust=True).mean()`.

    Maintained incrementally:
        num_t = x_t + d * num_{t-1}
        den_t = 1   + d * den_{t-1}
        value = num_t / den_t          with d = 0.5 ** (1 / halflife)
    """

    __slots__ = ("_decay", "_num", "_den", "_count")

    def __init__(self, halflife: float) -> None:
        if halflife <= 0:
            raise ValueError(f"halflife must be positive, got {halflife!r}")
        self._decay = 0.5 ** (1.0 / halflife)
        self._num = 0.0
        self._den = 0.0
        self._count = 0

    def reset(self) -> None:
        self._num = 0.0
        self._den = 0.0
        self._count = 0

    def add(self, value: float) -> None:
        self._num = value + self._decay * self._num
        self._den = 1.0 + self._decay * self._den
        self._count += 1

    def value(self, min_periods: int) -> float:
        if self._count < min_periods or self._den == 0.0:
            return NAN
        return self._num / self._den

    @property
    def count(self) -> int:
        return self._count


@dataclass
class _TeamState:
    """Mutable per-team state carried through the chronological pass."""

    elo: float
    ewma: dict[str, _Ewma]
    recent: deque  # trailing efficiency window of (shots, sot, goals)
    usable_prior: int = 0        # prior matches since the last history reset
    total_prior: int = 0         # all-time prior matches, never reset
    season: str | None = None
    season_matches: int = 0
    season_points: float = 0.0
    season_goal_diff: float = 0.0
    last_date: pd.Timestamp | None = None
    is_returning_season: bool = False

    def reset_history(self) -> None:
        """Drop rolling history (stale return or brand-new team)."""
        for accumulator in self.ewma.values():
            accumulator.reset()
        self.recent.clear()
        self.usable_prior = 0


# --------------------------------------------------------------------------
# Feature building
# --------------------------------------------------------------------------
def _validate_input_columns(matches: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in matches.columns]
    if missing:
        raise ValueError(f"matches frame is missing required columns: {missing}")


def _classify_team_seasons(matches: pd.DataFrame, order: Sequence[str]) -> dict[tuple[str, str], str]:
    """Label each (team, season) as 'new', 'stale' or 'continuing'.

    'stale' means the club was absent from the league for at least one complete
    season before returning — we hold no match history for the gap, so rolling
    history must not be resumed as though the matches were consecutive.
    """
    index_of = {season: i for i, season in enumerate(order)}
    appearances: dict[str, set[int]] = {}
    for season, home, away in zip(matches["Season"], matches["HomeTeam"], matches["AwayTeam"]):
        i = index_of[season]
        for team in (home, away):
            appearances.setdefault(team, set()).add(i)
    classification: dict[tuple[str, str], str] = {}
    for team, season_indices in appearances.items():
        indices = sorted(season_indices)
        for position, i in enumerate(indices):
            if position == 0:
                kind = "new"
            elif i - indices[position - 1] >= 2:
                kind = "stale"
            else:
                kind = "continuing"
            classification[(team, order[i])] = kind
    return classification


def _season_rosters(matches: pd.DataFrame) -> dict[str, set[str]]:
    """Clubs competing in each season (known before a ball is kicked)."""
    rosters: dict[str, set[str]] = {}
    for season, home, away in zip(matches["Season"], matches["HomeTeam"], matches["AwayTeam"]):
        roster = rosters.setdefault(season, set())
        roster.add(home)
        roster.add(away)
    return rosters


def build_features(
    matches: pd.DataFrame,
    elo_params: EloParams | EloParamSchedule,
    *,
    ewma_halflife: float = DEFAULT_EWMA_HALFLIFE,
    min_periods: int = DEFAULT_MIN_PERIODS,
    efficiency_window: int = DEFAULT_EFFICIENCY_WINDOW,
    rest_days_cap: int = DEFAULT_REST_DAYS_CAP,
) -> pd.DataFrame:
    """Build the pre-match feature matrix from a clean match frame.

    `elo_params` accepts either an `EloParamSchedule` — the safe production
    mode, where each season's Elo update/rollover uses parameters estimated
    only from strictly earlier seasons — or a single `EloParams`, applied to
    every season alike (backward-compatible/diagnostic mode; NOT causal at the
    parameter level, since a fold-wide estimate is computed partly from the
    very rows it would be used to generate — see the module docstring). An
    artifact built with a single `EloParams` must never be marked valid for
    model evaluation.

    Every row of `matches` produces exactly one output row — no rows are
    dropped, including cold-start rows, which carry NaN features plus explicit
    indicators. Missing values are never imputed here; imputation belongs to
    the model pipeline, where it can be fitted on training data only.
    """
    _validate_input_columns(matches)
    if matches.empty:
        raise ValueError("matches frame is empty")

    frame = matches.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    if isinstance(elo_params, EloParamSchedule):
        schedule = elo_params

        def resolve_params(season: str) -> EloParams:
            return schedule.params_for(season)

        base_initial_rating = schedule.initial_rating
    else:
        fixed_params = elo_params

        def resolve_params(season: str) -> EloParams:
            return fixed_params

        base_initial_rating = fixed_params.initial_rating

    order = _season_order(frame)
    first_season = order[0]
    season_kind = _classify_team_seasons(frame, order)
    rosters = _season_rosters(frame)
    states: dict[str, _TeamState] = {}
    started_seasons: set[str] = set()

    def team_state(team: str) -> _TeamState:
        state = states.get(team)
        if state is None:
            state = _TeamState(
                # Placeholder only: start_season() unconditionally overwrites
                # this for every team before its state is ever read, since
                # every team belongs to some season's roster and
                # start_season(season) always runs before that season's
                # matches are processed.
                elo=base_initial_rating,
                ewma={metric: _Ewma(ewma_halflife) for metric in EWMA_METRICS},
                recent=deque(maxlen=efficiency_window),
            )
            states[team] = state
        return state

    def start_season(season: str) -> None:
        """Roll the whole league into a new season, then re-centre the scale.

        Runs once per season, before any of that season's rows are emitted.
        Every input is known before a ball is kicked — the season's roster
        (promotion/relegation), each club's prior rating, a backward-looking
        new/stale/continuing classification, and this season's own Elo
        parameters (which, under a causal schedule, were estimated only from
        strictly earlier seasons) — so nothing here depends on a result from
        the season being started.
        """
        if season in started_seasons:
            return
        params = resolve_params(season)
        active = rosters[season]

        for team in sorted(active):  # sorted purely for determinism
            state = team_state(team)
            kind = season_kind[(team, season)]
            if season == first_season:
                # The dataset's opening season: these clubs are incumbents we
                # simply have no prior history for, not promoted sides.
                state.elo = params.initial_rating
                state.reset_history()
                state.is_returning_season = False
            elif kind in ("new", "stale"):
                state.elo = params.initial_rating + params.promoted_prior_delta
                state.reset_history()
                state.is_returning_season = kind == "stale"
            else:
                state.elo = params.initial_rating + params.season_shrink * (
                    state.elo - params.initial_rating
                )
                state.is_returning_season = False
            state.season = season
            state.season_matches = 0
            state.season_points = 0.0
            state.season_goal_diff = 0.0

        # Re-centre the active league on the initial rating. A uniform additive
        # shift: every pairwise difference (including the promoted gap) is
        # preserved exactly, while the absolute scale stays comparable across
        # seasons. Deterministic, and estimated from no data.
        mean_elo = sum(team_state(team).elo for team in active) / len(active)
        shift = params.initial_rating - mean_elo
        if shift != 0.0:
            for team in active:
                team_state(team).elo += shift

        started_seasons.add(season)

    def rest_days(state: _TeamState, date: pd.Timestamp) -> float:
        if state.last_date is None:
            return NAN
        return float(min((date - state.last_date).days, rest_days_cap))

    def season_to_date(state: _TeamState) -> tuple[float, float]:
        if state.season_matches < 1:
            return NAN, NAN
        return (
            state.season_points / state.season_matches,
            state.season_goal_diff / state.season_matches,
        )

    def efficiency(state: _TeamState) -> tuple[float, float]:
        if len(state.recent) < min_periods:
            return NAN, NAN
        shots = sum(item[0] for item in state.recent)
        sot = sum(item[1] for item in state.recent)
        goals = sum(item[2] for item in state.recent)
        sot_ratio = (sot / shots) if shots > 0 else NAN
        conversion = (goals / sot) if sot > 0 else NAN
        return sot_ratio, conversion

    rows: list[dict[str, Any]] = []

    for _date, day in frame.groupby("Date", sort=True):
        pending: list[tuple[_TeamState, _TeamState, Any]] = []

        # ---- Phase 1: emit every feature row for this date from prior state.
        # Rows are read as dicts rather than namedtuples so that a column named
        # "AS" can never be silently renamed by itertuples' identifier mangling.
        for match in day.to_dict("records"):
            start_season(match["Season"])
            home_state = team_state(match["HomeTeam"])
            away_state = team_state(match["AwayTeam"])

            home_ewma = {m: home_state.ewma[m].value(min_periods) for m in EWMA_METRICS}
            away_ewma = {m: away_state.ewma[m].value(min_periods) for m in EWMA_METRICS}
            home_std_ppg, home_std_gd = season_to_date(home_state)
            away_std_ppg, away_std_gd = season_to_date(away_state)
            home_sot_ratio, home_conversion = efficiency(home_state)
            away_sot_ratio, away_conversion = efficiency(away_state)

            rows.append(
                {
                    # Metadata
                    "Season": match["Season"],
                    "Date": match["Date"],
                    "HomeTeam": match["HomeTeam"],
                    "AwayTeam": match["AwayTeam"],
                    "home_matches_played_season": home_state.season_matches,
                    "away_matches_played_season": away_state.season_matches,
                    "home_prior_matches_all_time": home_state.total_prior,
                    "away_prior_matches_all_time": away_state.total_prior,
                    "home_rest_days": rest_days(home_state, match["Date"]),
                    "away_rest_days": rest_days(away_state, match["Date"]),
                    "home_is_returning": home_state.is_returning_season,
                    "away_is_returning": away_state.is_returning_season,
                    # Elo
                    "home_elo": home_state.elo,
                    "away_elo": away_state.elo,
                    "elo_diff": home_state.elo - away_state.elo,
                    # EWMA strength
                    "home_ewma_gf": home_ewma["gf"],
                    "home_ewma_ga": home_ewma["ga"],
                    "home_ewma_sot_for": home_ewma["sot_for"],
                    "home_ewma_sot_against": home_ewma["sot_against"],
                    "home_ewma_shots_for": home_ewma["shots_for"],
                    "home_ewma_shots_against": home_ewma["shots_against"],
                    "home_ewma_ppg": home_ewma["ppg"],
                    "away_ewma_gf": away_ewma["gf"],
                    "away_ewma_ga": away_ewma["ga"],
                    "away_ewma_sot_for": away_ewma["sot_for"],
                    "away_ewma_sot_against": away_ewma["sot_against"],
                    "away_ewma_shots_for": away_ewma["shots_for"],
                    "away_ewma_shots_against": away_ewma["shots_against"],
                    "away_ewma_ppg": away_ewma["ppg"],
                    # Matchup differences
                    "diff_ewma_ppg": home_ewma["ppg"] - away_ewma["ppg"],
                    "diff_ewma_sot_diff": (
                        (home_ewma["sot_for"] - home_ewma["sot_against"])
                        - (away_ewma["sot_for"] - away_ewma["sot_against"])
                    ),
                    "diff_ewma_gd": (
                        (home_ewma["gf"] - home_ewma["ga"])
                        - (away_ewma["gf"] - away_ewma["ga"])
                    ),
                    "home_attack_vs_away_defence_goals": home_ewma["gf"] - away_ewma["ga"],
                    "away_attack_vs_home_defence_goals": away_ewma["gf"] - home_ewma["ga"],
                    "home_attack_vs_away_defence_sot": (
                        home_ewma["sot_for"] - away_ewma["sot_against"]
                    ),
                    # Season-to-date
                    "home_std_ppg": home_std_ppg,
                    "away_std_ppg": away_std_ppg,
                    "diff_std_ppg": home_std_ppg - away_std_ppg,
                    "diff_std_gd": home_std_gd - away_std_gd,
                    # Efficiency
                    "home_sot_ratio": home_sot_ratio,
                    "away_sot_ratio": away_sot_ratio,
                    "home_conversion": home_conversion,
                    "away_conversion": away_conversion,
                    # History / cold start
                    "home_history_depth": float(min(home_state.usable_prior, HISTORY_DEPTH_CAP)),
                    "away_history_depth": float(min(away_state.usable_prior, HISTORY_DEPTH_CAP)),
                    "home_is_cold_start": float(home_state.usable_prior < min_periods),
                    "away_is_cold_start": float(away_state.usable_prior < min_periods),
                    # Target
                    RESULT_COLUMN: match["FTR"],
                    TARGET_COLUMN: TARGET_MAPPING[match["FTR"]],
                }
            )
            pending.append((home_state, away_state, match))

        # ---- Phase 2: only now apply this date's results to state.
        # Elo deltas are computed from pre-date ratings for every fixture
        # before any of them are applied, so the update is independent of the
        # order fixtures happen to appear in within the date. Each match's own
        # Season selects its Elo parameters, so the update itself is also
        # causal under a schedule (never informed by that season's own or any
        # later outcomes).
        elo_updates: list[tuple[_TeamState, _TeamState, float]] = []
        for home_state, away_state, match in pending:
            match_params = resolve_params(match["Season"])
            expected_home = 1.0 / (
                1.0
                + 10.0
                ** (
                    -(home_state.elo + match_params.home_advantage - away_state.elo)
                    / 400.0
                )
            )
            score_home = {"H": 1.0, "D": 0.5, "A": 0.0}[match["FTR"]]
            elo_updates.append(
                (home_state, away_state, match_params.k_factor * (score_home - expected_home))
            )

        for home_state, away_state, delta in elo_updates:
            home_state.elo += delta
            away_state.elo -= delta

        for home_state, away_state, match in pending:
            home_points = {"H": 3.0, "D": 1.0, "A": 0.0}[match["FTR"]]
            away_points = {"A": 3.0, "D": 1.0, "H": 0.0}[match["FTR"]]

            _record_match(
                home_state,
                goals_for=match["FTHG"],
                goals_against=match["FTAG"],
                sot_for=match["HST"],
                sot_against=match["AST"],
                shots_for=match["HS"],
                shots_against=match["AS"],
                points=home_points,
                date=match["Date"],
            )
            _record_match(
                away_state,
                goals_for=match["FTAG"],
                goals_against=match["FTHG"],
                sot_for=match["AST"],
                sot_against=match["HST"],
                shots_for=match["AS"],
                shots_against=match["HS"],
                points=away_points,
                date=match["Date"],
            )

    features = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return _cast_output_dtypes(features)


def _record_match(
    state: _TeamState,
    *,
    goals_for: float,
    goals_against: float,
    sot_for: float,
    sot_against: float,
    shots_for: float,
    shots_against: float,
    points: float,
    date: pd.Timestamp,
) -> None:
    """Fold one completed match into a team's state (called only in phase 2)."""
    state.ewma["gf"].add(float(goals_for))
    state.ewma["ga"].add(float(goals_against))
    state.ewma["sot_for"].add(float(sot_for))
    state.ewma["sot_against"].add(float(sot_against))
    state.ewma["shots_for"].add(float(shots_for))
    state.ewma["shots_against"].add(float(shots_against))
    state.ewma["ppg"].add(float(points))
    state.recent.append((float(shots_for), float(sot_for), float(goals_for)))
    state.usable_prior += 1
    state.total_prior += 1
    state.season_matches += 1
    state.season_points += points
    state.season_goal_diff += float(goals_for) - float(goals_against)
    state.last_date = date


def _cast_output_dtypes(features: pd.DataFrame) -> pd.DataFrame:
    for column in FEATURE_COLUMNS:
        features[column] = features[column].astype("float64")
    for column in (
        "home_matches_played_season",
        "away_matches_played_season",
        "home_prior_matches_all_time",
        "away_prior_matches_all_time",
    ):
        features[column] = features[column].astype("int64")
    for column in ("home_rest_days", "away_rest_days"):
        features[column] = features[column].astype("float64")
    for column in ("home_is_returning", "away_is_returning"):
        features[column] = features[column].astype("bool")
    features[TARGET_COLUMN] = features[TARGET_COLUMN].astype("int64")
    return features


# --------------------------------------------------------------------------
# Output validation
# --------------------------------------------------------------------------
def validate_feature_frame(
    features: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    min_periods: int = DEFAULT_MIN_PERIODS,
    initial_rating: float = DEFAULT_INITIAL_RATING,
) -> list[str]:
    """Return a list of contract violations (empty means the frame is sound).

    Combines static column-contract checks (constants against constants, so
    they catch a bad edit to this module) with data-dependent checks that can
    only fail on an actual bad build.
    """
    problems: list[str] = []

    # ---- Static column contract.
    missing = [c for c in OUTPUT_COLUMNS if c not in features.columns]
    if missing:
        problems.append(f"missing expected columns: {missing}")
    unexpected = [c for c in features.columns if c not in OUTPUT_COLUMNS]
    if unexpected:
        problems.append(f"unexpected columns present: {unexpected}")

    leaked = [c for c in INADMISSIBLE_COLUMNS if c in FEATURE_COLUMNS]
    if leaked:
        problems.append(f"current-match columns present in FEATURE_COLUMNS: {leaked}")

    overlap = set(METADATA_COLUMNS) & set(FEATURE_COLUMNS)
    if overlap:
        problems.append(f"columns claimed as both metadata and feature: {sorted(overlap)}")

    if RESULT_COLUMN in FEATURE_COLUMNS or TARGET_COLUMN in FEATURE_COLUMNS:
        problems.append("target column present in FEATURE_COLUMNS")

    if set(LINEAR_SAFE_FEATURE_COLUMNS) - set(FEATURE_COLUMNS):
        problems.append("LINEAR_SAFE_FEATURE_COLUMNS contains unknown columns")

    if missing:  # every check below needs the columns to exist
        return problems

    # ---- Data-dependent checks.
    if expected_rows is not None and len(features) != expected_rows:
        problems.append(f"expected {expected_rows} rows, found {len(features)}")

    if len(features) == 0:
        problems.append("feature frame is empty")
        return problems

    duplicates = int(features.duplicated(MATCH_KEY_COLUMNS).sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate match keys {MATCH_KEY_COLUMNS}")

    for column in FEATURE_COLUMNS:
        if features[column].dtype != "float64":
            problems.append(f"feature {column!r} has dtype {features[column].dtype}, expected float64")

    numeric = features[FEATURE_COLUMNS]
    infinite = int(numeric.isin([float("inf"), float("-inf")]).to_numpy().sum())
    if infinite:
        problems.append(f"{infinite} infinite value(s) in FEATURE_COLUMNS")

    all_nan_rows = int(numeric.isna().all(axis=1).sum())
    if all_nan_rows:
        problems.append(f"{all_nan_rows} row(s) have every feature NaN")

    unexpected_nan = [
        column
        for column in NON_NULLABLE_FEATURE_COLUMNS
        if int(features[column].isna().sum()) > 0
    ]
    if unexpected_nan:
        problems.append(
            f"NaN present in features that must always be populated: {unexpected_nan}"
        )

    # Target consistency with FTR under the fixed mapping.
    bad_target = set(features[TARGET_COLUMN].unique()) - set(TARGET_MAPPING.values())
    if bad_target:
        problems.append(f"target contains unexpected values: {sorted(bad_target)}")
    expected_target = features[RESULT_COLUMN].map(TARGET_MAPPING)
    if not expected_target.reset_index(drop=True).equals(
        features[TARGET_COLUMN].reset_index(drop=True).astype(expected_target.dtype)
    ):
        problems.append("target column disagrees with FTR under the fixed mapping")

    # Cold-start indicators must agree with history depth.
    for side in ("home", "away"):
        depth = features[f"{side}_history_depth"]
        flag = features[f"{side}_is_cold_start"]
        expected_flag = (depth < min_periods).astype("float64")
        mismatches = int((flag != expected_flag).sum())
        if mismatches:
            problems.append(
                f"{side}_is_cold_start disagrees with {side}_history_depth "
                f"on {mismatches} row(s)"
            )
        if int((depth < 0).sum()) or int((depth > HISTORY_DEPTH_CAP).sum()):
            problems.append(f"{side}_history_depth outside [0, {HISTORY_DEPTH_CAP}]")

    # A cold-start side must not carry EWMA values, and vice versa.
    for side in ("home", "away"):
        cold = features[f"{side}_is_cold_start"] > 0
        populated_when_cold = int((cold & features[f"{side}_ewma_ppg"].notna()).sum())
        if populated_when_cold:
            problems.append(
                f"{populated_when_cold} row(s) have {side}_ewma_ppg populated "
                f"while {side}_is_cold_start is set"
            )

    # Elo plausibility: a wide sanity band, not a fit to today's exact output.
    for column in ("home_elo", "away_elo"):
        values = features[column]
        if float(values.min()) < PLAUSIBLE_ELO_MIN or float(values.max()) > PLAUSIBLE_ELO_MAX:
            problems.append(
                f"{column} outside the plausible band "
                f"[{PLAUSIBLE_ELO_MIN}, {PLAUSIBLE_ELO_MAX}]: "
                f"observed [{values.min():.1f}, {values.max():.1f}]"
            )
    pooled_mean = float(
        pd.concat([features["home_elo"], features["away_elo"]]).mean()
    )
    if abs(pooled_mean - initial_rating) > 100.0:
        problems.append(
            f"pooled mean Elo {pooled_mean:.1f} is more than 100 points from the "
            f"initial rating {initial_rating:.1f}; the rating scale has drifted"
        )

    return problems


def cold_start_row_count(features: pd.DataFrame) -> int:
    """Number of rows where at least one side lacks sufficient usable history."""
    return int(
        ((features["home_is_cold_start"] > 0) | (features["away_is_cold_start"] > 0)).sum()
    )


def active_season_mean_elo(features: pd.DataFrame) -> pd.Series:
    """Mean pre-match Elo of each season's *first* appearance per club.

    Reads the re-centred scale directly: for every season, each club's rating as
    it stood at that club's opening fixture. Useful for confirming the scale is
    stable across seasons.
    """
    home = features[["Season", "Date", "HomeTeam", "home_elo"]].rename(
        columns={"HomeTeam": "Team", "home_elo": "elo"}
    )
    away = features[["Season", "Date", "AwayTeam", "away_elo"]].rename(
        columns={"AwayTeam": "Team", "away_elo": "elo"}
    )
    stacked = pd.concat([home, away], ignore_index=True).sort_values("Date")
    openers = stacked.groupby(["Season", "Team"], as_index=False).first()
    return openers.groupby("Season")["elo"].mean()

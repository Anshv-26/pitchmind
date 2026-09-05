"""Leakage-safe pre-match feature engineering for PitchMind.

Every feature emitted for a match is a function of matches that finished
strictly *before* that match's date — the current match never contributes to
its own feature row.

Three rules drive the design:

1. **Date batching.** Matches are processed grouped by date. All feature rows
   for date D are emitted from state as of the end of dates < D; only then are
   D's results applied to state. Same-date alphabetical order therefore carries
   no temporal information, and one fixture's result can never reach another
   fixture played the same day.

2. **Explicit Elo parameters.** Elo parameters change feature *values*, so a
   feature matrix is only valid for evaluating the fold it was built for.
   Parameters are passed in explicitly and carry provenance; no module-level
   constant holds a data-estimated value. Use `assert_artifact_valid_for`
   before any fit or scoring.

3. **Stable Elo scale.** After each season's rollover the active clubs are
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
# only be scored with an artifact whose Elo parameters were estimated through
# exactly the paired cutoff — chronological precedence alone is not enough, or
# a stale Fold-1 artifact would qualify for the sealed test.
INTENDED_TRAINING_CUTOFF: dict[str, str] = {
    "2022_23": "2021_22",
    "2023_24": "2022_23",
    "2024_25": "2023_24",
    "2025_26": "2024_25",
}

# A-priori placeholders used ONLY for the canonical artifact. These are
# conventional values, not estimates, so they carry empty provenance and the
# resulting artifact is deliberately rejected for model evaluation.
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
    "rebuild features with EloParams produced by estimate_elo_params() for "
    "that fold's training window."
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
# Elo parameters and provenance
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
    strictly before the evaluation window. `assert_artifact_valid_for` adds the
    stricter requirement that the fold pairing is the intended one.
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


def build_provenance(
    *,
    elo_params: EloParams,
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
    notes: str = "",
) -> dict[str, Any]:
    """Assemble the provenance record for a feature artifact (pure; no I/O)."""
    return {
        "artifact_purpose": purpose,
        "valid_for_model_evaluation": bool(valid_for_evaluation),
        "earliest_valid_evaluation_season": earliest_valid_evaluation_season,
        "intended_evaluation_seasons": list(intended_evaluation_seasons),
        "training_cutoff": elo_params.training_cutoff,
        "notes": notes,
        "sealed_season": SEALED_SEASON,
        "elo_params": elo_params_to_dict(elo_params),
        "estimated_from_seasons": list(elo_params.estimated_from_seasons),
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
    """Reconstruct the EloParams that produced a feature artifact."""
    return elo_params_from_dict(load_provenance(parquet_path)["elo_params"])


def assert_artifact_valid_for(
    parquet_path: str | Path,
    evaluation_seasons: Iterable[str],
    *,
    require_intended_pairing: bool = True,
    expected_source_sha256: str | None = None,
    check_contents: bool = True,
) -> EloParams:
    """Assert a feature artifact may be used to evaluate `evaluation_seasons`.

    Intended to be called by the training/experiment entry point before any fit
    or score, so evaluation integrity does not depend on the modeller
    remembering the rules. Raises on every failure mode; returns the artifact's
    EloParams on success.

    Checks, in order: sidecar present and well-formed; artifact not marked
    canonical/not-for-evaluation; parameters carry estimation provenance and
    precede the evaluation window; the fold pairing is the approved one; the
    stored schema matches this module; and the artifact's contents match its
    own sidecar.
    """
    seasons = list(evaluation_seasons)
    if not seasons:
        raise ValueError("evaluation_seasons must not be empty")

    path = Path(parquet_path)
    payload = load_provenance(path)
    params = elo_params_from_dict(payload["elo_params"])

    if not payload.get("valid_for_model_evaluation", False):
        raise ValueError(
            f"artifact {path} is marked not valid for model evaluation "
            f"({payload.get('artifact_purpose', 'no purpose recorded')}). "
            f"Build a fold artifact with --estimate-through."
        )

    # Temporal guard (also rejects a-priori/canonical parameters).
    assert_params_precede(params, seasons)

    if require_intended_pairing:
        assert_intended_fold_pairing(params, seasons)

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

    if check_contents:
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
    elo_params: EloParams,
    *,
    ewma_halflife: float = DEFAULT_EWMA_HALFLIFE,
    min_periods: int = DEFAULT_MIN_PERIODS,
    efficiency_window: int = DEFAULT_EFFICIENCY_WINDOW,
    rest_days_cap: int = DEFAULT_REST_DAYS_CAP,
) -> pd.DataFrame:
    """Build the pre-match feature matrix from a clean match frame.

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
                elo=elo_params.initial_rating + elo_params.promoted_prior_delta,
                ewma={metric: _Ewma(ewma_halflife) for metric in EWMA_METRICS},
                recent=deque(maxlen=efficiency_window),
            )
            states[team] = state
        return state

    def start_season(season: str) -> None:
        """Roll the whole league into a new season, then re-centre the scale.

        Runs once per season, before any of that season's rows are emitted.
        Every input is known before a ball is kicked — the season's roster
        (promotion/relegation), each club's prior rating, and a backward-looking
        new/stale/continuing classification — so nothing here depends on a
        result from the season being started.
        """
        if season in started_seasons:
            return
        active = rosters[season]

        for team in sorted(active):  # sorted purely for determinism
            state = team_state(team)
            kind = season_kind[(team, season)]
            if season == first_season:
                # The dataset's opening season: these clubs are incumbents we
                # simply have no prior history for, not promoted sides.
                state.elo = elo_params.initial_rating
                state.reset_history()
                state.is_returning_season = False
            elif kind in ("new", "stale"):
                state.elo = elo_params.initial_rating + elo_params.promoted_prior_delta
                state.reset_history()
                state.is_returning_season = kind == "stale"
            else:
                state.elo = elo_params.initial_rating + elo_params.season_shrink * (
                    state.elo - elo_params.initial_rating
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
        shift = elo_params.initial_rating - mean_elo
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
        # order fixtures happen to appear in within the date.
        elo_updates: list[tuple[_TeamState, _TeamState, float]] = []
        for home_state, away_state, match in pending:
            expected_home = 1.0 / (
                1.0
                + 10.0
                ** (
                    -(home_state.elo + elo_params.home_advantage - away_state.elo)
                    / 400.0
                )
            )
            score_home = {"H": 1.0, "D": 0.5, "A": 0.0}[match["FTR"]]
            elo_updates.append(
                (home_state, away_state, elo_params.k_factor * (score_home - expected_home))
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

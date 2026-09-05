"""Leakage and correctness tests for the feature-engineering layer.

Synthetic fixtures are tiny so expected values can be worked out by hand. The
real-data tests at the end run the adversarial probes from the Opus audit
against the actual 4,180-row pipeline, so those probes are regression-protected
rather than one-off manual checks.

Tests assert the contract, never whatever the implementation happens to produce.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.feature_engineering import (  # noqa: E402
    DEFAULT_EWMA_HALFLIFE,
    DEFAULT_INITIAL_RATING,
    DEFAULT_MIN_PERIODS,
    DERIVED_FEATURE_COLUMNS,
    EXPECTED_DATASET_START_SEASON,
    FEATURE_COLUMNS,
    INADMISSIBLE_COLUMNS,
    INTENDED_TRAINING_CUTOFF,
    LINEAR_SAFE_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    NON_NULLABLE_FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    RESULT_COLUMN,
    SEALED_SEASON,
    TARGET_COLUMN,
    TARGET_MAPPING,
    EloParams,
    active_season_mean_elo,
    assert_artifact_valid_for,
    assert_intended_fold_pairing,
    assert_params_precede,
    build_features,
    build_provenance,
    canonical_elo_params,
    elo_params_from_dict,
    elo_params_to_dict,
    estimate_elo_params,
    intended_evaluation_seasons_for,
    load_artifact_params,
    load_provenance,
    sidecar_path_for,
    validate_feature_frame,
)

MATCHES_PATH = REPO_ROOT / "data" / "processed" / "matches.parquet"
SYNTHETIC_START = "2019_20"


# --------------------------------------------------------------------------
# Fixture helpers (deliberately not named test_*, so pytest does not collect them)
# --------------------------------------------------------------------------
def make_matches(records: list[dict]) -> pd.DataFrame:
    """Build a clean-match frame; FTR is derived so it always agrees with goals."""
    rows = []
    for record in records:
        fthg, ftag = record["fthg"], record["ftag"]
        ftr = "H" if fthg > ftag else ("A" if fthg < ftag else "D")
        rows.append(
            {
                "Season": record["season"],
                "Date": pd.Timestamp(record["date"]),
                "HomeTeam": record["home"],
                "AwayTeam": record["away"],
                "FTHG": fthg,
                "FTAG": ftag,
                "FTR": ftr,
                "HS": record.get("hs", 10),
                "AS": record.get("as_", 10),
                "HST": record.get("hst", 5),
                "AST": record.get("ast", 5),
            }
        )
    return pd.DataFrame(rows)


def make_params(**overrides) -> EloParams:
    """EloParams carrying provenance, so the evaluation guard lets them through."""
    base = {
        "home_advantage": 0.0,
        "season_shrink": 1.0,
        "promoted_prior_delta": 0.0,
        "k_factor": 20.0,
        "initial_rating": 1500.0,
        "estimated_from_seasons": ("2019_20",),
    }
    base.update(overrides)
    return EloParams(**base)


def estimate_synthetic(matches: pd.DataFrame, through: str) -> EloParams:
    """Estimate on a synthetic league, whose history starts at 2019_20."""
    return estimate_elo_params(matches, through, expected_start_season=SYNTHETIC_START)


def row_on(features: pd.DataFrame, home: str, date: str) -> pd.Series:
    """The single row for a given home team on a given date."""
    match = features[
        (features["HomeTeam"] == home) & (features["Date"] == pd.Timestamp(date))
    ]
    assert len(match) == 1, f"expected one {home} row on {date}, got {len(match)}"
    return match.iloc[0]


def build_synthetic_league() -> pd.DataFrame:
    """Six seasons, four clubs each, exactly one newcomer per season.

    Strength ranks rotate between seasons so points per game vary (the shrink
    regression needs variance in x). Fixtures 0, 5 and 10 of every season are
    forced draws, which deterministically guarantees the newcomer takes at
    least one point per season — its mean match score therefore lies strictly
    inside (0, 1), where the Elo inversion is defined.
    """
    season_teams = {
        "2019_20": ["A", "B", "C", "D"],
        "2020_21": ["A", "B", "C", "E"],
        "2021_22": ["A", "B", "E", "F"],
        "2022_23": ["A", "B", "E", "G"],
        "2023_24": ["A", "B", "E", "H"],
        "2024_25": ["A", "B", "E", "I"],
    }
    records = []
    for season_index, (season, teams) in enumerate(season_teams.items()):
        continuing = teams[:-1]
        shift = season_index % len(continuing)
        rotated = continuing[shift:] + continuing[:shift]
        ranks = {team: position for position, team in enumerate(rotated)}
        ranks[teams[-1]] = len(teams)  # the newcomer is always weakest
        start = pd.Timestamp(f"{season.split('_')[0]}-08-10")

        day = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                if day % 5 == 0:
                    fthg, ftag = 1, 1
                elif ranks[home] < ranks[away]:
                    fthg, ftag = 2, 0
                else:
                    fthg, ftag = 0, 2
                records.append(
                    {
                        "season": season,
                        "date": start + pd.Timedelta(days=7 * day),
                        "home": home,
                        "away": away,
                        "fthg": fthg,
                        "ftag": ftag,
                    }
                )
                day += 1
    return make_matches(records)


# --------------------------------------------------------------------------
# 1. Current-match exclusion
# --------------------------------------------------------------------------
def test_current_match_never_influences_its_own_features():
    records = [
        {"season": "2019_20", "date": "2019-08-10", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-17", "home": "A", "away": "C", "fthg": 2, "ftag": 1},
        {"season": "2019_20", "date": "2019-08-24", "home": "A", "away": "D", "fthg": 3, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-31", "home": "A", "away": "B",
         "fthg": 1, "ftag": 0, "hs": 10, "hst": 5},
    ]
    matches = make_matches(records)
    baseline = build_features(matches, make_params())

    # Rewrite only the final match's own statistics. FTR is untouched, so the
    # sole change is to inputs generated by the match being predicted.
    mutated = matches.copy()
    mutated.loc[3, "HS"] = 99
    mutated.loc[3, "HST"] = 44
    changed = build_features(mutated, make_params())

    before = row_on(baseline, "A", "2019-08-31")[FEATURE_COLUMNS]
    after = row_on(changed, "A", "2019-08-31")[FEATURE_COLUMNS]
    pdt.assert_series_equal(before, after, check_names=False)


def test_current_match_goals_do_not_influence_its_own_features():
    """Stronger than the statistics case: mutate GOALS while holding the result class.

    1-0 and 5-0 are both home wins, so FTR, the Elo update direction and the
    points awarded are unchanged — only the goal counts differ. That match's own
    feature row must be byte-identical.
    """
    records = [
        {"season": "2019_20", "date": "2019-08-10", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-17", "home": "A", "away": "C", "fthg": 2, "ftag": 1},
        {"season": "2019_20", "date": "2019-08-24", "home": "A", "away": "D", "fthg": 3, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-31", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
    ]
    matches = make_matches(records)
    baseline = build_features(matches, make_params())

    mutated = matches.copy()
    mutated.loc[3, "FTHG"] = 5           # 1-0 -> 5-0, still "H"
    assert mutated.loc[3, "FTR"] == "H"

    changed = build_features(mutated, make_params())
    before = row_on(baseline, "A", "2019-08-31")[FEATURE_COLUMNS]
    after = row_on(changed, "A", "2019-08-31")[FEATURE_COLUMNS]
    pdt.assert_series_equal(before, after, check_names=False)


# --------------------------------------------------------------------------
# 2. Shift before aggregate (hand-computed EWMA)
# --------------------------------------------------------------------------
def test_ewma_is_shifted_and_matches_hand_computation():
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-08", "home": "A", "away": "C", "fthg": 2, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-15", "home": "A", "away": "D", "fthg": 3, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-22", "home": "A", "away": "B", "fthg": 9, "ftag": 0},
    ]
    features = build_features(make_matches(records), make_params())
    final = row_on(features, "A", "2019-08-22")

    decay = 0.5 ** (1.0 / DEFAULT_EWMA_HALFLIFE)
    # Adjusted EWMA over the three PRIOR matches only: goals 1, 2, 3.
    expected = (3 + decay * 2 + decay**2 * 1) / (1 + decay + decay**2)
    assert final["home_ewma_gf"] == pytest.approx(expected)

    # The unshifted value would fold in this match's 9 goals, so this
    # assertion genuinely detects a missing shift.
    unshifted = (9 + decay * 3 + decay**2 * 2 + decay**3 * 1) / (
        1 + decay + decay**2 + decay**3
    )
    assert not math.isclose(final["home_ewma_gf"], unshifted, rel_tol=1e-6)


def test_ewma_is_nan_below_min_periods():
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-08", "home": "A", "away": "C", "fthg": 2, "ftag": 0},
    ]
    features = build_features(make_matches(records), make_params())
    first = row_on(features, "A", "2019-08-01")
    second = row_on(features, "A", "2019-08-08")

    assert math.isnan(first["home_ewma_gf"])
    assert math.isnan(second["home_ewma_gf"])  # 1 prior match, min_periods=3
    assert first["home_is_cold_start"] == 1.0
    assert first["home_history_depth"] == 0.0
    assert second["home_history_depth"] == 1.0
    assert DEFAULT_MIN_PERIODS == 3


# --------------------------------------------------------------------------
# 3. Elo pre-match update order
# --------------------------------------------------------------------------
def test_elo_is_read_before_update_and_applied_after():
    params = make_params(promoted_prior_delta=-100.0, home_advantage=0.0, k_factor=20.0)
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-08", "home": "A", "away": "C", "fthg": 0, "ftag": 0},
    ]
    features = build_features(make_matches(records), params)

    # This is the dataset's first season, so all clubs start at the initial
    # rating (they are incumbents we lack history for, not promoted sides), and
    # re-centring is a no-op because the mean is already 1500.
    first = row_on(features, "A", "2019-08-01")
    assert first["home_elo"] == pytest.approx(1500.0)
    assert first["away_elo"] == pytest.approx(1500.0)
    assert first["elo_diff"] == pytest.approx(0.0)

    # Home win at equal ratings with no home advantage: E = 0.5,
    # delta = 20 * (1 - 0.5) = +10, applied only after the first row was emitted.
    second = row_on(features, "A", "2019-08-08")
    assert second["home_elo"] == pytest.approx(1510.0)
    assert second["away_elo"] == pytest.approx(1500.0)  # C has not played yet


# --------------------------------------------------------------------------
# 4. Same-date batching / order independence
# --------------------------------------------------------------------------
def _four_team_history() -> list[dict]:
    """Enough matches that A, B, C and D each have three priors."""
    return [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-02", "home": "C", "away": "D", "fthg": 2, "ftag": 1},
        {"season": "2019_20", "date": "2019-08-08", "home": "B", "away": "C", "fthg": 0, "ftag": 1},
        {"season": "2019_20", "date": "2019-08-09", "home": "D", "away": "A", "fthg": 1, "ftag": 1},
        {"season": "2019_20", "date": "2019-08-15", "home": "A", "away": "C", "fthg": 3, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-16", "home": "B", "away": "D", "fthg": 2, "ftag": 2},
    ]


def test_same_date_fixture_order_does_not_change_features():
    records = _four_team_history() + [
        {"season": "2019_20", "date": "2019-08-22", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-22", "home": "C", "away": "D", "fthg": 0, "ftag": 2},
    ]
    matches = make_matches(records)
    forward = build_features(matches, make_params())
    reversed_input = build_features(matches.iloc[::-1].reset_index(drop=True), make_params())
    pdt.assert_frame_equal(forward, reversed_input)


def test_same_date_match_does_not_see_another_same_date_result():
    """A fixture played the same day must not inform another; a day earlier must."""
    history = _four_team_history()
    heavy_defeat = {
        "season": "2019_20", "home": "A", "away": "B", "fthg": 5, "ftag": 0,
        "hs": 25, "hst": 15, "as_": 2, "ast": 0,
    }
    later_match = {
        "season": "2019_20", "date": "2019-08-22", "home": "B", "away": "D", "fthg": 0, "ftag": 1,
    }

    same_day = make_matches(history + [{**heavy_defeat, "date": "2019-08-22"}, later_match])
    staggered = make_matches(history + [{**heavy_defeat, "date": "2019-08-21"}, later_match])

    params = make_params(k_factor=20.0)
    same_day_row = row_on(build_features(same_day, params), "B", "2019-08-22")
    staggered_row = row_on(build_features(staggered, params), "B", "2019-08-22")

    # Only in the staggered build is B's heavy defeat already known.
    assert staggered_row["home_elo"] < same_day_row["home_elo"]
    assert staggered_row["home_ewma_ga"] > same_day_row["home_ewma_ga"]


# --------------------------------------------------------------------------
# 5. Season reset behaviour
# --------------------------------------------------------------------------
def test_season_to_date_is_nan_on_first_match_of_season():
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-08", "home": "A", "away": "B", "fthg": 2, "ftag": 0},
        {"season": "2020_21", "date": "2020-08-01", "home": "A", "away": "B", "fthg": 0, "ftag": 1},
    ]
    features = build_features(make_matches(records), make_params())
    first = row_on(features, "A", "2019-08-01")
    second = row_on(features, "A", "2019-08-08")
    new_season = row_on(features, "A", "2020-08-01")

    assert math.isnan(first["home_std_ppg"])
    assert second["home_std_ppg"] == pytest.approx(3.0)   # one prior win
    assert math.isnan(new_season["home_std_ppg"])         # reset at the boundary
    assert new_season["home_matches_played_season"] == 0
    # All-time counters do not reset, and neither does EWMA for a continuing club.
    assert new_season["home_prior_matches_all_time"] == 2


def test_elo_season_shrink_applies_at_the_boundary():
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2020_21", "date": "2020-08-01", "home": "A", "away": "B", "fthg": 0, "ftag": 0},
    ]
    matches = make_matches(records)

    # Both clubs continue, and the two-club league is symmetric about 1500, so
    # re-centring is a no-op here and the shrink is visible directly.
    carried = build_features(matches, make_params(season_shrink=1.0, k_factor=20.0))
    assert row_on(carried, "A", "2020-08-01")["home_elo"] == pytest.approx(1510.0)

    reverted = build_features(matches, make_params(season_shrink=0.0, k_factor=20.0))
    assert row_on(reverted, "A", "2020-08-01")["home_elo"] == pytest.approx(1500.0)


# --------------------------------------------------------------------------
# 6. Stale-history reset
# --------------------------------------------------------------------------
def _stale_fixture() -> pd.DataFrame:
    """X plays 2019/20 and 2021/22 (absent for all of 2020/21); Y plays all three."""
    records = [
        {"season": "2019_20", "date": date, "home": "X", "away": "Y", "fthg": 3, "ftag": 0}
        for date in ["2019-08-01", "2019-08-08", "2019-08-15", "2019-08-22"]
    ]
    records += [
        {"season": "2020_21", "date": date, "home": "Y", "away": "Z", "fthg": 1, "ftag": 0}
        for date in ["2020-08-01", "2020-08-08", "2020-08-15", "2020-08-22"]
    ]
    records.append(
        {"season": "2021_22", "date": "2021-08-01", "home": "X", "away": "Y", "fthg": 0, "ftag": 0}
    )
    return make_matches(records)


def test_stale_history_resets_after_a_full_season_absent():
    params = make_params(promoted_prior_delta=-100.0, season_shrink=1.0)
    features = build_features(_stale_fixture(), params)
    returning = row_on(features, "X", "2021-08-01")

    # X was absent for a complete season: rolling history is dropped, because we
    # hold no match record for the gap.
    assert math.isnan(returning["home_ewma_gf"])
    assert math.isnan(returning["home_sot_ratio"])
    assert returning["home_history_depth"] == 0.0
    assert returning["home_is_cold_start"] == 1.0
    assert bool(returning["home_is_returning"]) is True
    # The all-time counter still records the earlier spell.
    assert returning["home_prior_matches_all_time"] == 4

    # Re-centring makes the absolute rating depend on the rest of the league, so
    # assert the invariants it must preserve: the returning club sits below the
    # incumbent, and the active league averages exactly the initial rating.
    assert returning["home_elo"] < returning["away_elo"]
    assert (returning["home_elo"] + returning["away_elo"]) / 2 == pytest.approx(1500.0)


def test_adjacent_season_return_does_not_reset_history():
    params = make_params(promoted_prior_delta=-100.0, season_shrink=1.0)
    features = build_features(_stale_fixture(), params)
    returning = row_on(features, "X", "2021-08-01")

    # Y played continuously, so its history carries across both boundaries.
    assert not math.isnan(returning["away_ewma_gf"])
    assert returning["away_history_depth"] == 8.0
    assert returning["away_is_cold_start"] == 0.0
    assert bool(returning["away_is_returning"]) is False


# --------------------------------------------------------------------------
# 7. Cold-start rows are retained
# --------------------------------------------------------------------------
def test_no_rows_are_dropped_and_cold_start_rows_carry_indicators():
    matches = build_synthetic_league()
    features = build_features(matches, make_params())
    assert len(features) == len(matches)

    cold = features[(features["home_is_cold_start"] > 0) | (features["away_is_cold_start"] > 0)]
    assert len(cold) > 0, "fixture should contain cold-start rows"
    for _, row in cold.iterrows():
        if row["home_is_cold_start"] > 0:
            assert math.isnan(row["home_ewma_ppg"])
        if row["away_is_cold_start"] > 0:
            assert math.isnan(row["away_ewma_ppg"])

    assert cold["home_elo"].notna().all()
    assert cold["away_elo"].notna().all()
    assert set(cold[TARGET_COLUMN]).issubset(set(TARGET_MAPPING.values()))


# --------------------------------------------------------------------------
# 8. Rest days
# --------------------------------------------------------------------------
def test_rest_days_first_appearance_gap_and_cap():
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-06", "home": "A", "away": "C", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-10-01", "home": "A", "away": "D", "fthg": 1, "ftag": 0},
    ]
    features = build_features(make_matches(records), make_params())

    assert math.isnan(row_on(features, "A", "2019-08-01")["home_rest_days"])  # first ever
    assert row_on(features, "A", "2019-08-06")["home_rest_days"] == pytest.approx(5.0)
    # Real gap is 56 days; capped at 21.
    assert row_on(features, "A", "2019-10-01")["home_rest_days"] == pytest.approx(21.0)


# --------------------------------------------------------------------------
# 9. Safe ratios
# --------------------------------------------------------------------------
def test_efficiency_uses_sum_over_sum_not_mean_of_ratios():
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B",
         "fthg": 1, "ftag": 0, "hs": 10, "hst": 5},
        {"season": "2019_20", "date": "2019-08-08", "home": "A", "away": "C",
         "fthg": 0, "ftag": 1, "hs": 1, "hst": 1},
        {"season": "2019_20", "date": "2019-08-15", "home": "A", "away": "D",
         "fthg": 2, "ftag": 0, "hs": 10, "hst": 4},
        {"season": "2019_20", "date": "2019-08-22", "home": "A", "away": "B",
         "fthg": 0, "ftag": 0, "hs": 7, "hst": 3},
    ]
    features = build_features(make_matches(records), make_params())
    final = row_on(features, "A", "2019-08-22")

    assert final["home_sot_ratio"] == pytest.approx(10 / 21)   # (5+1+4) / (10+1+10)
    assert final["home_conversion"] == pytest.approx(3 / 10)   # (1+0+2) / (5+1+4)

    mean_of_ratios = (5 / 10 + 1 / 1 + 4 / 10) / 3
    assert not math.isclose(final["home_sot_ratio"], mean_of_ratios, rel_tol=1e-6)


def test_zero_denominator_yields_nan_not_zero_or_inf():
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B",
         "fthg": 0, "ftag": 1, "hs": 6, "hst": 0},
        {"season": "2019_20", "date": "2019-08-08", "home": "A", "away": "C",
         "fthg": 0, "ftag": 1, "hs": 4, "hst": 0},
        {"season": "2019_20", "date": "2019-08-15", "home": "A", "away": "D",
         "fthg": 0, "ftag": 1, "hs": 5, "hst": 0},
        {"season": "2019_20", "date": "2019-08-22", "home": "A", "away": "B",
         "fthg": 1, "ftag": 0, "hs": 8, "hst": 3},
    ]
    features = build_features(make_matches(records), make_params())
    final = row_on(features, "A", "2019-08-22")

    assert math.isnan(final["home_conversion"])            # 0 shots on target
    assert not math.isinf(final["home_conversion"])
    assert final["home_sot_ratio"] == pytest.approx(0.0)   # 0 / 15 is well defined


def test_efficiency_is_nan_below_min_periods():
    records = [
        {"season": "2019_20", "date": "2019-08-01", "home": "A", "away": "B", "fthg": 1, "ftag": 0},
        {"season": "2019_20", "date": "2019-08-08", "home": "A", "away": "C", "fthg": 1, "ftag": 0},
    ]
    features = build_features(make_matches(records), make_params())
    second = row_on(features, "A", "2019-08-08")
    assert math.isnan(second["home_sot_ratio"])
    assert math.isnan(second["home_conversion"])


# --------------------------------------------------------------------------
# 10. Column contract
# --------------------------------------------------------------------------
def test_feature_column_contract():
    assert len(FEATURE_COLUMNS) == 35
    assert len(set(FEATURE_COLUMNS)) == 35

    for column in INADMISSIBLE_COLUMNS:
        assert column not in FEATURE_COLUMNS, f"{column} is a current-match outcome"

    assert not set(METADATA_COLUMNS) & set(FEATURE_COLUMNS)
    assert RESULT_COLUMN not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert TARGET_MAPPING == {"H": 0, "D": 1, "A": 2}


def test_built_frame_satisfies_validation():
    features = build_features(build_synthetic_league(), make_params())
    assert validate_feature_frame(features, expected_rows=len(features)) == []
    assert list(features.columns) == OUTPUT_COLUMNS
    for column in FEATURE_COLUMNS:
        assert features[column].dtype == "float64"


def test_target_encoding_matches_ftr():
    features = build_features(build_synthetic_league(), make_params())
    expected = features[RESULT_COLUMN].map(TARGET_MAPPING)
    assert features[TARGET_COLUMN].tolist() == expected.tolist()


# --------------------------------------------------------------------------
# 11. Sealed-season guard
# --------------------------------------------------------------------------
def test_estimate_rejects_the_sealed_season():
    with pytest.raises(ValueError, match="sealed"):
        estimate_synthetic(build_synthetic_league(), SEALED_SEASON)


def test_estimate_rejects_seasons_after_the_sealed_season():
    with pytest.raises(ValueError, match="sealed"):
        estimate_synthetic(build_synthetic_league(), "2026_27")


def test_params_estimated_through_dev_data_may_evaluate_the_sealed_season():
    params = estimate_synthetic(build_synthetic_league(), "2024_25")
    assert SEALED_SEASON not in params.estimated_from_seasons
    assert_params_precede(params, [SEALED_SEASON])  # must not raise


# --------------------------------------------------------------------------
# 12. Fold-isolation guard
# --------------------------------------------------------------------------
def test_guard_rejects_parameters_estimated_into_the_evaluation_window():
    late = estimate_synthetic(build_synthetic_league(), "2023_24")
    with pytest.raises(ValueError, match="not strictly earlier"):
        assert_params_precede(late, ["2022_23"])


def test_guard_accepts_correctly_scoped_parameters():
    fold_one = estimate_synthetic(build_synthetic_league(), "2021_22")
    assert_params_precede(fold_one, ["2022_23"])  # must not raise


def test_guard_rejects_parameters_estimated_in_the_evaluation_season_itself():
    params = estimate_synthetic(build_synthetic_league(), "2022_23")
    with pytest.raises(ValueError, match="not strictly earlier"):
        assert_params_precede(params, ["2022_23"])


def test_guard_rejects_the_canonical_apriori_artifact():
    params = canonical_elo_params()
    assert params.estimated_from_seasons == ()
    assert params.is_estimated is False
    with pytest.raises(ValueError, match="no estimation provenance"):
        assert_params_precede(params, ["2022_23"])


def test_estimated_params_record_their_training_window():
    params = estimate_synthetic(build_synthetic_league(), "2021_22")
    assert params.estimated_from_seasons == ("2019_20", "2020_21", "2021_22")
    assert params.training_cutoff == "2021_22"
    assert params.is_estimated is True


# --------------------------------------------------------------------------
# 12b. Intended fold pairing (precedence alone is not enough)
# --------------------------------------------------------------------------
def test_intended_pairing_table_matches_the_approved_scheme():
    assert INTENDED_TRAINING_CUTOFF == {
        "2022_23": "2021_22",
        "2023_24": "2022_23",
        "2024_25": "2023_24",
        "2025_26": "2024_25",
    }
    assert intended_evaluation_seasons_for("2024_25") == ["2025_26"]


def test_fold_one_params_are_rejected_for_the_sealed_test_despite_preceding_it():
    """The precise gap the audit flagged: old enough, but not the intended pair."""
    fold_one = EloParams(
        home_advantage=41.7, season_shrink=0.832, promoted_prior_delta=-103.5,
        estimated_from_seasons=("2015_16", "2021_22"),
    )
    # Chronological precedence is satisfied...
    assert_params_precede(fold_one, ["2025_26"])
    # ...but the intended pairing is not.
    with pytest.raises(ValueError, match="must be scored with parameters"):
        assert_intended_fold_pairing(fold_one, ["2025_26"])


def test_intended_pairing_accepts_the_approved_combination():
    final = EloParams(
        home_advantage=43.0, season_shrink=0.758, promoted_prior_delta=-122.3,
        estimated_from_seasons=("2015_16", "2024_25"),
    )
    assert_intended_fold_pairing(final, ["2025_26"])  # must not raise


def test_intended_pairing_rejects_an_unknown_evaluation_season():
    params = EloParams(
        home_advantage=43.0, season_shrink=0.8, promoted_prior_delta=-100.0,
        estimated_from_seasons=("2015_16", "2019_20"),
    )
    with pytest.raises(ValueError, match="no approved training cutoff"):
        assert_intended_fold_pairing(params, ["2020_21"])


# --------------------------------------------------------------------------
# 13. Parameter sensitivity
# --------------------------------------------------------------------------
def test_features_actually_depend_on_elo_parameters():
    """If parameters did not move feature values, the fold guard would be a no-op."""
    matches = build_synthetic_league()
    a = build_features(matches, make_params(home_advantage=0.0, promoted_prior_delta=0.0))
    b = build_features(matches, make_params(home_advantage=80.0, promoted_prior_delta=-150.0))
    assert not a["home_elo"].equals(b["home_elo"])
    assert not a["elo_diff"].equals(b["elo_diff"])

    c = build_features(matches, make_params(season_shrink=0.0))
    d = build_features(matches, make_params(season_shrink=1.0))
    assert not c["home_elo"].equals(d["home_elo"])


def test_different_fold_params_produce_different_features():
    matches = build_synthetic_league()
    fold_one = build_features(matches, estimate_synthetic(matches, "2021_22"))
    fold_three = build_features(matches, estimate_synthetic(matches, "2023_24"))
    assert not fold_one["home_elo"].equals(fold_three["home_elo"])


# --------------------------------------------------------------------------
# 14. Provenance round-trip
# --------------------------------------------------------------------------
def test_elo_params_round_trip_through_json():
    original = estimate_synthetic(build_synthetic_league(), "2021_22")
    restored = elo_params_from_dict(json.loads(json.dumps(elo_params_to_dict(original))))
    assert restored == original


def test_guard_still_bites_after_a_provenance_round_trip():
    original = estimate_synthetic(build_synthetic_league(), "2023_24")
    restored = elo_params_from_dict(json.loads(json.dumps(elo_params_to_dict(original))))
    with pytest.raises(ValueError, match="not strictly earlier"):
        assert_params_precede(restored, ["2022_23"])


def test_provenance_record_marks_the_canonical_artifact_as_not_evaluable():
    provenance = build_provenance(
        elo_params=canonical_elo_params(),
        ewma_halflife=DEFAULT_EWMA_HALFLIFE,
        min_periods=DEFAULT_MIN_PERIODS,
        efficiency_window=10,
        rest_days_cap=21,
        source_path="data/processed/matches.parquet",
        source_sha256="abc123",
        n_rows=4180,
        purpose="canonical",
        valid_for_evaluation=False,
        notes="not valid for evaluation",
    )
    payload = json.loads(json.dumps(provenance))

    assert payload["valid_for_model_evaluation"] is False
    assert payload["estimated_from_seasons"] == []
    assert payload["training_cutoff"] is None
    assert payload["n_features"] == 35
    assert payload["target_mapping"] == {"H": 0, "D": 1, "A": 2}
    assert payload["sealed_season"] == SEALED_SEASON
    assert elo_params_from_dict(payload["elo_params"]) == canonical_elo_params()


# --------------------------------------------------------------------------
# 15. Artifact validation API (reads a real sidecar from disk)
# --------------------------------------------------------------------------
def _write_artifact(tmp_path: Path, elo_params: EloParams, *, valid: bool, name="features.parquet") -> Path:
    matches = build_synthetic_league()
    features = build_features(matches, elo_params)
    path = tmp_path / name
    features.to_parquet(path, index=False)
    provenance = build_provenance(
        elo_params=elo_params,
        ewma_halflife=DEFAULT_EWMA_HALFLIFE,
        min_periods=DEFAULT_MIN_PERIODS,
        efficiency_window=10,
        rest_days_cap=21,
        source_path="synthetic",
        source_sha256="deadbeef",
        n_rows=len(features),
        purpose="test artifact",
        valid_for_evaluation=valid,
        intended_evaluation_seasons=intended_evaluation_seasons_for(
            elo_params.training_cutoff or ""
        ),
        notes="",
    )
    sidecar_path_for(path).write_text(json.dumps(provenance, indent=2))
    return path


def test_load_artifact_params_round_trips_from_disk(tmp_path):
    params = EloParams(
        home_advantage=41.7, season_shrink=0.832, promoted_prior_delta=-103.5,
        estimated_from_seasons=("2019_20", "2021_22"),
    )
    path = _write_artifact(tmp_path, params, valid=True)
    assert load_artifact_params(path) == params
    assert load_provenance(path)["training_cutoff"] == "2021_22"


def test_missing_sidecar_fails_loudly(tmp_path):
    params = EloParams(1.0, 0.8, -100.0, estimated_from_seasons=("2019_20",))
    path = _write_artifact(tmp_path, params, valid=True)
    sidecar_path_for(path).unlink()
    with pytest.raises(FileNotFoundError, match="no provenance sidecar"):
        assert_artifact_valid_for(path, ["2022_23"])


def test_malformed_sidecar_fails_loudly(tmp_path):
    params = EloParams(1.0, 0.8, -100.0, estimated_from_seasons=("2019_20",))
    path = _write_artifact(tmp_path, params, valid=True)
    sidecar_path_for(path).write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        assert_artifact_valid_for(path, ["2022_23"])


def test_sidecar_missing_required_key_fails_loudly(tmp_path):
    params = EloParams(1.0, 0.8, -100.0, estimated_from_seasons=("2019_20",))
    path = _write_artifact(tmp_path, params, valid=True)
    payload = json.loads(sidecar_path_for(path).read_text())
    del payload["elo_params"]
    sidecar_path_for(path).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing required key"):
        assert_artifact_valid_for(path, ["2022_23"])


def test_row_count_mismatch_between_artifact_and_sidecar_fails(tmp_path):
    params = EloParams(1.0, 0.8, -100.0, estimated_from_seasons=("2019_20", "2021_22"))
    path = _write_artifact(tmp_path, params, valid=True)
    payload = json.loads(sidecar_path_for(path).read_text())
    payload["n_rows"] = 999999
    sidecar_path_for(path).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="out of sync"):
        assert_artifact_valid_for(path, ["2022_23"])


def test_canonical_artifact_is_rejected_for_evaluation(tmp_path):
    path = _write_artifact(tmp_path, canonical_elo_params(), valid=False)
    with pytest.raises(ValueError, match="not valid for model evaluation"):
        assert_artifact_valid_for(path, ["2022_23"])


def test_artifact_validation_enforces_the_intended_pairing(tmp_path):
    fold_one = EloParams(
        home_advantage=41.7, season_shrink=0.832, promoted_prior_delta=-103.5,
        estimated_from_seasons=("2019_20", "2021_22"),
    )
    path = _write_artifact(tmp_path, fold_one, valid=True)
    # Correct pairing passes.
    assert_artifact_valid_for(path, ["2022_23"])
    # Chronologically fine for 2024_25, but not the intended artifact.
    with pytest.raises(ValueError, match="must be scored with parameters"):
        assert_artifact_valid_for(path, ["2024_25"])
    # Opting out of pairing leaves only the temporal check, which passes.
    assert_artifact_valid_for(path, ["2024_25"], require_intended_pairing=False)


def test_artifact_validation_rejects_absent_evaluation_season(tmp_path):
    fold_one = EloParams(
        home_advantage=41.7, season_shrink=0.832, promoted_prior_delta=-103.5,
        estimated_from_seasons=("2019_20", "2021_22"),
    )
    path = _write_artifact(tmp_path, fold_one, valid=True)
    payload = json.loads(sidecar_path_for(path).read_text())
    sidecar_path_for(path).write_text(json.dumps(payload))
    # The synthetic league has no 2025_26 rows.
    with pytest.raises(ValueError, match="does not contain evaluation season"):
        assert_artifact_valid_for(
            path, ["2025_26"], require_intended_pairing=False
        )


def test_artifact_validation_detects_source_hash_mismatch(tmp_path):
    fold_one = EloParams(
        home_advantage=41.7, season_shrink=0.832, promoted_prior_delta=-103.5,
        estimated_from_seasons=("2019_20", "2021_22"),
    )
    path = _write_artifact(tmp_path, fold_one, valid=True)
    with pytest.raises(ValueError, match="source sha256"):
        assert_artifact_valid_for(path, ["2022_23"], expected_source_sha256="other")


# --------------------------------------------------------------------------
# 16. Estimator input guard
# --------------------------------------------------------------------------
def test_estimate_rejects_a_frame_that_does_not_start_at_the_expected_season():
    matches = build_synthetic_league()
    with pytest.raises(ValueError, match="begins at"):
        estimate_elo_params(matches, "2021_22")  # default expects 2015_16


def test_estimate_rejects_a_truncated_frame_missing_early_seasons():
    matches = build_synthetic_league()
    truncated = matches[matches["Season"] != "2019_20"]
    with pytest.raises(ValueError, match="begins at"):
        estimate_elo_params(truncated, "2022_23", expected_start_season=SYNTHETIC_START)


def test_expected_start_season_constant_is_centralised():
    assert EXPECTED_DATASET_START_SEASON == "2015_16"


# --------------------------------------------------------------------------
# 17. Elo scale: first-season initialisation and per-season re-centring
# --------------------------------------------------------------------------
def test_first_season_clubs_start_at_the_initial_rating_not_the_promoted_prior():
    """The dataset's opening season holds incumbents, not 20 promoted sides."""
    params = make_params(promoted_prior_delta=-100.0)
    features = build_features(build_synthetic_league(), params)
    first_season = features[features["Season"] == "2019_20"]
    opening_date = first_season["Date"].min()
    openers = first_season[first_season["Date"] == opening_date]
    for _, row in openers.iterrows():
        assert row["home_elo"] == pytest.approx(1500.0)
        assert row["away_elo"] == pytest.approx(1500.0)


def test_every_season_is_recentred_on_the_initial_rating():
    params = make_params(promoted_prior_delta=-150.0, season_shrink=0.75)
    features = build_features(build_synthetic_league(), params)
    means = active_season_mean_elo(features)
    assert len(means) == 6
    for season, mean_elo in means.items():
        assert mean_elo == pytest.approx(1500.0), f"{season} not re-centred"


def _season_opening_elos(features: pd.DataFrame, season: str) -> dict[str, float]:
    """Each club's rating at its own first fixture of a season.

    A club's rating only changes when it plays, so this is exactly the
    post-rollover, post-re-centring value.
    """
    rows = features[features["Season"] == season]
    home = rows[["Date", "HomeTeam", "home_elo"]].rename(
        columns={"HomeTeam": "Team", "home_elo": "elo"}
    )
    away = rows[["Date", "AwayTeam", "away_elo"]].rename(
        columns={"AwayTeam": "Team", "away_elo": "elo"}
    )
    stacked = pd.concat([home, away], ignore_index=True).sort_values("Date")
    return stacked.groupby("Team")["elo"].first().to_dict()


def test_elo_differences_are_invariant_to_the_initial_rating_level():
    """Re-centring makes the scale purely relative: only the level should move.

    Every rating is `initial_rating + something independent of it`, and the Elo
    expectancy depends only on differences — so shifting the initial rating by
    500 must shift every rating by exactly 500 and leave elo_diff untouched.
    """
    matches = build_synthetic_league()
    common = {"promoted_prior_delta": -100.0, "season_shrink": 0.8}
    a = build_features(matches, make_params(initial_rating=1500.0, **common))
    b = build_features(matches, make_params(initial_rating=2000.0, **common))

    pdt.assert_series_equal(a["elo_diff"], b["elo_diff"])
    assert np.allclose(b["home_elo"] - a["home_elo"], 500.0)
    assert np.allclose(b["away_elo"] - a["away_elo"], 500.0)


def test_promoted_clubs_sit_below_incumbents_after_recentring():
    params = make_params(promoted_prior_delta=-150.0, season_shrink=0.8)
    features = build_features(build_synthetic_league(), params)

    # "I" joins in 2024_25 and is that season's only newcomer.
    openers = _season_opening_elos(features, "2024_25")
    assert set(openers) == {"A", "B", "E", "I"}
    assert openers["I"] < min(openers[team] for team in ("A", "B", "E"))
    # And the gap is the promoted prior relative to the re-centred league.
    assert openers["I"] < DEFAULT_INITIAL_RATING


def test_recentring_does_not_break_same_date_order_independence():
    matches = build_synthetic_league()
    params = make_params(promoted_prior_delta=-120.0, season_shrink=0.8)
    forward = build_features(matches, params)
    shuffled = build_features(matches.sample(frac=1.0, random_state=3).reset_index(drop=True), params)
    pdt.assert_frame_equal(forward, shuffled)


# --------------------------------------------------------------------------
# 18. Linear-safe feature subset
# --------------------------------------------------------------------------
def test_linear_safe_subset_excludes_every_derived_column():
    assert set(LINEAR_SAFE_FEATURE_COLUMNS).issubset(set(FEATURE_COLUMNS))
    assert not set(LINEAR_SAFE_FEATURE_COLUMNS) & set(DERIVED_FEATURE_COLUMNS)
    assert len(LINEAR_SAFE_FEATURE_COLUMNS) == len(FEATURE_COLUMNS) - len(DERIVED_FEATURE_COLUMNS)
    # The full set is retained for tree models.
    assert len(FEATURE_COLUMNS) == 35


def test_derived_columns_really_are_reconstructible():
    """Justifies excluding them: each is an exact function of retained columns."""
    f = build_features(build_synthetic_league(), make_params())
    recon = {
        "elo_diff": f.home_elo - f.away_elo,
        "diff_ewma_ppg": f.home_ewma_ppg - f.away_ewma_ppg,
        "diff_ewma_gd": (f.home_ewma_gf - f.home_ewma_ga) - (f.away_ewma_gf - f.away_ewma_ga),
        "diff_ewma_sot_diff": (
            (f.home_ewma_sot_for - f.home_ewma_sot_against)
            - (f.away_ewma_sot_for - f.away_ewma_sot_against)
        ),
        "home_attack_vs_away_defence_goals": f.home_ewma_gf - f.away_ewma_ga,
        "away_attack_vs_home_defence_goals": f.away_ewma_gf - f.home_ewma_ga,
        "home_attack_vs_away_defence_sot": f.home_ewma_sot_for - f.away_ewma_sot_against,
        "diff_std_ppg": f.home_std_ppg - f.away_std_ppg,
    }
    for name, series in recon.items():
        assert np.allclose(
            f[name].to_numpy(dtype=float), series.to_numpy(dtype=float), equal_nan=True
        ), f"{name} should be an exact reconstruction"
    # diff_std_gd is NOT reconstructible: home/away season-to-date GD are not emitted.
    assert "diff_std_gd" in LINEAR_SAFE_FEATURE_COLUMNS


# --------------------------------------------------------------------------
# 19. Strengthened validation catches injected faults
# --------------------------------------------------------------------------
def _good_frame() -> pd.DataFrame:
    return build_features(build_synthetic_league(), make_params())


def test_validation_catches_duplicate_match_keys():
    f = _good_frame()
    broken = pd.concat([f, f.iloc[[0]]], ignore_index=True)
    assert any("duplicate match keys" in p for p in validate_feature_frame(broken))


def test_validation_catches_infinite_values():
    f = _good_frame()
    f.loc[0, "home_ewma_gf"] = float("inf")
    assert any("infinite value" in p for p in validate_feature_frame(f))


def test_validation_catches_all_nan_row():
    f = _good_frame()
    f.loc[0, FEATURE_COLUMNS] = float("nan")
    problems = validate_feature_frame(f)
    assert any("every feature NaN" in p for p in problems)


def test_validation_catches_nan_in_a_non_nullable_feature():
    f = _good_frame()
    f.loc[0, "home_elo"] = float("nan")
    assert any("must always be populated" in p for p in validate_feature_frame(f))


def test_validation_catches_target_disagreeing_with_ftr():
    f = _good_frame()
    f.loc[0, TARGET_COLUMN] = (int(f.loc[0, TARGET_COLUMN]) + 1) % 3
    assert any("disagrees with FTR" in p for p in validate_feature_frame(f))


def test_validation_catches_cold_start_flag_inconsistency():
    f = _good_frame()
    f.loc[0, "home_is_cold_start"] = 1.0 - f.loc[0, "home_is_cold_start"]
    assert any("disagrees with home_history_depth" in p for p in validate_feature_frame(f))


def test_validation_catches_implausible_elo():
    f = _good_frame()
    f.loc[:, "home_elo"] = 5000.0
    problems = validate_feature_frame(f)
    assert any("plausible band" in p or "drifted" in p for p in problems)


def test_validation_catches_wrong_row_count():
    f = _good_frame()
    assert any("expected" in p for p in validate_feature_frame(f, expected_rows=len(f) + 1))


def test_non_nullable_features_are_the_expected_ones():
    assert set(NON_NULLABLE_FEATURE_COLUMNS) == {
        "home_elo", "away_elo", "elo_diff",
        "home_history_depth", "away_history_depth",
        "home_is_cold_start", "away_is_cold_start",
    }


# --------------------------------------------------------------------------
# 20. REAL-DATA regression probes (ported from the adversarial audit)
# --------------------------------------------------------------------------
requires_real_data = pytest.mark.skipif(
    not MATCHES_PATH.exists(), reason="data/processed/matches.parquet not built"
)


@pytest.fixture(scope="module")
def real_matches() -> pd.DataFrame:
    return pd.read_parquet(MATCHES_PATH)


@pytest.fixture(scope="module")
def real_params(real_matches) -> EloParams:
    # A real fold cutoff, so the probes exercise production-shaped parameters.
    return estimate_elo_params(real_matches, "2023_24")


@pytest.fixture(scope="module")
def real_features(real_matches, real_params) -> pd.DataFrame:
    return build_features(real_matches, real_params)


@requires_real_data
def test_real_data_shape_and_validation(real_matches, real_features):
    assert len(real_features) == len(real_matches) == 4180
    assert validate_feature_frame(real_features, expected_rows=len(real_matches)) == []


@requires_real_data
def test_real_data_deterministic_rebuild(real_matches, real_params, real_features):
    again = build_features(real_matches, real_params)
    pdt.assert_frame_equal(real_features, again)


@requires_real_data
def test_real_data_same_date_order_independence(real_matches, real_params, real_features):
    shuffled = real_matches.sample(frac=1.0, random_state=11).reset_index(drop=True)
    pdt.assert_frame_equal(real_features, build_features(shuffled, real_params))


@requires_real_data
@pytest.mark.parametrize("boundary_season", ["2018_19", "2021_22", "2023_24"])
def test_real_data_truncation_invariance(real_matches, real_params, real_features, boundary_season):
    """Building on history truncated at a season boundary must reproduce it exactly.

    Truncation is at season boundaries because the per-season re-centring uses
    that season's roster, which is a pre-season scheduling fact; slicing
    mid-season would hide clubs that had not yet played and legitimately change
    the roster. Results are what must not leak, and they cannot.
    """
    cutoff = real_matches.loc[real_matches["Season"] == boundary_season, "Date"].max()
    truncated = real_matches[real_matches["Date"] <= cutoff]
    rebuilt = build_features(truncated, real_params).reset_index(drop=True)
    expected = real_features[real_features["Date"] <= cutoff].reset_index(drop=True)
    pdt.assert_frame_equal(rebuilt, expected)


@requires_real_data
@pytest.mark.parametrize("cutoff", ["2019-05-01", "2022-05-01", "2024-05-01"])
def test_real_data_future_shuffle_invariance(real_matches, real_params, real_features, cutoff):
    """Permuting every result after D must not change any feature row at or before D."""
    boundary = pd.Timestamp(cutoff)
    shuffled = real_matches.copy()
    future_index = shuffled.index[shuffled["Date"] > boundary].to_numpy()
    permuted = np.random.default_rng(0).permutation(future_index)
    outcome_columns = ["FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"]
    shuffled.loc[future_index, outcome_columns] = real_matches.loc[
        permuted, outcome_columns
    ].to_numpy()

    rebuilt = build_features(shuffled, real_params)
    before = real_features[real_features["Date"] <= boundary][FEATURE_COLUMNS].reset_index(drop=True)
    after = rebuilt[rebuilt["Date"] <= boundary][FEATURE_COLUMNS].reset_index(drop=True)
    assert len(before) > 1000
    pdt.assert_frame_equal(before, after)


@requires_real_data
def test_real_data_elo_scale_is_recentred_every_season(real_features):
    means = active_season_mean_elo(real_features)
    assert len(means) == 11
    for season, mean_elo in means.items():
        assert mean_elo == pytest.approx(DEFAULT_INITIAL_RATING), f"{season} not re-centred"


@requires_real_data
def test_real_data_no_feature_is_suspiciously_correlated_with_the_target(real_features):
    """Leakage canary. Computed on development seasons only — never the sealed season."""
    dev = real_features[real_features["Season"] != SEALED_SEASON]
    target = dev[TARGET_COLUMN].to_numpy(dtype=float)
    for column in FEATURE_COLUMNS:
        values = dev[column].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if mask.sum() < 500 or np.std(values[mask]) == 0:
            continue
        correlation = abs(np.corrcoef(values[mask], target[mask])[0, 1])
        assert correlation < 0.6, f"{column} correlates {correlation:.3f} with the target"


@requires_real_data
def test_real_data_linear_safe_subset_is_full_rank(real_features):
    complete = real_features[LINEAR_SAFE_FEATURE_COLUMNS].dropna().to_numpy(dtype=float)
    assert complete.shape[0] > 3000
    assert np.linalg.matrix_rank(complete) == len(LINEAR_SAFE_FEATURE_COLUMNS)


@requires_real_data
def test_real_data_full_feature_set_is_rank_deficient_as_documented(real_features):
    """Documents *why* the linear-safe subset exists; trees are unaffected."""
    complete = real_features[FEATURE_COLUMNS].dropna().to_numpy(dtype=float)
    assert np.linalg.matrix_rank(complete) < len(FEATURE_COLUMNS)


@requires_real_data
def test_real_data_sealed_season_never_enters_estimation(real_matches):
    for cutoff in ["2021_22", "2022_23", "2023_24", "2024_25"]:
        params = estimate_elo_params(real_matches, cutoff)
        assert SEALED_SEASON not in params.estimated_from_seasons
        assert params.training_cutoff == cutoff

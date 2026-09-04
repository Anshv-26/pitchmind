"""Inspect raw Premier League season CSVs downloaded to data/raw/.

Read-only reporting: no data is cleaned, transformed, or written anywhere.
Run scripts/download_data.py first to populate data/raw/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from download_data import RAW_DIR, SEASONS

# Descriptive-only: prefixes of columns that are bookmaker/betting-market
# derived across the football-data.co.uk seasons inspected so far. This is
# used purely to report presence in raw data; it is NOT used to filter or
# transform anything here. The future processing pipeline will select an
# explicit whitelist of approved football-performance columns rather than
# relying on a blacklist like this one.
BETTING_COLUMN_PREFIXES = (
    "B365", "BW", "BF", "BV", "BMGM", "IW", "LB", "PS", "WH", "VC",
    "CL", "1XB", "Max", "Avg", "Bb", "AHh",
)


def is_betting_column(column: str) -> bool:
    return any(column.startswith(prefix) for prefix in BETTING_COLUMN_PREFIXES)


def load_season(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def date_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if "Date" not in df.columns:
        return None, None
    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    if dates.isna().all():
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def unique_teams(df: pd.DataFrame) -> int:
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        return 0
    return int(pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel()).size)


def main() -> int:
    candidate_paths = {label: RAW_DIR / f"{label}.csv" for label in SEASONS}
    present = {label: path for label, path in candidate_paths.items() if path.exists()}
    missing = [label for label in SEASONS if label not in present]

    if not present:
        print(f"No season CSV files found in {RAW_DIR}. Run scripts/download_data.py first.")
        return 1

    print(f"Inspecting {len(present)} season file(s) in {RAW_DIR}")
    if missing:
        print(f"Missing season files (not yet downloaded): {', '.join(missing)}")
    print()

    per_season_columns: dict[str, set[str]] = {}
    per_season_info: dict[str, dict] = {}

    header = f"{'Season':<10}{'Rows':>6}{'Teams':>7}{'Cols':>6}  {'Date range'}"
    print(header)
    print("-" * len(header))

    for label in sorted(present):
        df = load_season(present[label])
        cols = set(df.columns)
        per_season_columns[label] = cols
        d_min, d_max = date_range(df)
        teams = unique_teams(df)
        per_season_info[label] = {
            "rows": len(df),
            "teams": teams,
            "cols": len(cols),
            "date_min": d_min,
            "date_max": d_max,
            "df": df,
        }
        date_str = f"{d_min} -> {d_max}" if d_min else "N/A"
        print(f"{label:<10}{len(df):>6}{teams:>7}{len(cols):>6}  {date_str}")
    print()

    print("Integrity checks (informational - a completed 20-team season")
    print("should have 380 rows and 20 unique teams; not enforced as a")
    print("hard rule so an incomplete/current season can be inspected too):")
    any_flag = False
    for label in sorted(per_season_info):
        info = per_season_info[label]
        flags = []
        if info["rows"] != 380:
            flags.append(f"rows={info['rows']} (expected 380)")
        if info["teams"] != 20:
            flags.append(f"teams={info['teams']} (expected 20)")
        if flags:
            any_flag = True
            print(f"  {label}: " + "; ".join(flags))
    if not any_flag:
        print("  all seasons: 380 rows / 20 teams as expected")
    print()

    all_columns = set().union(*per_season_columns.values())
    common_columns = set.intersection(*per_season_columns.values())
    varying_columns = all_columns - common_columns

    print(f"Columns common to all {len(per_season_columns)} seasons ({len(common_columns)}):")
    print("  " + ", ".join(sorted(common_columns)))
    print()

    print(f"Columns that vary between seasons ({len(varying_columns)}):")
    for col in sorted(varying_columns):
        seasons_with = sorted(l for l in per_season_columns if col in per_season_columns[l])
        seasons_without = sorted(l for l in per_season_columns if l not in seasons_with)
        print(
            f"  {col}: present in {len(seasons_with)}/{len(per_season_columns)} seasons"
            f" (missing from: {', '.join(seasons_without) if seasons_without else 'none'})"
        )
    print()

    print("Missingness in common columns (only columns with null values shown):")
    any_missing = False
    for label in sorted(per_season_info):
        df = per_season_info[label]["df"]
        null_counts = df[sorted(common_columns)].isna().sum()
        nonzero = null_counts[null_counts > 0]
        if not nonzero.empty:
            any_missing = True
            print(f"  {label}:")
            for col, count in nonzero.items():
                print(f"    {col}: {count} missing")
    if not any_missing:
        print("  none found in common columns")
    print()

    print("Betting-related columns detected (descriptive only - not used for")
    print("modelling; future processing will whitelist approved columns")
    print("rather than rely on this blacklist):")
    for label in sorted(per_season_columns):
        betting_cols = sorted(c for c in per_season_columns[label] if is_betting_column(c))
        print(f"  {label}: {len(betting_cols)} betting-related column(s)")
    all_betting_cols = sorted(c for c in all_columns if is_betting_column(c))
    print(f"  Total distinct betting-related columns seen across all seasons: {len(all_betting_cols)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

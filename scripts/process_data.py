"""Combine the raw Premier League season CSVs into one validated, clean,
chronologically ordered dataset for downstream feature engineering.

Reads only from data/raw/ (never modified) and writes a single Parquet
file to data/processed/matches.parquet. Only an explicit whitelist of
football-performance columns is retained — no bookmaker/betting columns
are carried forward. No feature engineering, target creation, or model
training happens here.

A small number of confirmed source-data typos are corrected in memory via
KNOWN_SOURCE_CORRECTIONS (see below) — data/raw/ itself is never edited.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from download_data import RAW_DIR, SEASONS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "matches.parquet"

# Explicit whitelist of football-performance columns to retain. Deliberately
# a whitelist rather than a betting-column blacklist, so a new bookmaker
# column introduced in a future season can never accidentally leak through.
#
# Referee is metadata, not a feature: it is retained here because it has
# complete coverage across all seasons and may support future
# referee-tendency features, but it must NOT automatically become an input
# to the first match-prediction model. It can only be used once we confirm
# referee assignments are reliably known for upcoming fixtures at inference
# time (unlike final match statistics, referees are sometimes announced
# closer to kickoff than other pre-match information).
WHITELIST_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR",
    "Referee",
]

NUMERIC_COLUMNS = [
    "FTHG", "FTAG", "HTHG", "HTAG",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]

# Final column order in the output dataset.
OUTPUT_COLUMNS = ["Season"] + WHITELIST_COLUMNS

# A completed 20-team Premier League season has exactly 380 matches and 20
# unique teams. Deviations are reported as warnings, not hard failures, so
# an in-progress/current season can still be processed in future.
EXPECTED_COMPLETED_ROWS = 380
EXPECTED_TEAMS = 20


@dataclass(frozen=True)
class SourceCorrection:
    """A manually verified fix for one confirmed typo in an immutable raw
    file. Applied in memory only, during processing — data/raw/ is never
    written to. Matched narrowly on (season, date, home team, away team) so
    it can only ever touch the one intended row, and only if the existing
    value still equals what was verified when the correction was written
    (see apply_known_source_corrections)."""

    season: str
    date: str  # ISO "YYYY-MM-DD", matched against the parsed Date column
    home_team: str
    away_team: str
    column: str
    expected_current_value: object
    corrected_value: object
    reason: str


# Confirmed source-data errors in the raw football-data.co.uk files.
# Each entry here is a single verified correction, never a general rule.
KNOWN_SOURCE_CORRECTIONS: list[SourceCorrection] = [
    SourceCorrection(
        season="2021_22",
        date="2021-08-15",
        home_team="Newcastle",
        away_team="West Ham",
        column="AS",
        expected_current_value=8,
        corrected_value=18,
        reason=(
            "Source typo: raw AS=8 violates AST<=AS (AST=9). Independent "
            "match records confirm West Ham recorded AS=18, AST=9."
        ),
    ),
]


@dataclass
class CorrectionLogEntry:
    season: str
    date: str
    home_team: str
    away_team: str
    column: str
    old_value: object
    new_value: object
    reason: str


@dataclass
class SeasonInfo:
    season: str
    rows: int
    teams: int
    dropped_columns: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    new_teams: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrections: list[CorrectionLogEntry] = field(default_factory=list)


@dataclass
class Violation:
    season: str
    rule: str
    detail: str


def load_raw_season(season_label: str) -> pd.DataFrame:
    path = RAW_DIR / f"{season_label}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing raw file for season {season_label}: {path} "
            f"(run scripts/download_data.py first)"
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def select_whitelist_columns(df: pd.DataFrame, season_label: str) -> tuple[pd.DataFrame, int]:
    missing = [c for c in WHITELIST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"season {season_label} is missing required columns: {missing}")
    dropped = len(df.columns) - len(WHITELIST_COLUMNS)
    return df[WHITELIST_COLUMNS].copy(), dropped


def parse_match_dates(df: pd.DataFrame) -> pd.DataFrame:
    # football-data.co.uk uses dd/mm/yy in some seasons and dd/mm/yyyy in
    # others (consistently within a single season's file); format="mixed"
    # parses each row's format without pandas warning about inference cost.
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    return df


def normalize_referee_text(df: pd.DataFrame) -> pd.DataFrame:
    # Whitespace stripping only - no renaming/aliasing logic yet.
    df = df.copy()
    df["Referee"] = df["Referee"].str.strip()
    return df


def apply_known_source_corrections(
    df: pd.DataFrame, season_label: str
) -> tuple[pd.DataFrame, list[CorrectionLogEntry]]:
    """Apply the KNOWN_SOURCE_CORRECTIONS entries for this season to an
    in-memory copy of the data. data/raw/ is never modified.

    Each correction must match exactly one row (season + date + home team +
    away team) and the row's current value in the target column must equal
    exactly what was verified when the correction was written. Either
    condition failing raises a clear error rather than silently applying
    (or skipping) the correction - this is a narrow, auditable patch, not a
    general repair mechanism.
    """
    df = df.copy()
    log_entries: list[CorrectionLogEntry] = []

    for correction in KNOWN_SOURCE_CORRECTIONS:
        if correction.season != season_label:
            continue

        target_date = pd.Timestamp(correction.date)
        mask = (
            (df["Date"] == target_date)
            & (df["HomeTeam"] == correction.home_team)
            & (df["AwayTeam"] == correction.away_team)
        )
        matched_rows = df.index[mask]

        if len(matched_rows) != 1:
            raise ValueError(
                f"known source correction for {correction.season} "
                f"{correction.date} {correction.home_team} v "
                f"{correction.away_team} matched {len(matched_rows)} row(s), "
                f"expected exactly 1 - refusing to apply correction"
            )

        row_index = matched_rows[0]
        current_value = df.at[row_index, correction.column]
        if current_value != correction.expected_current_value:
            raise ValueError(
                f"known source correction for {correction.season} "
                f"{correction.date} {correction.home_team} v "
                f"{correction.away_team} column {correction.column}: "
                f"expected current value {correction.expected_current_value!r}, "
                f"found {current_value!r} - refusing to apply correction "
                f"(row may already be fixed upstream, or the correction is stale)"
            )

        df.at[row_index, correction.column] = correction.corrected_value
        log_entries.append(
            CorrectionLogEntry(
                season=correction.season,
                date=correction.date,
                home_team=correction.home_team,
                away_team=correction.away_team,
                column=correction.column,
                old_value=current_value,
                new_value=correction.corrected_value,
                reason=correction.reason,
            )
        )

    return df, log_entries


def validate_season_rows(df: pd.DataFrame, season_label: str) -> list[Violation]:
    """Row-level integrity checks. Every check below passes with zero
    violations against the real source data as of this writing; failures
    are treated as hard errors (see module docstring / process report)
    rather than silently repaired, since this is immutable historical data
    and a violation likely signals either a genuine new source problem or
    a bug in this script."""
    violations: list[Violation] = []

    self_matches = df[df["HomeTeam"] == df["AwayTeam"]]
    for _, row in self_matches.iterrows():
        violations.append(
            Violation(season_label, "home_equals_away", f"{row['Date'].date()} {row['HomeTeam']}")
        )

    negative_rows = df[(df[NUMERIC_COLUMNS] < 0).any(axis=1)]
    for _, row in negative_rows.iterrows():
        violations.append(
            Violation(
                season_label, "negative_statistic",
                f"{row['Date'].date()} {row['HomeTeam']} v {row['AwayTeam']}",
            )
        )

    expected_ftr = pd.Series("D", index=df.index)
    expected_ftr[df["FTHG"] > df["FTAG"]] = "H"
    expected_ftr[df["FTHG"] < df["FTAG"]] = "A"
    for _, row in df[df["FTR"] != expected_ftr].iterrows():
        violations.append(
            Violation(
                season_label, "ftr_inconsistent",
                f"{row['Date'].date()} {row['HomeTeam']} v {row['AwayTeam']} "
                f"FTHG={row['FTHG']} FTAG={row['FTAG']} FTR={row['FTR']}",
            )
        )

    expected_htr = pd.Series("D", index=df.index)
    expected_htr[df["HTHG"] > df["HTAG"]] = "H"
    expected_htr[df["HTHG"] < df["HTAG"]] = "A"
    for _, row in df[df["HTR"] != expected_htr].iterrows():
        violations.append(
            Violation(
                season_label, "htr_inconsistent",
                f"{row['Date'].date()} {row['HomeTeam']} v {row['AwayTeam']} "
                f"HTHG={row['HTHG']} HTAG={row['HTAG']} HTR={row['HTR']}",
            )
        )

    for _, row in df[df["HST"] > df["HS"]].iterrows():
        violations.append(
            Violation(
                season_label, "hst_exceeds_hs",
                f"{row['Date'].date()} {row['HomeTeam']} v {row['AwayTeam']} "
                f"HS={row['HS']} HST={row['HST']}",
            )
        )

    for _, row in df[df["AST"] > df["AS"]].iterrows():
        violations.append(
            Violation(
                season_label, "ast_exceeds_as",
                f"{row['Date'].date()} {row['HomeTeam']} v {row['AwayTeam']} "
                f"AS={row['AS']} AST={row['AST']}",
            )
        )

    for _, row in df[df["HTHG"] > df["FTHG"]].iterrows():
        violations.append(
            Violation(
                season_label, "hthg_exceeds_fthg",
                f"{row['Date'].date()} {row['HomeTeam']} v {row['AwayTeam']} "
                f"HTHG={row['HTHG']} FTHG={row['FTHG']}",
            )
        )

    for _, row in df[df["HTAG"] > df["FTAG"]].iterrows():
        violations.append(
            Violation(
                season_label, "htag_exceeds_ftag",
                f"{row['Date'].date()} {row['HomeTeam']} v {row['AwayTeam']} "
                f"HTAG={row['HTAG']} FTAG={row['FTAG']}",
            )
        )

    return violations


def validate_no_duplicate_matches(combined: pd.DataFrame) -> list[Violation]:
    dup_mask = combined.duplicated(subset=["Date", "HomeTeam", "AwayTeam"], keep=False)
    violations = []
    for _, row in combined[dup_mask].iterrows():
        violations.append(
            Violation(
                row["Season"], "duplicate_match",
                f"{row['Date'].date()} {row['HomeTeam']} v {row['AwayTeam']}",
            )
        )
    return violations


def process_season(season_label: str, seen_teams: set[str]) -> tuple[pd.DataFrame, SeasonInfo, list[Violation]]:
    raw_df = load_raw_season(season_label)
    df, dropped = select_whitelist_columns(raw_df, season_label)
    df = parse_match_dates(df)
    df["Season"] = season_label
    df = normalize_referee_text(df)
    df, corrections = apply_known_source_corrections(df, season_label)

    violations = validate_season_rows(df, season_label)

    teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
    new_teams = sorted(teams - seen_teams)

    warnings = []
    if len(df) != EXPECTED_COMPLETED_ROWS:
        warnings.append(f"{len(df)} rows found, expected {EXPECTED_COMPLETED_ROWS}")
    if len(teams) != EXPECTED_TEAMS:
        warnings.append(f"{len(teams)} unique teams found, expected {EXPECTED_TEAMS}")

    info = SeasonInfo(
        season=season_label,
        rows=len(df),
        teams=len(teams),
        dropped_columns=dropped,
        date_min=df["Date"].min(),
        date_max=df["Date"].max(),
        new_teams=new_teams,
        warnings=warnings,
        corrections=corrections,
    )
    return df, info, violations


def _print_report(season_infos: list[SeasonInfo], combined: pd.DataFrame) -> None:
    print()
    header = f"{'Season':<10}{'Rows':>6}{'Teams':>7}{'Dropped':>9}  {'Date range'}"
    print(header)
    print("-" * len(header))
    for info in season_infos:
        date_str = f"{info.date_min.date()} -> {info.date_max.date()}"
        print(f"{info.season:<10}{info.rows:>6}{info.teams:>7}{info.dropped_columns:>9}  {date_str}")
    print()

    any_warning = False
    for info in season_infos:
        if info.warnings:
            any_warning = True
            print(f"  warning [{info.season}]: " + "; ".join(info.warnings))
    if not any_warning:
        print("  no row/team-count warnings")
    print()

    for info in season_infos:
        if info.new_teams:
            print(f"  new team name(s) first seen in {info.season}: {', '.join(info.new_teams)}")
    print()

    any_correction = False
    for info in season_infos:
        for c in info.corrections:
            any_correction = True
            print(
                f"  known source correction applied [{c.season}] "
                f"{c.date} {c.home_team} v {c.away_team}: "
                f"{c.column} {c.old_value!r} -> {c.new_value!r} ({c.reason})"
            )
    if not any_correction:
        print("  no known source corrections applied")
    print()

    print(f"Combined dataset: {len(combined)} rows, {len(combined.columns)} columns")
    print(f"Columns: {', '.join(combined.columns)}")
    print(f"Date range: {combined['Date'].min().date()} -> {combined['Date'].max().date()}")
    print()
    print("dtypes:")
    for col, dtype in combined.dtypes.items():
        print(f"  {col}: {dtype}")
    print()
    print(
        "Note: rows are sorted by (Date, HomeTeam, AwayTeam) for reproducible "
        "ordering only. Same-date fixtures are NOT in true kickoff order - "
        "future feature engineering must build each team's history from "
        "genuinely prior match dates only, never from this alphabetical "
        "tiebreak."
    )
    print()
    size_bytes = OUTPUT_PATH.stat().st_size
    print(f"Wrote {OUTPUT_PATH} ({size_bytes:,} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine raw Premier League season CSVs into one clean, whitelisted, validated dataset"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        choices=sorted(SEASONS),
        help="Subset of season labels to process (default: all)",
    )
    args = parser.parse_args(argv)

    season_labels = args.seasons or list(SEASONS)

    season_frames: list[pd.DataFrame] = []
    season_infos: list[SeasonInfo] = []
    all_violations: list[Violation] = []
    seen_teams: set[str] = set()

    for season_label in season_labels:
        print(f"[{season_label}] processing ...")
        try:
            df, info, violations = process_season(season_label, seen_teams)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  FAILED: {exc}")
            return 1

        seen_teams |= set(df["HomeTeam"]) | set(df["AwayTeam"])
        all_violations.extend(violations)
        season_frames.append(df)
        season_infos.append(info)
        print(f"  {info.rows} rows, {info.dropped_columns} non-whitelisted column(s) dropped")

    combined = pd.concat(season_frames, ignore_index=True)
    all_violations.extend(validate_no_duplicate_matches(combined))

    if all_violations:
        print()
        print(f"VALIDATION FAILED - {len(all_violations)} violation(s) found:")
        for v in all_violations:
            print(f"  [{v.season}] {v.rule}: {v.detail}")
        print()
        print("No output file written.")
        return 1

    combined = combined.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    combined = combined[OUTPUT_COLUMNS]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)

    _print_report(season_infos, combined)
    return 0


if __name__ == "__main__":
    sys.exit(main())

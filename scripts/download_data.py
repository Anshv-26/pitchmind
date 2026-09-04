"""Download historical Premier League match CSVs from football-data.co.uk.

Each season is fetched, validated, and saved independently to
``data/raw/{season_label}.csv``. Raw response bytes are preserved exactly
as received — files are never parsed and re-serialized. This script is
safe to re-run: a season already present with a valid file is skipped
unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REQUEST_TIMEOUT_SECONDS = 30

# Columns every football-data.co.uk match row must have for the file to be
# usable at all. Kept minimal and football-specific (never a betting column)
# so validation doesn't break if a season's set of bookmaker columns changes.
REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")

# A completed 20-team Premier League season has exactly 380 matches and 20
# unique teams. These are reported as warnings, not treated as hard
# validation failures, so an in-progress/current season (fewer matches,
# possibly fewer teams seen so far) can be downloaded later without the
# script assuming every season is complete.
EXPECTED_COMPLETED_ROWS = 380
EXPECTED_TEAMS = 20

# Readable season label -> football-data.co.uk season code.
SEASONS: dict[str, str] = {
    "2015_16": "1516",
    "2016_17": "1617",
    "2017_18": "1718",
    "2018_19": "1819",
    "2019_20": "1920",
    "2020_21": "2021",
    "2021_22": "2122",
    "2022_23": "2223",
    "2023_24": "2324",
    "2024_25": "2425",
    "2025_26": "2526",
}


@dataclass
class DownloadResult:
    season: str
    status: str  # "downloaded", "skipped", or "failed"
    reason: str = ""
    rows: int | None = None
    teams: int | None = None
    warnings: list[str] = field(default_factory=list)


def season_calendar_years(season_label: str) -> tuple[int, int]:
    """Return the two calendar years a season label spans, e.g. (2015, 2016)."""
    start_str, end_suffix = season_label.split("_")
    start_year = int(start_str)
    end_year = (start_year // 100) * 100 + int(end_suffix)
    return start_year, end_year


def validate_content(
    content: bytes, season_label: str
) -> tuple[bool, str, pd.DataFrame | None]:
    """Validate raw CSV bytes for a season.

    Checks that the content parses as CSV, has the core match columns, and
    that match dates fall within the season's two expected calendar years.
    The calendar-year check (rather than a strict August-May window) is
    what catches football-data.co.uk silently redirecting an unknown
    season code to an unrelated season's file.
    """
    if not content or not content.strip():
        return False, "empty response body", None

    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception as exc:
        return False, f"could not parse CSV: {exc}", None

    if df.empty:
        return False, "parsed CSV has zero data rows", None

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, f"missing required columns: {missing_cols}", None

    # football-data.co.uk uses dd/mm/yy in some seasons and dd/mm/yyyy in
    # others (consistently within a single season's file); format="mixed"
    # parses each row's format without pandas warning about inference cost.
    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    if dates.isna().all():
        return False, "no valid dates could be parsed from Date column", None

    start_year, end_year = season_calendar_years(season_label)
    expected_years = {start_year, end_year}
    found_years = set(dates.dropna().dt.year.unique().tolist())
    unexpected_years = found_years - expected_years
    if unexpected_years:
        return False, (
            f"match dates fall outside expected {start_year}/{end_year} "
            f"season (found years: {sorted(unexpected_years)}) - likely "
            f"wrong season code or an unexpected server redirect"
        ), None

    return True, "", df


def _count_unique_teams(df: pd.DataFrame) -> int:
    return int(pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel()).size)


def _integrity_warnings(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    rows = len(df)
    teams = _count_unique_teams(df)
    if rows != EXPECTED_COMPLETED_ROWS:
        warnings.append(
            f"{rows} rows found, expected {EXPECTED_COMPLETED_ROWS} for a "
            f"completed 20-team season"
        )
    if teams != EXPECTED_TEAMS:
        warnings.append(f"{teams} unique teams found, expected {EXPECTED_TEAMS}")
    return warnings


def _atomic_write(dest_path: Path, content: bytes) -> None:
    """Write bytes to dest_path atomically (temp file + rename)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=dest_path.parent, prefix=f".{dest_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def download_season(
    session: requests.Session,
    season_label: str,
    season_code: str,
    dest_dir: Path,
    force: bool = False,
) -> DownloadResult:
    dest_path = dest_dir / f"{season_label}.csv"

    if dest_path.exists() and not force:
        existing_content = dest_path.read_bytes()
        ok, reason, df = validate_content(existing_content, season_label)
        if ok:
            return DownloadResult(
                season=season_label,
                status="skipped",
                reason="valid file already present",
                rows=len(df),
                teams=_count_unique_teams(df),
                warnings=_integrity_warnings(df),
            )
        print(
            f"  existing file for {season_label} failed validation "
            f"({reason}) - re-downloading"
        )

    url = BASE_URL.format(code=season_code)
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        return DownloadResult(season=season_label, status="failed", reason=f"network error: {exc}")

    content = response.content
    ok, reason, df = validate_content(content, season_label)
    if not ok:
        return DownloadResult(season=season_label, status="failed", reason=reason)

    _atomic_write(dest_path, content)

    return DownloadResult(
        season=season_label,
        status="downloaded",
        rows=len(df),
        teams=_count_unique_teams(df),
        warnings=_integrity_warnings(df),
    )


def _print_result(result: DownloadResult) -> None:
    if result.status == "downloaded":
        print(f"  downloaded - {result.rows} rows, {result.teams} teams")
    elif result.status == "skipped":
        print(f"  skipped - {result.reason} ({result.rows} rows, {result.teams} teams)")
    else:
        print(f"  FAILED - {result.reason}")
    for warning in result.warnings:
        print(f"  warning: {warning}")


def _print_summary(results: list[DownloadResult]) -> None:
    downloaded = [r for r in results if r.status == "downloaded"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    print(
        f"Summary: {len(downloaded)} downloaded, {len(skipped)} skipped, "
        f"{len(failed)} failed (of {len(results)} seasons)"
    )
    if failed:
        print("Failed seasons:")
        for r in failed:
            print(f"  - {r.season}: {r.reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download historical Premier League CSVs from football-data.co.uk"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        choices=sorted(SEASONS),
        help="Subset of season labels to (re)download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid file already exists",
    )
    args = parser.parse_args(argv)

    seasons_to_run = {label: SEASONS[label] for label in (args.seasons or SEASONS)}

    session = requests.Session()
    session.headers.update({"User-Agent": "pitchmind-data-pipeline/1.0"})

    results: list[DownloadResult] = []
    for season_label, season_code in seasons_to_run.items():
        print(f"[{season_label}] fetching season code {season_code} ...")
        result = download_season(session, season_label, season_code, RAW_DIR, force=args.force)
        results.append(result)
        _print_result(result)

    print()
    _print_summary(results)

    return 0 if all(r.status != "failed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

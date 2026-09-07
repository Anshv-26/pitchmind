"""Verify whether OUR API-Football key actually covers Premier League 2026/27.

API-Football's free plan restricts which seasons are accessible, and it
signals that restriction with **HTTP 200 and an empty payload**, not an error
status. A naive "did it return 200?" check would therefore report success for
a plan that cannot see the current season at all. This probe treats
`200 + empty response` as FAILURE.

Budget: at most 3 requests. Run it manually, never from pytest, never on
import, and never in a loop.

    python scripts/probe_live_provider.py

Requires PITCHMIND_API_FOOTBALL_KEY in the environment. Prints a single
machine-greppable verdict line:

    CURRENT_EPL_FREE_ACCESS: YES
    CURRENT_EPL_FREE_ACCESS: NO
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from backend.app.core.config import (
    CURRENT_SEASON_START_YEAR,
    PREMIER_LEAGUE_API_FOOTBALL_ID,
    load_settings,
)
from backend.app.services.football_data.api_football import BASE_URL

MAX_REQUESTS = 3
VERDICT_YES = "CURRENT_EPL_FREE_ACCESS: YES"
VERDICT_NO = "CURRENT_EPL_FREE_ACCESS: NO"


class ProbeBudget:
    """Hard cap so a probe can never quietly burn the daily quota."""

    def __init__(self, limit: int = MAX_REQUESTS) -> None:
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise RuntimeError(f"probe exceeded its {self.limit}-request budget")
        self.used += 1


def _get(client: httpx.Client, path: str, params: dict, api_key: str, budget: ProbeBudget) -> dict:
    budget.spend()
    response = client.get(
        f"{BASE_URL}{path}", params=params, headers={"x-apisports-key": api_key}
    )
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "json": response.json() if response.headers.get("content-type", "").startswith("application/json") else None,
    }


def evaluate_league_response(result: dict) -> tuple[bool, str]:
    """Decide whether a /leagues response proves current-season access.

    Rejects, in order: non-200; provider-level `errors`; empty `response`
    (the free-plan restriction signal); and a league whose `seasons` array
    does not actually contain the current season.
    """
    if result["status_code"] == 429:
        return False, "rate limited (HTTP 429) - quota exhausted or too many requests"
    if result["status_code"] != 200:
        return False, f"HTTP {result['status_code']}"

    payload = result["json"]
    if not isinstance(payload, dict):
        return False, "response body was not a JSON object"

    errors = payload.get("errors")
    if errors:
        return False, f"provider reported errors: {errors!r}"

    entries = payload.get("response")
    if not isinstance(entries, list) or not entries:
        # The critical case: HTTP 200 with nothing in it.
        return False, (
            "HTTP 200 but empty 'response' - the plan does not expose league "
            f"{PREMIER_LEAGUE_API_FOOTBALL_ID} season {CURRENT_SEASON_START_YEAR}"
        )

    for entry in entries:
        seasons = entry.get("seasons") if isinstance(entry, dict) else None
        if not isinstance(seasons, list):
            continue
        for season in seasons:
            if isinstance(season, dict) and season.get("year") == CURRENT_SEASON_START_YEAR:
                coverage = season.get("coverage") or {}
                fixtures_coverage = coverage.get("fixtures") or {}
                details = {
                    "events": bool(fixtures_coverage.get("events")),
                    "lineups": bool(fixtures_coverage.get("lineups")),
                    "statistics_fixtures": bool(fixtures_coverage.get("statistics_fixtures")),
                    "standings": bool(coverage.get("standings")),
                }
                return True, f"season {CURRENT_SEASON_START_YEAR} present; coverage flags: {details}"

    return False, f"league returned but season {CURRENT_SEASON_START_YEAR} not listed in its seasons"


def evaluate_fixtures_response(result: dict) -> tuple[bool, str]:
    """Confirm fixtures actually exist for the current season."""
    if result["status_code"] == 429:
        return False, "rate limited (HTTP 429)"
    if result["status_code"] != 200:
        return False, f"HTTP {result['status_code']}"
    payload = result["json"]
    if not isinstance(payload, dict):
        return False, "response body was not a JSON object"
    if payload.get("errors"):
        return False, f"provider reported errors: {payload['errors']!r}"
    entries = payload.get("response")
    if not isinstance(entries, list) or not entries:
        return False, "HTTP 200 but no fixtures returned for the current season"
    return True, f"{len(entries)} fixture(s) returned"


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    if not settings.has_api_football_key:
        print("No API key found in PITCHMIND_API_FOOTBALL_KEY.")
        print(f"{VERDICT_NO}  reason: no API key configured")
        return 1

    budget = ProbeBudget()
    api_key = settings.api_football_key or ""
    timeout = httpx.Timeout(
        connect=settings.http_connect_timeout,
        read=settings.http_read_timeout,
        write=settings.http_read_timeout,
        pool=settings.http_connect_timeout,
    )

    print(f"Probing API-Football for league {PREMIER_LEAGUE_API_FOOTBALL_ID}, "
          f"season {CURRENT_SEASON_START_YEAR} (max {budget.limit} requests) ...")

    with httpx.Client(timeout=timeout) as client:
        try:
            league_result = _get(
                client,
                "/leagues",
                {"id": PREMIER_LEAGUE_API_FOOTBALL_ID, "season": CURRENT_SEASON_START_YEAR},
                api_key,
                budget,
            )
        except httpx.HTTPError as exc:
            print(f"{VERDICT_NO}  reason: transport error contacting provider ({exc})")
            return 1

        # Quota headers, when present, are useful context for the operator.
        remaining_day = league_result["headers"].get("x-ratelimit-requests-remaining")
        remaining_min = league_result["headers"].get("x-ratelimit-remaining")
        if remaining_day is not None or remaining_min is not None:
            print(f"  quota: {remaining_day} remaining today, {remaining_min} remaining this minute")

        league_ok, league_reason = evaluate_league_response(league_result)
        print(f"  league/season check: {'PASS' if league_ok else 'FAIL'} - {league_reason}")
        if not league_ok:
            print(f"{VERDICT_NO}  reason: {league_reason}")
            return 1

        try:
            fixtures_result = _get(
                client,
                "/fixtures",
                {
                    "league": PREMIER_LEAGUE_API_FOOTBALL_ID,
                    "season": CURRENT_SEASON_START_YEAR,
                    "next": 1,
                },
                api_key,
                budget,
            )
        except httpx.HTTPError as exc:
            print(f"{VERDICT_NO}  reason: transport error fetching fixtures ({exc})")
            return 1

        fixtures_ok, fixtures_reason = evaluate_fixtures_response(fixtures_result)
        print(f"  fixtures check: {'PASS' if fixtures_ok else 'FAIL'} - {fixtures_reason}")
        if not fixtures_ok:
            print(f"{VERDICT_NO}  reason: {fixtures_reason}")
            return 1

    print(f"  requests used: {budget.used}/{budget.limit}")
    print(VERDICT_YES)
    print(
        "\nNext step: construct ApiFootballProvider(..., live_verified=True) only now "
        "that current-season access is proven."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Real-integration verification for football-data.org (MANUAL, not pytest).

Makes AT MOST 3 real requests against football-data.org using the configured
free token, to verify PitchMind can genuinely retrieve and normalize current
Premier League competition/season info, fixtures, and standings.

    python scripts/verify_football_data_org.py

Requires PITCHMIND_FOOTBALL_DATA_ORG_KEY in the environment (loaded from a
local, gitignored `.env` if present). The token is never printed, logged, or
included in any error message - only a boolean "configured" flag and its
length are ever shown.

This script deliberately reuses the REAL adapter's `_get` (its actual
authentication, timeout and error handling) rather than a parallel HTTP
implementation, so what is verified here is the adapter's actual behaviour.
Response headers are observed via an httpx event hook attached to the SAME
client the adapter uses - no extra request, no separate implementation.

Request budget: exactly 3, hard-capped, no retries, no loops.
    1. GET /competitions/PL            -> competition + currentSeason info
    2. GET /competitions/PL/matches    -> fixtures (via provider.get_fixtures)
    3. GET /competitions/PL/standings  -> standings (via provider.get_standings)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from backend.app.core.config import load_settings
from backend.app.services.football_data.errors import (
    FootballDataError,
    MalformedProviderPayload,
    UnknownTeam,
)
from backend.app.services.football_data.football_data_org import FootballDataOrgProvider
from backend.app.services.football_data.teams import default_registry

MAX_REQUESTS = 3

# Response-header keyword scan - deliberately NOT a fixed list of exact header
# names, since we must not assume names before seeing them. Anything whose
# name contains one of these (case-insensitive) is reported as rate-limit
# metadata. "auth"/"token"/"key"/"cookie" are excluded so nothing resembling a
# credential is ever echoed, even defensively (response headers should never
# carry our token back, but this is a deliberate belt-and-braces filter).
_RATE_LIMIT_KEYWORDS = (
    "ratelimit", "rate-limit", "requests-available", "quota", "retry-after", "throttl", "x-request",
)
_NEVER_PRINT_KEYWORDS = ("auth", "token", "key", "cookie")


class RequestBudget:
    """Hard cap so this script can never quietly exceed its request budget."""

    def __init__(self, limit: int = MAX_REQUESTS) -> None:
        self.limit = limit
        self.used = 0

    def spend(self, label: str) -> None:
        if self.used >= self.limit:
            raise RuntimeError(f"verification exceeded its {self.limit}-request budget at {label!r}")
        self.used += 1
        print(f"  [request {self.used}/{self.limit}] {label}")


def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
    """All response headers except anything that could resemble a credential."""
    return {
        name: value
        for name, value in headers.items()
        if not any(bad in name.lower() for bad in _NEVER_PRINT_KEYWORDS)
    }


def extract_rate_limit_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if any(keyword in name.lower() for keyword in _RATE_LIMIT_KEYWORDS)
    }


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    print("=== football-data.org real-integration verification ===")
    print(
        f"PITCHMIND_FOOTBALL_DATA_ORG_KEY configured: {settings.has_football_data_org_key} "
        f"(length={len(settings.football_data_org_key) if settings.football_data_org_key else 0})"
    )
    if not settings.has_football_data_org_key:
        print("\nFOOTBALL_DATA_ORG_VERIFICATION: FAIL - no API key configured")
        return 1

    # Observe every response's headers via an event hook on the SAME client
    # the adapter uses - not a parallel request path.
    captured_headers: list[dict[str, str]] = []

    def on_response(response: httpx.Response) -> None:
        captured_headers.append(
            {"__url__": str(response.url.path), "__status__": str(response.status_code), **_safe_headers(response.headers)}
        )

    client = httpx.Client(
        timeout=httpx.Timeout(
            connect=settings.http_connect_timeout,
            read=settings.http_read_timeout,
            write=settings.http_read_timeout,
            pool=settings.http_connect_timeout,
        ),
        event_hooks={"response": [on_response]},
    )

    budget = RequestBudget()
    provider = FootballDataOrgProvider(settings, client=client)
    is_2026_27 = False

    try:
        # ---- Request 1: competition + current-season info --------------
        try:
            budget.spend("GET /competitions/PL")
            competition_payload = provider._get("/competitions/PL")
        except FootballDataError as exc:
            print(f"\nFOOTBALL_DATA_ORG_AUTH: FAIL - {exc}")
            print("FOOTBALL_DATA_ORG_VERIFICATION: FAIL")
            return 1
        print("FOOTBALL_DATA_ORG_AUTH: PASS (X-Auth-Token accepted)")

        current_season = competition_payload.get("currentSeason") or {}
        start_date = current_season.get("startDate")
        end_date = current_season.get("endDate")
        print(f"  competition name: {competition_payload.get('name')!r}")
        print(
            f"  currentSeason: startDate={start_date} endDate={end_date} "
            f"matchday={current_season.get('currentMatchday')}"
        )
        is_2026_27 = bool(start_date and str(start_date).startswith("2026"))
        print(f"CURRENT_PL_SEASON: {start_date} -> {end_date}")
        print(f"CURRENT_PL_ACCESS: {'YES' if is_2026_27 else 'NO'}")

        # ---- Request 2: fixtures ----------------------------------------
        try:
            budget.spend("GET /competitions/PL/matches")
            matches_payload = provider._get("/competitions/PL/matches")
        except FootballDataError as exc:
            print(f"\nFIXTURES_NORMALIZED: FAIL - {exc}")
            print("FOOTBALL_DATA_ORG_VERIFICATION: FAIL")
            return 1

        raw_matches = matches_payload.get("matches")
        if not isinstance(raw_matches, list) or not raw_matches:
            print("\nFIXTURES_NORMALIZED: FAIL - HTTP 200 but 'matches' was empty/missing")
            print("FOOTBALL_DATA_ORG_VERIFICATION: FAIL")
            return 1
        fixtures_raw_count = len(raw_matches)

        all_team_names: set[str] = set()
        for raw_match in raw_matches:
            for side in ("homeTeam", "awayTeam"):
                name = (raw_match.get(side) or {}).get("name")
                if name:
                    all_team_names.add(name)

        unresolved_fixture_teams: set[str] = set()
        fixtures_normalized = 0
        malformed_fixtures = 0
        for raw_match in raw_matches:
            try:
                provider.parse_fixture(raw_match, "2026_27")
                fixtures_normalized += 1
            except UnknownTeam as exc:
                unresolved_fixture_teams.add(exc.raw_value)
            except MalformedProviderPayload:
                malformed_fixtures += 1

        print(f"FIXTURES_NORMALIZED: {fixtures_normalized}/{fixtures_raw_count}"
              + (f" ({malformed_fixtures} malformed)" if malformed_fixtures else ""))

        # ---- Request 3: standings ----------------------------------------
        try:
            budget.spend("GET /competitions/PL/standings")
            standings_payload = provider._get("/competitions/PL/standings")
        except FootballDataError as exc:
            print(f"\nSTANDINGS_NORMALIZED: FAIL - {exc}")
            print("FOOTBALL_DATA_ORG_VERIFICATION: FAIL")
            return 1

        total_table: list[dict] = []
        for group in standings_payload.get("standings") or []:
            if isinstance(group, dict) and group.get("type") == "TOTAL":
                total_table = group.get("table") or []
                break
        if not total_table:
            print("\nSTANDINGS_NORMALIZED: FAIL - HTTP 200 but no TOTAL standings table")
            print("FOOTBALL_DATA_ORG_VERIFICATION: FAIL")
            return 1
        standings_raw_count = len(total_table)

        unresolved_standing_teams: set[str] = set()
        standings_normalized = 0
        malformed_standings = 0
        for row in total_table:
            team_name = (row.get("team") or {}).get("name")
            if team_name:
                all_team_names.add(team_name)
            try:
                provider.parse_standing_row(row)
                standings_normalized += 1
            except UnknownTeam as exc:
                unresolved_standing_teams.add(exc.raw_value)
            except MalformedProviderPayload:
                malformed_standings += 1

        print(f"STANDINGS_NORMALIZED: {standings_normalized}/{standings_raw_count}"
              + (f" ({malformed_standings} malformed)" if malformed_standings else ""))

    finally:
        client.close()

    # ---- Team identity summary -------------------------------------------
    unresolved_all = unresolved_fixture_teams | unresolved_standing_teams
    resolved_count = len(all_team_names) - len(unresolved_all)
    print(f"\nTEAM_IDENTITIES_RESOLVED: {resolved_count}/{len(all_team_names)}")
    print(f"  all current PL team names seen ({len(all_team_names)}): {sorted(all_team_names)}")
    if unresolved_all:
        print(f"  UNRESOLVED TEAM NAMES: {sorted(unresolved_all)}")

    # ---- Real provider IDs, for updating the registry (no guessing) ------
    print("\nOBSERVED PROVIDER TEAM IDS (name -> football_data.org numeric id):")
    seen_ids: dict[str, int] = {}
    for raw_match in raw_matches:
        for side in ("homeTeam", "awayTeam"):
            team = raw_match.get(side) or {}
            if team.get("name") and team.get("id") is not None:
                seen_ids[team["name"]] = team["id"]
    for row in total_table:
        team = row.get("team") or {}
        if team.get("name") and team.get("id") is not None:
            seen_ids[team["name"]] = team["id"]
    for name in sorted(seen_ids):
        print(f"  {name!r}: {seen_ids[name]}")

    # ---- Rate-limit / quota headers ---------------------------------------
    print(f"\nRATE_LIMIT_HEADERS (all non-credential response headers, by request):")
    any_rate_limit_found = False
    for captured in captured_headers:
        rate_limit = extract_rate_limit_headers(captured)
        if rate_limit:
            any_rate_limit_found = True
            print(f"  {captured['__url__']} (HTTP {captured['__status__']}): {rate_limit}")
    if not any_rate_limit_found:
        print("  none found by keyword scan; full header set per response:")
        for captured in captured_headers:
            shown = {k: v for k, v in captured.items() if not k.startswith("__")}
            print(f"    {captured['__url__']}: {shown}")

    print(f"\nRequests used: {budget.used}/{budget.limit}")
    all_passed = (
        is_2026_27
        and fixtures_normalized == fixtures_raw_count
        and standings_normalized == standings_raw_count
        and not unresolved_all
    )
    print(f"FOOTBALL_DATA_ORG_VERIFICATION: {'PASS' if all_passed else 'PARTIAL/FAIL - see details above'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

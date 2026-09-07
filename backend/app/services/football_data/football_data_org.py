"""football-data.org adapter - fixtures, results and standings only.

Capability scope is set by the provider's **verified free contract**: the free
tier covers competitions/fixtures/standings with delayed scores at 10
requests/minute, and does NOT include live scores, lineups, goalscorers,
bookings, or match statistics (those sit behind paid tiers/add-ons).

This adapter therefore declares only FIXTURES and STANDINGS. It does not
implement live methods at all, so asking it for live data raises
`UnsupportedCapability` via `BaseProvider` rather than returning an empty list
that a caller could mistake for "no matches are live".
"""

from __future__ import annotations

import httpx

from backend.app.core.config import (
    PREMIER_LEAGUE_FOOTBALL_DATA_ORG_CODE,
    FootballDataSettings,
)
from backend.app.services.football_data.errors import (
    MalformedProviderPayload,
    ProviderRateLimited,
    ProviderUnavailable,
)
from backend.app.services.football_data.models import (
    Fixture,
    FixtureStatus,
    StandingRow,
)
from backend.app.services.football_data.provider import BaseProvider, Capability
from backend.app.services.football_data.teams import (
    PROVIDER_FOOTBALL_DATA_ORG,
    TeamRegistry,
    default_registry,
)

BASE_URL = "https://api.football-data.org/v4"

# Provider status string -> normalized status. Anything absent maps to
# UNKNOWN rather than being guessed into SCHEDULED/FINISHED.
_STATUS_MAP: dict[str, FixtureStatus] = {
    "SCHEDULED": FixtureStatus.SCHEDULED,
    "TIMED": FixtureStatus.SCHEDULED,
    "IN_PLAY": FixtureStatus.LIVE,
    "PAUSED": FixtureStatus.HALF_TIME,
    "FINISHED": FixtureStatus.FINISHED,
    "POSTPONED": FixtureStatus.POSTPONED,
    "SUSPENDED": FixtureStatus.SUSPENDED,
    "CANCELLED": FixtureStatus.CANCELLED,
    "AWARDED": FixtureStatus.FINISHED,
}


def map_status(raw: str | None) -> FixtureStatus:
    if raw is None:
        return FixtureStatus.UNKNOWN
    return _STATUS_MAP.get(str(raw).upper(), FixtureStatus.UNKNOWN)


class FootballDataOrgProvider(BaseProvider):
    """Adapter for football-data.org v4 (free tier scope)."""

    name = "football_data_org"
    _CAPABILITIES = frozenset({Capability.FIXTURES, Capability.STANDINGS})

    def __init__(
        self,
        settings: FootballDataSettings,
        *,
        client: httpx.Client | None = None,
        registry: TeamRegistry | None = None,
        competition_code: str = PREMIER_LEAGUE_FOOTBALL_DATA_ORG_CODE,
    ) -> None:
        self._settings = settings
        self._client = client
        self._registry = registry if registry is not None else default_registry()
        self._competition_code = competition_code

    def capabilities(self) -> frozenset[Capability]:
        """With no API key configured this provider can do nothing - it
        reports an empty capability set rather than failing at import."""
        if not self._settings.has_football_data_org_key:
            return frozenset()
        return self._CAPABILITIES

    # ---- HTTP ----------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> dict:
        """One request, no automatic retries.

        Retries are deliberately absent: on a constrained free quota a silent
        retry storm wastes the budget that live data depends on.
        """
        if not self._settings.has_football_data_org_key:
            raise ProviderUnavailable(
                f"{self.name}: no API key configured "
                f"(set PITCHMIND_FOOTBALL_DATA_ORG_KEY)"
            )
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=self._settings.http_connect_timeout,
                    read=self._settings.http_read_timeout,
                    write=self._settings.http_read_timeout,
                    pool=self._settings.http_connect_timeout,
                )
            )
        try:
            response = client.get(
                f"{BASE_URL}{path}",
                params=params,
                headers={"X-Auth-Token": self._settings.football_data_org_key or ""},
            )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(f"{self.name}: request timed out ({exc})") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{self.name}: transport error ({exc})") from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise ProviderRateLimited(
                f"{self.name}: rate limited (HTTP 429)",
                retry_after_seconds=float(retry_after) if retry_after else None,
            )
        if response.status_code >= 500:
            raise ProviderUnavailable(f"{self.name}: server error (HTTP {response.status_code})")
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"{self.name}: request rejected (HTTP {response.status_code})"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MalformedProviderPayload(f"{self.name}: response body was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise MalformedProviderPayload(
                f"{self.name}: expected a JSON object, got {type(payload).__name__}"
            )
        return payload

    # ---- Normalization -------------------------------------------------
    def parse_fixture(self, raw: dict, season_label: str) -> Fixture:
        """Normalize one match object. Raises rather than guessing on
        anything structural that is missing."""
        try:
            match_id = raw["id"]
            home_raw = raw["homeTeam"]
            away_raw = raw["awayTeam"]
            kickoff = raw["utcDate"]
        except (KeyError, TypeError) as exc:
            raise MalformedProviderPayload(f"{self.name}: match missing required field ({exc})") from exc

        home_name = home_raw.get("name") or home_raw.get("shortName")
        away_name = away_raw.get("name") or away_raw.get("shortName")
        if not home_name or not away_name:
            raise MalformedProviderPayload(f"{self.name}: match {match_id} has no usable team names")

        score = raw.get("score") or {}
        full_time = score.get("fullTime") or {}

        return Fixture(
            provider_fixture_id=str(match_id),
            competition_code=self._competition_code,
            season_label=season_label,
            home_team=self._registry.team_ref(
                home_name,
                provider=PROVIDER_FOOTBALL_DATA_ORG,
                provider_team_id=str(home_raw["id"]) if home_raw.get("id") is not None else None,
            ),
            away_team=self._registry.team_ref(
                away_name,
                provider=PROVIDER_FOOTBALL_DATA_ORG,
                provider_team_id=str(away_raw["id"]) if away_raw.get("id") is not None else None,
            ),
            kickoff_utc=kickoff,
            status=map_status(raw.get("status")),
            matchday=raw.get("matchday"),
            home_score=full_time.get("home"),
            away_score=full_time.get("away"),
        )

    def parse_standing_row(self, raw: dict) -> StandingRow:
        try:
            team_raw = raw["team"]
            team_name = team_raw.get("name") or team_raw.get("shortName")
            if not team_name:
                raise KeyError("team.name")
            return StandingRow(
                position=raw["position"],
                team=self._registry.team_ref(
                    team_name,
                    provider=PROVIDER_FOOTBALL_DATA_ORG,
                    provider_team_id=str(team_raw["id"]) if team_raw.get("id") is not None else None,
                ),
                played=raw["playedGames"],
                won=raw["won"],
                drawn=raw["draw"],
                lost=raw["lost"],
                goals_for=raw["goalsFor"],
                goals_against=raw["goalsAgainst"],
                goal_difference=raw["goalDifference"],
                points=raw["points"],
            )
        except (KeyError, TypeError) as exc:
            raise MalformedProviderPayload(
                f"{self.name}: standings row missing required field ({exc})"
            ) from exc

    # ---- Provider interface -------------------------------------------
    def get_fixtures(self, season_label: str) -> list[Fixture]:
        self.require(Capability.FIXTURES)
        payload = self._get(f"/competitions/{self._competition_code}/matches")
        matches = payload.get("matches")
        if matches is None:
            raise MalformedProviderPayload(f"{self.name}: response has no 'matches' key")
        if not isinstance(matches, list):
            raise MalformedProviderPayload(f"{self.name}: 'matches' was not a list")
        return [self.parse_fixture(match, season_label) for match in matches]

    def get_standings(self, season_label: str) -> list[StandingRow]:
        self.require(Capability.STANDINGS)
        payload = self._get(f"/competitions/{self._competition_code}/standings")
        standings = payload.get("standings")
        if not isinstance(standings, list):
            raise MalformedProviderPayload(f"{self.name}: response has no 'standings' list")
        for group in standings:
            if isinstance(group, dict) and group.get("type") == "TOTAL":
                table = group.get("table")
                if not isinstance(table, list):
                    raise MalformedProviderPayload(f"{self.name}: TOTAL standings had no 'table' list")
                return [self.parse_standing_row(row) for row in table]
        raise MalformedProviderPayload(f"{self.name}: no TOTAL standings group in response")

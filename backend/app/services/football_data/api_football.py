"""API-Football (api-sports.io) adapter - live match state.

This is the only layer in PitchMind with verified free-tier access to live PL
score, minute, events, lineups and match statistics. Its live capability is
nonetheless treated as **runtime-conditional**: whether a given free key
actually covers Premier League season 2026 is unverified until
`scripts/probe_live_provider.py` says so. Construct with
`live_verified=False` (the default) and the adapter declares no live
capability, so nothing can silently depend on unproven access.

Batching: `/fixtures?ids=A-B-C` returns events, lineups and statistics
embedded for several fixtures in ONE request, so `get_live_matches()` costs a
single call regardless of how many matches are in play. That is what makes a
lazy 5-minute refresh affordable on a 100-request/day free quota.

Every parser tolerates missing fields. A statistic the provider omits stays
`None`; it is never coerced to 0.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from backend.app.core.config import PREMIER_LEAGUE_API_FOOTBALL_ID, FootballDataSettings
from backend.app.services.football_data.errors import (
    MalformedProviderPayload,
    ProviderRateLimited,
    ProviderUnavailable,
)
from backend.app.services.football_data.models import (
    Fixture,
    FixtureStatus,
    Lineup,
    LineupPlayer,
    LiveMatchState,
    MatchEvent,
    MatchEventType,
    MatchStats,
    ProviderFreshness,
    utcnow,
)
from backend.app.services.football_data.provider import BaseProvider, Capability
from backend.app.services.football_data.teams import (
    PROVIDER_API_FOOTBALL,
    TeamRegistry,
    default_registry,
)

BASE_URL = "https://v3.football.api-sports.io"

# Documented cap for the batched `ids` parameter.
MAX_BATCHED_FIXTURE_IDS = 20

_STATUS_MAP: dict[str, FixtureStatus] = {
    "TBD": FixtureStatus.SCHEDULED,
    "NS": FixtureStatus.SCHEDULED,
    "1H": FixtureStatus.LIVE,
    "2H": FixtureStatus.LIVE,
    "ET": FixtureStatus.EXTRA_TIME,
    "BT": FixtureStatus.EXTRA_TIME,
    "P": FixtureStatus.PENALTIES,
    "HT": FixtureStatus.HALF_TIME,
    "FT": FixtureStatus.FINISHED,
    "AET": FixtureStatus.FINISHED,
    "PEN": FixtureStatus.FINISHED,
    "SUSP": FixtureStatus.SUSPENDED,
    "INT": FixtureStatus.SUSPENDED,
    "PST": FixtureStatus.POSTPONED,
    "CANC": FixtureStatus.CANCELLED,
    "ABD": FixtureStatus.ABANDONED,
    "AWD": FixtureStatus.FINISHED,
    "WO": FixtureStatus.FINISHED,
    "LIVE": FixtureStatus.LIVE,
}

# Provider statistic label -> MatchStats field.
_STAT_FIELD_MAP: dict[str, str] = {
    "total shots": "shots",
    "shots on goal": "shots_on_target",
    "ball possession": "possession_percent",
    "corner kicks": "corners",
    "fouls": "fouls",
    "yellow cards": "yellow_cards",
    "red cards": "red_cards",
    "offsides": "offsides",
    "total passes": "passes",
    "passes %": "pass_accuracy_percent",
    "goalkeeper saves": "saves",
}

_PERCENT_FIELDS = {"possession_percent", "pass_accuracy_percent"}


def map_status(raw: str | None) -> FixtureStatus:
    if raw is None:
        return FixtureStatus.UNKNOWN
    return _STATUS_MAP.get(str(raw).upper(), FixtureStatus.UNKNOWN)


def _parse_stat_value(field: str, raw_value: object) -> int | float | None:
    """Provider statistic values arrive as ints, `null`, or strings like
    "56%". `None` and unparseable values stay `None` - never 0."""
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    text = str(raw_value).strip()
    if text == "" or text.lower() in {"null", "none"}:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if number < 0:
        return None
    if field in _PERCENT_FIELDS:
        return min(100.0, number)
    return int(number)


def parse_match_stats(raw_statistics: list | None) -> MatchStats | None:
    """Build MatchStats from one team's statistics list.

    Returns `None` when the provider supplied no statistics block at all -
    distinct from a block whose individual entries are null.
    """
    if not raw_statistics:
        return None
    values: dict[str, int | float | None] = {}
    for item in raw_statistics:
        if not isinstance(item, dict):
            continue
        label = str(item.get("type") or "").strip().lower()
        field = _STAT_FIELD_MAP.get(label)
        if field is None:
            continue
        values[field] = _parse_stat_value(field, item.get("value"))
    if not values:
        return None
    return MatchStats(**values)


def _map_event_type(raw_type: object, raw_detail: object) -> MatchEventType:
    event_type = str(raw_type or "").strip().lower()
    detail = str(raw_detail or "").strip().lower()
    if event_type == "goal":
        if "own goal" in detail:
            return MatchEventType.OWN_GOAL
        if "penalty" in detail and "missed" in detail:
            return MatchEventType.PENALTY_MISSED
        if "penalty" in detail:
            return MatchEventType.PENALTY_GOAL
        return MatchEventType.GOAL
    if event_type == "card":
        if "yellow" in detail and "second" in detail:
            return MatchEventType.SECOND_YELLOW_CARD
        if "yellow" in detail:
            return MatchEventType.YELLOW_CARD
        if "red" in detail:
            return MatchEventType.RED_CARD
        return MatchEventType.UNKNOWN
    if event_type in {"subst", "substitution"}:
        return MatchEventType.SUBSTITUTION
    if event_type == "var":
        return MatchEventType.VAR
    return MatchEventType.UNKNOWN


class ApiFootballProvider(BaseProvider):
    """Adapter for API-Football v3."""

    name = "api_football"
    _LIVE_CAPABILITIES = frozenset(
        {
            Capability.FIXTURES,
            Capability.STANDINGS,
            Capability.LIVE_SCORE,
            Capability.EVENTS,
            Capability.LINEUPS,
            Capability.MATCH_STATISTICS,
        }
    )

    def __init__(
        self,
        settings: FootballDataSettings,
        *,
        client: httpx.Client | None = None,
        registry: TeamRegistry | None = None,
        league_id: int = PREMIER_LEAGUE_API_FOOTBALL_ID,
        season_year: int | None = None,
        live_verified: bool = False,
    ) -> None:
        self._settings = settings
        self._client = client
        self._registry = registry if registry is not None else default_registry()
        self._league_id = league_id
        self._season_year = season_year
        # Live capability stays OFF until a probe verifies this key really
        # covers the current PL season - see scripts/probe_live_provider.py.
        self._live_verified = live_verified

    def capabilities(self) -> frozenset[Capability]:
        if not self._settings.has_api_football_key:
            return frozenset()
        if not self._live_verified:
            return frozenset()
        return self._LIVE_CAPABILITIES

    # ---- HTTP ----------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> dict:
        if not self._settings.has_api_football_key:
            raise ProviderUnavailable(
                f"{self.name}: no API key configured (set PITCHMIND_API_FOOTBALL_KEY)"
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
                headers={"x-apisports-key": self._settings.api_football_key or ""},
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
            raise ProviderUnavailable(f"{self.name}: request rejected (HTTP {response.status_code})")

        try:
            payload = response.json()
        except ValueError as exc:
            raise MalformedProviderPayload(f"{self.name}: response body was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise MalformedProviderPayload(
                f"{self.name}: expected a JSON object, got {type(payload).__name__}"
            )
        # API-Football reports application-level problems in `errors` while
        # still returning HTTP 200 - treat that as a failure, not success.
        errors = payload.get("errors")
        if errors:
            raise ProviderUnavailable(f"{self.name}: provider reported errors {errors!r}")
        return payload

    # ---- Normalization -------------------------------------------------
    def parse_fixture(self, raw: dict, season_label: str) -> Fixture:
        fixture_raw = raw.get("fixture")
        teams_raw = raw.get("teams")
        if not isinstance(fixture_raw, dict) or not isinstance(teams_raw, dict):
            raise MalformedProviderPayload(f"{self.name}: entry missing 'fixture'/'teams' objects")

        home_raw = teams_raw.get("home") or {}
        away_raw = teams_raw.get("away") or {}
        home_name = home_raw.get("name")
        away_name = away_raw.get("name")
        if not home_name or not away_name:
            raise MalformedProviderPayload(f"{self.name}: fixture has no usable team names")

        kickoff = fixture_raw.get("date")
        if kickoff is None:
            timestamp = fixture_raw.get("timestamp")
            if timestamp is None:
                raise MalformedProviderPayload(f"{self.name}: fixture has no kickoff time")
            kickoff = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)

        fixture_id = fixture_raw.get("id")
        if fixture_id is None:
            raise MalformedProviderPayload(f"{self.name}: fixture has no id")

        goals = raw.get("goals") or {}
        status_raw = (fixture_raw.get("status") or {}).get("short")

        return Fixture(
            provider_fixture_id=str(fixture_id),
            competition_code=str(self._league_id),
            season_label=season_label,
            home_team=self._registry.team_ref(
                home_name,
                provider=PROVIDER_API_FOOTBALL,
                provider_team_id=str(home_raw["id"]) if home_raw.get("id") is not None else None,
            ),
            away_team=self._registry.team_ref(
                away_name,
                provider=PROVIDER_API_FOOTBALL,
                provider_team_id=str(away_raw["id"]) if away_raw.get("id") is not None else None,
            ),
            kickoff_utc=kickoff,
            status=map_status(status_raw),
            home_score=goals.get("home"),
            away_score=goals.get("away"),
        )

    def parse_events(self, raw_events: list | None, fixture: Fixture) -> tuple[MatchEvent, ...]:
        if not raw_events:
            return ()
        events: list[MatchEvent] = []
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            team_name = (item.get("team") or {}).get("name")
            team_canonical_id = None
            if team_name:
                try:
                    team_canonical_id = self._registry.resolve(
                        team_name, provider=PROVIDER_API_FOOTBALL
                    ).canonical_id
                except Exception:
                    team_canonical_id = None
            elapsed = (item.get("time") or {}).get("elapsed")
            events.append(
                MatchEvent(
                    type=_map_event_type(item.get("type"), item.get("detail")),
                    minute=int(elapsed) if isinstance(elapsed, (int, float)) else None,
                    team_canonical_id=team_canonical_id,
                    player_name=(item.get("player") or {}).get("name"),
                    assist_name=(item.get("assist") or {}).get("name"),
                    detail=item.get("detail"),
                )
            )
        return tuple(events)

    def parse_lineup(self, raw_lineup: dict | None) -> Lineup | None:
        if not isinstance(raw_lineup, dict):
            return None
        team_name = (raw_lineup.get("team") or {}).get("name")
        if not team_name:
            return None
        canonical_id = self._registry.resolve(team_name, provider=PROVIDER_API_FOOTBALL).canonical_id

        def _players(entries: list | None) -> tuple[LineupPlayer, ...]:
            if not entries:
                return ()
            players: list[LineupPlayer] = []
            for entry in entries:
                player = (entry or {}).get("player") if isinstance(entry, dict) else None
                if not isinstance(player, dict):
                    continue
                name = player.get("name")
                if not name:
                    continue
                number = player.get("number")
                players.append(
                    LineupPlayer(
                        name=name,
                        number=int(number) if isinstance(number, (int, float)) else None,
                        position=player.get("pos"),
                    )
                )
            return tuple(players)

        return Lineup(
            team_canonical_id=canonical_id,
            formation=raw_lineup.get("formation"),
            starting_xi=_players(raw_lineup.get("startXI")),
            substitutes=_players(raw_lineup.get("substitutes")),
            coach_name=(raw_lineup.get("coach") or {}).get("name"),
        )

    def parse_live_match_state(
        self, raw: dict, season_label: str, *, fetched_at: datetime | None = None
    ) -> LiveMatchState:
        """Normalize ONE fixture entry - including its embedded events,
        lineups and statistics - into a single snapshot."""
        fixture = self.parse_fixture(raw, season_label)
        timestamp = fetched_at if fetched_at is not None else utcnow()

        home_stats = away_stats = None
        statistics = raw.get("statistics")
        if isinstance(statistics, list):
            for block in statistics:
                if not isinstance(block, dict):
                    continue
                team_name = (block.get("team") or {}).get("name")
                if not team_name:
                    continue
                try:
                    canonical_id = self._registry.resolve(
                        team_name, provider=PROVIDER_API_FOOTBALL
                    ).canonical_id
                except Exception:
                    continue
                parsed = parse_match_stats(block.get("statistics"))
                if canonical_id == fixture.home_team.canonical_id:
                    home_stats = parsed
                elif canonical_id == fixture.away_team.canonical_id:
                    away_stats = parsed

        home_lineup = away_lineup = None
        lineups = raw.get("lineups")
        if isinstance(lineups, list):
            for block in lineups:
                parsed_lineup = self.parse_lineup(block)
                if parsed_lineup is None:
                    continue
                if parsed_lineup.team_canonical_id == fixture.home_team.canonical_id:
                    home_lineup = parsed_lineup
                elif parsed_lineup.team_canonical_id == fixture.away_team.canonical_id:
                    away_lineup = parsed_lineup

        elapsed = ((raw.get("fixture") or {}).get("status") or {}).get("elapsed")

        return LiveMatchState(
            fixture=fixture,
            provider=self.name,
            minute=int(elapsed) if isinstance(elapsed, (int, float)) else None,
            home_score=fixture.home_score,
            away_score=fixture.away_score,
            home_stats=home_stats,
            away_stats=away_stats,
            events=self.parse_events(raw.get("events"), fixture),
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            freshness=ProviderFreshness(
                provider=self.name,
                fetched_at=timestamp,
                last_successful_refresh=timestamp,
                is_stale=False,
            ),
        )

    # ---- Provider interface -------------------------------------------
    def _season_param(self, season_label: str) -> int:
        if self._season_year is not None:
            return self._season_year
        # "2026_27" -> 2026
        return int(season_label.split("_")[0])

    def get_fixtures(self, season_label: str) -> list[Fixture]:
        self.require(Capability.FIXTURES)
        payload = self._get(
            "/fixtures", params={"league": self._league_id, "season": self._season_param(season_label)}
        )
        entries = payload.get("response")
        if not isinstance(entries, list):
            raise MalformedProviderPayload(f"{self.name}: 'response' was not a list")
        return [self.parse_fixture(entry, season_label) for entry in entries]

    def get_live_matches(self, season_label: str | None = None) -> list[LiveMatchState]:
        """All in-play PL matches in ONE request (`?live=...`)."""
        self.require(Capability.LIVE_SCORE)
        payload = self._get("/fixtures", params={"live": str(self._league_id)})
        entries = payload.get("response")
        if not isinstance(entries, list):
            raise MalformedProviderPayload(f"{self.name}: 'response' was not a list")
        fetched_at = utcnow()
        resolved_label = season_label or (str(self._season_year) if self._season_year else "unknown")
        return [
            self.parse_live_match_state(entry, resolved_label, fetched_at=fetched_at)
            for entry in entries
        ]

    def get_live_match_states_by_ids(
        self, provider_fixture_ids: list[str], season_label: str
    ) -> list[LiveMatchState]:
        """Batched fetch: `/fixtures?ids=A-B-C` returns events, lineups and
        statistics embedded, so many fixtures cost ONE request."""
        self.require(Capability.LIVE_SCORE)
        if not provider_fixture_ids:
            return []
        if len(provider_fixture_ids) > MAX_BATCHED_FIXTURE_IDS:
            raise ValueError(
                f"{self.name}: at most {MAX_BATCHED_FIXTURE_IDS} fixture ids per batched request"
            )
        payload = self._get("/fixtures", params={"ids": "-".join(str(i) for i in provider_fixture_ids)})
        entries = payload.get("response")
        if not isinstance(entries, list):
            raise MalformedProviderPayload(f"{self.name}: 'response' was not a list")
        fetched_at = utcnow()
        return [self.parse_live_match_state(entry, season_label, fetched_at=fetched_at) for entry in entries]

    def get_live_match_state(self, provider_fixture_id: str, season_label: str = "unknown") -> LiveMatchState:
        self.require(Capability.LIVE_SCORE)
        states = self.get_live_match_states_by_ids([provider_fixture_id], season_label)
        if not states:
            raise MalformedProviderPayload(
                f"{self.name}: no fixture returned for id {provider_fixture_id!r}"
            )
        return states[0]

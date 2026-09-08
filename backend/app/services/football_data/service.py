"""FootballDataService - current and live football facts, lazily cached.

Two clearly separate concerns:

* **Current / pre-match** (standings, fixtures, results) - cheap, cached for
  tens of minutes, served from football-data.org.
* **Live** (score, minute, events, lineups, statistics) - expensive, cached
  for `live_ttl_seconds` (default 300), served from API-Football or the
  replay provider.

The live path is strictly lazy. A provider request happens only when a user
asks AND the cached snapshot has aged past the TTL. There is no background
poller anywhere in this package; with no users there are no requests. One
refreshed snapshot answers every live question - score, minute, possession,
shots, corners, cards, substitutions - because the whole `LiveMatchState` is
cached as a unit rather than per statistic.

SEALED-SEASON BOUNDARY
----------------------
This module deals only in *current football facts*. It never imports
`backend.app.ml`, never reads the sealed 2025/26 outcomes, never regenerates
features, and never writes to `data/processed/` or `models/`. Displaying the
2026/27 table is allowed; updating ML state from it is not, and there is no
code path here that could.
"""

from __future__ import annotations

from datetime import datetime

from backend.app.core.config import CURRENT_SEASON_LABEL, FootballDataSettings, load_settings
from backend.app.services.football_data.cache import CacheEntry, TTLCache
from backend.app.services.football_data.errors import ProviderUnavailable, UnsupportedCapability
from backend.app.services.football_data.models import (
    Fixture,
    LiveMatchState,
    ProviderFreshness,
    StandingRow,
    utcnow,
)
from backend.app.services.football_data.provider import Capability, FootballDataProvider

_LIVE_MATCHES_KEY = "live:matches"
_STANDINGS_KEY = "current:standings"
_FIXTURES_KEY = "current:fixtures"


class FootballDataService:
    """Orchestrates providers + cache. Holds no football logic of its own."""

    def __init__(
        self,
        *,
        current_provider: FootballDataProvider | None = None,
        live_provider: FootballDataProvider | None = None,
        settings: FootballDataSettings | None = None,
        cache: TTLCache | None = None,
        season_label: str = CURRENT_SEASON_LABEL,
    ) -> None:
        self._settings = settings if settings is not None else load_settings()
        self._current_provider = current_provider
        self._live_provider = live_provider
        self._cache = cache if cache is not None else TTLCache()
        self._season_label = season_label

    @property
    def cache(self) -> TTLCache:
        return self._cache

    @property
    def settings(self) -> FootballDataSettings:
        return self._settings

    @property
    def live_ttl_seconds(self) -> int:
        return self._settings.live_ttl_seconds

    @property
    def season_label(self) -> str:
        return self._season_label

    @property
    def current_provider_name(self) -> str | None:
        """Name of the configured current-data provider, or None if absent."""
        return None if self._current_provider is None else self._current_provider.name

    @property
    def live_provider_name(self) -> str | None:
        """Name of the configured live provider, or None if absent."""
        return None if self._live_provider is None else self._live_provider.name

    # ---- Freshness -----------------------------------------------------
    def _freshness_from_entry(
        self, entry: CacheEntry, provider_name: str, ttl_seconds: int
    ) -> ProviderFreshness:
        return ProviderFreshness(
            provider=provider_name,
            fetched_at=entry.fetched_at,
            last_successful_refresh=entry.last_successful_refresh,
            is_stale=entry.is_stale,
            ttl_seconds=ttl_seconds,
        )

    # ---- Current / pre-match ------------------------------------------
    def _require_current_provider(self) -> FootballDataProvider:
        if self._current_provider is None:
            raise ProviderUnavailable("no current-data provider is configured")
        return self._current_provider

    def get_standings(self, *, now: datetime | None = None) -> list[StandingRow]:
        provider = self._require_current_provider()
        if Capability.STANDINGS not in provider.capabilities():
            raise UnsupportedCapability(
                f"provider {provider.name!r} cannot supply standings "
                f"(declared: {sorted(c.value for c in provider.capabilities())})"
            )
        entry = self._cache.get_or_refresh(
            _STANDINGS_KEY,
            self._settings.standings_ttl_seconds,
            lambda: provider.get_standings(self._season_label),
            now=now,
        )
        return entry.value

    def get_fixtures(self, *, now: datetime | None = None) -> list[Fixture]:
        provider = self._require_current_provider()
        if Capability.FIXTURES not in provider.capabilities():
            raise UnsupportedCapability(
                f"provider {provider.name!r} cannot supply fixtures "
                f"(declared: {sorted(c.value for c in provider.capabilities())})"
            )
        entry = self._cache.get_or_refresh(
            _FIXTURES_KEY,
            self._settings.fixtures_ttl_seconds,
            lambda: provider.get_fixtures(self._season_label),
            now=now,
        )
        return entry.value

    def get_upcoming_fixtures(
        self, *, limit: int | None = None, now: datetime | None = None
    ) -> list[Fixture]:
        reference = now if now is not None else utcnow()
        upcoming = [f for f in self.get_fixtures(now=now) if f.kickoff_utc >= reference]
        upcoming.sort(key=lambda f: f.kickoff_utc)
        return upcoming[:limit] if limit is not None else upcoming

    def get_recent_results(
        self, *, limit: int | None = None, now: datetime | None = None
    ) -> list[Fixture]:
        finished = [f for f in self.get_fixtures(now=now) if f.status.is_finished]
        finished.sort(key=lambda f: f.kickoff_utc, reverse=True)
        return finished[:limit] if limit is not None else finished

    def get_team_recent_results(
        self, canonical_team_id: str, *, limit: int = 5, now: datetime | None = None
    ) -> list[Fixture]:
        results = [
            f
            for f in self.get_recent_results(now=now)
            if canonical_team_id in (f.home_team.canonical_id, f.away_team.canonical_id)
        ]
        return results[:limit]

    # ---- Live ----------------------------------------------------------
    def _require_live_provider(self) -> FootballDataProvider:
        if self._live_provider is None:
            raise ProviderUnavailable(
                "no live provider is configured (API-Football live access is "
                "unverified until scripts/probe_live_provider.py succeeds; the "
                "replay provider can be used for demos)"
            )
        capabilities = self._live_provider.capabilities()
        if Capability.LIVE_SCORE not in capabilities:
            raise UnsupportedCapability(
                f"provider {self._live_provider.name!r} does not support live score "
                f"(declared: {sorted(c.value for c in capabilities)})"
            )
        return self._live_provider

    def get_live_matches(self, *, now: datetime | None = None) -> list[LiveMatchState]:
        """All in-play matches, lazily refreshed at most once per TTL window.

        Repeated calls inside one window perform ZERO provider requests - the
        same snapshot answers every live question asked during that window.
        """
        provider = self._require_live_provider()
        entry = self._cache.get_or_refresh(
            _LIVE_MATCHES_KEY,
            self._settings.live_ttl_seconds,
            lambda: list(provider.get_live_matches()),
            now=now,
        )
        freshness = self._freshness_from_entry(entry, provider.name, self._settings.live_ttl_seconds)
        # Re-state age on every read: the football facts are unchanged, but a
        # cached snapshot must always report its true age, never look current.
        return [state.with_freshness(freshness) for state in entry.value]

    def get_live_match_state(
        self, provider_fixture_id: str, *, now: datetime | None = None
    ) -> LiveMatchState | None:
        """One match's snapshot, served from the same shared live cache.

        Returns `None` when that fixture is not currently in play. That is a
        genuine "no live state" answer, distinct from `UnsupportedCapability`
        (provider cannot do live) and from `ProviderUnavailable` (refresh
        failed with nothing cached) - both of which raise.
        """
        for state in self.get_live_matches(now=now):
            if state.fixture.provider_fixture_id == provider_fixture_id:
                return state
        return None

    def live_cache_age_seconds(self, *, now: datetime | None = None) -> float | None:
        """Age of the live snapshot, for "last updated N seconds ago"."""
        entry = self._cache.peek(_LIVE_MATCHES_KEY)
        return None if entry is None else entry.age_seconds(now=now)

    # Cache kinds callers may ask about, so no caller has to know the internal
    # cache key strings.
    CACHE_KINDS: dict[str, str] = {
        "live": _LIVE_MATCHES_KEY,
        "standings": _STANDINGS_KEY,
        "fixtures": _FIXTURES_KEY,
    }

    def cache_entry_for(self, kind: str) -> CacheEntry | None:
        """The cached entry for one data kind, or None if nothing is cached.

        Exposed so the serving layer can report freshness/staleness provenance
        without duplicating this module's private cache-key constants.
        """
        try:
            key = self.CACHE_KINDS[kind]
        except KeyError:
            raise ValueError(
                f"unknown cache kind {kind!r}; expected one of {sorted(self.CACHE_KINDS)}"
            ) from None
        return self._cache.peek(key)

    def ttl_seconds_for(self, kind: str) -> int:
        """The configured TTL for one data kind."""
        ttls = {
            "live": self._settings.live_ttl_seconds,
            "standings": self._settings.standings_ttl_seconds,
            "fixtures": self._settings.fixtures_ttl_seconds,
        }
        try:
            return ttls[kind]
        except KeyError:
            raise ValueError(f"unknown cache kind {kind!r}; expected one of {sorted(ttls)}") from None

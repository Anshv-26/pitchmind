"""Deterministic current/live football tools.

This is the boundary a future Claude Agent SDK tool will call. Agents get
small typed functions here - they never touch provider HTTP clients, vendor
payloads, or cache internals.

Nothing in this module calls an LLM. Every number returned came from a
provider response normalized by `services.football_data`, or from the
deterministic replay script.
"""

from __future__ import annotations

from datetime import datetime

from backend.app.services.football_data.errors import ProviderUnavailable, UnknownTeam
from backend.app.services.football_data.models import Fixture, FixtureStatus
from backend.app.services.football_data.replay import ReplayProvider
from backend.app.services.football_data.service import FootballDataService
from backend.app.services.football_data.teams import TeamRegistry, default_registry
from backend.app.tools.schemas import (
    DataProvenance,
    FixturesResponse,
    LiveMatchesResponse,
    LiveMatchStateResponse,
    SourceKind,
    StandingsResponse,
)


def _provenance(
    service: FootballDataService,
    kind: str,
    provider_name: str,
    *,
    now: datetime | None = None,
) -> DataProvenance:
    """Build provenance from the service's own cache entry.

    A replay-backed provider is labelled `REPLAY` so downstream consumers
    cannot mistake scripted demo data for real live football.
    """
    entry = service.cache_entry_for(kind)
    source_kind = SourceKind.REPLAY if provider_name == ReplayProvider.name else SourceKind.REAL_PROVIDER
    if entry is None:
        return DataProvenance(
            provider=provider_name,
            source_kind=source_kind,
            ttl_seconds=service.ttl_seconds_for(kind),
        )
    return DataProvenance(
        provider=provider_name,
        source_kind=source_kind,
        fetched_at=entry.fetched_at,
        cache_age_seconds=entry.age_seconds(now=now),
        ttl_seconds=service.ttl_seconds_for(kind),
        is_stale=entry.is_stale,
        last_successful_refresh=entry.last_successful_refresh,
    )


def _require_configured_current_provider(service: FootballDataService) -> None:
    """Turn "no credentials configured" into an actionable availability error.

    Without a key the provider honestly declares an EMPTY capability set, which
    the service reports as `UnsupportedCapability`. At the serving boundary
    that reads as "this feature does not exist", when the truth is "this
    deployment is not configured" - a materially different thing for an
    operator (and for an agent deciding whether to retry elsewhere). Raise the
    configuration-shaped error instead, and let the genuine capability check
    below still apply when a key IS present.
    """
    if not service.settings.has_football_data_org_key:
        raise ProviderUnavailable(
            "current-data provider is not configured: set "
            "PITCHMIND_FOOTBALL_DATA_ORG_KEY to enable standings and fixtures"
        )


def get_current_standings(
    service: FootballDataService, *, now: datetime | None = None
) -> StandingsResponse:
    """The current Premier League table, normalized and cache-backed."""
    _require_configured_current_provider(service)
    rows = service.get_standings(now=now)
    return StandingsResponse(
        season_label=service.season_label,
        count=len(rows),
        provenance=_provenance(service, "standings", service.current_provider_name or "unconfigured", now=now),
        standings=rows,
    )


def get_fixtures(
    service: FootballDataService,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    status: FixtureStatus | None = None,
    team: str | None = None,
    limit: int | None = None,
    registry: TeamRegistry | None = None,
    now: datetime | None = None,
) -> FixturesResponse:
    """Current-season matches with typed filters.

    One function covers fixtures AND results - a "result" is simply a fixture
    with `status=FINISHED`, so a separate implementation would duplicate
    logic for no benefit.

    `team` accepts any known spelling (historical, provider, or alias) and is
    resolved through canonical identity. An unknown club raises `UnknownTeam`
    rather than being fuzzy-matched onto a different club.
    """
    _require_configured_current_provider(service)
    fixtures: list[Fixture] = service.get_fixtures(now=now)

    canonical_team_id: str | None = None
    if team is not None:
        registry = registry if registry is not None else default_registry()
        canonical_team_id = registry.resolve(team).canonical_id  # raises UnknownTeam
        fixtures = [
            f
            for f in fixtures
            if canonical_team_id in (f.home_team.canonical_id, f.away_team.canonical_id)
        ]

    if status is not None:
        fixtures = [f for f in fixtures if f.status is status]
    if date_from is not None:
        fixtures = [f for f in fixtures if f.kickoff_utc >= date_from]
    if date_to is not None:
        fixtures = [f for f in fixtures if f.kickoff_utc <= date_to]

    fixtures.sort(key=lambda f: f.kickoff_utc)
    if limit is not None:
        fixtures = fixtures[:limit]

    return FixturesResponse(
        season_label=service.season_label,
        count=len(fixtures),
        provenance=_provenance(service, "fixtures", service.current_provider_name or "unconfigured", now=now),
        filters_applied={
            "team": canonical_team_id,
            "status": status.value if status is not None else None,
            "date_from": date_from.isoformat() if date_from is not None else None,
            "date_to": date_to.isoformat() if date_to is not None else None,
        },
        fixtures=fixtures,
    )


def get_live_matches(
    service: FootballDataService, *, now: datetime | None = None
) -> LiveMatchesResponse:
    """Every match currently in play, from one lazily-refreshed snapshot.

    Provenance says explicitly whether this is a real provider or the
    deterministic replay script. Real live data is currently served by NO
    provider: API-Football's free tier was verified not to expose the current
    Premier League season, so live is replay-backed for development.
    """
    matches = service.get_live_matches(now=now)
    return LiveMatchesResponse(
        count=len(matches),
        provenance=_provenance(service, "live", service.live_provider_name or "unconfigured", now=now),
        matches=matches,
    )


def get_live_match_state(
    service: FootballDataService, provider_fixture_id: str, *, now: datetime | None = None
) -> LiveMatchStateResponse:
    """One match's live snapshot, served from the same shared cache.

    `match=None` means that fixture is not currently in play - a genuine
    answer, not an error.
    """
    state = service.get_live_match_state(provider_fixture_id, now=now)
    return LiveMatchStateResponse(
        provenance=_provenance(service, "live", service.live_provider_name or "unconfigured", now=now),
        match=state,
    )


__all__ = [
    "get_current_standings",
    "get_fixtures",
    "get_live_matches",
    "get_live_match_state",
    "UnknownTeam",
]

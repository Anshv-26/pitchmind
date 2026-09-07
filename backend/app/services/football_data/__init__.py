"""Current and live Premier League data for PitchMind.

Provider-neutral: nothing outside this package sees a vendor response shape.
Deliberately independent of `backend.app.ml` - this layer reports current
football facts and never touches frozen models, features, or the sealed
2025/26 season.
"""

from backend.app.services.football_data.cache import CacheEntry, TTLCache
from backend.app.services.football_data.errors import (
    CurrentSeasonNotAvailable,
    FootballDataError,
    MalformedProviderPayload,
    ProviderRateLimited,
    ProviderUnavailable,
    UnknownTeam,
    UnsupportedCapability,
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
    StandingRow,
    TeamRef,
)
from backend.app.services.football_data.provider import (
    BaseProvider,
    Capability,
    FootballDataProvider,
)
from backend.app.services.football_data.service import FootballDataService
from backend.app.services.football_data.teams import (
    CanonicalTeam,
    TeamRegistry,
    default_registry,
    normalize_team_name,
)

__all__ = [
    "CacheEntry",
    "TTLCache",
    "CurrentSeasonNotAvailable",
    "FootballDataError",
    "MalformedProviderPayload",
    "ProviderRateLimited",
    "ProviderUnavailable",
    "UnknownTeam",
    "UnsupportedCapability",
    "Fixture",
    "FixtureStatus",
    "Lineup",
    "LineupPlayer",
    "LiveMatchState",
    "MatchEvent",
    "MatchEventType",
    "MatchStats",
    "ProviderFreshness",
    "StandingRow",
    "TeamRef",
    "BaseProvider",
    "Capability",
    "FootballDataProvider",
    "FootballDataService",
    "CanonicalTeam",
    "TeamRegistry",
    "default_registry",
    "normalize_team_name",
]

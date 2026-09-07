"""The provider contract every football-data source implements.

Deliberately small. Methods exist because PitchMind needs the data, not to
mirror a vendor's endpoint list: API-Football returns score, events, lineups
and statistics for a fixture in one response, so those become fields on a
single `LiveMatchState` rather than four separate calls. That keeps one
refresh able to answer many different user questions.

Capabilities are declared, not inferred. A provider that cannot do live
scores says so via `capabilities()`, and calling the method raises
`UnsupportedCapability`. It must never return `[]`, because "this provider
cannot do live" and "there is no live football right now" are different
answers and confusing them would let PitchMind mislead a user.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from backend.app.services.football_data.errors import UnsupportedCapability
from backend.app.services.football_data.models import Fixture, LiveMatchState, StandingRow


class Capability(str, Enum):
    FIXTURES = "fixtures"
    STANDINGS = "standings"
    LIVE_SCORE = "live_score"
    EVENTS = "events"
    LINEUPS = "lineups"
    MATCH_STATISTICS = "match_statistics"
    PLAYER_STATISTICS = "player_statistics"
    INJURIES = "injuries"


@runtime_checkable
class FootballDataProvider(Protocol):
    """Structural contract. Adapters need not inherit from anything."""

    name: str

    def capabilities(self) -> frozenset[Capability]: ...

    def get_standings(self, season_label: str) -> list[StandingRow]: ...

    def get_fixtures(self, season_label: str) -> list[Fixture]: ...

    def get_live_matches(self) -> list[LiveMatchState]: ...

    def get_live_match_state(self, provider_fixture_id: str) -> LiveMatchState: ...


class BaseProvider:
    """Shared capability enforcement.

    Subclasses declare `_CAPABILITIES` and implement only what they support;
    every unsupported method raises `UnsupportedCapability` by default.
    """

    name: str = "base"
    _CAPABILITIES: frozenset[Capability] = frozenset()

    def capabilities(self) -> frozenset[Capability]:
        return self._CAPABILITIES

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities()

    def require(self, capability: Capability) -> None:
        """Guard called at the top of every capability-gated method."""
        if not self.supports(capability):
            raise UnsupportedCapability(
                f"provider {self.name!r} does not support {capability.value!r}; "
                f"declared capabilities: {sorted(c.value for c in self.capabilities())}"
            )

    # Default implementations: refuse rather than fabricate.
    def get_standings(self, season_label: str) -> list[StandingRow]:
        self.require(Capability.STANDINGS)
        raise NotImplementedError

    def get_fixtures(self, season_label: str) -> list[Fixture]:
        self.require(Capability.FIXTURES)
        raise NotImplementedError

    def get_live_matches(self) -> list[LiveMatchState]:
        self.require(Capability.LIVE_SCORE)
        raise NotImplementedError

    def get_live_match_state(self, provider_fixture_id: str) -> LiveMatchState:
        self.require(Capability.LIVE_SCORE)
        raise NotImplementedError

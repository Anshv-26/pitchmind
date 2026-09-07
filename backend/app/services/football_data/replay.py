"""Deterministic replay provider - live behaviour without a live match.

Purpose: demonstrate PitchMind's live features when no PL match is in play,
when the API quota is unavailable, or during an interview demo where a real
live match cannot be relied upon.

Critically, this is **not a separate fake code path**. The replay provider
implements the same `FootballDataProvider` contract and emits the same
`LiveMatchState` schema as API-Football, so the cache, the service, and every
later tool/agent behave identically. Nothing downstream knows or cares which
provider produced a snapshot.

Progression is fully deterministic: the same step index always yields the same
snapshot, so demos and tests are reproducible.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend.app.services.football_data.models import (
    Fixture,
    FixtureStatus,
    LiveMatchState,
    MatchEvent,
    MatchEventType,
    MatchStats,
    ProviderFreshness,
    utcnow,
)
from backend.app.services.football_data.provider import BaseProvider, Capability
from backend.app.services.football_data.teams import TeamRegistry, default_registry

REPLAY_SEASON_LABEL = "replay"
REPLAY_FIXTURE_ID = "replay-1"
REPLAY_KICKOFF = datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc)


class ReplayProvider(BaseProvider):
    """Steps through a scripted match. `advance()` moves one step forward."""

    name = "replay"
    _CAPABILITIES = frozenset(
        {
            Capability.FIXTURES,
            Capability.LIVE_SCORE,
            Capability.EVENTS,
            Capability.MATCH_STATISTICS,
        }
    )

    def __init__(
        self,
        *,
        registry: TeamRegistry | None = None,
        home_team: str = "Arsenal",
        away_team: str = "Liverpool",
        step: int = 0,
    ) -> None:
        self._registry = registry if registry is not None else default_registry()
        self._home_name = home_team
        self._away_name = away_team
        self._step = step

    # ---- Replay control ------------------------------------------------
    @property
    def step(self) -> int:
        return self._step

    @property
    def step_count(self) -> int:
        return len(_SCRIPT)

    def advance(self, steps: int = 1) -> None:
        """Move forward through the script, clamping at the final step."""
        self._step = min(self._step + steps, len(_SCRIPT) - 1)

    def reset(self) -> None:
        self._step = 0

    def set_step(self, step: int) -> None:
        if not 0 <= step < len(_SCRIPT):
            raise IndexError(f"replay step must be in [0, {len(_SCRIPT) - 1}], got {step}")
        self._step = step

    # ---- Snapshot construction ----------------------------------------
    def _fixture(self, status: FixtureStatus, home_score: int, away_score: int) -> Fixture:
        return Fixture(
            provider_fixture_id=REPLAY_FIXTURE_ID,
            competition_code="PL",
            season_label=REPLAY_SEASON_LABEL,
            home_team=self._registry.team_ref(self._home_name, provider=self.name),
            away_team=self._registry.team_ref(self._away_name, provider=self.name),
            kickoff_utc=REPLAY_KICKOFF,
            status=status,
            home_score=home_score,
            away_score=away_score,
        )

    def _state_for_step(self, step: int, *, fetched_at: datetime | None = None) -> LiveMatchState:
        script = _SCRIPT[step]
        timestamp = fetched_at if fetched_at is not None else utcnow()
        fixture = self._fixture(script["status"], script["home_score"], script["away_score"])

        home_id = fixture.home_team.canonical_id
        away_id = fixture.away_team.canonical_id
        events = tuple(
            MatchEvent(
                type=event["type"],
                minute=event["minute"],
                team_canonical_id=home_id if event["team"] == "home" else away_id,
                player_name=event.get("player"),
                detail=event.get("detail"),
            )
            for event in script["events"]
        )

        return LiveMatchState(
            fixture=fixture,
            provider=self.name,
            minute=script["minute"],
            home_score=script["home_score"],
            away_score=script["away_score"],
            home_stats=MatchStats(**script["home_stats"]),
            away_stats=MatchStats(**script["away_stats"]),
            events=events,
            freshness=ProviderFreshness(
                provider=self.name,
                fetched_at=timestamp,
                last_successful_refresh=timestamp,
                is_stale=False,
            ),
        )

    # ---- Provider interface -------------------------------------------
    def get_fixtures(self, season_label: str) -> list[Fixture]:
        self.require(Capability.FIXTURES)
        script = _SCRIPT[self._step]
        return [self._fixture(script["status"], script["home_score"], script["away_score"])]

    def get_live_matches(self, season_label: str | None = None) -> list[LiveMatchState]:
        self.require(Capability.LIVE_SCORE)
        state = self._state_for_step(self._step)
        return [state] if state.fixture.status.is_in_play else []

    def get_live_match_state(
        self, provider_fixture_id: str = REPLAY_FIXTURE_ID, season_label: str = REPLAY_SEASON_LABEL
    ) -> LiveMatchState:
        self.require(Capability.LIVE_SCORE)
        return self._state_for_step(self._step)


# The scripted match. Statistics deliberately include `None` entries so demos
# exercise the "provider did not report this" path rather than always showing
# a number.
_SCRIPT: tuple[dict, ...] = (
    {
        "minute": 20,
        "status": FixtureStatus.LIVE,
        "home_score": 0,
        "away_score": 0,
        "home_stats": {"shots": 3, "shots_on_target": 1, "possession_percent": 54.0, "corners": 2},
        "away_stats": {"shots": 4, "shots_on_target": 2, "possession_percent": 46.0, "corners": 1},
        "events": (),
    },
    {
        "minute": 35,
        "status": FixtureStatus.LIVE,
        "home_score": 1,
        "away_score": 0,
        "home_stats": {"shots": 6, "shots_on_target": 3, "possession_percent": 56.0, "corners": 3},
        "away_stats": {"shots": 5, "shots_on_target": 2, "possession_percent": 44.0, "corners": 1},
        "events": (
            {"type": MatchEventType.GOAL, "minute": 31, "team": "home", "player": "Replay Forward"},
        ),
    },
    {
        "minute": 45,
        "status": FixtureStatus.HALF_TIME,
        "home_score": 1,
        "away_score": 0,
        "home_stats": {"shots": 7, "shots_on_target": 3, "possession_percent": 55.0, "corners": 3},
        "away_stats": {"shots": 6, "shots_on_target": 2, "possession_percent": 45.0, "corners": 2},
        "events": (
            {"type": MatchEventType.GOAL, "minute": 31, "team": "home", "player": "Replay Forward"},
            {
                "type": MatchEventType.YELLOW_CARD,
                "minute": 40,
                "team": "away",
                "player": "Replay Midfielder",
            },
        ),
    },
    {
        "minute": 60,
        "status": FixtureStatus.LIVE,
        "home_score": 1,
        "away_score": 1,
        # Corners deliberately unreported here: exercises None != 0.
        "home_stats": {"shots": 9, "shots_on_target": 4, "possession_percent": 52.0, "corners": None},
        "away_stats": {"shots": 9, "shots_on_target": 5, "possession_percent": 48.0, "corners": None},
        "events": (
            {"type": MatchEventType.GOAL, "minute": 31, "team": "home", "player": "Replay Forward"},
            {
                "type": MatchEventType.YELLOW_CARD,
                "minute": 40,
                "team": "away",
                "player": "Replay Midfielder",
            },
            {"type": MatchEventType.GOAL, "minute": 54, "team": "away", "player": "Replay Striker"},
            {
                "type": MatchEventType.SUBSTITUTION,
                "minute": 58,
                "team": "home",
                "player": "Replay Substitute",
            },
        ),
    },
    {
        "minute": 90,
        "status": FixtureStatus.FINISHED,
        "home_score": 2,
        "away_score": 1,
        "home_stats": {
            "shots": 14,
            "shots_on_target": 6,
            "possession_percent": 53.0,
            "corners": 7,
            "yellow_cards": 1,
        },
        "away_stats": {
            "shots": 11,
            "shots_on_target": 5,
            "possession_percent": 47.0,
            "corners": 4,
            "yellow_cards": 2,
        },
        "events": (
            {"type": MatchEventType.GOAL, "minute": 31, "team": "home", "player": "Replay Forward"},
            {
                "type": MatchEventType.YELLOW_CARD,
                "minute": 40,
                "team": "away",
                "player": "Replay Midfielder",
            },
            {"type": MatchEventType.GOAL, "minute": 54, "team": "away", "player": "Replay Striker"},
            {
                "type": MatchEventType.SUBSTITUTION,
                "minute": 58,
                "team": "home",
                "player": "Replay Substitute",
            },
            {"type": MatchEventType.GOAL, "minute": 77, "team": "home", "player": "Replay Substitute"},
        ),
    },
)

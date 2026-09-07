"""Provider-neutral schemas for current and live football data.

Nothing outside this package should ever see a provider's raw response shape.
Adapters parse vendor JSON into these models; the service, tools, and (later)
the API and agent layer depend only on what is here.

Two rules drive the design:

1. **A missing statistic is `None`, never `0`.** "The provider did not supply
   shots-on-target" and "there have been zero shots on target" are different
   football facts, and collapsing them would let PitchMind state something
   untrue. Every field in `MatchStats` is therefore optional, and nothing in
   this module defaults a statistic to zero.

2. **Every fetched object carries its own freshness.** `ProviderFreshness`
   travels with the data, so a consumer can always say "last updated 2 minutes
   ago" and can never accidentally present a stale snapshot as current.

All datetimes are timezone-aware (UTC). Naive datetimes are rejected by
validation rather than silently assumed to be UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    """Timezone-aware current time. Single source so tests can monkeypatch it."""
    return datetime.now(timezone.utc)


class _Base(BaseModel):
    """Immutable, extra-rejecting base. `extra="forbid"` means a provider
    field we did not model shows up as a loud validation error during
    development rather than being silently dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class FixtureStatus(str, Enum):
    """Normalized match status across providers.

    `UNKNOWN` exists so an unrecognised provider status is representable
    without crashing and without being misreported as SCHEDULED or FINISHED -
    the two statuses a wrong guess would most damage.
    """

    SCHEDULED = "SCHEDULED"
    LIVE = "LIVE"
    HALF_TIME = "HALF_TIME"
    EXTRA_TIME = "EXTRA_TIME"
    PENALTIES = "PENALTIES"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    SUSPENDED = "SUSPENDED"
    ABANDONED = "ABANDONED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_in_play(self) -> bool:
        """True only while the ball is actually in play (or at the break).
        Drives whether the service is allowed to spend a live provider call."""
        return self in {
            FixtureStatus.LIVE,
            FixtureStatus.HALF_TIME,
            FixtureStatus.EXTRA_TIME,
            FixtureStatus.PENALTIES,
        }

    @property
    def is_finished(self) -> bool:
        return self is FixtureStatus.FINISHED

    @property
    def is_not_played(self) -> bool:
        """Postponed/suspended/abandoned/cancelled: there is no meaningful
        live state to refresh, and no result to report."""
        return self in {
            FixtureStatus.POSTPONED,
            FixtureStatus.SUSPENDED,
            FixtureStatus.ABANDONED,
            FixtureStatus.CANCELLED,
        }


class MatchEventType(str, Enum):
    GOAL = "GOAL"
    OWN_GOAL = "OWN_GOAL"
    PENALTY_GOAL = "PENALTY_GOAL"
    PENALTY_MISSED = "PENALTY_MISSED"
    YELLOW_CARD = "YELLOW_CARD"
    SECOND_YELLOW_CARD = "SECOND_YELLOW_CARD"
    RED_CARD = "RED_CARD"
    SUBSTITUTION = "SUBSTITUTION"
    VAR = "VAR"
    UNKNOWN = "UNKNOWN"


class TeamRef(_Base):
    """A canonical team reference. `canonical_id` is PitchMind's own stable
    identifier (see `teams.py`) - never a provider id, so the same club is the
    same object across historical data, current data and live data."""

    canonical_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    provider_team_id: str | None = None


class ProviderFreshness(_Base):
    """Travels with every fetched object so freshness is never guesswork.

    `is_stale` means the value is older than its TTL and a refresh was either
    not attempted or failed - it must be surfaced to the user, not hidden.
    """

    provider: str = Field(min_length=1)
    fetched_at: datetime
    last_successful_refresh: datetime | None = None
    is_stale: bool = False
    ttl_seconds: int | None = None
    provider_timestamp: datetime | None = None

    @field_validator("fetched_at", "last_successful_refresh", "provider_timestamp")
    @classmethod
    def _require_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("datetimes must be timezone-aware")
        return value

    def age_seconds(self, *, now: datetime | None = None) -> float:
        """Seconds since this data was fetched. Never negative."""
        reference = now if now is not None else utcnow()
        return max(0.0, (reference - self.fetched_at).total_seconds())


class Fixture(_Base):
    """A scheduled, in-progress or completed match."""

    provider_fixture_id: str = Field(min_length=1)
    competition_code: str = Field(min_length=1)
    season_label: str = Field(min_length=1)
    home_team: TeamRef
    away_team: TeamRef
    kickoff_utc: datetime
    status: FixtureStatus = FixtureStatus.UNKNOWN
    matchday: int | None = None
    home_score: int | None = None
    away_score: int | None = None

    @field_validator("kickoff_utc")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("kickoff_utc must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _reject_same_team(self) -> "Fixture":
        if self.home_team.canonical_id == self.away_team.canonical_id:
            raise ValueError("home and away team must differ")
        return self


class StandingRow(_Base):
    """One row of a league table."""

    position: int = Field(ge=1)
    team: TeamRef
    played: int = Field(ge=0)
    won: int = Field(ge=0)
    drawn: int = Field(ge=0)
    lost: int = Field(ge=0)
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    goal_difference: int
    points: int


class MatchEvent(_Base):
    """A discrete in-match occurrence. `minute` is optional because providers
    sometimes omit it for pre-match or administrative events."""

    type: MatchEventType
    minute: int | None = Field(default=None, ge=0)
    team_canonical_id: str | None = None
    player_name: str | None = None
    assist_name: str | None = None
    detail: str | None = None


class MatchStats(_Base):
    """Per-side match statistics.

    EVERY field is optional and defaults to `None`. A provider that omits a
    statistic must leave it `None`; writing 0 would assert a football fact
    that was never reported. Consumers must render `None` as "not reported",
    never as zero.
    """

    shots: int | None = Field(default=None, ge=0)
    shots_on_target: int | None = Field(default=None, ge=0)
    possession_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    corners: int | None = Field(default=None, ge=0)
    fouls: int | None = Field(default=None, ge=0)
    yellow_cards: int | None = Field(default=None, ge=0)
    red_cards: int | None = Field(default=None, ge=0)
    offsides: int | None = Field(default=None, ge=0)
    passes: int | None = Field(default=None, ge=0)
    pass_accuracy_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    saves: int | None = Field(default=None, ge=0)


class LineupPlayer(_Base):
    name: str = Field(min_length=1)
    number: int | None = Field(default=None, ge=0)
    position: str | None = None


class Lineup(_Base):
    """One team's lineup. Empty lists mean "provider reported none"; a lineup
    that was never reported at all is represented by omitting the Lineup."""

    team_canonical_id: str = Field(min_length=1)
    formation: str | None = None
    starting_xi: tuple[LineupPlayer, ...] = ()
    substitutes: tuple[LineupPlayer, ...] = ()
    coach_name: str | None = None


class LiveMatchState(_Base):
    """One normalized snapshot of a match, whatever its source.

    This is the single shape the rest of PitchMind consumes. A snapshot from
    API-Football and a snapshot from the deterministic replay provider are
    indistinguishable to consumers by design - there is no separate "demo"
    code path.

    One snapshot answers many questions (score, minute, possession, shots,
    cards, subs, events), which is exactly why the service refreshes all of
    these fields together rather than per-statistic.
    """

    fixture: Fixture
    provider: str = Field(min_length=1)
    minute: int | None = Field(default=None, ge=0)
    home_score: int | None = Field(default=None, ge=0)
    away_score: int | None = Field(default=None, ge=0)
    home_stats: MatchStats | None = None
    away_stats: MatchStats | None = None
    events: tuple[MatchEvent, ...] = ()
    home_lineup: Lineup | None = None
    away_lineup: Lineup | None = None
    freshness: ProviderFreshness

    @property
    def status(self) -> FixtureStatus:
        return self.fixture.status

    @property
    def is_in_play(self) -> bool:
        return self.fixture.status.is_in_play

    def with_freshness(self, freshness: ProviderFreshness) -> "LiveMatchState":
        """Return a copy carrying updated freshness. Used when serving a
        cached snapshot that must now be flagged stale - the football facts
        are unchanged, only their age is re-stated."""
        return self.model_copy(update={"freshness": freshness})

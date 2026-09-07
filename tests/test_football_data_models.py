"""Schema and team-identity tests for the current/live football data layer.

Fully offline: no network, no API key, no provider contact.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.football_data.errors import UnknownTeam  # noqa: E402
from backend.app.services.football_data.models import (  # noqa: E402
    Fixture,
    FixtureStatus,
    LiveMatchState,
    MatchEvent,
    MatchEventType,
    MatchStats,
    ProviderFreshness,
    StandingRow,
    TeamRef,
)
from backend.app.services.football_data.teams import (  # noqa: E402
    CanonicalTeam,
    TeamRegistry,
    default_registry,
    normalize_team_name,
)

UTC_NOW = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)


def _team(canonical_id: str, name: str) -> TeamRef:
    return TeamRef(canonical_id=canonical_id, canonical_name=name)


def _fixture(**overrides) -> Fixture:
    base = dict(
        provider_fixture_id="1",
        competition_code="PL",
        season_label="2026_27",
        home_team=_team("arsenal", "Arsenal"),
        away_team=_team("liverpool", "Liverpool"),
        kickoff_utc=UTC_NOW,
        status=FixtureStatus.LIVE,
    )
    base.update(overrides)
    return Fixture(**base)


# --------------------------------------------------------------------------
# Timezone awareness
# --------------------------------------------------------------------------
def test_naive_kickoff_is_rejected():
    with pytest.raises(ValidationError):
        _fixture(kickoff_utc=datetime(2026, 9, 5, 15, 0))  # no tzinfo


def test_naive_freshness_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        ProviderFreshness(provider="x", fetched_at=datetime(2026, 9, 5, 15, 0))


def test_timezone_aware_timestamps_are_accepted():
    freshness = ProviderFreshness(provider="x", fetched_at=UTC_NOW)
    assert freshness.fetched_at.tzinfo is not None


# --------------------------------------------------------------------------
# MatchStats: missing is None, never 0
# --------------------------------------------------------------------------
def test_match_stats_default_to_none_not_zero():
    stats = MatchStats()
    for field in MatchStats.model_fields:
        assert getattr(stats, field) is None, f"{field} defaulted to a value instead of None"


def test_missing_statistic_is_distinguishable_from_zero():
    unreported = MatchStats(shots=5)
    genuinely_zero = MatchStats(shots=5, corners=0)
    assert unreported.corners is None
    assert genuinely_zero.corners == 0
    assert unreported.corners != genuinely_zero.corners


def test_match_stats_reject_impossible_values():
    with pytest.raises(ValidationError):
        MatchStats(shots=-1)
    with pytest.raises(ValidationError):
        MatchStats(possession_percent=101.0)


# --------------------------------------------------------------------------
# Fixture / status semantics
# --------------------------------------------------------------------------
def test_fixture_rejects_same_team_on_both_sides():
    with pytest.raises(ValidationError):
        _fixture(away_team=_team("arsenal", "Arsenal"))


@pytest.mark.parametrize(
    "status",
    [FixtureStatus.LIVE, FixtureStatus.HALF_TIME, FixtureStatus.EXTRA_TIME, FixtureStatus.PENALTIES],
)
def test_in_play_statuses(status):
    assert status.is_in_play


@pytest.mark.parametrize(
    "status",
    [FixtureStatus.POSTPONED, FixtureStatus.SUSPENDED, FixtureStatus.ABANDONED, FixtureStatus.CANCELLED],
)
def test_not_played_statuses_are_not_in_play(status):
    assert status.is_not_played
    assert not status.is_in_play
    assert not status.is_finished


def test_unknown_status_is_neither_scheduled_nor_finished():
    """An unrecognised provider status must not be guessed into a status that
    would mislead - it is explicitly UNKNOWN."""
    assert FixtureStatus.UNKNOWN is not FixtureStatus.SCHEDULED
    assert not FixtureStatus.UNKNOWN.is_finished
    assert not FixtureStatus.UNKNOWN.is_in_play


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------
def test_freshness_age_is_never_negative():
    freshness = ProviderFreshness(provider="x", fetched_at=UTC_NOW)
    assert freshness.age_seconds(now=UTC_NOW - timedelta(seconds=30)) == 0.0


def test_freshness_age_computes_correctly():
    freshness = ProviderFreshness(provider="x", fetched_at=UTC_NOW)
    assert freshness.age_seconds(now=UTC_NOW + timedelta(seconds=94)) == pytest.approx(94.0)


def test_live_state_json_serializable():
    state = LiveMatchState(
        fixture=_fixture(),
        provider="replay",
        minute=63,
        home_score=1,
        away_score=0,
        home_stats=MatchStats(shots=9),
        events=(MatchEvent(type=MatchEventType.GOAL, minute=31),),
        freshness=ProviderFreshness(provider="replay", fetched_at=UTC_NOW),
    )
    decoded = json.loads(state.model_dump_json())
    assert decoded["minute"] == 63
    assert decoded["home_stats"]["corners"] is None  # missing stays null in JSON


def test_with_freshness_preserves_football_facts():
    state = LiveMatchState(
        fixture=_fixture(),
        provider="replay",
        minute=63,
        home_score=1,
        away_score=0,
        freshness=ProviderFreshness(provider="replay", fetched_at=UTC_NOW),
    )
    stale = state.with_freshness(
        ProviderFreshness(provider="replay", fetched_at=UTC_NOW, is_stale=True)
    )
    assert stale.freshness.is_stale is True
    assert stale.model_dump(exclude={"freshness"}) == state.model_dump(exclude={"freshness"})


def test_standing_row_validates():
    row = StandingRow(
        position=1, team=_team("arsenal", "Arsenal"), played=5, won=4, drawn=1, lost=0,
        goals_for=12, goals_against=3, goal_difference=9, points=13,
    )
    assert row.points == 13
    with pytest.raises(ValidationError):
        StandingRow(
            position=0, team=_team("arsenal", "Arsenal"), played=5, won=4, drawn=1, lost=0,
            goals_for=12, goals_against=3, goal_difference=9, points=13,
        )


# --------------------------------------------------------------------------
# Team identity
# --------------------------------------------------------------------------
def test_all_historical_names_resolve():
    """Every club name in the frozen historical dataset must resolve."""
    registry = default_registry()
    historical = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley",
        "Cardiff", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Huddersfield",
        "Hull", "Ipswich", "Leeds", "Leicester", "Liverpool", "Luton", "Man City",
        "Man United", "Middlesbrough", "Newcastle", "Norwich", "Nott'm Forest",
        "Sheffield United", "Southampton", "Stoke", "Sunderland", "Swansea",
        "Tottenham", "Watford", "West Brom", "West Ham", "Wolves",
    ]
    assert len(historical) == 34
    for name in historical:
        assert registry.resolve(name).canonical_id


@pytest.mark.parametrize(
    "provider_name,expected",
    [
        ("Manchester City", "man_city"),
        ("Manchester City FC", "man_city"),
        ("Manchester United", "man_united"),
        ("Man Utd", "man_united"),
        ("Nottingham Forest", "nottingham_forest"),
        ("Nott'm Forest", "nottingham_forest"),
        ("Brighton & Hove Albion FC", "brighton"),
        ("Wolverhampton Wanderers", "wolves"),
        ("Spurs", "tottenham"),
        ("AFC Bournemouth", "bournemouth"),
        ("West Bromwich Albion", "west_brom"),
        ("Sheffield Utd", "sheffield_united"),
    ],
)
def test_provider_spellings_resolve_to_canonical_ids(provider_name, expected):
    assert default_registry().resolve(provider_name).canonical_id == expected


def test_normalization_strips_suffixes_and_punctuation():
    assert normalize_team_name("Nott'm Forest") == "nottm forest"
    assert normalize_team_name("Manchester City FC") == "manchester city"
    assert normalize_team_name("  Arsenal  AFC ") == "arsenal"


def test_normalization_removes_straight_apostrophe_without_splitting_the_word():
    """An apostrophe must be REMOVED, not turned into a space - turning it
    into a space would incorrectly split one token into two."""
    assert normalize_team_name("Nott'm Forest") == "nottm forest"
    assert "nott m forest" != normalize_team_name("Nott'm Forest")


def test_normalization_removes_curly_apostrophe_the_same_way():
    assert normalize_team_name("Nott’m Forest") == "nottm forest"
    assert normalize_team_name("Nott’m Forest") == normalize_team_name("Nott'm Forest")


def test_normalization_collapses_repeated_whitespace():
    assert normalize_team_name("Nott'm    Forest") == "nottm forest"
    assert normalize_team_name("  Nott'm Forest  ") == "nottm forest"
    assert normalize_team_name("Nott'm\tForest") == "nottm forest"


def test_normalization_still_separates_unrelated_punctuation_with_a_space():
    """Non-apostrophe punctuation (e.g. '&') is a genuine word separator and
    must still become a space, not be removed."""
    assert normalize_team_name("Brighton & Hove Albion") == "brighton hove albion"
    assert normalize_team_name("Brighton&Hove Albion") == "brighton hove albion"


def test_known_aliases_still_resolve_after_the_apostrophe_fix():
    registry = default_registry()
    assert registry.resolve("Nott'm Forest").canonical_id == "nottingham_forest"
    assert registry.resolve("Nottingham Forest").canonical_id == "nottingham_forest"
    assert registry.resolve("Nottm Forest").canonical_id == "nottingham_forest"
    assert registry.resolve("Notts Forest").canonical_id == "nottingham_forest"
    assert registry.resolve("Manchester City FC").canonical_id == "man_city"
    assert registry.resolve("Spurs").canonical_id == "tottenham"


def test_unknown_team_raises_and_names_the_value():
    with pytest.raises(UnknownTeam) as excinfo:
        default_registry().resolve("Wrexham")
    assert "Wrexham" in str(excinfo.value)


def test_no_fuzzy_matching_between_similar_clubs():
    """Sheffield Wednesday must NOT be silently matched to Sheffield United."""
    registry = default_registry()
    assert registry.resolve("Sheffield United").canonical_id == "sheffield_united"
    with pytest.raises(UnknownTeam):
        registry.resolve("Sheffield Wednesday")


def test_current_only_team_has_identity_without_historical_history():
    """A promoted club can exist canonically while having no ML history -
    the expected case for 2026/27 newcomers."""
    registry = default_registry()
    registry.register(
        CanonicalTeam(
            canonical_id="newcomer_fc",
            canonical_name="Newcomer FC",
            aliases=("Newcomer",),
            historical_ml_history_available=False,
        )
    )
    assert registry.resolve("Newcomer").canonical_id == "newcomer_fc"
    assert registry.has_historical_ml_history("newcomer_fc") is False
    assert registry.has_historical_ml_history("arsenal") is True


def test_ambiguous_alias_registration_is_refused():
    registry = TeamRegistry()
    with pytest.raises(ValueError, match="already mapped"):
        registry.register(CanonicalTeam("fake", "Fake Club", aliases=("Arsenal FC",)))


def test_duplicate_canonical_id_is_refused():
    registry = TeamRegistry()
    with pytest.raises(ValueError, match="duplicate canonical_id"):
        registry.register(CanonicalTeam("arsenal", "Arsenal Again"))


def test_resolve_by_provider_id_raises_until_ids_are_populated():
    """Provider numeric ids are not guessed; resolving one before it is
    verified must fail loudly rather than return the wrong club."""
    with pytest.raises(UnknownTeam):
        default_registry().resolve_by_provider_id("api_football", 50)

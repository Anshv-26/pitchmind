"""Provider adapter tests - capability enforcement, normalization, errors.

Every provider request in this file is served by a mocked httpx transport.
No network, no API key, no real provider is ever contacted.
"""

from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import FootballDataSettings  # noqa: E402
from backend.app.services.football_data.api_football import (  # noqa: E402
    ApiFootballProvider,
    parse_match_stats,
)
from backend.app.services.football_data.errors import (  # noqa: E402
    MalformedProviderPayload,
    ProviderRateLimited,
    ProviderUnavailable,
    UnsupportedCapability,
)
from backend.app.services.football_data.football_data_org import FootballDataOrgProvider  # noqa: E402
from backend.app.services.football_data.models import FixtureStatus  # noqa: E402
from backend.app.services.football_data.provider import Capability  # noqa: E402


def make_settings(**overrides) -> FootballDataSettings:
    base = dict(
        football_data_org_key="test-key",
        api_football_key="test-key",
        live_ttl_seconds=300,
        standings_ttl_seconds=1800,
        fixtures_ttl_seconds=1800,
        metadata_ttl_seconds=21600,
        http_connect_timeout=5.0,
        http_read_timeout=10.0,
    )
    base.update(overrides)
    return FootballDataSettings(**base)


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def json_handler(payload, status_code: int = 200, headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload, headers=headers or {})

    return handler


# --------------------------------------------------------------------------
# football-data.org
# --------------------------------------------------------------------------
FD_ORG_MATCHES = {
    "matches": [
        {
            "id": 501,
            "utcDate": "2026-09-05T14:00:00Z",
            "status": "FINISHED",
            "matchday": 4,
            "homeTeam": {"id": 57, "name": "Arsenal FC"},
            "awayTeam": {"id": 64, "name": "Liverpool FC"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        },
        {
            "id": 502,
            "utcDate": "2026-09-06T13:00:00Z",
            "status": "POSTPONED",
            "matchday": 4,
            "homeTeam": {"id": 65, "name": "Manchester City FC"},
            "awayTeam": {"id": 66, "name": "Manchester United FC"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
    ]
}

FD_ORG_STANDINGS = {
    "standings": [
        {
            "type": "TOTAL",
            "table": [
                {
                    "position": 1,
                    "team": {"id": 57, "name": "Arsenal FC"},
                    "playedGames": 4,
                    "won": 3,
                    "draw": 1,
                    "lost": 0,
                    "goalsFor": 9,
                    "goalsAgainst": 2,
                    "goalDifference": 7,
                    "points": 10,
                }
            ],
        }
    ]
}


def test_football_data_org_declares_only_verified_free_capabilities():
    """The free contract has no live scores, lineups, events or statistics -
    the adapter must not claim them."""
    provider = FootballDataOrgProvider(make_settings())
    capabilities = provider.capabilities()
    assert capabilities == frozenset({Capability.FIXTURES, Capability.STANDINGS})
    for forbidden in (
        Capability.LIVE_SCORE,
        Capability.EVENTS,
        Capability.LINEUPS,
        Capability.MATCH_STATISTICS,
    ):
        assert forbidden not in capabilities


def test_football_data_org_live_raises_unsupported_not_empty_list():
    """Unsupported must raise, never return [] - an empty list would be
    indistinguishable from 'no matches are live right now'."""
    provider = FootballDataOrgProvider(make_settings())
    with pytest.raises(UnsupportedCapability):
        provider.get_live_matches()
    with pytest.raises(UnsupportedCapability):
        provider.get_live_match_state("1")


def test_football_data_org_without_key_has_no_capabilities():
    provider = FootballDataOrgProvider(make_settings(football_data_org_key=None))
    assert provider.capabilities() == frozenset()
    with pytest.raises(UnsupportedCapability):
        provider.get_fixtures("2026_27")


def test_football_data_org_normalizes_fixtures():
    provider = FootballDataOrgProvider(
        make_settings(), client=mock_client(json_handler(FD_ORG_MATCHES))
    )
    fixtures = provider.get_fixtures("2026_27")
    assert len(fixtures) == 2

    finished = fixtures[0]
    assert finished.provider_fixture_id == "501"
    assert finished.home_team.canonical_id == "arsenal"
    assert finished.away_team.canonical_id == "liverpool"
    assert finished.status is FixtureStatus.FINISHED
    assert (finished.home_score, finished.away_score) == (2, 1)
    assert finished.kickoff_utc.tzinfo is not None

    postponed = fixtures[1]
    assert postponed.status is FixtureStatus.POSTPONED
    assert postponed.home_score is None  # unplayed: None, not 0
    assert postponed.home_team.canonical_id == "man_city"


def test_football_data_org_normalizes_standings():
    provider = FootballDataOrgProvider(
        make_settings(), client=mock_client(json_handler(FD_ORG_STANDINGS))
    )
    rows = provider.get_standings("2026_27")
    assert len(rows) == 1
    assert rows[0].position == 1
    assert rows[0].team.canonical_id == "arsenal"
    assert rows[0].points == 10


def test_football_data_org_unknown_status_becomes_unknown():
    payload = {"matches": [dict(FD_ORG_MATCHES["matches"][0], status="SOMETHING_NEW")]}
    provider = FootballDataOrgProvider(make_settings(), client=mock_client(json_handler(payload)))
    assert provider.get_fixtures("2026_27")[0].status is FixtureStatus.UNKNOWN


# --------------------------------------------------------------------------
# Real-verification regressions (see scripts/verify_football_data_org.py):
# a real free-tier call against the 2026/27 season revealed a current-only
# club ("Coventry City FC") absent from the historical registry, and two
# genuine quota headers the adapter previously discarded.
# --------------------------------------------------------------------------
def test_football_data_org_resolves_a_current_only_promoted_club():
    """Coventry City FC has no historical ML data but IS a valid current
    canonical identity - the real-verification scenario that surfaced this."""
    payload = {
        "matches": [
            {
                "id": 999999,
                "utcDate": "2026-09-12T14:00:00Z",
                "status": "SCHEDULED",
                "matchday": 4,
                "homeTeam": {"id": 1076, "name": "Coventry City FC"},
                "awayTeam": {"id": 57, "name": "Arsenal FC"},
                "score": {"fullTime": {"home": None, "away": None}},
            }
        ]
    }
    provider = FootballDataOrgProvider(make_settings(), client=mock_client(json_handler(payload)))
    fixture = provider.get_fixtures("2026_27")[0]
    assert fixture.home_team.canonical_id == "coventry_city"

    from backend.app.services.football_data.teams import default_registry

    assert default_registry().has_historical_ml_history("coventry_city") is False


def test_football_data_org_verified_provider_ids_resolve():
    """IDs verified against the real API (not guessed) resolve correctly."""
    from backend.app.services.football_data.teams import default_registry

    registry = default_registry()
    assert registry.resolve_by_provider_id("football_data_org", 57).canonical_id == "arsenal"
    assert registry.resolve_by_provider_id("football_data_org", "73").canonical_id == "tottenham"
    assert registry.resolve_by_provider_id("football_data_org", 1076).canonical_id == "coventry_city"


def test_football_data_org_captures_real_rate_limit_headers():
    """The provider's real free-tier responses carry
    X-Requests-Available-Minute / X-RequestCounter-Reset - verified by a real
    call, not assumed. The adapter must preserve them, not discard them."""
    provider = FootballDataOrgProvider(
        make_settings(),
        client=mock_client(
            json_handler(
                FD_ORG_MATCHES,
                headers={"X-Requests-Available-Minute": "8", "X-RequestCounter-Reset": "59"},
            )
        ),
    )
    assert provider.last_rate_limit is None  # nothing yet before any request
    provider.get_fixtures("2026_27")
    rate_limit = provider.last_rate_limit
    assert rate_limit is not None
    assert rate_limit.requests_available_this_minute == 8
    assert rate_limit.reset_seconds == 59
    assert rate_limit.retry_after_seconds is None


def test_football_data_org_rate_limit_missing_headers_stay_none():
    """No quota headers on this response -> fields stay None, never 0."""
    provider = FootballDataOrgProvider(make_settings(), client=mock_client(json_handler(FD_ORG_MATCHES)))
    provider.get_fixtures("2026_27")
    assert provider.last_rate_limit.requests_available_this_minute is None
    assert provider.last_rate_limit.reset_seconds is None


def test_football_data_org_rate_limit_captured_even_on_429():
    provider = FootballDataOrgProvider(
        make_settings(),
        client=mock_client(
            json_handler(
                {},
                status_code=429,
                headers={"Retry-After": "42", "X-Requests-Available-Minute": "0"},
            )
        ),
    )
    with pytest.raises(ProviderRateLimited):
        provider.get_fixtures("2026_27")
    assert provider.last_rate_limit.requests_available_this_minute == 0
    assert provider.last_rate_limit.retry_after_seconds == 42.0


@pytest.mark.parametrize("status_code", [500, 502, 503])
def test_server_errors_raise_provider_unavailable(status_code):
    provider = FootballDataOrgProvider(
        make_settings(), client=mock_client(json_handler({}, status_code=status_code))
    )
    with pytest.raises(ProviderUnavailable):
        provider.get_fixtures("2026_27")


def test_rate_limit_raises_provider_rate_limited_with_retry_after():
    provider = FootballDataOrgProvider(
        make_settings(),
        client=mock_client(json_handler({}, status_code=429, headers={"Retry-After": "42"})),
    )
    with pytest.raises(ProviderRateLimited) as excinfo:
        provider.get_fixtures("2026_27")
    assert excinfo.value.retry_after_seconds == 42.0


def test_malformed_json_raises_malformed_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "application/json"})

    provider = FootballDataOrgProvider(make_settings(), client=mock_client(handler))
    with pytest.raises(MalformedProviderPayload):
        provider.get_fixtures("2026_27")


def test_missing_required_field_raises_malformed_payload():
    payload = {"matches": [{"id": 1, "homeTeam": {"name": "Arsenal FC"}}]}  # no awayTeam/utcDate
    provider = FootballDataOrgProvider(make_settings(), client=mock_client(json_handler(payload)))
    with pytest.raises(MalformedProviderPayload):
        provider.get_fixtures("2026_27")


def test_timeout_raises_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = FootballDataOrgProvider(make_settings(), client=mock_client(handler))
    with pytest.raises(ProviderUnavailable, match="timed out"):
        provider.get_fixtures("2026_27")


# --------------------------------------------------------------------------
# API-Football
# --------------------------------------------------------------------------
API_FOOTBALL_LIVE = {
    "errors": [],
    "response": [
        {
            "fixture": {
                "id": 9001,
                "date": "2026-09-05T14:00:00+00:00",
                "status": {"short": "2H", "elapsed": 63},
            },
            "teams": {"home": {"id": 42, "name": "Arsenal"}, "away": {"id": 40, "name": "Liverpool"}},
            "goals": {"home": 1, "away": 1},
            "events": [
                {
                    "time": {"elapsed": 31},
                    "team": {"name": "Arsenal"},
                    "player": {"name": "Player One"},
                    "assist": {"name": "Player Two"},
                    "type": "Goal",
                    "detail": "Normal Goal",
                },
                {
                    "time": {"elapsed": 40},
                    "team": {"name": "Liverpool"},
                    "player": {"name": "Player Three"},
                    "type": "Card",
                    "detail": "Yellow Card",
                },
                {
                    "time": {"elapsed": 58},
                    "team": {"name": "Arsenal"},
                    "player": {"name": "Player Four"},
                    "type": "subst",
                    "detail": "Substitution 1",
                },
            ],
            "lineups": [
                {
                    "team": {"name": "Arsenal"},
                    "formation": "4-3-3",
                    "startXI": [{"player": {"name": "Keeper", "number": 1, "pos": "G"}}],
                    "substitutes": [{"player": {"name": "Bench", "number": 12, "pos": "M"}}],
                    "coach": {"name": "The Manager"},
                }
            ],
            "statistics": [
                {
                    "team": {"name": "Arsenal"},
                    "statistics": [
                        {"type": "Total Shots", "value": 9},
                        {"type": "Shots on Goal", "value": 4},
                        {"type": "Ball Possession", "value": "56%"},
                        {"type": "Corner Kicks", "value": None},
                    ],
                },
                {
                    "team": {"name": "Liverpool"},
                    "statistics": [
                        {"type": "Total Shots", "value": 9},
                        {"type": "Shots on Goal", "value": 5},
                        {"type": "Ball Possession", "value": "44%"},
                        {"type": "Corner Kicks", "value": 0},
                    ],
                },
            ],
        }
    ],
}


def live_provider(payload=API_FOOTBALL_LIVE, **kwargs) -> ApiFootballProvider:
    return ApiFootballProvider(
        make_settings(), client=mock_client(json_handler(payload)), live_verified=True, **kwargs
    )


def test_api_football_live_capability_is_off_until_verified():
    """Live access is conditional: unverified keys declare NO capabilities,
    so nothing can depend on unproven current-season access."""
    provider = ApiFootballProvider(make_settings(), live_verified=False)
    assert provider.capabilities() == frozenset()
    with pytest.raises(UnsupportedCapability):
        provider.get_live_matches()


def test_api_football_without_key_has_no_capabilities():
    provider = ApiFootballProvider(make_settings(api_football_key=None), live_verified=True)
    assert provider.capabilities() == frozenset()


def test_api_football_verified_declares_live_capabilities():
    capabilities = live_provider().capabilities()
    assert Capability.LIVE_SCORE in capabilities
    assert Capability.EVENTS in capabilities
    assert Capability.LINEUPS in capabilities
    assert Capability.MATCH_STATISTICS in capabilities


def test_api_football_normalizes_full_live_state():
    states = live_provider().get_live_matches(season_label="2026_27")
    assert len(states) == 1
    state = states[0]

    assert state.fixture.provider_fixture_id == "9001"
    assert state.fixture.home_team.canonical_id == "arsenal"
    assert state.fixture.away_team.canonical_id == "liverpool"
    assert state.fixture.status is FixtureStatus.LIVE
    assert state.minute == 63
    assert (state.home_score, state.away_score) == (1, 1)
    assert state.freshness.provider == "api_football"
    assert state.freshness.fetched_at.tzinfo is not None


def test_api_football_parses_shots_sot_possession_and_corners():
    state = live_provider().get_live_matches(season_label="2026_27")[0]
    assert state.home_stats.shots == 9
    assert state.home_stats.shots_on_target == 4
    assert state.home_stats.possession_percent == pytest.approx(56.0)
    assert state.away_stats.shots_on_target == 5


def test_api_football_null_statistic_stays_none_while_zero_stays_zero():
    """The core None-vs-0 distinction, end to end through the parser."""
    state = live_provider().get_live_matches(season_label="2026_27")[0]
    assert state.home_stats.corners is None  # provider sent null
    assert state.away_stats.corners == 0  # provider sent 0


def test_api_football_parses_goal_card_and_substitution_events():
    state = live_provider().get_live_matches(season_label="2026_27")[0]
    types = [event.type.value for event in state.events]
    assert types == ["GOAL", "YELLOW_CARD", "SUBSTITUTION"]
    goal = state.events[0]
    assert goal.minute == 31
    assert goal.team_canonical_id == "arsenal"
    assert goal.player_name == "Player One"
    assert goal.assist_name == "Player Two"


def test_api_football_parses_lineups():
    state = live_provider().get_live_matches(season_label="2026_27")[0]
    assert state.home_lineup is not None
    assert state.home_lineup.formation == "4-3-3"
    assert state.home_lineup.starting_xi[0].name == "Keeper"
    assert state.home_lineup.coach_name == "The Manager"
    assert state.away_lineup is None  # not supplied -> None, not an empty lineup


def test_api_football_one_response_yields_one_snapshot_answering_many_questions():
    """Score, minute, possession, shots, corners, cards and subs all come
    from a SINGLE response - never one request per statistic."""
    state = live_provider().get_live_matches(season_label="2026_27")[0]
    assert state.home_score is not None
    assert state.minute is not None
    assert state.home_stats.possession_percent is not None
    assert state.home_stats.shots is not None
    assert state.away_stats.corners is not None
    assert any(e.type.value == "YELLOW_CARD" for e in state.events)
    assert any(e.type.value == "SUBSTITUTION" for e in state.events)


@pytest.mark.parametrize(
    "short,expected",
    [
        ("NS", FixtureStatus.SCHEDULED),
        ("1H", FixtureStatus.LIVE),
        ("HT", FixtureStatus.HALF_TIME),
        ("ET", FixtureStatus.EXTRA_TIME),
        ("P", FixtureStatus.PENALTIES),
        ("FT", FixtureStatus.FINISHED),
        ("PST", FixtureStatus.POSTPONED),
        ("SUSP", FixtureStatus.SUSPENDED),
        ("ABD", FixtureStatus.ABANDONED),
        ("CANC", FixtureStatus.CANCELLED),
        ("WEIRD", FixtureStatus.UNKNOWN),
    ],
)
def test_api_football_status_mapping(short, expected):
    payload = {
        "errors": [],
        "response": [
            {
                "fixture": {"id": 1, "date": "2026-09-05T14:00:00+00:00", "status": {"short": short}},
                "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Liverpool"}},
                "goals": {"home": None, "away": None},
            }
        ],
    }
    state = live_provider(payload).get_live_matches(season_label="2026_27")[0]
    assert state.fixture.status is expected


def test_api_football_reports_provider_level_errors_despite_http_200():
    """API-Football signals problems in `errors` while returning HTTP 200 -
    that must be a failure, not an empty success."""
    payload = {"errors": {"token": "invalid"}, "response": []}
    with pytest.raises(ProviderUnavailable, match="reported errors"):
        live_provider(payload).get_live_matches(season_label="2026_27")


def test_api_football_batching_rejects_oversized_request():
    provider = live_provider()
    with pytest.raises(ValueError, match="at most 20"):
        provider.get_live_match_states_by_ids([str(i) for i in range(21)], "2026_27")


def test_api_football_batching_uses_one_request_for_many_fixtures():
    """Batched ids must produce exactly ONE HTTP call."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=API_FOOTBALL_LIVE)

    provider = ApiFootballProvider(
        make_settings(), client=mock_client(handler), live_verified=True
    )
    provider.get_live_match_states_by_ids(["9001", "9002", "9003"], "2026_27")
    assert len(calls) == 1
    assert "ids=9001-9002-9003" in str(calls[0].url)


def test_parse_match_stats_returns_none_when_no_block_supplied():
    assert parse_match_stats(None) is None
    assert parse_match_stats([]) is None


def test_parse_match_stats_ignores_unrecognised_labels():
    assert parse_match_stats([{"type": "Some New Metric", "value": 5}]) is None

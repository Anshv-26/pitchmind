"""Cache, freshness, service and sealed-season-safety tests.

Fully offline. The provider-request economics are asserted directly: N reads
inside one TTL window must cost exactly ONE provider request, and no user
activity must cost zero.
"""

from __future__ import annotations

import ast
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import (  # noqa: E402
    DEFAULT_LIVE_TTL_SECONDS,
    ENV_LIVE_TTL_SECONDS,
    FootballDataSettings,
    load_settings,
)
from backend.app.services.football_data.cache import TTLCache  # noqa: E402
from backend.app.services.football_data.errors import (  # noqa: E402
    ProviderUnavailable,
    UnsupportedCapability,
)
from backend.app.services.football_data.football_data_org import FootballDataOrgProvider  # noqa: E402
from backend.app.services.football_data.models import FixtureStatus  # noqa: E402
from backend.app.services.football_data.replay import ReplayProvider  # noqa: E402
from backend.app.services.football_data.service import FootballDataService  # noqa: E402

T0 = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)


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


class CountingProvider(ReplayProvider):
    """Replay provider that records how many live fetches it served."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.live_calls = 0
        self.fail_next = False

    def get_live_matches(self, season_label: str | None = None):
        if self.fail_next:
            raise ProviderUnavailable("simulated provider outage")
        self.live_calls += 1
        return super().get_live_matches(season_label)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def test_default_live_ttl_is_five_minutes():
    assert DEFAULT_LIVE_TTL_SECONDS == 300


def test_live_ttl_is_configuration_driven(monkeypatch):
    """The TTL must be lowerable to 60s/30s later without a code change."""
    monkeypatch.setenv(ENV_LIVE_TTL_SECONDS, "60")
    assert load_settings().live_ttl_seconds == 60
    monkeypatch.setenv(ENV_LIVE_TTL_SECONDS, "30")
    assert load_settings().live_ttl_seconds == 30


def test_invalid_ttl_is_rejected_loudly(monkeypatch):
    monkeypatch.setenv(ENV_LIVE_TTL_SECONDS, "0")
    with pytest.raises(ValueError):
        load_settings()
    monkeypatch.setenv(ENV_LIVE_TTL_SECONDS, "not-a-number")
    with pytest.raises(ValueError):
        load_settings()


def test_settings_load_without_any_credentials(monkeypatch):
    """pytest must never need an API key."""
    monkeypatch.delenv("PITCHMIND_API_FOOTBALL_KEY", raising=False)
    monkeypatch.delenv("PITCHMIND_FOOTBALL_DATA_ORG_KEY", raising=False)
    settings = load_settings()
    assert settings.has_api_football_key is False
    assert settings.has_football_data_org_key is False


# --------------------------------------------------------------------------
# Cache mechanics
# --------------------------------------------------------------------------
def test_cache_miss_fetches_once():
    cache = TTLCache()
    calls = []
    cache.get_or_refresh("k", 300, lambda: calls.append(1) or "v", now=T0)
    assert len(calls) == 1


def test_cache_hit_within_ttl_performs_zero_requests():
    cache = TTLCache()
    calls = []

    def refresh():
        calls.append(1)
        return "v"

    cache.get_or_refresh("k", 300, refresh, now=T0)
    for offset in (1, 60, 150, 299):
        cache.get_or_refresh("k", 300, refresh, now=T0 + timedelta(seconds=offset))
    assert len(calls) == 1, "reads inside the TTL window must not hit the provider"


def test_expired_cache_refreshes_exactly_once():
    cache = TTLCache()
    calls = []

    def refresh():
        calls.append(1)
        return f"v{len(calls)}"

    cache.get_or_refresh("k", 300, refresh, now=T0)
    entry = cache.get_or_refresh("k", 300, refresh, now=T0 + timedelta(seconds=300))
    assert len(calls) == 2
    assert entry.value == "v2"
    assert entry.is_stale is False


def test_refresh_failure_serves_stale_value_with_flag_and_last_success():
    cache = TTLCache()
    state = {"fail": False}

    def refresh():
        if state["fail"]:
            raise ProviderUnavailable("down")
        return "good"

    cache.get_or_refresh("k", 300, refresh, now=T0)
    state["fail"] = True
    entry = cache.get_or_refresh("k", 300, refresh, now=T0 + timedelta(seconds=400))

    assert entry.value == "good"  # last good value still served
    assert entry.is_stale is True  # but explicitly flagged
    assert entry.last_successful_refresh == T0  # provenance preserved
    assert entry.age_seconds(now=T0 + timedelta(seconds=400)) == pytest.approx(400.0)


def test_refresh_failure_with_no_cached_value_raises():
    """An empty success would be a lie - the error must propagate."""
    cache = TTLCache()
    with pytest.raises(ProviderUnavailable):
        cache.get_or_refresh("k", 300, lambda: (_ for _ in ()).throw(ProviderUnavailable("down")), now=T0)


def test_no_background_polling_threads_are_started():
    """Nothing in this package may start a thread, task, or timer."""
    import threading

    before = threading.active_count()
    cache = TTLCache()
    cache.get_or_refresh("k", 300, lambda: "v", now=T0)
    service = FootballDataService(live_provider=CountingProvider(), settings=make_settings())
    service.get_live_matches(now=T0)
    assert threading.active_count() == before


# --------------------------------------------------------------------------
# Service: live economics
# --------------------------------------------------------------------------
def test_no_user_activity_causes_zero_provider_calls():
    provider = CountingProvider()
    FootballDataService(live_provider=provider, settings=make_settings())
    assert provider.live_calls == 0, "constructing the service must not fetch anything"


def test_twenty_user_reads_in_one_window_cost_one_provider_request():
    """The central economics claim: one refreshed snapshot serves many
    different user questions."""
    provider = CountingProvider(step=3)
    service = FootballDataService(live_provider=provider, settings=make_settings())

    answers = []
    for i in range(20):
        now = T0 + timedelta(seconds=i * 10)  # 0..190s, all inside the 300s TTL
        state = service.get_live_matches(now=now)[0]
        answers.append(
            (
                state.home_score,
                state.away_score,
                state.minute,
                state.home_stats.shots,
                state.home_stats.shots_on_target,
                state.home_stats.possession_percent,
                state.away_stats.corners,
                len(state.events),
            )
        )

    assert provider.live_calls == 1, "20 reads must cost exactly one provider request"
    assert len(set(answers)) == 1, "all reads must see the same snapshot"


def test_read_after_ttl_expiry_refreshes_once():
    provider = CountingProvider(step=3)
    service = FootballDataService(live_provider=provider, settings=make_settings())
    service.get_live_matches(now=T0)
    service.get_live_matches(now=T0 + timedelta(seconds=299))
    assert provider.live_calls == 1
    service.get_live_matches(now=T0 + timedelta(seconds=300))
    assert provider.live_calls == 2


def test_service_reports_stale_snapshot_rather_than_pretending_it_is_current():
    provider = CountingProvider(step=3)
    service = FootballDataService(live_provider=provider, settings=make_settings())
    service.get_live_matches(now=T0)

    provider.fail_next = True
    later = T0 + timedelta(seconds=394)
    states = service.get_live_matches(now=later)

    assert states[0].freshness.is_stale is True
    assert states[0].freshness.last_successful_refresh == T0
    # Enables "last updated 394 seconds ago" rather than silence or a lie.
    assert states[0].freshness.age_seconds(now=later) == pytest.approx(394.0)
    assert service.live_cache_age_seconds(now=later) == pytest.approx(394.0)


def test_configurable_ttl_changes_refresh_behaviour():
    provider = CountingProvider(step=3)
    service = FootballDataService(live_provider=provider, settings=make_settings(live_ttl_seconds=60))
    service.get_live_matches(now=T0)
    service.get_live_matches(now=T0 + timedelta(seconds=59))
    assert provider.live_calls == 1
    service.get_live_matches(now=T0 + timedelta(seconds=60))
    assert provider.live_calls == 2


def test_service_without_live_provider_raises_rather_than_returning_empty():
    service = FootballDataService(settings=make_settings())
    with pytest.raises(ProviderUnavailable):
        service.get_live_matches(now=T0)


def test_service_rejects_live_from_a_provider_that_cannot_do_live():
    service = FootballDataService(
        live_provider=FootballDataOrgProvider(make_settings()), settings=make_settings()
    )
    with pytest.raises(UnsupportedCapability):
        service.get_live_matches(now=T0)


def test_get_live_match_state_returns_none_for_a_fixture_not_in_play():
    """Distinct from raising: this is a genuine 'that match is not live'."""
    service = FootballDataService(live_provider=CountingProvider(step=3), settings=make_settings())
    assert service.get_live_match_state("no-such-fixture", now=T0) is None
    assert service.get_live_match_state("replay-1", now=T0) is not None


def test_finished_match_is_not_reported_as_live():
    service = FootballDataService(live_provider=CountingProvider(step=4), settings=make_settings())
    assert service.get_live_matches(now=T0) == []


# --------------------------------------------------------------------------
# Replay mode
# --------------------------------------------------------------------------
def test_replay_progresses_through_the_scripted_match():
    provider = ReplayProvider()
    observed = []
    for step in range(provider.step_count):
        provider.set_step(step)
        state = provider.get_live_match_state()
        observed.append((state.minute, state.home_score, state.away_score, len(state.events)))

    assert observed[0][:3] == (20, 0, 0)
    assert observed[1][:3] == (35, 1, 0)
    assert observed[-1][:3] == (90, 2, 1)
    minutes = [row[0] for row in observed]
    assert minutes == sorted(minutes), "replay minutes must advance monotonically"
    event_counts = [row[3] for row in observed]
    assert event_counts == sorted(event_counts), "events must accumulate"


def test_replay_is_deterministic_for_football_facts():
    a = ReplayProvider(step=3).get_live_match_state()
    b = ReplayProvider(step=3).get_live_match_state()
    assert a.model_dump(exclude={"freshness"}) == b.model_dump(exclude={"freshness"})


def test_replay_emits_the_same_schema_as_a_real_provider():
    from backend.app.services.football_data.models import LiveMatchState

    state = ReplayProvider(step=1).get_live_match_state()
    assert isinstance(state, LiveMatchState)
    assert state.freshness.provider == "replay"
    assert state.fixture.home_team.canonical_id == "arsenal"


def test_replay_exercises_the_missing_statistic_path():
    """Step 3 deliberately reports no corners, so demos show 'not reported'
    rather than a fabricated zero."""
    state = ReplayProvider(step=3).get_live_match_state()
    assert state.home_stats.corners is None
    assert state.home_stats.shots is not None


def test_replay_advance_clamps_at_the_final_step():
    provider = ReplayProvider()
    provider.advance(100)
    assert provider.step == provider.step_count - 1


def test_replay_works_through_the_service_identically():
    """The rest of PitchMind must not care whether state came from
    API-Football or the replay provider."""
    service = FootballDataService(live_provider=ReplayProvider(step=1), settings=make_settings())
    state = service.get_live_matches(now=T0)[0]
    assert state.fixture.status is FixtureStatus.LIVE
    assert state.freshness.ttl_seconds == 300


# --------------------------------------------------------------------------
# Sealed-season / ML isolation
# --------------------------------------------------------------------------
FOOTBALL_DATA_PACKAGE = REPO_ROOT / "backend" / "app" / "services" / "football_data"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_live_layer_never_imports_the_ml_package():
    """Structural proof that current/live data cannot touch frozen models,
    feature generation, or the sealed season."""
    offenders = {}
    for module_path in FOOTBALL_DATA_PACKAGE.glob("*.py"):
        ml_imports = {m for m in _imported_modules(module_path) if m.startswith("backend.app.ml")}
        if ml_imports:
            offenders[module_path.name] = ml_imports
    assert not offenders, f"football_data must not import backend.app.ml: {offenders}"


def test_live_layer_never_references_the_sealed_season():
    for module_path in FOOTBALL_DATA_PACKAGE.glob("*.py"):
        source = module_path.read_text()
        assert "2025_26" not in source, f"{module_path.name} references the sealed season"


def test_live_layer_performs_no_persistence_writes():
    """No path in this package may persist anything - it must never mutate
    historical artifacts, frozen features, or model files.

    Checked on actual CALL nodes rather than by searching text, because the
    module docstrings legitimately name `data/processed`/`models/` when
    documenting exactly this guarantee.
    """
    forbidden_calls = {"to_parquet", "to_csv", "dump", "write_text", "write_bytes", "savefig"}
    offenders: dict[str, set[str]] = {}
    for module_path in FOOTBALL_DATA_PACKAGE.glob("*.py"):
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in forbidden_calls:
                offenders.setdefault(module_path.name, set()).add(name)
            # open(..., "w"/"a") is a write regardless of target.
            if name == "open":
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and (
                        "w" in arg.value or "a" in arg.value
                    ):
                        offenders.setdefault(module_path.name, set()).add("open(write)")
    assert not offenders, f"football_data must not persist anything: {offenders}"


def test_live_layer_does_not_import_pandas_or_sklearn():
    """It has no business doing ML work at all."""
    for module_path in FOOTBALL_DATA_PACKAGE.glob("*.py"):
        modules = _imported_modules(module_path)
        assert not any(m.startswith(("pandas", "sklearn", "numpy")) for m in modules), module_path.name


def test_probe_script_is_never_executed_on_import():
    """The capability probe must only run when invoked deliberately."""
    source = (REPO_ROOT / "scripts" / "probe_live_provider.py").read_text()
    assert 'if __name__ == "__main__":' in source
    tree = ast.parse(source)
    top_level_calls = [
        node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert not top_level_calls, "probe script must not call anything at import time"


def test_probe_rejects_http_200_with_empty_response():
    """HTTP 200 + empty data is the free-plan restriction signal, and must be
    treated as failure, not success."""
    from scripts.probe_live_provider import evaluate_league_response

    ok, reason = evaluate_league_response({"status_code": 200, "headers": {}, "json": {"response": []}})
    assert ok is False
    assert "empty" in reason.lower()


def test_probe_rejects_provider_errors_despite_http_200():
    from scripts.probe_live_provider import evaluate_league_response

    ok, reason = evaluate_league_response(
        {"status_code": 200, "headers": {}, "json": {"errors": {"plan": "upgrade required"}, "response": []}}
    )
    assert ok is False
    assert "errors" in reason.lower()


def test_probe_rejects_league_without_the_current_season():
    from scripts.probe_live_provider import evaluate_league_response

    payload = {"response": [{"league": {"id": 39}, "seasons": [{"year": 2023}]}]}
    ok, reason = evaluate_league_response({"status_code": 200, "headers": {}, "json": payload})
    assert ok is False
    assert "not listed" in reason.lower()


def test_probe_accepts_a_genuine_current_season_response():
    from backend.app.core.config import CURRENT_SEASON_START_YEAR
    from scripts.probe_live_provider import evaluate_league_response

    payload = {
        "response": [
            {
                "league": {"id": 39},
                "seasons": [
                    {
                        "year": CURRENT_SEASON_START_YEAR,
                        "coverage": {
                            "fixtures": {"events": True, "lineups": True, "statistics_fixtures": True},
                            "standings": True,
                        },
                    }
                ],
            }
        ]
    }
    ok, reason = evaluate_league_response({"status_code": 200, "headers": {}, "json": payload})
    assert ok is True
    assert "coverage flags" in reason


def test_probe_budget_is_capped():
    from scripts.probe_live_provider import ProbeBudget

    budget = ProbeBudget(limit=3)
    for _ in range(3):
        budget.spend()
    with pytest.raises(RuntimeError, match="budget"):
        budget.spend()

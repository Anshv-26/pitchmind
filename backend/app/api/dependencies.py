"""Wiring for the API layer.

The `FootballDataService` is built LAZILY and cached process-wide. Both
properties matter:

* **Lazy** - no provider is constructed and no network call is made at import
  time, so the app imports and `/health` responds with no credentials at all.
* **Cached** - the service owns the TTL cache. Rebuilding it per request would
  silently discard every cached snapshot and turn a 5-minute lazy refresh into
  a provider call per request, which is exactly what the cache exists to
  prevent.

Live provider policy (current, honest):
    football-data.org -> fixtures / results / standings   (REAL)
    ReplayProvider    -> live match state                  (REPLAY, development)
    API-Football      -> adapter exists but its free tier was verified NOT to
                         expose Premier League season 2026, so it is NOT wired
                         in as a live provider here.
"""

from __future__ import annotations

from backend.app.core.config import FootballDataSettings, load_settings
from backend.app.services.football_data.football_data_org import FootballDataOrgProvider
from backend.app.services.football_data.replay import ReplayProvider
from backend.app.services.football_data.service import FootballDataService

_service: FootballDataService | None = None


def build_service(settings: FootballDataSettings | None = None) -> FootballDataService:
    """Construct a service. Does not touch the network."""
    settings = settings if settings is not None else load_settings()
    return FootballDataService(
        current_provider=FootballDataOrgProvider(settings),
        # Replay is the ONLY live source today. It is labelled REPLAY in every
        # response's provenance so it can never pass as real live football.
        live_provider=ReplayProvider(),
        settings=settings,
    )


def get_service() -> FootballDataService:
    """FastAPI dependency: the shared, cache-owning service instance."""
    global _service
    if _service is None:
        _service = build_service()
    return _service


def set_service(service: FootballDataService | None) -> None:
    """Replace (or clear) the shared instance. Used by tests to inject fakes."""
    global _service
    _service = service


def reset_service() -> None:
    set_service(None)

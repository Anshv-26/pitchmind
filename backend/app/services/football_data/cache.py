"""Lazy, on-demand TTL cache for football data.

There is deliberately **no background polling loop**. Nothing here starts a
thread, task or timer. A provider request happens only when a user asks a
question AND the cached value has aged past its TTL. If nobody uses
PitchMind, PitchMind makes zero provider requests.

    user question
          |
    check cache
          |
    fresh (age < ttl) ---> return cached value, no network
    expired / missing ---> one refresh, store, return
    refresh failed ------> return last good value flagged is_stale=True
    refresh failed, no prior value ---> raise

The stale path is the important one: PitchMind must be able to say "last
updated 4 minutes ago" rather than either failing outright or - far worse -
presenting an old snapshot as if it were current.

One cached snapshot answers many questions (score, minute, possession, shots,
corners, cards, subs), which is why `LiveMatchState` is cached whole rather
than per-statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Generic, TypeVar

from backend.app.services.football_data.errors import FootballDataError
from backend.app.services.football_data.models import utcnow

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached value plus the timestamps needed to reason about freshness."""

    value: T
    fetched_at: datetime
    last_successful_refresh: datetime
    is_stale: bool = False

    def age_seconds(self, *, now: datetime | None = None) -> float:
        reference = now if now is not None else utcnow()
        return max(0.0, (reference - self.fetched_at).total_seconds())

    def is_expired(self, ttl_seconds: int, *, now: datetime | None = None) -> bool:
        return self.age_seconds(now=now) >= ttl_seconds


class TTLCache:
    """In-memory, single-process TTL cache.

    No Redis, no SQLite - V1 is a single process and premature infrastructure
    would be dead weight. The upgrade path (SQLite for multi-process, Redis
    for multi-instance) keeps this same `get_or_refresh` interface.

    Instrumented with `refresh_count` / `hit_count` / `miss_count` so tests can
    assert the economics directly: N reads inside one TTL window must perform
    exactly ONE provider request.
    """

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self.refresh_count = 0
        self.hit_count = 0
        self.miss_count = 0

    def peek(self, key: str) -> CacheEntry | None:
        """Inspect without affecting counters or triggering a refresh."""
        return self._entries.get(key)

    def store(self, key: str, value: T, *, now: datetime | None = None) -> CacheEntry[T]:
        timestamp = now if now is not None else utcnow()
        entry = CacheEntry(
            value=value, fetched_at=timestamp, last_successful_refresh=timestamp, is_stale=False
        )
        self._entries[key] = entry
        return entry

    def invalidate(self, key: str) -> None:
        self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()

    def get_or_refresh(
        self,
        key: str,
        ttl_seconds: int,
        refresh: Callable[[], T],
        *,
        now: datetime | None = None,
    ) -> CacheEntry[T]:
        """Return a fresh value, refreshing at most once, never polling.

        On refresh failure with a previous value present, the previous value is
        returned flagged `is_stale=True` with `last_successful_refresh`
        preserved - the caller can then report the true age instead of
        pretending the data is current. With no previous value, the provider
        error propagates: an empty success would be a lie.
        """
        reference = now if now is not None else utcnow()
        entry = self._entries.get(key)

        if entry is not None and not entry.is_expired(ttl_seconds, now=reference):
            self.hit_count += 1
            return entry

        self.miss_count += 1
        try:
            self.refresh_count += 1
            value = refresh()
        except FootballDataError:
            if entry is None:
                raise
            # Keep serving the last good value, but say so.
            stale_entry = CacheEntry(
                value=entry.value,
                fetched_at=entry.fetched_at,
                last_successful_refresh=entry.last_successful_refresh,
                is_stale=True,
            )
            self._entries[key] = stale_entry
            return stale_entry

        return self.store(key, value, now=reference)

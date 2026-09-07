"""Environment-driven configuration for PitchMind's runtime services.

Every value here is read from the environment, never hard-coded, and no
credential is ever committed. A missing API key is NOT an import-time error:
providers report themselves unavailable through their capability set instead
(see `services.football_data.provider`), so the application - and the test
suite - runs fine with no credentials at all.

TTLs are configuration-driven on purpose. V1 runs a deliberately relaxed
five-minute live cache because PitchMind is being demoed to family/friends,
not competing with Flashscore. If cheap API credits are bought later, the
live TTL drops to 60s or 30s by changing one environment variable - no code
change, no redesign.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Environment variable names, in one place so nothing guesses at them.
ENV_FOOTBALL_DATA_ORG_KEY = "PITCHMIND_FOOTBALL_DATA_ORG_KEY"
ENV_API_FOOTBALL_KEY = "PITCHMIND_API_FOOTBALL_KEY"
ENV_LIVE_TTL_SECONDS = "PITCHMIND_LIVE_TTL_SECONDS"
ENV_STANDINGS_TTL_SECONDS = "PITCHMIND_STANDINGS_TTL_SECONDS"
ENV_FIXTURES_TTL_SECONDS = "PITCHMIND_FIXTURES_TTL_SECONDS"
ENV_METADATA_TTL_SECONDS = "PITCHMIND_METADATA_TTL_SECONDS"
ENV_HTTP_CONNECT_TIMEOUT = "PITCHMIND_HTTP_CONNECT_TIMEOUT"
ENV_HTTP_READ_TIMEOUT = "PITCHMIND_HTTP_READ_TIMEOUT"

# V1 default: ~5 minutes. Deliberately relaxed for a small-scale demo; see
# module docstring. Lower it via PITCHMIND_LIVE_TTL_SECONDS, not by editing code.
DEFAULT_LIVE_TTL_SECONDS = 300
DEFAULT_STANDINGS_TTL_SECONDS = 30 * 60
DEFAULT_FIXTURES_TTL_SECONDS = 30 * 60
DEFAULT_METADATA_TTL_SECONDS = 6 * 60 * 60

DEFAULT_HTTP_CONNECT_TIMEOUT = 5.0
DEFAULT_HTTP_READ_TIMEOUT = 10.0

# The Premier League, as identified by each provider.
PREMIER_LEAGUE_FOOTBALL_DATA_ORG_CODE = "PL"
PREMIER_LEAGUE_API_FOOTBALL_ID = 39

# The real-world season currently being played. Note this is NOT an ML
# training season: see `services.football_data.service` for the hard boundary
# between "current football facts" (allowed) and "ML state" (frozen).
CURRENT_SEASON_START_YEAR = 2026
CURRENT_SEASON_LABEL = "2026_27"


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _key_from_env(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


@dataclass(frozen=True)
class FootballDataSettings:
    """Resolved settings. Built by `load_settings()`, never at import time,
    so tests and callers can vary the environment freely."""

    football_data_org_key: str | None
    api_football_key: str | None
    live_ttl_seconds: int
    standings_ttl_seconds: int
    fixtures_ttl_seconds: int
    metadata_ttl_seconds: int
    http_connect_timeout: float
    http_read_timeout: float

    @property
    def has_football_data_org_key(self) -> bool:
        return self.football_data_org_key is not None

    @property
    def has_api_football_key(self) -> bool:
        return self.api_football_key is not None


def load_settings() -> FootballDataSettings:
    """Read settings from the environment. Safe to call with no credentials."""
    return FootballDataSettings(
        football_data_org_key=_key_from_env(ENV_FOOTBALL_DATA_ORG_KEY),
        api_football_key=_key_from_env(ENV_API_FOOTBALL_KEY),
        live_ttl_seconds=_int_from_env(ENV_LIVE_TTL_SECONDS, DEFAULT_LIVE_TTL_SECONDS),
        standings_ttl_seconds=_int_from_env(ENV_STANDINGS_TTL_SECONDS, DEFAULT_STANDINGS_TTL_SECONDS),
        fixtures_ttl_seconds=_int_from_env(ENV_FIXTURES_TTL_SECONDS, DEFAULT_FIXTURES_TTL_SECONDS),
        metadata_ttl_seconds=_int_from_env(ENV_METADATA_TTL_SECONDS, DEFAULT_METADATA_TTL_SECONDS),
        http_connect_timeout=_float_from_env(ENV_HTTP_CONNECT_TIMEOUT, DEFAULT_HTTP_CONNECT_TIMEOUT),
        http_read_timeout=_float_from_env(ENV_HTTP_READ_TIMEOUT, DEFAULT_HTTP_READ_TIMEOUT),
    )

"""Typed errors for the football-data layer.

Every failure mode here is explicit. Nothing in this package converts a
failure into an empty-but-successful response: an empty fixture list must
always mean "the provider reported no fixtures", never "something broke".
"""

from __future__ import annotations


class FootballDataError(Exception):
    """Base class for every error raised by this package."""


class ProviderUnavailable(FootballDataError):
    """The provider could not be reached, timed out, returned 5xx, or has no
    credentials configured. Distinct from "returned no data"."""


class ProviderRateLimited(ProviderUnavailable):
    """HTTP 429 or an exhausted request quota.

    Subclasses ProviderUnavailable so a caller that only cares "can I refresh
    right now?" can catch one type, while a caller that wants to back off
    specifically can catch this one.
    """

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class UnsupportedCapability(FootballDataError):
    """A provider was asked for something it does not support - for example
    live scores from football-data.org's free contract.

    Raised rather than returning `[]`/`None`, so an unsupported capability can
    never be silently mistaken for "there is no live football right now".
    """


class UnknownTeam(FootballDataError):
    """A team name/id could not be resolved to a canonical identity.

    Deliberately fatal: fuzzy-matching an unknown club would risk mapping
    (say) Sheffield United onto Sheffield Wednesday, and a wrong team is worse
    than a loud failure.
    """

    def __init__(self, raw_value: str, *, provider: str | None = None) -> None:
        source = f" (provider={provider})" if provider else ""
        super().__init__(
            f"could not resolve team {raw_value!r}{source} to a canonical identity; "
            f"add an explicit alias/provider id to teams.py - fuzzy matching is "
            f"deliberately not used"
        )
        self.raw_value = raw_value
        self.provider = provider


class MalformedProviderPayload(FootballDataError):
    """The provider responded, but the body was not valid JSON or did not
    match the shape this adapter requires."""


class CurrentSeasonNotAvailable(FootballDataError):
    """The provider/plan does not expose the requested (current) season.

    This is the failure the API-Football capability probe is designed to
    detect: a free plan may return HTTP 200 with an empty payload for a season
    it does not actually cover.
    """

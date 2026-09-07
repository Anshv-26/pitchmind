"""Canonical team identity across historical, current and live data.

The same club must be the same object whether it arrives as `Man City` from
the historical Football-Data CSVs, `Manchester City FC` from
football-data.org, or `Manchester City` from API-Football. That reconciliation
happens here and nowhere else.

**No fuzzy matching.** Edit-distance matching silently maps Sheffield United
onto Sheffield Wednesday, or Manchester City onto Manchester United - and a
confidently wrong team is far worse than a loud failure. Resolution is:

    exact alias hit  ->  normalized-form hit  ->  UnknownTeam

where normalization only casefolds, strips punctuation/diacritics and drops
club-type suffixes (FC/AFC/CF). Anything unresolved raises `UnknownTeam`
naming the raw string, so a human adds an explicit alias.

**Historical ML history is a separate flag from identity.** A club promoted
into the current season legitimately has a canonical identity while having no
rows in our frozen historical dataset. `historical_ml_history_available=False`
records that honestly; nothing here fabricates historical state for such a
club. This is the expected path for 2026/27 newcomers, which by construction
are absent from the 34 clubs seen in seasons 2015/16-2025/26.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from backend.app.services.football_data.errors import UnknownTeam
from backend.app.services.football_data.models import TeamRef

# Provider keys used in `provider_ids`.
PROVIDER_FOOTBALL_DATA_ORG = "football_data_org"
PROVIDER_API_FOOTBALL = "api_football"

# Suffixes stripped during normalization. Deliberately short: these are
# club-type markers, not distinguishing parts of any English club's name.
_STRIPPED_SUFFIXES = ("fc", "afc", "cf")


@dataclass(frozen=True)
class CanonicalTeam:
    """One club's stable identity.

    `provider_ids` is intentionally EMPTY by default. Real provider numeric
    ids are not guessed here - they are populated only from a verified probe
    against a live account, because an invented id would silently fetch the
    wrong club's data.
    """

    canonical_id: str
    canonical_name: str
    historical_names: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    provider_ids: dict[str, str] = field(default_factory=dict)
    historical_ml_history_available: bool = True

    def to_ref(self, *, provider_team_id: str | None = None) -> TeamRef:
        return TeamRef(
            canonical_id=self.canonical_id,
            canonical_name=self.canonical_name,
            provider_team_id=provider_team_id,
        )


def normalize_team_name(raw: str) -> str:
    """Casefold, strip accents/punctuation and drop a trailing club-type
    suffix. `Nott'm Forest` -> `nottm forest`; `Manchester City FC` ->
    `manchester city`.

    Apostrophes (straight `'` and curly `'`) are REMOVED, not turned into
    spaces: they occur mid-word in club names like `Nott'm Forest`, and
    replacing one with a space would incorrectly split a single token into
    two (`Nott'm` -> `Nott m` instead of `Nottm`). Every other punctuation
    character still becomes a space, which is the correct behaviour for
    separators such as `&` in `Brighton & Hove Albion`.
    """
    decomposed = unicodedata.normalize("NFKD", raw)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    without_apostrophes = without_accents.replace("'", "").replace("’", "")
    cleaned = "".join(
        ch if ch.isalnum() or ch.isspace() else " " for ch in without_apostrophes.casefold()
    )
    tokens = cleaned.split()
    while tokens and tokens[-1] in _STRIPPED_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Registry: every club appearing in the frozen historical dataset
# (34 clubs across seasons 2015/16-2025/26, verified against matches.parquet).
# `historical_names` are the exact Football-Data CSV spellings.
# --------------------------------------------------------------------------
_CANONICAL_TEAMS: tuple[CanonicalTeam, ...] = (
    CanonicalTeam("arsenal", "Arsenal", ("Arsenal",), ("Arsenal FC",)),
    CanonicalTeam("aston_villa", "Aston Villa", ("Aston Villa",), ("Aston Villa FC", "Villa")),
    CanonicalTeam("bournemouth", "Bournemouth", ("Bournemouth",), ("AFC Bournemouth", "Bournemouth AFC")),
    CanonicalTeam("brentford", "Brentford", ("Brentford",), ("Brentford FC",)),
    CanonicalTeam(
        "brighton",
        "Brighton & Hove Albion",
        ("Brighton",),
        ("Brighton & Hove Albion FC", "Brighton and Hove Albion", "Brighton Hove Albion"),
    ),
    CanonicalTeam("burnley", "Burnley", ("Burnley",), ("Burnley FC",)),
    CanonicalTeam("cardiff", "Cardiff City", ("Cardiff",), ("Cardiff City FC",)),
    CanonicalTeam("chelsea", "Chelsea", ("Chelsea",), ("Chelsea FC",)),
    CanonicalTeam("crystal_palace", "Crystal Palace", ("Crystal Palace",), ("Crystal Palace FC",)),
    CanonicalTeam("everton", "Everton", ("Everton",), ("Everton FC",)),
    CanonicalTeam("fulham", "Fulham", ("Fulham",), ("Fulham FC",)),
    CanonicalTeam(
        "huddersfield", "Huddersfield Town", ("Huddersfield",), ("Huddersfield Town AFC", "Huddersfield Town FC")
    ),
    CanonicalTeam("hull", "Hull City", ("Hull",), ("Hull City AFC", "Hull City FC")),
    CanonicalTeam("ipswich", "Ipswich Town", ("Ipswich",), ("Ipswich Town FC",)),
    CanonicalTeam("leeds", "Leeds United", ("Leeds",), ("Leeds United FC", "Leeds Utd")),
    CanonicalTeam("leicester", "Leicester City", ("Leicester",), ("Leicester City FC",)),
    CanonicalTeam("liverpool", "Liverpool", ("Liverpool",), ("Liverpool FC",)),
    CanonicalTeam("luton", "Luton Town", ("Luton",), ("Luton Town FC",)),
    CanonicalTeam("man_city", "Manchester City", ("Man City",), ("Manchester City FC", "Man. City")),
    CanonicalTeam(
        "man_united",
        "Manchester United",
        ("Man United",),
        ("Manchester United FC", "Man Utd", "Man. United", "Manchester Utd"),
    ),
    CanonicalTeam("middlesbrough", "Middlesbrough", ("Middlesbrough",), ("Middlesbrough FC",)),
    CanonicalTeam(
        "newcastle", "Newcastle United", ("Newcastle",), ("Newcastle United FC", "Newcastle Utd")
    ),
    CanonicalTeam("norwich", "Norwich City", ("Norwich",), ("Norwich City FC",)),
    CanonicalTeam(
        "nottingham_forest",
        "Nottingham Forest",
        ("Nott'm Forest",),
        ("Nottingham Forest FC", "Nottm Forest", "Notts Forest"),
    ),
    CanonicalTeam(
        "sheffield_united", "Sheffield United", ("Sheffield United",), ("Sheffield United FC", "Sheffield Utd")
    ),
    CanonicalTeam("southampton", "Southampton", ("Southampton",), ("Southampton FC",)),
    CanonicalTeam("stoke", "Stoke City", ("Stoke",), ("Stoke City FC",)),
    CanonicalTeam("sunderland", "Sunderland", ("Sunderland",), ("Sunderland AFC", "Sunderland FC")),
    CanonicalTeam("swansea", "Swansea City", ("Swansea",), ("Swansea City AFC", "Swansea City FC")),
    CanonicalTeam(
        "tottenham", "Tottenham Hotspur", ("Tottenham",), ("Tottenham Hotspur FC", "Spurs")
    ),
    CanonicalTeam("watford", "Watford", ("Watford",), ("Watford FC",)),
    CanonicalTeam(
        "west_brom",
        "West Bromwich Albion",
        ("West Brom",),
        ("West Bromwich Albion FC", "West Bromwich Albion", "West Bromwich"),
    ),
    CanonicalTeam("west_ham", "West Ham United", ("West Ham",), ("West Ham United FC", "West Ham Utd")),
    CanonicalTeam(
        "wolves",
        "Wolverhampton Wanderers",
        ("Wolves",),
        ("Wolverhampton Wanderers FC", "Wolverhampton"),
    ),
)


class TeamRegistry:
    """Explicit-mapping team resolution. Construct once and share."""

    def __init__(self, teams: tuple[CanonicalTeam, ...] = _CANONICAL_TEAMS) -> None:
        self._by_canonical_id: dict[str, CanonicalTeam] = {}
        self._by_normalized_name: dict[str, CanonicalTeam] = {}
        self._by_provider_id: dict[tuple[str, str], CanonicalTeam] = {}
        for team in teams:
            self.register(team)

    def register(self, team: CanonicalTeam) -> None:
        """Add a club. Used to seed the registry and to add current-season
        newcomers that have no historical ML history."""
        if team.canonical_id in self._by_canonical_id:
            raise ValueError(f"duplicate canonical_id {team.canonical_id!r}")
        self._by_canonical_id[team.canonical_id] = team

        spellings = {team.canonical_name, *team.historical_names, *team.aliases}
        for spelling in spellings:
            key = normalize_team_name(spelling)
            existing = self._by_normalized_name.get(key)
            if existing is not None and existing.canonical_id != team.canonical_id:
                raise ValueError(
                    f"name {spelling!r} (normalized {key!r}) is already mapped to "
                    f"{existing.canonical_id!r}; refusing an ambiguous alias"
                )
            self._by_normalized_name[key] = team

        for provider, provider_id in team.provider_ids.items():
            self._by_provider_id[(provider, str(provider_id))] = team

    def resolve(self, raw_name: str, *, provider: str | None = None) -> CanonicalTeam:
        """Resolve a provider/historical spelling to a canonical club.

        Raises `UnknownTeam` rather than guessing - see module docstring.
        """
        if raw_name is None or not str(raw_name).strip():
            raise UnknownTeam(repr(raw_name), provider=provider)
        key = normalize_team_name(str(raw_name))
        team = self._by_normalized_name.get(key)
        if team is None:
            raise UnknownTeam(str(raw_name), provider=provider)
        return team

    def resolve_by_provider_id(self, provider: str, provider_id: str | int) -> CanonicalTeam:
        """Resolve by a provider's numeric id, once those ids are populated
        from a verified probe. Raises `UnknownTeam` while they are not."""
        team = self._by_provider_id.get((provider, str(provider_id)))
        if team is None:
            raise UnknownTeam(f"{provider}:{provider_id}", provider=provider)
        return team

    def get(self, canonical_id: str) -> CanonicalTeam:
        team = self._by_canonical_id.get(canonical_id)
        if team is None:
            raise UnknownTeam(canonical_id)
        return team

    def team_ref(self, raw_name: str, *, provider: str | None = None, provider_team_id: str | None = None) -> TeamRef:
        """Resolve and return the schema-level reference in one step."""
        return self.resolve(raw_name, provider=provider).to_ref(provider_team_id=provider_team_id)

    def has_historical_ml_history(self, canonical_id: str) -> bool:
        return self.get(canonical_id).historical_ml_history_available

    @property
    def canonical_ids(self) -> tuple[str, ...]:
        return tuple(self._by_canonical_id)

    def __len__(self) -> int:
        return len(self._by_canonical_id)


def default_registry() -> TeamRegistry:
    """A fresh registry seeded with the 34 historical clubs."""
    return TeamRegistry()

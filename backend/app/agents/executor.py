"""The single agent-layer doorway into PitchMind's deterministic tools.

    orchestrator / specialists
              |
        ToolExecutor          <- this module
              |
      backend/app/tools/*     <- existing application tools (unchanged)
              |
        EvidenceLedger

Nothing in the agent layer may reach past this doorway. ToolExecutor calls the
already-built application tools and re-implements none of their logic: it
validates arguments, dispatches through an explicit registry, records the
result as evidence, and returns a reference to it.

THREE PROPERTIES THAT MATTER

1. **Explicit dispatch.** Tool names map to handlers through a hand-written
   registry - never `getattr(module, name)` on a caller-supplied string. An
   unapproved name fails deterministically instead of reaching arbitrary code.

2. **Request-scoped deduplication.** Identical calls within one request reuse
   the first evidence item. This is NOT the provider TTL cache (which governs
   network freshness); it stops the same deterministic operation being
   performed twice while answering one question - e.g. the research and
   tactical specialists both asking for Arsenal's fixtures.

3. **Structured, typed failure.** An expected application error becomes a
   `ToolFailure` with a specific `ToolErrorCode`, never a fabricated success
   and never a bare "something went wrong". Unexpected exceptions propagate.

SEALED BOUNDARY: this module only calls existing inference/application tools.
It trains nothing, fits nothing, writes nothing, and never touches the sealed
season.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.agents.contracts import (
    EvidenceItem,
    EvidenceKind,
    ToolErrorCode,
    ToolExecutionResult,
    ToolFailure,
    ToolName,
)
from backend.app.core.config import load_settings
from backend.app.services.football_data.errors import (
    CurrentSeasonNotAvailable,
    MalformedProviderPayload,
    ProviderRateLimited,
    ProviderUnavailable,
    UnknownTeam,
    UnsupportedCapability,
)
from backend.app.services.football_data.models import FixtureStatus
from backend.app.services.football_data.service import FootballDataService
from backend.app.services.football_data.teams import TeamRegistry, default_registry
from backend.app.tools import football_data_tools, prediction_tools, scoreline_tools
from backend.app.tools.schemas import (
    OutcomePredictionRequest,
    ScorelinePredictionRequest,
)

# Expected application errors -> typed codes. ORDER MATTERS: ProviderRateLimited
# subclasses ProviderUnavailable, so it must be tested first or a rate limit
# would be misreported as a generic outage.
_ERROR_CODES: tuple[tuple[type[Exception], ToolErrorCode], ...] = (
    (UnknownTeam, ToolErrorCode.UNKNOWN_TEAM),
    (scoreline_tools.TeamNotInScoreModel, ToolErrorCode.TEAM_NOT_IN_SCORE_MODEL),
    (prediction_tools.ModelArtifactUnavailable, ToolErrorCode.MODEL_ARTIFACT_UNAVAILABLE),
    (ProviderRateLimited, ToolErrorCode.PROVIDER_RATE_LIMITED),
    (CurrentSeasonNotAvailable, ToolErrorCode.CURRENT_SEASON_NOT_AVAILABLE),
    (MalformedProviderPayload, ToolErrorCode.MALFORMED_PROVIDER_PAYLOAD),
    (UnsupportedCapability, ToolErrorCode.UNSUPPORTED_CAPABILITY),
    (ProviderUnavailable, ToolErrorCode.PROVIDER_UNAVAILABLE),
)


# --------------------------------------------------------------------------
# Argument models - one per tool, all extra="forbid"
# --------------------------------------------------------------------------
class _Args(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StandingsArgs(_Args):
    """No arguments: the table is for the configured current season."""


class FixturesArgs(_Args):
    team: str | None = None
    status: FixtureStatus | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    limit: int | None = Field(default=None, ge=1, le=500)


class LiveMatchesArgs(_Args):
    """No arguments: returns every match currently in play."""


class LiveMatchStateArgs(_Args):
    provider_fixture_id: str = Field(min_length=1)


# Prediction/scoreline reuse the EXISTING application request schemas rather
# than declaring near-duplicates, so validation cannot drift from the tools.
OutcomeArgs = OutcomePredictionRequest
ScorelineArgs = ScorelinePredictionRequest


@dataclass(frozen=True)
class _ToolSpec:
    """How one approved tool is validated, invoked and provenance-extracted."""

    args_model: type[BaseModel]
    evidence_kind: EvidenceKind
    needs_service: bool
    invoke: Callable[["ToolExecutor", BaseModel], BaseModel]
    provenance_field: str


def _standings(executor: "ToolExecutor", args: BaseModel) -> BaseModel:
    return football_data_tools.get_current_standings(executor.require_service(), now=executor.now)


def _fixtures(executor: "ToolExecutor", args: FixturesArgs) -> BaseModel:
    return football_data_tools.get_fixtures(
        executor.require_service(),
        team=args.team,
        status=args.status,
        date_from=args.date_from,
        date_to=args.date_to,
        limit=args.limit,
        registry=executor.registry,
        now=executor.now,
    )


def _live_matches(executor: "ToolExecutor", args: BaseModel) -> BaseModel:
    return football_data_tools.get_live_matches(executor.require_service(), now=executor.now)


def _live_match_state(executor: "ToolExecutor", args: LiveMatchStateArgs) -> BaseModel:
    return football_data_tools.get_live_match_state(
        executor.require_service(), args.provider_fixture_id, now=executor.now
    )


def _outcome(executor: "ToolExecutor", args: OutcomePredictionRequest) -> BaseModel:
    return prediction_tools.run_outcome_prediction(args, artifact=executor.outcome_artifact)


def _explanation(executor: "ToolExecutor", args: OutcomePredictionRequest) -> BaseModel:
    return prediction_tools.explain_outcome_prediction(args, artifact=executor.outcome_artifact)


def _scoreline(executor: "ToolExecutor", args: ScorelinePredictionRequest) -> BaseModel:
    return scoreline_tools.get_scoreline_prediction(
        args, artifact=executor.score_artifact, registry=executor.registry
    )


# The ONLY tools an agent may invoke. A name outside this mapping is refused.
TOOL_REGISTRY: dict[ToolName, _ToolSpec] = {
    ToolName.GET_CURRENT_STANDINGS: _ToolSpec(
        StandingsArgs, EvidenceKind.CURRENT_DATA, True, _standings, "provenance"
    ),
    ToolName.GET_FIXTURES: _ToolSpec(
        FixturesArgs, EvidenceKind.CURRENT_DATA, True, _fixtures, "provenance"
    ),
    ToolName.GET_LIVE_MATCHES: _ToolSpec(
        LiveMatchesArgs, EvidenceKind.LIVE_STATE, True, _live_matches, "provenance"
    ),
    ToolName.GET_LIVE_MATCH_STATE: _ToolSpec(
        LiveMatchStateArgs, EvidenceKind.LIVE_STATE, True, _live_match_state, "provenance"
    ),
    ToolName.RUN_OUTCOME_PREDICTION: _ToolSpec(
        OutcomeArgs, EvidenceKind.MODEL_PREDICTION, False, _outcome, "model_provenance"
    ),
    ToolName.EXPLAIN_OUTCOME_PREDICTION: _ToolSpec(
        OutcomeArgs, EvidenceKind.MODEL_EXPLANATION, False, _explanation, "model_provenance"
    ),
    ToolName.GET_SCORELINE_PREDICTION: _ToolSpec(
        ScorelineArgs, EvidenceKind.SCORE_MODEL, False, _scoreline, "model_provenance"
    ),
}


class EvidenceLedger:
    """Request-scoped record of everything deterministically established.

    Evidence ids are `E1`, `E2`, ... - deterministic within one ledger, simple,
    non-secret, and safe to show in a public execution trace. They carry no
    prompt or reasoning content.
    """

    def __init__(self) -> None:
        self._items: list[EvidenceItem] = []
        self._by_id: dict[str, EvidenceItem] = {}
        self._by_dedup_key: dict[str, EvidenceItem] = {}
        self._failures: list[ToolFailure] = []

    def next_evidence_id(self) -> str:
        return f"E{len(self._items) + 1}"

    def add(self, item: EvidenceItem) -> EvidenceItem:
        self._items.append(item)
        self._by_id[item.evidence_id] = item
        self._by_dedup_key[item.dedup_key] = item
        return item

    def record_failure(self, failure: ToolFailure) -> ToolFailure:
        """Failures are tracked separately - never as successful evidence."""
        self._failures.append(failure)
        return failure

    def find_by_dedup_key(self, dedup_key: str) -> EvidenceItem | None:
        return self._by_dedup_key.get(dedup_key)

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self._by_id.get(evidence_id)

    @property
    def items(self) -> tuple[EvidenceItem, ...]:
        return tuple(self._items)

    @property
    def failures(self) -> tuple[ToolFailure, ...]:
        return tuple(self._failures)

    def __len__(self) -> int:
        return len(self._items)

    def to_json(self) -> str:
        return json.dumps(
            {
                "evidence": [item.model_dump(mode="json") for item in self._items],
                "failures": [failure.model_dump(mode="json") for failure in self._failures],
            },
            sort_keys=True,
        )


@dataclass
class ToolExecutor:
    """Validates, dispatches and records deterministic tool calls.

    Dependencies are injected so tests can supply a fake service, the replay
    provider, and local artifacts with no network. This reuses the existing
    service/provider architecture rather than adding a second one.
    """

    service: FootballDataService | None = None
    registry: TeamRegistry = field(default_factory=default_registry)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    outcome_artifact: Any = None
    score_artifact: Any = None
    now: datetime | None = None

    # ---- helpers -------------------------------------------------------
    def require_service(self) -> FootballDataService:
        if self.service is None:
            raise _ServiceNotConfigured("no football data service is configured")
        return self.service

    def _canonical_team(self, value: str) -> str:
        """Canonical id for dedup purposes, falling back to the raw string.

        Two specialists asking for "Arsenal" and "Arsenal FC" are asking the
        same question, so the dedup key should collapse them. An unresolvable
        name falls back to its raw form here and the TOOL raises the typed
        `UnknownTeam` - error semantics stay owned by the tool, not by key
        construction.
        """
        try:
            return self.registry.resolve(value).canonical_id
        except UnknownTeam:
            return value.strip().casefold()

    def _dedup_key(self, tool_name: ToolName, args: BaseModel) -> str:
        """Stable key: tool name + canonically-serialized validated arguments.

        Validating into a Pydantic model already normalizes field ORDER, and
        `sort_keys` makes the serialization canonical regardless. Python's
        `hash()` is deliberately not used - it is salted per process and would
        not be stable or externally meaningful.
        """
        payload = args.model_dump(mode="json")
        for team_field in ("team", "home_team", "away_team"):
            value = payload.get(team_field)
            if isinstance(value, str) and value:
                payload[team_field] = self._canonical_team(value)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return f"{tool_name.value}:{canonical}"

    def _scrub(self, message: str) -> str:
        """Redact configured credentials from any outbound message.

        The tools never place a key in an exception message, so this is defence
        in depth - but an evidence ledger destined for a public trace is
        exactly the wrong place to discover that assumption was wrong.
        """
        text = str(message)
        try:
            settings = load_settings()
        except ValueError:
            return text
        for secret in (settings.football_data_org_key, settings.api_football_key):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text

    def _fail(
        self, tool_name: ToolName | None, code: ToolErrorCode, message: str, dedup_key: str | None = None
    ) -> ToolExecutionResult:
        failure = self.ledger.record_failure(
            ToolFailure(
                tool_name=tool_name,
                error_code=code,
                message=self._scrub(message),
                dedup_key=dedup_key,
            )
        )
        return ToolExecutionResult(ok=False, failure=failure)

    # ---- execution -----------------------------------------------------
    def execute(self, tool_name: ToolName | str, arguments: dict[str, Any] | None = None) -> ToolExecutionResult:
        """Execute one approved deterministic tool and record its evidence."""
        arguments = dict(arguments or {})

        # 1. Explicit registry lookup - never dynamic attribute access.
        try:
            resolved_name = ToolName(tool_name)
        except ValueError:
            return self._fail(None, ToolErrorCode.UNKNOWN_TOOL, f"unknown tool {tool_name!r}")
        spec = TOOL_REGISTRY.get(resolved_name)
        if spec is None:
            return self._fail(None, ToolErrorCode.UNKNOWN_TOOL, f"tool {resolved_name.value!r} is not agent-callable")

        # 2. Validate arguments BEFORE anything is executed.
        try:
            args = spec.args_model(**arguments)
        except ValidationError as exc:
            return self._fail(
                resolved_name,
                ToolErrorCode.INVALID_ARGUMENTS,
                f"invalid arguments for {resolved_name.value}: {exc.error_count()} validation error(s)",
            )

        # 3. Request-scoped dedup: identical call -> reuse existing evidence.
        dedup_key = self._dedup_key(resolved_name, args)
        existing = self.ledger.find_by_dedup_key(dedup_key)
        if existing is not None:
            return ToolExecutionResult(ok=True, evidence=existing, deduplicated=True)

        # 4. Execute the existing application tool.
        try:
            response = spec.invoke(self, args)
        except _ServiceNotConfigured as exc:
            return self._fail(resolved_name, ToolErrorCode.SERVICE_NOT_CONFIGURED, str(exc), dedup_key)
        except Exception as exc:  # noqa: BLE001 - re-raised below unless expected
            for error_type, code in _ERROR_CODES:
                if isinstance(exc, error_type):
                    return self._fail(resolved_name, code, str(exc), dedup_key)
            # Unexpected: a programming error must fail loudly, not be masked.
            raise

        # 5. Record structured evidence.
        payload = response.model_dump(mode="json")
        provenance = getattr(response, spec.provenance_field, None)
        source_kind = getattr(provenance, "source_kind", None)
        item = EvidenceItem(
            evidence_id=self.ledger.next_evidence_id(),
            tool_name=resolved_name,
            evidence_kind=spec.evidence_kind,
            args=args.model_dump(mode="json"),
            payload=payload,
            source_kind=getattr(source_kind, "value", str(source_kind)) if source_kind else "UNKNOWN",
            is_stale=bool(getattr(provenance, "is_stale", False)),
            dedup_key=dedup_key,
        )
        return ToolExecutionResult(ok=True, evidence=self.ledger.add(item))


class _ServiceNotConfigured(RuntimeError):
    """Internal signal: a service-backed tool was called with no service."""


__all__ = [
    "TOOL_REGISTRY",
    "EvidenceLedger",
    "FixturesArgs",
    "LiveMatchStateArgs",
    "LiveMatchesArgs",
    "OutcomeArgs",
    "ScorelineArgs",
    "StandingsArgs",
    "ToolExecutor",
]

"""PitchMind's agent layer.

Stages 1-2: deterministic intent routing, execution planning, and the single
doorway (`ToolExecutor`) through which the agent layer reaches PitchMind's
deterministic tools. No Claude runtime and no network access yet.

The deliberate design is hybrid: Python owns routing, tier selection and
budgets; Claude is invoked only in higher tiers where reasoning or delegation
materially improves the answer.
"""

from backend.app.agents.contracts import (
    CapabilityStatus,
    EvidenceItem,
    EvidenceKind,
    ExecutionPlan,
    ExecutionTier,
    Intent,
    RationaleCode,
    ResolvedEntity,
    RoutingDecision,
    RoutingGateCriteria,
    RoutingResult,
    SpecialistName,
    ToolErrorCode,
    ToolExecutionResult,
    ToolFailure,
    ToolName,
)
from backend.app.agents.executor import TOOL_REGISTRY, EvidenceLedger, ToolExecutor
from backend.app.agents.intents import classify_intent, normalize_question
from backend.app.agents.planner import extract_teams, route_and_plan

__all__ = [
    "CapabilityStatus",
    "ExecutionPlan",
    "ExecutionTier",
    "Intent",
    "RationaleCode",
    "ResolvedEntity",
    "RoutingDecision",
    "RoutingGateCriteria",
    "RoutingResult",
    "SpecialistName",
    "ToolErrorCode",
    "ToolExecutionResult",
    "ToolFailure",
    "ToolName",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceLedger",
    "ToolExecutor",
    "TOOL_REGISTRY",
    "classify_intent",
    "normalize_question",
    "extract_teams",
    "route_and_plan",
]

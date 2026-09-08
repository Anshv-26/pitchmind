"""PitchMind's agent layer.

Stage 1 scope: deterministic intent routing and execution planning only.
Nothing here executes a tool, calls Claude, or touches the network.

The deliberate design is hybrid: Python owns routing, tier selection and
budgets; Claude is invoked only in higher tiers where reasoning or delegation
materially improves the answer.
"""

from backend.app.agents.contracts import (
    CapabilityStatus,
    ExecutionPlan,
    ExecutionTier,
    Intent,
    RationaleCode,
    ResolvedEntity,
    RoutingDecision,
    RoutingGateCriteria,
    RoutingResult,
    SpecialistName,
    ToolName,
)
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
    "ToolName",
    "classify_intent",
    "normalize_question",
    "extract_teams",
    "route_and_plan",
]

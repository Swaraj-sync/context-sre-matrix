"""Typed models for the Context-Aware SRE benchmark."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


class TeamContext(str, Enum):
    """Primary team objective currently in focus."""

    PAYMENTS = "payments_consistency"
    SEARCH = "search_latency"
    BATCH = "batch_cost_efficiency"


class PRDecision(str, Enum):
    """Decision on the incoming pull request."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ROLLBACK = "ROLLBACK"


class WriteMode(str, Enum):
    """Database write strategy toggle."""

    SYNC = "SYNC"
    ASYNC = "ASYNC"


class ComputeShift(str, Enum):
    """Short-lived compute reallocation action."""

    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"
    UP = "UP"


class InfraToggles(BaseModel):
    """Operational dials available to the agent each step."""

    cache_ttl_s: int = Field(
        default=60,
        ge=15,
        le=300,
        description="Cache TTL in seconds. Higher values reduce latency but can increase risk.",
    )
    write_mode: WriteMode = Field(
        default=WriteMode.SYNC, description="Synchronous writes are safer but can increase latency."
    )
    compute_shift: ComputeShift = Field(
        default=ComputeShift.NEUTRAL,
        description="Temporary compute reallocation for a service group.",
    )


class PRImpact(BaseModel):
    """Expected metric deltas if a PR is approved."""

    latency_delta_ms: float
    availability_risk_delta_pct: float
    compute_cost_delta_pct: float
    error_rate_delta_pct: float
    consistency_delta: float = Field(
        description="Positive improves consistency, negative degrades consistency."
    )
    bottleneck: str = Field(description="Subsystem most impacted by the PR.")


class PullRequestSignal(BaseModel):
    """Incoming PR represented as an impact vector."""

    pr_id: str
    title: str
    service: str
    impact: PRImpact


class SystemHealth(BaseModel):
    """Live telemetry snapshot shown to the agent."""

    latency_ms: float
    availability_risk_pct: float
    compute_cost_index: float
    error_rate_pct: float
    cpu_utilization_pct: float
    ram_utilization_pct: float
    network_saturation_pct: float
    consistency_gap: float = Field(
        ge=0.0, le=1.0, description="0 means perfect consistency, 1 means severe inconsistency."
    )


class RewardBreakdown(BaseModel):
    """Transparent decomposition of the per-step score."""

    total: float = 0.0
    primary_objective: float = 0.0
    secondary_objective: float = 0.0
    decision_alignment: float = 0.0
    penalties: float = 0.0
    safety_breaches: List[str] = Field(default_factory=list)


class SREAction(Action):
    """Action emitted by the policy."""

    pr_decision: PRDecision
    target_pr_id: Optional[str] = Field(
        default=None,
        description="Optional PR identifier to rollback. If omitted, most recent approved PR is used.",
    )
    infra_toggles: InfraToggles = Field(default_factory=InfraToggles)


class SREObservation(Observation):
    """Observation returned by reset() and step()."""

    task_id: int
    scenario_id: str
    active_context: TeamContext
    incoming_pr: PullRequestSignal
    system_health: SystemHealth
    step_budget_remaining: int
    incident_active: bool
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    message: str = ""


class SREState(State):
    """Internal episode state used for introspection and grading."""

    task_id: int = 1
    scenario_id: str = ""
    seed: Optional[int] = None
    active_context: TeamContext = TeamContext.PAYMENTS
    incident_root_pr_id: Optional[str] = None
    incident_active: bool = False
    incident_resolved: bool = False
    cumulative_reward: float = 0.0
    approved_pr_ids: List[str] = Field(default_factory=list)
    context_switches_seen: int = 0
    episode_score: Optional[float] = None
    grade_metrics: Dict[str, float] = Field(default_factory=dict)

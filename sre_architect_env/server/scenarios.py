"""Scenario catalog for easy/medium/hard tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

try:
    from models import TeamContext
except ImportError:
    from sre_architect_env.models import TeamContext


@dataclass(frozen=True)
class PRScenario:
    """Single PR with deterministic impact vector."""

    pr_id: str
    title: str
    service: str
    latency_delta_ms: float
    availability_risk_delta_pct: float
    compute_cost_delta_pct: float
    error_rate_delta_pct: float
    consistency_delta: float
    bottleneck: str


@dataclass(frozen=True)
class HealthProfile:
    """Initial health profile sampled at reset()."""

    latency_ms: float
    availability_risk_pct: float
    compute_cost_index: float
    error_rate_pct: float
    cpu_utilization_pct: float
    ram_utilization_pct: float
    network_saturation_pct: float
    consistency_gap: float


@dataclass(frozen=True)
class TaskScenario:
    """Task scenario with context schedule and PR stream."""

    task_id: int
    scenario_id: str
    context_timeline: Tuple[TeamContext, ...]
    pr_stream: Tuple[PRScenario, ...]
    incident_root_pr_id: Optional[str] = None

    def validate(self) -> None:
        if len(self.context_timeline) != len(self.pr_stream):
            raise ValueError(
                f"{self.scenario_id} has mismatched timeline/pr length: "
                f"{len(self.context_timeline)} != {len(self.pr_stream)}"
            )


def _pr(
    pr_id: str,
    title: str,
    service: str,
    latency: float,
    risk: float,
    cost: float,
    error: float,
    consistency: float,
    bottleneck: str,
) -> PRScenario:
    return PRScenario(
        pr_id=pr_id,
        title=title,
        service=service,
        latency_delta_ms=latency,
        availability_risk_delta_pct=risk,
        compute_cost_delta_pct=cost,
        error_rate_delta_pct=error,
        consistency_delta=consistency,
        bottleneck=bottleneck,
    )


TASK_SCENARIOS: Dict[int, Tuple[TaskScenario, ...]] = {
    1: (
        TaskScenario(
            task_id=1,
            scenario_id="payments_consistency_a",
            context_timeline=(
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
            ),
            pr_stream=(
                _pr(
                    "pr-pay-cache-readthrough",
                    "Read-through cache for payment lookups",
                    "payments-api",
                    -8.0,
                    0.0030,
                    1.8,
                    0.20,
                    -0.20,
                    "db",
                ),
                _pr(
                    "pr-pay-sync-write-hardening",
                    "Enforce synchronous write quorum",
                    "ledger-writer",
                    5.5,
                    -0.0040,
                    2.5,
                    -0.15,
                    0.35,
                    "db",
                ),
                _pr(
                    "pr-pay-batch-settlement",
                    "Batch settlement compaction",
                    "settlement-worker",
                    -1.5,
                    0.0015,
                    -3.0,
                    0.25,
                    -0.15,
                    "cpu",
                ),
                _pr(
                    "pr-pay-idempotency-keys",
                    "Idempotency guard at gateway",
                    "payments-gateway",
                    1.2,
                    -0.0028,
                    0.8,
                    -0.30,
                    0.20,
                    "network",
                ),
                _pr(
                    "pr-pay-async-ledger-replicator",
                    "Async ledger replication for throughput",
                    "ledger-replicator",
                    -10.5,
                    0.0060,
                    -1.2,
                    0.45,
                    -0.45,
                    "db",
                ),
                _pr(
                    "pr-pay-db-index-tuning",
                    "Composite index optimization",
                    "payments-db",
                    -4.0,
                    -0.0012,
                    1.0,
                    -0.10,
                    0.10,
                    "db",
                ),
            ),
        ),
        TaskScenario(
            task_id=1,
            scenario_id="payments_consistency_b",
            context_timeline=(
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
            ),
            pr_stream=(
                _pr(
                    "pr-pay-fast-write-queue",
                    "Queue based write fanout",
                    "write-queue",
                    -9.0,
                    0.0045,
                    -0.5,
                    0.30,
                    -0.30,
                    "network",
                ),
                _pr(
                    "pr-pay-fraud-rule-cache",
                    "Fraud rule cache coalescing",
                    "fraud-service",
                    -3.5,
                    0.0012,
                    2.0,
                    -0.05,
                    -0.10,
                    "cpu",
                ),
                _pr(
                    "pr-pay-commit-log-audit",
                    "Commit log audit enforcement",
                    "ledger-writer",
                    2.3,
                    -0.0030,
                    1.5,
                    -0.15,
                    0.30,
                    "db",
                ),
                _pr(
                    "pr-pay-auth-precompute",
                    "Precompute auth token claims",
                    "auth-service",
                    -2.1,
                    0.0008,
                    1.4,
                    0.04,
                    -0.05,
                    "cpu",
                ),
                _pr(
                    "pr-pay-ledger-write-coalescer",
                    "Write coalescer for ledger sink",
                    "ledger-sink",
                    -7.0,
                    0.0058,
                    -1.7,
                    0.55,
                    -0.40,
                    "db",
                ),
                _pr(
                    "pr-pay-read-replica-stabilizer",
                    "Replica consistency reconciler",
                    "payments-db",
                    1.8,
                    -0.0022,
                    1.1,
                    -0.22,
                    0.25,
                    "db",
                ),
            ),
        ),
    ),
    2: (
        TaskScenario(
            task_id=2,
            scenario_id="context_switch_a",
            context_timeline=(
                TeamContext.SEARCH,
                TeamContext.SEARCH,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.SEARCH,
                TeamContext.PAYMENTS,
                TeamContext.SEARCH,
            ),
            pr_stream=(
                _pr(
                    "pr-search-hot-key-cache",
                    "Hot-key cache warm path",
                    "search-api",
                    -12.0,
                    0.0020,
                    2.8,
                    0.10,
                    -0.05,
                    "memory",
                ),
                _pr(
                    "pr-search-async-index-refresh",
                    "Async index refresh queue",
                    "indexer",
                    -6.5,
                    0.0032,
                    -0.8,
                    0.35,
                    -0.20,
                    "db",
                ),
                _pr(
                    "pr-payments-write-fsync",
                    "Force fsync on transaction commit",
                    "ledger-writer",
                    4.4,
                    -0.0032,
                    1.9,
                    -0.20,
                    0.28,
                    "db",
                ),
                _pr(
                    "pr-payments-cache-invalidation-delay",
                    "Delayed invalidation strategy",
                    "payments-cache",
                    -5.0,
                    0.0038,
                    -1.0,
                    0.22,
                    -0.22,
                    "memory",
                ),
                _pr(
                    "pr-search-query-plan-hints",
                    "Static planner hints",
                    "query-planner",
                    -7.2,
                    0.0007,
                    1.1,
                    -0.05,
                    0.05,
                    "cpu",
                ),
                _pr(
                    "pr-payments-dual-write-guard",
                    "Dual write guard rails",
                    "payment-router",
                    2.7,
                    -0.0026,
                    1.0,
                    -0.18,
                    0.24,
                    "network",
                ),
                _pr(
                    "pr-search-edge-response-cache",
                    "Edge cache for search responses",
                    "search-edge",
                    -9.0,
                    0.0025,
                    2.2,
                    0.15,
                    -0.10,
                    "network",
                ),
            ),
        ),
        TaskScenario(
            task_id=2,
            scenario_id="context_switch_b",
            context_timeline=(
                TeamContext.PAYMENTS,
                TeamContext.SEARCH,
                TeamContext.SEARCH,
                TeamContext.PAYMENTS,
                TeamContext.SEARCH,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
            ),
            pr_stream=(
                _pr(
                    "pr-payments-optimistic-flush",
                    "Optimistic async flush",
                    "ledger-writer",
                    -7.8,
                    0.0040,
                    -0.3,
                    0.42,
                    -0.32,
                    "db",
                ),
                _pr(
                    "pr-search-shard-preload",
                    "Shard preload for hot partitions",
                    "search-sharder",
                    -8.7,
                    0.0015,
                    2.0,
                    0.06,
                    -0.08,
                    "memory",
                ),
                _pr(
                    "pr-search-result-coalescing",
                    "Result coalescing pipeline",
                    "search-api",
                    -4.0,
                    0.0005,
                    1.0,
                    -0.04,
                    0.03,
                    "cpu",
                ),
                _pr(
                    "pr-payments-consistency-checkpoint",
                    "Checkpoint consistency validation",
                    "payments-gateway",
                    2.0,
                    -0.0025,
                    1.2,
                    -0.20,
                    0.25,
                    "db",
                ),
                _pr(
                    "pr-search-adaptive-ttl",
                    "Adaptive cache TTL by query class",
                    "cache-service",
                    -6.2,
                    0.0021,
                    1.7,
                    0.12,
                    -0.06,
                    "memory",
                ),
                _pr(
                    "pr-payments-retry-backoff",
                    "Retry backoff for downstream failures",
                    "payment-router",
                    0.8,
                    -0.0016,
                    0.5,
                    -0.10,
                    0.08,
                    "network",
                ),
                _pr(
                    "pr-payments-serializable-window",
                    "Serializable window enforcement",
                    "ledger-writer",
                    3.1,
                    -0.0030,
                    1.8,
                    -0.14,
                    0.30,
                    "db",
                ),
            ),
        ),
    ),
    3: (
        TaskScenario(
            task_id=3,
            scenario_id="incident_triage_a",
            context_timeline=(
                TeamContext.SEARCH,
                TeamContext.SEARCH,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.SEARCH,
                TeamContext.PAYMENTS,
                TeamContext.BATCH,
                TeamContext.PAYMENTS,
            ),
            pr_stream=(
                _pr(
                    "pr-search-cache-burst",
                    "Aggressive cache burst strategy",
                    "search-api",
                    -9.5,
                    0.0025,
                    2.0,
                    0.10,
                    -0.08,
                    "memory",
                ),
                _pr(
                    "pr-inc-async-write-rollout",
                    "Global async write rollout",
                    "global-io",
                    -13.0,
                    0.0180,
                    -2.5,
                    1.40,
                    -0.55,
                    "db",
                ),
                _pr(
                    "pr-payments-fraud-sampler",
                    "Fraud sampler optimization",
                    "fraud-service",
                    -1.8,
                    0.0010,
                    -0.8,
                    0.20,
                    -0.10,
                    "cpu",
                ),
                _pr(
                    "pr-payments-commit-guard",
                    "Commit guard hotfix",
                    "ledger-writer",
                    2.0,
                    -0.0030,
                    1.0,
                    -0.15,
                    0.25,
                    "db",
                ),
                _pr(
                    "pr-search-rate-limiter",
                    "Rate limiter smoothing",
                    "search-edge",
                    1.5,
                    -0.0008,
                    0.2,
                    -0.12,
                    0.05,
                    "network",
                ),
                _pr(
                    "pr-payments-ledger-trace",
                    "Ledger trace propagation",
                    "payments-api",
                    0.7,
                    -0.0006,
                    0.6,
                    -0.05,
                    0.04,
                    "cpu",
                ),
                _pr(
                    "pr-batch-compression-pass",
                    "Batch compression pass",
                    "batch-worker",
                    0.2,
                    0.0001,
                    -2.0,
                    0.05,
                    0.02,
                    "cpu",
                ),
                _pr(
                    "pr-payments-replica-fence",
                    "Replica fence enforcement",
                    "payments-db",
                    1.6,
                    -0.0022,
                    1.2,
                    -0.20,
                    0.22,
                    "db",
                ),
            ),
            incident_root_pr_id="pr-inc-async-write-rollout",
        ),
        TaskScenario(
            task_id=3,
            scenario_id="incident_triage_b",
            context_timeline=(
                TeamContext.PAYMENTS,
                TeamContext.SEARCH,
                TeamContext.SEARCH,
                TeamContext.PAYMENTS,
                TeamContext.PAYMENTS,
                TeamContext.BATCH,
                TeamContext.SEARCH,
                TeamContext.PAYMENTS,
            ),
            pr_stream=(
                _pr(
                    "pr-payments-cache-queue",
                    "Queue-backed payment cache",
                    "payments-cache",
                    -6.8,
                    0.0038,
                    -1.3,
                    0.40,
                    -0.30,
                    "memory",
                ),
                _pr(
                    "pr-search-index-fast-commit",
                    "Fast commit index path",
                    "indexer",
                    -7.0,
                    0.0015,
                    -0.6,
                    0.30,
                    -0.18,
                    "db",
                ),
                _pr(
                    "pr-inc-shared-event-loop",
                    "Shared event loop for all services",
                    "runtime-core",
                    -11.5,
                    0.0160,
                    -2.0,
                    1.10,
                    -0.48,
                    "cpu",
                ),
                _pr(
                    "pr-payments-journal-integrity",
                    "Journal integrity checks",
                    "ledger-writer",
                    2.2,
                    -0.0032,
                    1.4,
                    -0.10,
                    0.28,
                    "db",
                ),
                _pr(
                    "pr-payments-key-rotation",
                    "Payment key rotation",
                    "payments-gateway",
                    0.5,
                    -0.0008,
                    0.4,
                    -0.03,
                    0.06,
                    "network",
                ),
                _pr(
                    "pr-batch-spot-rebalance",
                    "Spot rebalance policy",
                    "batch-scheduler",
                    0.0,
                    0.0003,
                    -3.2,
                    0.05,
                    0.00,
                    "cpu",
                ),
                _pr(
                    "pr-search-burst-control",
                    "Burst control hard limits",
                    "search-edge",
                    2.1,
                    -0.0012,
                    0.8,
                    -0.20,
                    0.04,
                    "network",
                ),
                _pr(
                    "pr-payments-write-audit",
                    "Write audit reconciler",
                    "ledger-reconciler",
                    1.8,
                    -0.0016,
                    1.0,
                    -0.12,
                    0.20,
                    "db",
                ),
            ),
            incident_root_pr_id="pr-inc-shared-event-loop",
        ),
    ),
}

for _task_entries in TASK_SCENARIOS.values():
    for _scenario in _task_entries:
        _scenario.validate()


HEALTHY_PROFILES: Dict[int, Tuple[HealthProfile, ...]] = {
    1: (
        HealthProfile(56.0, 0.0030, 102.0, 0.40, 52.0, 48.0, 44.0, 0.08),
        HealthProfile(60.0, 0.0022, 98.0, 0.35, 49.0, 46.0, 40.0, 0.06),
    ),
    2: (
        HealthProfile(63.0, 0.0045, 106.0, 0.55, 58.0, 54.0, 49.0, 0.10),
        HealthProfile(59.0, 0.0035, 104.0, 0.45, 55.0, 51.0, 47.0, 0.09),
    ),
    3: (
        HealthProfile(68.0, 0.0050, 110.0, 0.70, 61.0, 58.0, 52.0, 0.12),
        HealthProfile(66.0, 0.0040, 108.0, 0.62, 60.0, 56.0, 50.0, 0.11),
    ),
}

DEGRADED_PROFILES: Dict[int, Tuple[HealthProfile, ...]] = {
    1: (
        HealthProfile(88.0, 0.0110, 114.0, 1.20, 71.0, 69.0, 63.0, 0.22),
        HealthProfile(94.0, 0.0130, 118.0, 1.40, 75.0, 72.0, 66.0, 0.26),
    ),
    2: (
        HealthProfile(92.0, 0.0140, 116.0, 1.60, 73.0, 70.0, 67.0, 0.24),
        HealthProfile(98.0, 0.0165, 121.0, 1.90, 77.0, 74.0, 70.0, 0.28),
    ),
    3: (
        HealthProfile(104.0, 0.0200, 125.0, 2.40, 80.0, 77.0, 73.0, 0.34),
        HealthProfile(111.0, 0.0240, 130.0, 2.90, 84.0, 82.0, 76.0, 0.38),
    ),
}


def supported_task_ids() -> Tuple[int, ...]:
    """Return supported task identifiers."""
    return tuple(sorted(TASK_SCENARIOS.keys()))


def scenarios_for_task(task_id: int) -> Tuple[TaskScenario, ...]:
    """Return scenario variants for a task."""
    if task_id not in TASK_SCENARIOS:
        raise ValueError(f"Unsupported task id: {task_id}")
    return TASK_SCENARIOS[task_id]


def find_scenario(task_id: int, scenario_id: str) -> TaskScenario:
    """Lookup a scenario by task and scenario identifier."""
    for scenario in scenarios_for_task(task_id):
        if scenario.scenario_id == scenario_id:
            return scenario
    raise ValueError(f"Unknown scenario_id={scenario_id!r} for task {task_id}")


def health_profiles(task_id: int, degraded: bool) -> Tuple[HealthProfile, ...]:
    """Return available initial health profiles for a task."""
    table = DEGRADED_PROFILES if degraded else HEALTHY_PROFILES
    return table[task_id]


def iter_all_scenarios() -> Iterable[TaskScenario]:
    """Yield all task scenarios."""
    for task_id in supported_task_ids():
        for scenario in scenarios_for_task(task_id):
            yield scenario


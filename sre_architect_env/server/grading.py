"""Reward and grader utilities for the SRE benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional

try:
    from models import (
        ComputeShift,
        PRDecision,
        PullRequestSignal,
        RewardBreakdown,
        SREAction,
        SystemHealth,
        TeamContext,
        WriteMode,
    )
except ImportError:
    from sre_architect_env.models import (
        ComputeShift,
        PRDecision,
        PullRequestSignal,
        RewardBreakdown,
        SREAction,
        SystemHealth,
        TeamContext,
        WriteMode,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _primary_secondary_scores(context: TeamContext, health: SystemHealth) -> tuple[float, float]:
    """Compute objective-aligned scores for the given context."""
    if context == TeamContext.PAYMENTS:
        primary = _clamp01(1.0 - (health.availability_risk_pct / 0.015))
        consistency_score = _clamp01(1.0 - (health.consistency_gap / 0.45))
        reliability_score = _clamp01(1.0 - (health.error_rate_pct / 2.0))
        secondary = 0.65 * consistency_score + 0.35 * reliability_score
        return primary, secondary

    if context == TeamContext.SEARCH:
        primary = _clamp01((85.0 - health.latency_ms) / 35.0)
        risk_score = _clamp01(1.0 - (health.availability_risk_pct / 0.04))
        cost_score = _clamp01((170.0 - health.compute_cost_index) / 80.0)
        secondary = 0.70 * risk_score + 0.30 * cost_score
        return primary, secondary

    # TeamContext.BATCH
    primary = _clamp01((150.0 - health.compute_cost_index) / 70.0)
    latency_score = _clamp01((135.0 - health.latency_ms) / 55.0)
    risk_score = _clamp01(1.0 - (health.availability_risk_pct / 0.05))
    secondary = 0.50 * latency_score + 0.50 * risk_score
    return primary, secondary


def _decision_alignment(
    context: TeamContext,
    action: SREAction,
    incoming_pr: PullRequestSignal,
    rollback_target_matched_root: bool,
) -> float:
    """Score whether the selected PR decision is context-appropriate."""
    impact = incoming_pr.impact
    risky_for_payments = (
        impact.availability_risk_delta_pct > 0.0025 or impact.consistency_delta < -0.18
    )
    latency_boost = impact.latency_delta_ms < -5.0
    strong_safety_upgrade = (
        impact.availability_risk_delta_pct < -0.0020 and impact.consistency_delta > 0.12
    )

    if action.pr_decision == PRDecision.ROLLBACK:
        if rollback_target_matched_root:
            return 1.0
        return 0.55

    if context == TeamContext.PAYMENTS:
        if action.pr_decision == PRDecision.REJECT and risky_for_payments:
            return 1.0
        if action.pr_decision == PRDecision.APPROVE and strong_safety_upgrade:
            return 1.0
        if action.pr_decision == PRDecision.APPROVE and risky_for_payments:
            return 0.0
        return 0.65

    if context == TeamContext.SEARCH:
        if action.pr_decision == PRDecision.APPROVE and latency_boost:
            return 1.0
        if action.pr_decision == PRDecision.REJECT and latency_boost:
            return 0.35
        if action.pr_decision == PRDecision.APPROVE and impact.availability_risk_delta_pct > 0.0050:
            return 0.25
        return 0.70

    # TeamContext.BATCH
    if action.pr_decision == PRDecision.APPROVE and impact.compute_cost_delta_pct < -1.0:
        return 1.0
    if action.pr_decision == PRDecision.REJECT and impact.compute_cost_delta_pct < -1.0:
        return 0.40
    return 0.70


def _safety_breaches(
    context: TeamContext,
    health: SystemHealth,
    action: SREAction,
    incident_active: bool,
) -> List[str]:
    breaches: List[str] = []

    if context == TeamContext.PAYMENTS and health.availability_risk_pct > 0.010:
        breaches.append("payments_risk_breach")
    if context == TeamContext.PAYMENTS and action.infra_toggles.write_mode == WriteMode.ASYNC:
        breaches.append("payments_async_write_violation")
    if context == TeamContext.SEARCH and health.latency_ms > 95.0:
        breaches.append("search_latency_breach")
    if context == TeamContext.SEARCH and health.availability_risk_pct > 0.035:
        breaches.append("search_risk_breach")
    if context == TeamContext.BATCH and health.compute_cost_index > 155.0:
        breaches.append("batch_cost_breach")

    if health.error_rate_pct > 4.0:
        breaches.append("global_error_budget_breach")

    if incident_active and action.infra_toggles.compute_shift != ComputeShift.UP:
        breaches.append("incident_unmitigated")

    return breaches


def compute_step_reward(
    context: TeamContext,
    action: SREAction,
    incoming_pr: PullRequestSignal,
    health_before: SystemHealth,
    health_after: SystemHealth,
    incident_active: bool,
    rollback_target_matched_root: bool,
) -> RewardBreakdown:
    """Compute deterministic reward components and clamp to [0, 1]."""
    del health_before  # Reserved for future differential shaping.
    primary_score, secondary_score = _primary_secondary_scores(context, health_after)
    alignment_score = _decision_alignment(
        context=context,
        action=action,
        incoming_pr=incoming_pr,
        rollback_target_matched_root=rollback_target_matched_root,
    )
    breaches = _safety_breaches(context, health_after, action, incident_active)

    penalties = 0.18 * len(breaches)
    if incident_active and action.pr_decision != PRDecision.ROLLBACK:
        penalties += 0.08

    raw_total = (0.55 * primary_score) + (0.30 * secondary_score) + (0.15 * alignment_score) - penalties
    total = _clamp01(raw_total)

    return RewardBreakdown(
        total=round(total, 6),
        primary_objective=round(primary_score, 6),
        secondary_objective=round(secondary_score, 6),
        decision_alignment=round(alignment_score, 6),
        penalties=round(penalties, 6),
        safety_breaches=breaches,
    )


@dataclass(frozen=True)
class EpisodeStepSnapshot:
    """Step-level data retained for end-of-episode grading."""

    context: TeamContext
    reward_breakdown: RewardBreakdown
    health_after: SystemHealth


def _context_adaptation_score(steps: List[EpisodeStepSnapshot]) -> float:
    if len(steps) < 2:
        return 1.0

    switch_scores: List[float] = []
    for i in range(1, len(steps)):
        if steps[i - 1].context != steps[i].context:
            switch_scores.append(steps[i].reward_breakdown.decision_alignment)

    if not switch_scores:
        return 1.0
    return _clamp01(mean(switch_scores))


def _safety_score(steps: List[EpisodeStepSnapshot]) -> float:
    if not steps:
        return 0.0
    total_breaches = sum(len(step.reward_breakdown.safety_breaches) for step in steps)
    max_allowed = max(1, len(steps) * 3)
    return _clamp01(1.0 - (total_breaches / max_allowed))


def grade_episode(
    task_id: int,
    steps: List[EpisodeStepSnapshot],
    incident_root_pr_id: Optional[str],
    rolled_back_root: bool,
    incident_resolved: bool,
) -> Dict[str, float]:
    """Compute final task score and diagnostic metrics."""
    if not steps:
        return {
            "overall_score": 0.0,
            "avg_step_score": 0.0,
            "safety_score": 0.0,
            "adaptation_score": 0.0,
            "incident_score": 0.0,
        }

    avg_step_score = _clamp01(mean(step.reward_breakdown.total for step in steps))
    safety_score = _safety_score(steps)
    adaptation_score = _context_adaptation_score(steps)

    final_health = steps[-1].health_after
    stabilization = _clamp01(
        ((3.5 - final_health.error_rate_pct) / 3.5 + (0.035 - final_health.availability_risk_pct) / 0.035)
        / 2.0
    )
    incident_score = _clamp01(
        0.45 * (1.0 if rolled_back_root else 0.0)
        + 0.45 * (1.0 if incident_resolved else 0.0)
        + 0.10 * stabilization
    )

    if task_id == 1:
        overall = _clamp01(0.62 * avg_step_score + 0.38 * safety_score)
    elif task_id == 2:
        overall = _clamp01(0.45 * avg_step_score + 0.25 * safety_score + 0.30 * adaptation_score)
    else:
        # Hard task emphasizes incident diagnosis + rollback quality.
        overall = _clamp01(0.32 * avg_step_score + 0.18 * safety_score + 0.50 * incident_score)

    result: Dict[str, float] = {
        "overall_score": round(overall, 6),
        "avg_step_score": round(avg_step_score, 6),
        "safety_score": round(safety_score, 6),
        "adaptation_score": round(adaptation_score, 6),
        "incident_score": round(incident_score, 6),
    }
    if incident_root_pr_id:
        result["incident_root_present"] = 1.0
    return result


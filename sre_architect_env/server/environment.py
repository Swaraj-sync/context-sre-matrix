"""Core environment implementation for context-aware SRE decision making."""

from __future__ import annotations

import random
import uuid
from typing import Dict, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from models import (
        ComputeShift,
        PRDecision,
        PRImpact,
        PullRequestSignal,
        RewardBreakdown,
        SREAction,
        SREObservation,
        SREState,
        SystemHealth,
        TeamContext,
        WriteMode,
    )
    from server.grading import EpisodeStepSnapshot, compute_step_reward, grade_episode
    from server.scenarios import (
        HealthProfile,
        PRScenario,
        TaskScenario,
        find_scenario,
        health_profiles,
        scenarios_for_task,
        supported_task_ids,
    )
except ImportError:
    from sre_architect_env.models import (
        ComputeShift,
        PRDecision,
        PRImpact,
        PullRequestSignal,
        RewardBreakdown,
        SREAction,
        SREObservation,
        SREState,
        SystemHealth,
        TeamContext,
        WriteMode,
    )
    from sre_architect_env.server.grading import (
        EpisodeStepSnapshot,
        compute_step_reward,
        grade_episode,
    )
    from sre_architect_env.server.scenarios import (
        HealthProfile,
        PRScenario,
        TaskScenario,
        find_scenario,
        health_profiles,
        scenarios_for_task,
        supported_task_ids,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class SREArchitectEnvironment(Environment[SREAction, SREObservation, SREState]):
    """Deterministic, context-aware microservice SRE benchmark environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random()
        self._scenario: Optional[TaskScenario] = None
        self._health = SystemHealth(
            latency_ms=60.0,
            availability_risk_pct=0.003,
            compute_cost_index=100.0,
            error_rate_pct=0.5,
            cpu_utilization_pct=50.0,
            ram_utilization_pct=48.0,
            network_saturation_pct=45.0,
            consistency_gap=0.08,
        )
        self._state = SREState()
        self._step_index = 0
        self._done = False
        self._approved_impacts: Dict[str, PRImpact] = {}
        self._approved_order: list[str] = []
        self._episode_steps: list[EpisodeStepSnapshot] = []
        self._rolled_back_root = False
        self._last_message = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SREObservation:
        task_id = int(kwargs.get("task_id", 1))
        if task_id not in supported_task_ids():
            raise ValueError(f"Unsupported task id: {task_id}")

        self._rng = random.Random(seed)
        scenario_id = kwargs.get("scenario_id")
        if scenario_id:
            scenario = find_scenario(task_id, scenario_id)
        else:
            scenario = self._rng.choice(scenarios_for_task(task_id))
        self._scenario = scenario

        degraded_start = kwargs.get("degraded_start")
        if degraded_start is None:
            degraded_start = self._rng.random() < 0.45
        profile = self._rng.choice(health_profiles(task_id, bool(degraded_start)))
        self._health = self._profile_to_health(profile)
        self._normalize_health()

        self._step_index = 0
        self._done = False
        self._approved_impacts = {}
        self._approved_order = []
        self._episode_steps = []
        self._rolled_back_root = False

        active_context = scenario.context_timeline[0]
        self._state = SREState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            scenario_id=scenario.scenario_id,
            seed=seed,
            active_context=active_context,
            incident_root_pr_id=scenario.incident_root_pr_id,
            incident_active=False,
            incident_resolved=False,
            cumulative_reward=0.0,
            approved_pr_ids=[],
            context_switches_seen=0,
            episode_score=None,
            grade_metrics={},
        )

        if task_id == 3 and degraded_start and self._rng.random() < 0.40:
            self._state.incident_active = True
            self._health.error_rate_pct += 0.6
            self._health.availability_risk_pct += 0.004

        self._normalize_health()
        self._last_message = (
            f"Reset task={task_id} scenario={scenario.scenario_id} "
            f"degraded_start={bool(degraded_start)} active_context={active_context.value}"
        )
        return self._build_observation(
            reward=None,
            done=False,
            message=self._last_message,
            reward_breakdown=RewardBreakdown(),
        )

    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SREObservation:
        del timeout_s, kwargs

        if self._scenario is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        if self._done:
            return self._build_observation(
                reward=0.0,
                done=True,
                message="Episode already completed. Call reset() for a new episode.",
                reward_breakdown=RewardBreakdown(total=0.0),
            )

        context = self._scenario.context_timeline[self._step_index]
        self._state.active_context = context
        current_pr = self._scenario.pr_stream[self._step_index]
        current_signal = self._to_signal(current_pr)

        health_before = self._health.model_copy(deep=True)
        message_parts = []

        self._apply_infra_toggles(action)

        rollback_target = None
        rollback_target_matched_root = False
        if action.pr_decision == PRDecision.APPROVE:
            self._apply_pr_impact(current_signal.impact, current_pr.bottleneck, direction=1)
            self._approved_impacts[current_pr.pr_id] = current_signal.impact.model_copy(deep=True)
            self._approved_order.append(current_pr.pr_id)
            message_parts.append(f"Approved {current_pr.pr_id}")
        elif action.pr_decision == PRDecision.REJECT:
            message_parts.append(f"Rejected {current_pr.pr_id}")
        else:
            rollback_target = self._rollback(action.target_pr_id)
            if rollback_target:
                message_parts.append(f"Rolled back {rollback_target}")
                if rollback_target == self._state.incident_root_pr_id:
                    rollback_target_matched_root = True
                    self._rolled_back_root = True
            else:
                message_parts.append("Rollback requested but no approved PR matched target.")

        self._apply_background_drift(context)
        self._update_incident_state(
            current_pr=current_pr,
            action=action,
            rolled_back_pr_id=rollback_target,
        )
        self._normalize_health()

        health_after = self._health.model_copy(deep=True)
        reward_breakdown = compute_step_reward(
            context=context,
            action=action,
            incoming_pr=current_signal,
            health_before=health_before,
            health_after=health_after,
            incident_active=self._state.incident_active,
            rollback_target_matched_root=rollback_target_matched_root,
        )

        step_reward = reward_breakdown.total
        self._state.cumulative_reward = round(self._state.cumulative_reward + step_reward, 6)
        self._state.step_count += 1
        self._state.approved_pr_ids = list(self._approved_order)

        self._episode_steps.append(
            EpisodeStepSnapshot(
                context=context,
                reward_breakdown=reward_breakdown,
                health_after=health_after.model_copy(deep=True),
            )
        )

        self._step_index += 1
        self._done = self._step_index >= len(self._scenario.pr_stream)

        if self._done:
            final_grade = grade_episode(
                task_id=self._state.task_id,
                steps=self._episode_steps,
                incident_root_pr_id=self._state.incident_root_pr_id,
                rolled_back_root=self._rolled_back_root,
                incident_resolved=self._state.incident_resolved,
            )
            self._state.episode_score = final_grade["overall_score"]
            self._state.grade_metrics = final_grade
            message_parts.append(
                f"Episode complete. score={final_grade['overall_score']:.3f} "
                f"safety={final_grade['safety_score']:.3f}"
            )
        else:
            next_context = self._scenario.context_timeline[self._step_index]
            if next_context != context:
                self._state.context_switches_seen += 1
            self._state.active_context = next_context
            message_parts.append(f"Next context={next_context.value}")

        self._last_message = " ".join(message_parts)
        return self._build_observation(
            reward=step_reward,
            done=self._done,
            message=self._last_message,
            reward_breakdown=reward_breakdown,
        )

    @property
    def state(self) -> SREState:
        return self._state

    def _profile_to_health(self, profile: HealthProfile) -> SystemHealth:
        return SystemHealth(
            latency_ms=profile.latency_ms,
            availability_risk_pct=profile.availability_risk_pct,
            compute_cost_index=profile.compute_cost_index,
            error_rate_pct=profile.error_rate_pct,
            cpu_utilization_pct=profile.cpu_utilization_pct,
            ram_utilization_pct=profile.ram_utilization_pct,
            network_saturation_pct=profile.network_saturation_pct,
            consistency_gap=profile.consistency_gap,
        )

    def _to_signal(self, pr: PRScenario) -> PullRequestSignal:
        return PullRequestSignal(
            pr_id=pr.pr_id,
            title=pr.title,
            service=pr.service,
            impact=PRImpact(
                latency_delta_ms=pr.latency_delta_ms,
                availability_risk_delta_pct=pr.availability_risk_delta_pct,
                compute_cost_delta_pct=pr.compute_cost_delta_pct,
                error_rate_delta_pct=pr.error_rate_delta_pct,
                consistency_delta=pr.consistency_delta,
                bottleneck=pr.bottleneck,
            ),
        )

    def _terminal_signal(self) -> PullRequestSignal:
        return PullRequestSignal(
            pr_id="episode_complete",
            title="No incoming PR - episode complete",
            service="control-plane",
            impact=PRImpact(
                latency_delta_ms=0.0,
                availability_risk_delta_pct=0.0,
                compute_cost_delta_pct=0.0,
                error_rate_delta_pct=0.0,
                consistency_delta=0.0,
                bottleneck="none",
            ),
        )

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        message: str,
        reward_breakdown: RewardBreakdown,
    ) -> SREObservation:
        if self._scenario and self._step_index < len(self._scenario.pr_stream):
            incoming_pr = self._to_signal(self._scenario.pr_stream[self._step_index])
        else:
            incoming_pr = self._terminal_signal()

        step_budget_remaining = 0
        if self._scenario:
            step_budget_remaining = max(0, len(self._scenario.pr_stream) - self._step_index)

        return SREObservation(
            done=done,
            reward=reward,
            task_id=self._state.task_id,
            scenario_id=self._state.scenario_id,
            active_context=self._state.active_context,
            incoming_pr=incoming_pr,
            system_health=self._health.model_copy(deep=True),
            step_budget_remaining=step_budget_remaining,
            incident_active=self._state.incident_active,
            reward_breakdown=reward_breakdown,
            message=message,
        )

    def _apply_infra_toggles(self, action: SREAction) -> None:
        toggles = action.infra_toggles

        ttl_delta = toggles.cache_ttl_s - 60
        self._health.latency_ms -= ttl_delta * 0.05
        self._health.availability_risk_pct += ttl_delta * 0.00003
        self._health.consistency_gap += ttl_delta * 0.00035

        if toggles.write_mode == WriteMode.ASYNC:
            self._health.latency_ms -= 4.0
            self._health.availability_risk_pct += 0.0022
            self._health.error_rate_pct += 0.12
            self._health.consistency_gap += 0.08
        else:
            self._health.latency_ms += 1.5
            self._health.availability_risk_pct -= 0.0008
            self._health.error_rate_pct -= 0.03
            self._health.consistency_gap -= 0.04

        if toggles.compute_shift == ComputeShift.UP:
            self._health.latency_ms -= 5.0
            self._health.error_rate_pct -= 0.20
            self._health.compute_cost_index += 4.0
            self._health.cpu_utilization_pct -= 5.0
            self._health.ram_utilization_pct -= 4.0
            self._health.network_saturation_pct -= 3.0
        elif toggles.compute_shift == ComputeShift.DOWN:
            self._health.latency_ms += 4.0
            self._health.error_rate_pct += 0.18
            self._health.compute_cost_index -= 3.0
            self._health.cpu_utilization_pct += 3.0
            self._health.ram_utilization_pct += 2.0
            self._health.network_saturation_pct += 2.0

    def _apply_pr_impact(self, impact: PRImpact, bottleneck: str, direction: int) -> None:
        self._health.latency_ms += direction * impact.latency_delta_ms
        self._health.availability_risk_pct += direction * impact.availability_risk_delta_pct
        self._health.compute_cost_index += direction * impact.compute_cost_delta_pct
        self._health.error_rate_pct += direction * impact.error_rate_delta_pct
        self._health.consistency_gap -= direction * (impact.consistency_delta * 0.10)

        pressure = 1.0 if direction > 0 else -1.0
        if bottleneck == "db":
            self._health.cpu_utilization_pct += 1.5 * pressure
            self._health.network_saturation_pct += 1.0 * pressure
        elif bottleneck == "cpu":
            self._health.cpu_utilization_pct += 2.0 * pressure
            self._health.ram_utilization_pct += 0.5 * pressure
        elif bottleneck == "memory":
            self._health.ram_utilization_pct += 2.2 * pressure
            self._health.cpu_utilization_pct += 0.4 * pressure
        elif bottleneck == "network":
            self._health.network_saturation_pct += 2.0 * pressure
            self._health.cpu_utilization_pct += 0.5 * pressure

    def _rollback(self, target_pr_id: Optional[str]) -> Optional[str]:
        if not self._approved_order:
            return None

        rollback_id = target_pr_id or self._approved_order[-1]
        impact = self._approved_impacts.get(rollback_id)
        if impact is None:
            return None

        # We do not have the original bottleneck in approved map, use impact hint.
        self._apply_pr_impact(impact=impact, bottleneck=impact.bottleneck, direction=-1)
        self._approved_impacts.pop(rollback_id, None)
        self._approved_order = [pr_id for pr_id in self._approved_order if pr_id != rollback_id]
        return rollback_id

    def _apply_background_drift(self, context: TeamContext) -> None:
        if context == TeamContext.PAYMENTS:
            self._health.latency_ms += 0.8
            self._health.availability_risk_pct += 0.0006
            self._health.compute_cost_index += 0.5
        elif context == TeamContext.SEARCH:
            self._health.latency_ms += 1.2
            self._health.network_saturation_pct += 1.5
            self._health.compute_cost_index += 0.3
        else:
            self._health.latency_ms += 0.4
            self._health.compute_cost_index -= 0.2
            self._health.availability_risk_pct += 0.0002

    def _update_incident_state(
        self,
        current_pr: PRScenario,
        action: SREAction,
        rolled_back_pr_id: Optional[str],
    ) -> None:
        if self._state.task_id != 3:
            return

        root_id = self._state.incident_root_pr_id
        if not self._state.incident_active:
            should_trigger = False
            if action.pr_decision == PRDecision.APPROVE and root_id and current_pr.pr_id == root_id:
                should_trigger = True
            if self._health.error_rate_pct > 2.6 or self._health.availability_risk_pct > 0.022:
                should_trigger = True
            if should_trigger:
                self._state.incident_active = True

        if self._state.incident_active:
            self._health.latency_ms += 8.0
            self._health.error_rate_pct += 0.75
            self._health.availability_risk_pct += 0.004
            self._health.cpu_utilization_pct += 3.5
            self._health.network_saturation_pct += 4.0

            if rolled_back_pr_id and rolled_back_pr_id == root_id:
                self._rolled_back_root = True

            is_stabilizing_action = (
                action.infra_toggles.compute_shift == ComputeShift.UP
                and action.infra_toggles.write_mode == WriteMode.SYNC
            )
            if (
                self._rolled_back_root
                and is_stabilizing_action
                and self._health.error_rate_pct < 3.0
                and self._health.availability_risk_pct < 0.030
            ):
                self._state.incident_active = False
                self._state.incident_resolved = True

    def _normalize_health(self) -> None:
        self._health.latency_ms = round(_clamp(self._health.latency_ms, 15.0, 450.0), 6)
        self._health.availability_risk_pct = round(
            _clamp(self._health.availability_risk_pct, 0.0, 0.200), 6
        )
        self._health.compute_cost_index = round(
            _clamp(self._health.compute_cost_index, 30.0, 260.0), 6
        )
        self._health.error_rate_pct = round(_clamp(self._health.error_rate_pct, 0.0, 25.0), 6)
        self._health.cpu_utilization_pct = round(
            _clamp(self._health.cpu_utilization_pct, 0.0, 100.0), 6
        )
        self._health.ram_utilization_pct = round(
            _clamp(self._health.ram_utilization_pct, 0.0, 100.0), 6
        )
        self._health.network_saturation_pct = round(
            _clamp(self._health.network_saturation_pct, 0.0, 100.0), 6
        )
        self._health.consistency_gap = round(_clamp(self._health.consistency_gap, 0.0, 1.0), 6)

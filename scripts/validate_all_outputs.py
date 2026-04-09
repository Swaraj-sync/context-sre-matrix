#!/usr/bin/env python3
"""Check EVERY numeric output for exact 0.0 or 1.0 — step rewards, episode scores, all metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parents[1] / "sre_architect_env"
sys.path.insert(0, str(ENV_DIR))

from models import (
    ComputeShift, InfraToggles, PRDecision, SREAction, TeamContext, WriteMode,
)
from server.environment import SREArchitectEnvironment
from server.scenarios import iter_all_scenarios


def _make_policies():
    def always_reject(obs):
        return SREAction(pr_decision=PRDecision.REJECT, infra_toggles=InfraToggles(cache_ttl_s=60))

    def always_approve(obs):
        return SREAction(pr_decision=PRDecision.APPROVE, infra_toggles=InfraToggles(
            cache_ttl_s=120, write_mode=WriteMode.ASYNC, compute_shift=ComputeShift.UP))

    def always_rollback(obs):
        return SREAction(pr_decision=PRDecision.ROLLBACK, target_pr_id=obs.incoming_pr.pr_id,
            infra_toggles=InfraToggles(cache_ttl_s=45, write_mode=WriteMode.SYNC, compute_shift=ComputeShift.UP))

    def context_aware(obs):
        if obs.incident_active:
            return SREAction(pr_decision=PRDecision.ROLLBACK, target_pr_id=obs.incoming_pr.pr_id,
                infra_toggles=InfraToggles(cache_ttl_s=45, write_mode=WriteMode.SYNC, compute_shift=ComputeShift.UP))
        if obs.active_context == TeamContext.SEARCH:
            return SREAction(pr_decision=PRDecision.APPROVE, infra_toggles=InfraToggles(
                cache_ttl_s=120, write_mode=WriteMode.ASYNC, compute_shift=ComputeShift.UP))
        if obs.active_context == TeamContext.BATCH:
            return SREAction(pr_decision=PRDecision.APPROVE, infra_toggles=InfraToggles(
                cache_ttl_s=90, write_mode=WriteMode.SYNC, compute_shift=ComputeShift.DOWN))
        return SREAction(pr_decision=PRDecision.REJECT, infra_toggles=InfraToggles(
            cache_ttl_s=45, write_mode=WriteMode.SYNC, compute_shift=ComputeShift.NEUTRAL))

    return {"reject": always_reject, "approve": always_approve, "rollback": always_rollback, "aware": context_aware}


def main() -> int:
    policies = _make_policies()
    seeds = [0, 1, 7, 42, 99, 123, 256, 999]
    step_violations = []
    episode_violations = []
    total_runs = 0
    total_steps = 0

    for scenario in iter_all_scenarios():
        for degraded in (False, True):
            for seed in seeds:
                for pname, pfn in policies.items():
                    total_runs += 1
                    label = f"t={scenario.task_id} s={scenario.scenario_id} seed={seed} d={degraded} p={pname}"
                    env = SREArchitectEnvironment()
                    obs = env.reset(seed=seed, task_id=scenario.task_id,
                                    scenario_id=scenario.scenario_id, degraded_start=degraded)
                    step = 0
                    while not obs.done:
                        obs = env.step(pfn(obs))
                        step += 1
                        total_steps += 1
                        r = obs.reward
                        rb = obs.reward_breakdown

                        if r is not None and (r == 0.0 or r == 1.0):
                            step_violations.append(f"{label} step={step}: obs.reward={r}")
                        if rb.total == 0.0 or rb.total == 1.0:
                            step_violations.append(f"{label} step={step}: reward_breakdown.total={rb.total}")
                        if rb.primary_objective == 0.0 or rb.primary_objective == 1.0:
                            step_violations.append(f"{label} step={step}: primary_objective={rb.primary_objective}")
                        if rb.secondary_objective == 0.0 or rb.secondary_objective == 1.0:
                            step_violations.append(f"{label} step={step}: secondary_objective={rb.secondary_objective}")
                        if rb.decision_alignment == 0.0 or rb.decision_alignment == 1.0:
                            step_violations.append(f"{label} step={step}: decision_alignment={rb.decision_alignment}")

                    state = env.state
                    for key, val in state.grade_metrics.items():
                        if val == 0.0 or val == 1.0:
                            episode_violations.append(f"{label}: grade_metrics[{key}]={val}")
                    if state.episode_score is not None and (state.episode_score == 0.0 or state.episode_score == 1.0):
                        episode_violations.append(f"{label}: episode_score={state.episode_score}")

    print(f"Ran {total_runs} episodes, {total_steps} total steps.\n")

    if episode_violations:
        print(f"EPISODE SCORE VIOLATIONS ({len(episode_violations)}):")
        for v in episode_violations:
            print(f"  ✗ {v}")
    else:
        print("Episode scores: ALL strictly in (0, 1) ✓")

    if step_violations:
        print(f"\nSTEP REWARD VIOLATIONS ({len(step_violations)}):")
        for v in step_violations[:50]:
            print(f"  ✗ {v}")
        if len(step_violations) > 50:
            print(f"  ... and {len(step_violations) - 50} more")
    else:
        print("Step rewards: ALL strictly in (0, 1) ✓")

    if episode_violations or step_violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

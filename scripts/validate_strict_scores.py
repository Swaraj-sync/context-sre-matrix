#!/usr/bin/env python3
"""Exhaustive local validation: every task score must be strictly in (0, 1)."""

from __future__ import annotations

import sys
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parents[1] / "sre_architect_env"
sys.path.insert(0, str(ENV_DIR))

from models import (
    ComputeShift,
    InfraToggles,
    PRDecision,
    SREAction,
    TeamContext,
    WriteMode,
)
from server.environment import SREArchitectEnvironment
from server.scenarios import iter_all_scenarios, health_profiles


def _make_policies():
    """Return named policies that cover different scoring extremes."""

    def always_reject(obs):
        return SREAction(
            pr_decision=PRDecision.REJECT,
            infra_toggles=InfraToggles(cache_ttl_s=60),
        )

    def always_approve(obs):
        return SREAction(
            pr_decision=PRDecision.APPROVE,
            infra_toggles=InfraToggles(
                cache_ttl_s=120,
                write_mode=WriteMode.ASYNC,
                compute_shift=ComputeShift.UP,
            ),
        )

    def always_rollback(obs):
        return SREAction(
            pr_decision=PRDecision.ROLLBACK,
            target_pr_id=obs.incoming_pr.pr_id,
            infra_toggles=InfraToggles(
                cache_ttl_s=45,
                write_mode=WriteMode.SYNC,
                compute_shift=ComputeShift.UP,
            ),
        )

    def context_aware(obs):
        if obs.incident_active:
            return SREAction(
                pr_decision=PRDecision.ROLLBACK,
                target_pr_id=obs.incoming_pr.pr_id,
                infra_toggles=InfraToggles(
                    cache_ttl_s=45,
                    write_mode=WriteMode.SYNC,
                    compute_shift=ComputeShift.UP,
                ),
            )
        if obs.active_context == TeamContext.SEARCH:
            return SREAction(
                pr_decision=PRDecision.APPROVE,
                infra_toggles=InfraToggles(
                    cache_ttl_s=120,
                    write_mode=WriteMode.ASYNC,
                    compute_shift=ComputeShift.UP,
                ),
            )
        if obs.active_context == TeamContext.BATCH:
            return SREAction(
                pr_decision=PRDecision.APPROVE,
                infra_toggles=InfraToggles(
                    cache_ttl_s=90,
                    write_mode=WriteMode.SYNC,
                    compute_shift=ComputeShift.DOWN,
                ),
            )
        return SREAction(
            pr_decision=PRDecision.REJECT,
            infra_toggles=InfraToggles(
                cache_ttl_s=45,
                write_mode=WriteMode.SYNC,
                compute_shift=ComputeShift.NEUTRAL,
            ),
        )

    return {
        "always_reject": always_reject,
        "always_approve": always_approve,
        "always_rollback": always_rollback,
        "context_aware": context_aware,
    }


def run_episode(task_id, seed, scenario_id, degraded_start, policy_fn):
    env = SREArchitectEnvironment()
    obs = env.reset(
        seed=seed,
        task_id=task_id,
        scenario_id=scenario_id,
        degraded_start=degraded_start,
    )
    while not obs.done:
        obs = env.step(policy_fn(obs))
    return env.state


def main() -> int:
    policies = _make_policies()
    seeds = [0, 1, 7, 42, 99, 123, 256, 999]
    failures = []
    total_runs = 0

    for scenario in iter_all_scenarios():
        for degraded in (False, True):
            for seed in seeds:
                for policy_name, policy_fn in policies.items():
                    total_runs += 1
                    label = (
                        f"task={scenario.task_id} scenario={scenario.scenario_id} "
                        f"seed={seed} degraded={degraded} policy={policy_name}"
                    )
                    try:
                        state = run_episode(
                            scenario.task_id, seed, scenario.scenario_id, degraded, policy_fn
                        )
                    except Exception as exc:
                        failures.append(f"CRASH  {label}: {exc}")
                        continue

                    metrics = state.grade_metrics
                    for key, value in metrics.items():
                        if value <= 0.0 or value >= 1.0:
                            failures.append(
                                f"RANGE  {label}: {key}={value} (must be strictly in (0,1))"
                            )

                    score = state.episode_score
                    if score is not None and (score <= 0.0 or score >= 1.0):
                        failures.append(
                            f"RANGE  {label}: episode_score={score}"
                        )

    print(f"\nRan {total_runs} episodes across all scenarios/seeds/policies.\n")

    if failures:
        print(f"VALIDATION FAILED — {len(failures)} violation(s):\n")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("VALIDATION PASSED — all scores strictly in (0, 1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from models import (
    ComputeShift,
    InfraToggles,
    PRDecision,
    SREAction,
    TeamContext,
    WriteMode,
)
from server.environment import SREArchitectEnvironment


def _policy(obs):
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


def test_step_rewards_and_episode_scores_are_normalized():
    for task_id in (1, 2, 3):
        env = SREArchitectEnvironment()
        obs = env.reset(seed=7, task_id=task_id)
        assert obs.reward is None

        while not obs.done:
            obs = env.step(_policy(obs))
            assert 0.0 <= float(obs.reward or 0.0) <= 1.0
            assert 0.0 <= float(obs.reward_breakdown.total) <= 1.0

        assert env.state.episode_score is not None
        score = float(env.state.episode_score)
        assert 0.0 < score < 1.0, f"task {task_id}: episode_score={score} not in (0,1)"
        for key, val in env.state.grade_metrics.items():
            assert 0.0 < val < 1.0, f"task {task_id}: {key}={val} not in (0,1)"

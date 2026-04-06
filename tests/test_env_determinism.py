from __future__ import annotations

from models import InfraToggles, PRDecision, SREAction
from server.environment import SREArchitectEnvironment


def _rollout(task_id: int, seed: int):
    env = SREArchitectEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    trace = [
        (
            round(float(obs.system_health.latency_ms), 6),
            round(float(obs.system_health.availability_risk_pct), 6),
            obs.active_context.value,
        )
    ]

    while not obs.done:
        action = SREAction(
            pr_decision=PRDecision.REJECT,
            infra_toggles=InfraToggles(cache_ttl_s=60),
        )
        obs = env.step(action)
        trace.append(
            (
                round(float(obs.reward or 0.0), 6),
                round(float(obs.system_health.latency_ms), 6),
                round(float(obs.system_health.error_rate_pct), 6),
                obs.active_context.value,
                bool(obs.incident_active),
            )
        )

    return trace, env.state.episode_score


def test_seeded_rollout_is_deterministic_for_all_tasks():
    for task_id in (1, 2, 3):
        first_trace, first_score = _rollout(task_id=task_id, seed=17)
        second_trace, second_score = _rollout(task_id=task_id, seed=17)
        assert first_trace == second_trace
        assert first_score == second_score

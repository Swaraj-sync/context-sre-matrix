"""FastAPI app wiring for the SRE benchmark environment."""

from __future__ import annotations

from openenv.core.env_server.http_server import create_app

try:
    from models import SREAction, SREObservation
    from server.environment import SREArchitectEnvironment
    from server.scenarios import scenarios_for_task, supported_task_ids
except ImportError:
    from sre_architect_env.models import SREAction, SREObservation
    from sre_architect_env.server.environment import SREArchitectEnvironment
    from sre_architect_env.server.scenarios import scenarios_for_task, supported_task_ids


app = create_app(
    SREArchitectEnvironment,
    SREAction,
    SREObservation,
    env_name="sre_architect_env",
)


@app.get("/tasks")
def tasks() -> dict:
    """Expose task/scenario metadata for debugging and baseline tooling."""
    task_payload = []
    for task_id in supported_task_ids():
        scenarios = scenarios_for_task(task_id)
        task_payload.append(
            {
                "task_id": task_id,
                "scenario_count": len(scenarios),
                "scenario_ids": [scenario.scenario_id for scenario in scenarios],
                "step_counts": [len(scenario.pr_stream) for scenario in scenarios],
            }
        )
    return {"tasks": task_payload}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

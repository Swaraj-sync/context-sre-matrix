"""Typed OpenEnv client for the SRE benchmark environment."""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SREAction, SREObservation, SREState


class SREArchitectEnv(EnvClient[SREAction, SREObservation, SREState]):
    """Client for interacting with a deployed SRE benchmark environment."""

    def _step_payload(self, action: SREAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        observation_payload = payload.get("observation", {})
        observation_payload["done"] = payload.get(
            "done", observation_payload.get("done", False)
        )
        observation_payload["reward"] = payload.get(
            "reward", observation_payload.get("reward")
        )
        observation = SREObservation.model_validate(observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SREState:
        return SREState.model_validate(payload)

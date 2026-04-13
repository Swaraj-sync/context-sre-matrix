from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

from sre_architect_env.client import SREArchitectEnv
from sre_architect_env.models import SREAction


TASK_IDS = (1, 2, 3)
SEED = 7
REQUEST_TIMEOUT_S = 30
DEFAULT_ENV_BASE_URL = "http://localhost:8000"
_EPS = 1e-4

ALLOWED_DECISIONS = {"APPROVE", "REJECT", "ROLLBACK"}
ALLOWED_WRITE_MODES = {"SYNC", "ASYNC"}
ALLOWED_COMPUTE_SHIFTS = {"DOWN", "NEUTRAL", "UP"}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _strict_task_score(value: Optional[float]) -> float:
    """Normalize task scores to strict open interval (0, 1)."""
    if value is None:
        return _EPS
    return max(_EPS, min(1.0 - _EPS, float(value)))


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(text[start : end + 1])


def _sanitize_action(action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    pr_id = observation.get("incoming_pr", {}).get("pr_id")

    decision = str(action.get("pr_decision", "REJECT")).upper()
    if decision not in ALLOWED_DECISIONS:
        decision = "REJECT"

    target_pr_id = action.get("target_pr_id")
    if decision == "ROLLBACK":
        if not target_pr_id:
            target_pr_id = pr_id
    else:
        target_pr_id = None

    toggles = action.get("infra_toggles") or {}
    cache_ttl_s = toggles.get("cache_ttl_s", 60)
    try:
        cache_ttl_s = int(cache_ttl_s)
    except Exception:
        cache_ttl_s = 60
    cache_ttl_s = max(15, min(300, cache_ttl_s))

    write_mode = str(toggles.get("write_mode", "SYNC")).upper()
    if write_mode not in ALLOWED_WRITE_MODES:
        write_mode = "SYNC"

    compute_shift = str(toggles.get("compute_shift", "NEUTRAL")).upper()
    if compute_shift not in ALLOWED_COMPUTE_SHIFTS:
        compute_shift = "NEUTRAL"

    return {
        "pr_decision": decision,
        "target_pr_id": target_pr_id,
        "infra_toggles": {
            "cache_ttl_s": cache_ttl_s,
            "write_mode": write_mode,
            "compute_shift": compute_shift,
        },
    }


def _fallback_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    active_context = observation.get("active_context", "")
    incoming_pr = observation.get("incoming_pr", {})
    impact = incoming_pr.get("impact", {})
    health = observation.get("system_health", {})
    incident_active = bool(observation.get("incident_active", False))

    pr_id = incoming_pr.get("pr_id")
    latency_delta = float(impact.get("latency_delta_ms", 0.0))
    risk_delta = float(impact.get("availability_risk_delta_pct", 0.0))
    cost_delta = float(impact.get("compute_cost_delta_pct", 0.0))
    consistency_delta = float(impact.get("consistency_delta", 0.0))

    if incident_active:
        return {
            "pr_decision": "ROLLBACK",
            "target_pr_id": pr_id,
            "infra_toggles": {
                "cache_ttl_s": 45,
                "write_mode": "SYNC",
                "compute_shift": "UP",
            },
        }

    if active_context == "payments_consistency":
        risky = risk_delta > 0.0020 or consistency_delta < -0.10
        decision = "REJECT" if risky else "APPROVE"
        compute_shift = "UP" if float(health.get("error_rate_pct", 0.0)) > 1.8 else "NEUTRAL"
        return {
            "pr_decision": decision,
            "target_pr_id": None,
            "infra_toggles": {
                "cache_ttl_s": 45,
                "write_mode": "SYNC",
                "compute_shift": compute_shift,
            },
        }

    if active_context == "search_latency":
        aggressive_safe = latency_delta < -4.0 and risk_delta < 0.0040
        decision = "APPROVE" if aggressive_safe else "REJECT"
        compute_shift = "UP" if float(health.get("latency_ms", 999.0)) > 70.0 else "NEUTRAL"
        return {
            "pr_decision": decision,
            "target_pr_id": None,
            "infra_toggles": {
                "cache_ttl_s": 120,
                "write_mode": "ASYNC",
                "compute_shift": compute_shift,
            },
        }

    # batch_cost_efficiency
    decision = "APPROVE" if cost_delta < -1.0 else "REJECT"
    return {
        "pr_decision": decision,
        "target_pr_id": None,
        "infra_toggles": {
            "cache_ttl_s": 90,
            "write_mode": "SYNC",
            "compute_shift": "DOWN",
        },
    }


def _predict_action_with_llm(
    client: OpenAI,
    model_name: str,
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = {
        "task_id": observation.get("task_id"),
        "active_context": observation.get("active_context"),
        "incoming_pr": observation.get("incoming_pr", {}),
        "system_health": observation.get("system_health", {}),
        "incident_active": observation.get("incident_active", False),
    }

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a deterministic SRE policy. Return only minified JSON with keys "
                    "pr_decision, target_pr_id, infra_toggles. "
                    "infra_toggles must include cache_ttl_s, write_mode, compute_shift. "
                    "Use pr_decision from APPROVE/REJECT/ROLLBACK."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, separators=(",", ":")),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    parsed = _extract_json(content)
    return _sanitize_action(parsed, observation)


def _predict_action(client: OpenAI, model_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = _predict_action_with_llm(client, model_name, observation)
    except Exception:
        action = _fallback_action(observation)
    return _sanitize_action(action, observation)


def _log(tag: str, payload: Dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload, separators=(',', ':'))}")


def run_task(
    task_id: int,
    env: SREArchitectEnv,
    client: OpenAI,
    model_name: str,
) -> Dict[str, Any]:
    reset_result = env.reset(task_id=task_id, seed=SEED)
    observation_model = reset_result.observation
    observation = observation_model.model_dump(mode="json")

    _log(
        "[START]",
        {
            "task_id": task_id,
            "scenario_id": observation.get("scenario_id"),
            "seed": SEED,
        },
    )

    done = bool(reset_result.done)
    step_count = 0
    total_reward = 0.0

    while not done:
        action = _predict_action(client=client, model_name=model_name, observation=observation)
        typed_action = SREAction.model_validate(action)
        result = env.step(typed_action)
        obs_model = result.observation
        observation = obs_model.model_dump(mode="json")
        reward = float(result.reward or 0.0)
        step_count += 1
        total_reward += reward
        done = bool(result.done)

        _log(
            "[STEP]",
            {
                "task_id": task_id,
                "step": step_count,
                "action": action,
                "reward": round(reward, 6),
                "done": done,
            },
        )

    raw_episode_score = env.state().episode_score
    task_score = round(_strict_task_score(raw_episode_score), 6)

    _log(
        "[END]",
        {
            "task_id": task_id,
            "steps": step_count,
            "total_reward": round(total_reward, 6),
            "episode_score": raw_episode_score,
            "task_score": task_score,
        },
    )

    return {
        "task_id": task_id,
        "task_score": task_score,
        "steps": step_count,
        "total_reward": round(total_reward, 6),
        "final_context": observation.get("active_context"),
        "incident_active": bool(observation.get("incident_active", False)),
    }


def main() -> int:
    api_base_url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    env_base_url = os.getenv("ENV_BASE_URL", DEFAULT_ENV_BASE_URL).rstrip("/")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {hf_token}"})
    health_response = session.get(f"{env_base_url}/health", timeout=REQUEST_TIMEOUT_S)
    health_response.raise_for_status()
    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    task_results = []
    with SREArchitectEnv(base_url=env_base_url).sync() as env:
        for task_id in TASK_IDS:
            task_results.append(
                run_task(
                    task_id=task_id,
                    env=env,
                    client=client,
                    model_name=model_name,
                )
            )

    total_reward = sum(result["total_reward"] for result in task_results)
    total_steps = sum(result["steps"] for result in task_results)
    summary = {
        "tasks": task_results,
        "overall_total_reward": round(total_reward, 6),
        "overall_step_average_reward": round(total_reward / total_steps, 6)
        if total_steps
        else 0.0,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

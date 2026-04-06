#!/usr/bin/env python3
"""Local pre-validation checks for the Context-Aware SRE benchmark."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = REPO_ROOT / "sre_architect_env"
REQUIRED_FILES = [
    REPO_ROOT / "inference.py",
    REPO_ROOT / "tests" / "test_env_determinism.py",
    REPO_ROOT / "tests" / "test_reward_bounds.py",
    ENV_DIR / "openenv.yaml",
    ENV_DIR / "models.py",
    ENV_DIR / "client.py",
    ENV_DIR / "server" / "environment.py",
    ENV_DIR / "server" / "app.py",
    ENV_DIR / "server" / "Dockerfile",
]


def check_required_files() -> list[str]:
    failures = []
    for path in REQUIRED_FILES:
        if not path.exists():
            failures.append(f"Missing required file: {path}")
    return failures


def check_reward_ranges() -> list[str]:
    sys.path.insert(0, str(ENV_DIR))
    from models import InfraToggles, PRDecision, SREAction  # pylint: disable=import-error
    from server.environment import SREArchitectEnvironment  # pylint: disable=import-error

    failures = []
    for task_id in (1, 2, 3):
        env = SREArchitectEnvironment()
        obs = env.reset(seed=7, task_id=task_id)
        while not obs.done:
            obs = env.step(
                SREAction(
                    pr_decision=PRDecision.REJECT,
                    infra_toggles=InfraToggles(cache_ttl_s=60),
                )
            )
            reward = float(obs.reward or 0.0)
            if not (0.0 <= reward <= 1.0):
                failures.append(f"Task {task_id}: reward out of range: {reward}")
                break
        if env.state.episode_score is None or not (0.0 <= float(env.state.episode_score) <= 1.0):
            failures.append(
                f"Task {task_id}: invalid episode score: {env.state.episode_score}"
            )
    return failures


def check_openenv_validate() -> list[str]:
    executable = REPO_ROOT / "venv" / "bin" / "openenv"
    if not executable.exists():
        return ["openenv executable not found in ./venv/bin/openenv"]

    result = subprocess.run(  # noqa: S603
        [str(executable), "validate", "--verbose"],
        cwd=str(ENV_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [
            "openenv validate failed",
            result.stdout.strip(),
            result.stderr.strip(),
        ]
    return []


def main() -> int:
    checks = [
        ("required-files", check_required_files),
        ("reward-ranges", check_reward_ranges),
        ("openenv-validate", check_openenv_validate),
    ]

    failures = []
    for name, fn in checks:
        current = fn()
        if current:
            failures.extend([f"[{name}] {msg}" for msg in current])

    if failures:
        print("PRE-VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PRE-VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

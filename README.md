# Context-Aware SRE Matrix (OpenEnv)

This repository contains `sre_architect_env`, an OpenEnv benchmark where an agent plays the role of an on-call SRE in a simulated multi-service production system.

The agent receives:

- current team context/SLA objective,
- mathematically encoded PR impact vectors,
- live telemetry (latency/risk/cost/errors/utilization),

and must decide:

- PR action: `APPROVE`, `REJECT`, or `ROLLBACK`,
- bounded infra controls (`cache_ttl_s`, `write_mode`, `compute_shift`).

## Tasks

- **Task 1 (Easy):** single-objective Payments consistency.
- **Task 2 (Medium):** dynamic context switching between Search and Payments.
- **Task 3 (Hard):** incident triage requiring root-cause rollback and stabilization.

## Repository layout

- `sre_architect_env/`: OpenEnv environment package.
- `inference.py`: required root baseline inference script.
- `tests/`: determinism, reward bound, and logging format tests.
- `scripts/pre_validate.py`: local pre-submission validation checks.
- `docs/submission_constraints.md`: frozen submission checklist.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org "openenv-core[core]" pytest requests openai uv
```

## Run locally

```bash
# Terminal 1
cd sre_architect_env
uvicorn server.app:app --host 127.0.0.1 --port 8000

# Terminal 2
API_BASE_URL="https://your-llm-base-url/v1" \
MODEL_NAME="your-model" \
HF_TOKEN="your-token" \
ENV_BASE_URL="http://127.0.0.1:8000" \
python inference.py
```

## Validation

```bash
pytest -q tests
python scripts/pre_validate.py
cd sre_architect_env && openenv validate --verbose
```

## Notes

- The environment is deterministic for the same `(task_id, seed, action sequence)`.
- Rewards and episode scores are clamped to `[0.0, 1.0]`.
- Inference uses the typed OpenEnv client session for stable multi-step rollout behavior.

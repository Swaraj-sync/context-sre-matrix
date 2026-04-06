# Context-Aware SRE Benchmark (`sre_architect_env`)

`sre_architect_env` is an OpenEnv-compatible environment where an AI policy acts as an on-call SRE:

- Reads mathematically encoded PR impacts and live system telemetry.
- Chooses `APPROVE`, `REJECT`, or `ROLLBACK`.
- Tunes bounded infrastructure dials each step.
- Optimizes behavior for a context-specific SLA objective.

## Interface

- `reset(seed=None, task_id=...)` initializes a new scenario.
- `step(action)` applies decision + infra toggles and returns updated telemetry.
- `state()` returns typed internal state and grading metadata.

## Action Space

`SREAction`:

- `pr_decision`: `APPROVE | REJECT | ROLLBACK`
- `target_pr_id`: optional (used for rollback)
- `infra_toggles`:
  - `cache_ttl_s`: bounded integer `[15, 300]`
  - `write_mode`: `SYNC | ASYNC`
  - `compute_shift`: `DOWN | NEUTRAL | UP`

## Observation Space

`SREObservation` includes:

- `active_context`
- `incoming_pr` (impact vector)
- `system_health` telemetry
- `incident_active`
- `reward_breakdown`
- standard OpenEnv `done` and `reward`

## Tasks

- **Task 1 (Easy):** single-objective Payments consistency.
- **Task 2 (Medium):** dynamic context switching (Search/Payments).
- **Task 3 (Hard):** production incident triage with rollback.

Each task includes deterministic scenario variants and seeded stochastic starts (healthy or degraded profile) to avoid static memorization.

## Reward and grading

- Per-step reward is decomposed into:
  - primary objective alignment,
  - secondary objective alignment,
  - action/decision alignment,
  - explicit penalties for safety breaches.
- All step rewards are clamped to `[0.0, 1.0]`.
- Episode grading computes task-specific final score with diagnostics:
  - average step quality,
  - safety score,
  - context-switch adaptation score,
  - incident triage score (hard task).

## Telemetry fields

`system_health` exposes:

- `latency_ms`
- `availability_risk_pct`
- `compute_cost_index`
- `error_rate_pct`
- `cpu_utilization_pct`
- `ram_utilization_pct`
- `network_saturation_pct`
- `consistency_gap`

## Endpoint behavior

- `/health` is stateless and returns service health.
- For multi-step rollouts, use the OpenEnv typed client (`SREArchitectEnv`) which maintains a session.
- HTTP `reset`/`step`/`state` are exposed by OpenEnv and should still validate for compliance checks.

## Local Run

```bash
cd sre_architect_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Validation

```bash
cd sre_architect_env
openenv validate --verbose
```

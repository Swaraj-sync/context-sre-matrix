# Submission Constraints Checklist

This document freezes the non-negotiable constraints for the Scaler OpenEnv Round 1 submission.

## Required Artifacts

- [ ] Real-world environment with OpenEnv-compatible `reset()`, `step()`, and `state()`.
- [ ] Typed Action, Observation, and State models.
- [ ] `openenv.yaml` manifest in the environment package.
- [ ] At least 3 tasks (easy, medium, hard) with deterministic grader logic.
- [ ] Rewards and task scores normalized to the `[0.0, 1.0]` range.
- [ ] Root-level `inference.py` baseline script.
- [ ] Working `server/Dockerfile`.
- [ ] Environment README with action/observation spaces and setup.

## Inference Script Compliance

- [ ] `inference.py` is in repository root.
- [ ] Uses OpenAI Python client for LLM calls.
- [ ] Reads required environment variables:
  - [ ] `API_BASE_URL`
  - [ ] `MODEL_NAME`
  - [ ] `HF_TOKEN`
- [ ] Emits structured stdout logs with `[START]`, `[STEP]`, and `[END]` tags.
- [ ] Completes within infrastructure/runtime limits.

## Infrastructure and Validation

- [ ] Inference runtime under 20 minutes.
- [ ] Expected to run on 2 vCPU and 8 GB memory.
- [ ] Docker image builds successfully.
- [ ] `/health` endpoint responds with HTTP 200.
- [ ] Environment responds to `reset()` and `step()`.
- [ ] Pre-validation script passes.
- [ ] OpenEnv validation passes.

## Design Guardrails

- [ ] Deterministic replay for same task + seed.
- [ ] Bounded/discrete action space (no unbounded free-form controls).
- [ ] Safety penalties for SLA breaches.
- [ ] Partial-progress reward shaping (not sparse-only reward).
- [ ] Grader diagnostics explain score breakdown for debugging.

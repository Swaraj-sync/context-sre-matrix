#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/sre_architect_env"
OPENENV_BIN="${ROOT_DIR}/venv/bin/openenv"

if [[ ! -x "${OPENENV_BIN}" ]]; then
  echo "Missing openenv CLI at ${OPENENV_BIN}"
  exit 1
fi

if [[ -z "${HF_REPO_ID:-}" ]]; then
  echo "HF_REPO_ID is required (example: username/sre-architect-env)"
  exit 1
fi

# ADD THIS LINE RIGHT HERE:
cp "${ROOT_DIR}/inference.py" "${ENV_DIR}/inference.py"

cd "${ENV_DIR}"

echo "Running openenv validate..."
"${OPENENV_BIN}" validate --verbose

echo "Building Docker image via openenv build..."
"${OPENENV_BIN}" build

echo "Pushing to Hugging Face Space ${HF_REPO_ID}..."
"${OPENENV_BIN}" push --repo-id "${HF_REPO_ID}"

echo "Deployment complete."


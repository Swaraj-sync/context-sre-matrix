# Deployment Guide

## 1) Validate environment package

```bash
cd sre_architect_env
openenv validate --verbose
```

## 2) Build Docker image

```bash
cd sre_architect_env
openenv build
```

If Docker is unavailable, ensure your local daemon is running first (e.g. Docker Desktop or Colima).

## 3) Push to Hugging Face Space

```bash
cd sre_architect_env
openenv push --repo-id "$HF_REPO_ID"
```

Expected environment variables before push:

- `HF_TOKEN`
- `HF_REPO_ID`

## 4) Post-deploy checks

Replace `<SPACE_URL>` with your deployed endpoint:

```bash
curl -sS "<SPACE_URL>/health"
curl -sS -X POST "<SPACE_URL>/reset" -H 'Content-Type: application/json' -d '{"task_id":1,"seed":7}'
```

## 5) Run baseline against deployed env

```bash
API_BASE_URL="$API_BASE_URL" \
MODEL_NAME="$MODEL_NAME" \
HF_TOKEN="$HF_TOKEN" \
ENV_BASE_URL="<SPACE_URL>" \
python inference.py
```

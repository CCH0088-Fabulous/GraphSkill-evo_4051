#!/usr/bin/env bash
# Serve a complete LOCAL model only; hub downloads are explicitly disabled.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${PROJECT_ROOT}/.venv-linux"
ENV_PYTHON="${ENV_ROOT}/bin/python"
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/serve_model.sh /absolute/path/to/complete/local/model" >&2
  exit 2
fi
if [[ ! -d "$1" ]]; then
  echo "Model argument must be an existing local directory; remote repository IDs are not accepted." >&2
  exit 2
fi
MODEL_PATH="$(cd -- "$1" && pwd)"
if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Run bash scripts/bootstrap_linux.sh first." >&2
  exit 2
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
"${ENV_PYTHON}" -m repro.diagnostics --check-weights "${MODEL_PATH}"
exec "${ENV_ROOT}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${REPRO_MODEL_NAME:-Qwen/Qwen3.5-9B}" \
  --host "${REPRO_SERVE_HOST:-127.0.0.1}" \
  --port "${REPRO_SERVE_PORT:-8000}" \
  --dtype bfloat16 \
  --max-model-len "${REPRO_MAX_MODEL_LEN:-16384}" \
  --max-num-seqs "${REPRO_MAX_NUM_SEQS:-1}" \
  --gpu-memory-utilization "${REPRO_GPU_MEMORY_UTILIZATION:-0.85}" \
  --reasoning-parser qwen3

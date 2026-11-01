#!/usr/bin/env bash
# Install libraries and source code only. Never fetch model weights.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${PROJECT_ROOT}/.venv-linux"
ENV_PYTHON="${ENV_ROOT}/bin/python"
LOCK_PATH="${PROJECT_ROOT}/requirements/lock.linux.txt"
REFRESH_LOCK="${REFRESH_LOCK:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script in native Linux or WSL2 Ubuntu." >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/third_party/GraphSkillEvo/pyproject.toml" ]]; then
  echo "Missing official checkout; restore third_party/upstream.lock.json first." >&2
  exit 2
fi
"${PYTHON_BIN}" -c 'import sys; assert (3,11) <= sys.version_info < (3,15), "Python 3.11-3.14 required"'
cd -- "${PROJECT_ROOT}"
if [[ ! -x "${ENV_PYTHON}" ]]; then
  if ! "${PYTHON_BIN}" -m venv "${ENV_ROOT}"; then
    echo "Install the Python venv package for your distribution (Ubuntu: sudo apt install python3-venv)." >&2
    exit 2
  fi
fi
"${ENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
if [[ -f "${LOCK_PATH}" && "${REFRESH_LOCK}" != "1" ]]; then
  "${ENV_PYTHON}" -m pip install -r "${LOCK_PATH}"
else
  # vLLM resolves its matching CUDA/PyTorch stack. Do not pre-install a conflicting torch build.
  "${ENV_PYTHON}" -m pip install 'vllm>=0.17,<1'
fi
"${ENV_PYTHON}" -m pip install -e '.[data,dev]' -e third_party/GraphSkillEvo
"${ENV_PYTHON}" -m pip check
"${ENV_PYTHON}" -c 'import torch, vllm; print("vLLM:", vllm.__version__); print("torch:", torch.__version__); print("CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA GPU not visible"; print("GPU:", torch.cuda.get_device_name(0))'
mkdir -p -- "${PROJECT_ROOT}/requirements"
"${ENV_PYTHON}" -m pip freeze --exclude-editable > "${LOCK_PATH}.tmp"
mv -- "${LOCK_PATH}.tmp" "${LOCK_PATH}"
echo "Linux inference environment ready. Lock: ${LOCK_PATH}"
echo "No model weights were downloaded."


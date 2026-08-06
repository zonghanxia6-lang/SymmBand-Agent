#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${1:-conda}"
DEVICE="${2:-gpu}"
ENV_NAME="${ENV_NAME:-symmcd-band-agent}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"

usage() {
  echo "Usage: bash install.sh [conda|venv] [gpu|cpu]"
  echo "Examples:"
  echo "  bash install.sh conda gpu"
  echo "  bash install.sh venv cpu"
}

if [[ "${BACKEND}" != "conda" && "${BACKEND}" != "venv" ]]; then
  usage
  exit 2
fi
if [[ "${DEVICE}" != "gpu" && "${DEVICE}" != "cpu" ]]; then
  usage
  exit 2
fi

if [[ "${BACKEND}" == "conda" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Load Miniconda/Anaconda first." >&2
    exit 1
  fi
  conda env create --name "${ENV_NAME}" --file "${SCRIPT_DIR}/environment.yml"
  PYTHON=(conda run --no-capture-output --name "${ENV_NAME}" python)
  ENV_PYTHON="$(conda run --no-capture-output --name "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
else
  PYTHON311="${PYTHON311:-python3.11}"
  if ! command -v "${PYTHON311}" >/dev/null 2>&1; then
    echo "${PYTHON311} was not found. Install Python 3.11 or set PYTHON311." >&2
    exit 1
  fi
  "${PYTHON311}" -m venv "${VENV_DIR}"
  PYTHON=("${VENV_DIR}/bin/python")
  ENV_PYTHON="${VENV_DIR}/bin/python"
fi

"${PYTHON[@]}" -m pip install --upgrade pip setuptools wheel

if [[ "${DEVICE}" == "gpu" ]]; then
  "${PYTHON[@]}" -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
  PYG_WHEELS="https://data.pyg.org/whl/torch-2.5.1+cu121.html"
else
  "${PYTHON[@]}" -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
  PYG_WHEELS="https://data.pyg.org/whl/torch-2.5.1+cpu.html"
fi

"${PYTHON[@]}" -m pip install "torch-scatter==2.1.2" --find-links "${PYG_WHEELS}"
"${PYTHON[@]}" -m pip install "torch-geometric==2.7.0"
"${PYTHON[@]}" -m pip install \
  --constraint "${SCRIPT_DIR}/constraints.txt" \
  --requirement "${SCRIPT_DIR}/requirements-common.txt"
"${PYTHON[@]}" -m pip check
"${PYTHON[@]}" "${SCRIPT_DIR}/validate_environment.py" --python-only

echo
echo "Python environment installed: ${ENV_PYTHON}"
echo "Python-only validation passed. Re-run it with:"
echo "  ${ENV_PYTHON} ${SCRIPT_DIR}/validate_environment.py --python-only"
echo "Then edit config/env.agent.example and config/runtime.env.example as described in README.md."

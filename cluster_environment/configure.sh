#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYMMCD_ROOT="${SYMMCD_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYDANTIC_AI_SOURCE="${PYDANTIC_AI_SOURCE:-${SYMMCD_ROOT}/vendor/pydantic-ai}"
BAND_ANALYZER_ROOT="${BAND_ANALYZER_ROOT:-${SYMMCD_ROOT}/band_analysis}"
BAND_OUTPUT_ROOT="${BAND_OUTPUT_ROOT:-${SYMMCD_ROOT}/band-results}"
BAND_PYTHON="${BAND_PYTHON:-$(command -v python)}"
MACE_MODEL="${MACE_MODEL:-${SYMMCD_ROOT}/macemodel/2023-12-03-mace-128-L1_epoch-199.model}"
MACE_DEVICE="${MACE_DEVICE:-cuda}"
VASP_CMD="${VASP_CMD:-}"
VASP_GAMMA_CMD="${VASP_GAMMA_CMD:-${VASP_CMD}}"
VASP_NCL_CMD="${VASP_NCL_CMD:-${VASP_CMD}}"
POTCAR_ROOT="${POTCAR_ROOT:-}"
IRVSP_BIN="${IRVSP_BIN:-$(command -v irvsp || true)}"
LLM_MODEL="${LLM_MODEL:-deepseek-v4-flash}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.deepseek.com}"
LLM_API_KEY="${LLM_API_KEY:-}"

if [[ -z "${LLM_API_KEY}" && -f "${SYMMCD_ROOT}/.env.agent" ]]; then
  EXISTING_KEY_LINE="$(grep -m 1 '^LLM_API_KEY=' "${SYMMCD_ROOT}/.env.agent" || true)"
  LLM_API_KEY="${EXISTING_KEY_LINE#LLM_API_KEY=}"
fi
LLM_API_KEY="${LLM_API_KEY:-replace-with-your-deepseek-key}"

if [[ -z "${VASP_CMD}" ]]; then
  echo "Set VASP_CMD, for example: VASP_CMD='srun -n 32 /path/vasp_std'" >&2
  exit 1
fi
if [[ -z "${POTCAR_ROOT}" ]]; then
  echo "Set POTCAR_ROOT to the pymatgen POTCAR root directory." >&2
  exit 1
fi
if [[ -z "${IRVSP_BIN}" ]]; then
  echo "Set IRVSP_BIN or load the irvsp module before running this script." >&2
  exit 1
fi
if [[ ! -d "${PYDANTIC_AI_SOURCE}/pydantic_ai_slim/pydantic_ai" ]]; then
  echo "Invalid PYDANTIC_AI_SOURCE: ${PYDANTIC_AI_SOURCE}" >&2
  exit 1
fi
if [[ ! -f "${BAND_ANALYZER_ROOT}/agent_runner.py" ]]; then
  echo "Invalid BAND_ANALYZER_ROOT: ${BAND_ANALYZER_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MACE_MODEL}" ]]; then
  echo "Local MACE model not found: ${MACE_MODEL}" >&2
  exit 1
fi

mkdir -p "${BAND_OUTPUT_ROOT}"

cat > "${SCRIPT_DIR}/config/atomate2.yaml" <<EOF
VASP_CMD: "${VASP_CMD}"
VASP_GAMMA_CMD: "${VASP_GAMMA_CMD}"
VASP_NCL_CMD: "${VASP_NCL_CMD}"
VASP_INCAR_UPDATES:
  NCORE: 4
EOF

cat > "${SCRIPT_DIR}/config/jobflow.yaml" <<EOF
JOB_STORE:
  docs_store:
    type: JSONStore
    paths:
      - /tmp/symmcd_jobflow_docs.json
    read_only: false
  additional_stores:
    data:
      type: JSONStore
      paths:
        - /tmp/symmcd_jobflow_data.json
      read_only: false
EOF

IRVSP_DIR="$(cd "$(dirname "${IRVSP_BIN}")" && pwd)"
cat > "${SCRIPT_DIR}/config/runtime.env" <<EOF
#!/usr/bin/env bash
export SYMMCD_ROOT="${SYMMCD_ROOT}"
export PYDANTIC_AI_SOURCE="${PYDANTIC_AI_SOURCE}"
export BAND_ANALYZER_ROOT="${BAND_ANALYZER_ROOT}"
export BAND_PYTHON="${BAND_PYTHON}"
export BAND_OUTPUT_ROOT="${BAND_OUTPUT_ROOT}"
export ATOMATE2_CONFIG_FILE="${SCRIPT_DIR}/config/atomate2.yaml"
export JOBFLOW_CONFIG_FILE="${SCRIPT_DIR}/config/jobflow.yaml"
export PMG_VASP_PSP_DIR="${POTCAR_ROOT}"
export PATH="${IRVSP_DIR}:\${PATH}"
export OMP_NUM_THREADS=1
EOF
chmod 600 "${SCRIPT_DIR}/config/runtime.env"

cat > "${SYMMCD_ROOT}/.env.agent" <<EOF
LLM_BASE_URL=${LLM_BASE_URL}
LLM_MODEL=${LLM_MODEL}
LLM_API_KEY=${LLM_API_KEY}
PYDANTIC_AI_SOURCE=${PYDANTIC_AI_SOURCE}
MACE_MODEL=${MACE_MODEL}
MACE_DEVICE=${MACE_DEVICE}
BAND_ANALYZER_ROOT=${BAND_ANALYZER_ROOT}
BAND_PYTHON=${BAND_PYTHON}
BAND_OUTPUT_ROOT=${BAND_OUTPUT_ROOT}
BAND_TIMEOUT_SECONDS=0
EOF
chmod 600 "${SYMMCD_ROOT}/.env.agent"

echo "Cluster configuration written. Load it with:"
echo "  source ${SCRIPT_DIR}/config/runtime.env"
echo "Then run:"
echo "  python ${SCRIPT_DIR}/validate_environment.py"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ETCE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

export ETCE_ROOT="${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${PROJECT_ROOT}/results/phase4/server_logs"
mkdir -p "${LOG_DIR}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/22_phase4b_validation_rule_bank.py" \
  2>&1 | tee -a "${LOG_DIR}/phase4b_build.log"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/23_validate_phase4b.py" \
  2>&1 | tee -a "${LOG_DIR}/phase4b_validate.log"

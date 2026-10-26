#!/usr/bin/env bash
# User-run only. --preflight is CPU-only and does not query or submit Slurm.
set -euo pipefail
PROJECT_ROOT="${OFT_BATCH8_PROJECT_ROOT:-/home/caijiasheng25/robopatch_defense/roboticAttack-main}"
PYTHON_BIN="${OFT_BATCH8_PYTHON:-/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python}"
TASK_CODE="${PROJECT_ROOT}/research_runs/wrist_dino_batch8_when_v1/code"
export PYTHONPATH="${TASK_CODE}:${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 _CHECK_PEFT=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${PROJECT_ROOT}"
case "${1:-}" in
  ""|--preflight|--dry-run) ;;
  *) echo "Usage: bash submit_exploring_oft_batch8_when2000_v1.sh [--preflight|--dry-run]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then exit 2; fi
"${PYTHON_BIN}" -B "${TASK_CODE}/train.py" --preflight
for script in slurm/exploring_oft_dino/{142,143,144,145,146}_batch8_when_*.sbatch; do
  bash -n "${script}"
done
if [[ ${1:-} == --preflight || ${1:-} == --dry-run ]]; then
  echo "BATCH8_WHEN2000_PREFLIGHT_PASS; no Slurm submission"
  exit 0
fi
"${PYTHON_BIN}" -B "${TASK_CODE}/dino_batch8_common.py" --submit

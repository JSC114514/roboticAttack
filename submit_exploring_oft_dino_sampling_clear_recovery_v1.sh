#!/usr/bin/env bash
# Only an explicit unflagged user invocation queries Slurm and submits jobs.
set -euo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${OFT_SAMPLING_CLEAR_PROJECT_ROOT:-${SCRIPT_ROOT}}"
PYTHON_BIN="${OFT_SAMPLING_CLEAR_PYTHON:-/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python}"
export OFT_DINO_SAMPLING_CLEAR_CONFIG="${PROJECT_ROOT}/configs/exploring_oft_dino/object_dino_sampling_clear2000_v1.json"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 _CHECK_PEFT=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${PROJECT_ROOT}"
case "${1:-}" in
  ""|--preflight|--dry-run) ;;
  *) echo "Usage: bash submit_exploring_oft_dino_sampling_clear_recovery_v1.sh [--preflight|--dry-run]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo "Only one optional flag accepted" >&2; exit 2; fi
for script in slurm/exploring_oft_dino/{117,118,119,120,121}_clear_sampling_recovery_*.sbatch; do
  bash -n "${script}"
done
if [[ ${1:-} == --preflight || ${1:-} == --dry-run ]]; then
  exec "${PYTHON_BIN}" -B scripts/dino_sampling_clear_recovery_common.py --preflight
fi
exec "${PYTHON_BIN}" -B scripts/dino_sampling_clear_recovery_common.py --submit

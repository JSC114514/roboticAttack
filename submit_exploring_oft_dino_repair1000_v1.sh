#!/usr/bin/env bash
# User-run only: completed prefix/bank are reused; GPU submission is explicit.
set -euo pipefail
PROJECT_ROOT="${OFT_REPAIR_PROJECT_ROOT:-/home/caijiasheng25/robopatch_defense/roboticAttack-main}"
PYTHON_BIN="${OFT_REPAIR_PYTHON:-/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python}"
export OFT_DINO_REPAIR_CONFIG="${PROJECT_ROOT}/configs/exploring_oft_dino/object_dino_repair1000_v1.json"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"
case "${1:-}" in
  ""|--preflight|--dry-run) ;;
  *) echo "Usage: bash submit_exploring_oft_dino_repair1000_v1.sh [--preflight|--dry-run]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo "Only one optional flag accepted" >&2; exit 2; fi
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_REPAIR_wrapper.py --dry-run
"${PYTHON_BIN}" -B scripts/run_exploring_oft_dino_repair_cross_suite.py --preflight
for script in slurm/exploring_oft_dino/{102,103,104,105,106}_repair1000_*.sbatch; do
  bash -n "${script}"
done
if [[ ${1:-} == --preflight || ${1:-} == --dry-run ]]; then
  echo "REPAIR_PREFLIGHT_PASS; no queue query or Slurm submission"
  exit 0
fi
"${PYTHON_BIN}" -B scripts/dino_repair_common.py --submit

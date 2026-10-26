#!/usr/bin/env bash
# User-run only. --preflight/--dry-run perform CPU reads and never query or submit Slurm.
set -euo pipefail
PROJECT_ROOT="${OFT_REGISTER_PROJECT_ROOT:-/home/caijiasheng25/robopatch_defense/roboticAttack-main}"
PYTHON_BIN="${OFT_REGISTER_PYTHON:-/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python}"
export OFT_DINO_REGISTER_DELAYED_CONFIG="${PROJECT_ROOT}/configs/exploring_oft_dino/object_dino_register_delayed2000_v2.json"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 _CHECK_PEFT=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${PROJECT_ROOT}"
case "${1:-}" in
  ""|--preflight|--dry-run) ;;
  *) echo "Usage: bash submit_exploring_oft_dino_register_delayed2000_v2.sh [--preflight|--dry-run]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo "Only one optional flag accepted" >&2; exit 2; fi
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_REGISTER_DELAYED_wrapper.py --dry-run
for script in slurm/exploring_oft_dino/{132,133,134,135,136}_register_delayed_*.sbatch; do
  bash -n "${script}"
done
if [[ ${1:-} == --preflight || ${1:-} == --dry-run ]]; then
  echo "REGISTER_DELAYED2000_PREFLIGHT_PASS; no queue query or Slurm submission"
  exit 0
fi
"${PYTHON_BIN}" -B scripts/dino_register_delayed_common.py --submit

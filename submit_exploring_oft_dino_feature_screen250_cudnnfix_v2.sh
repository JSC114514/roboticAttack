#!/usr/bin/env bash
# USER-RUN ONLY. No cancellation, no changes to the running UADA job.
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
source "${PROJECT_ROOT}/research_runs/wrist_dino_feature_screen250_v1/slurm/environment.sh"
REPAIR_ROOT="${EXPERIMENT_ROOT}/repair_cudnn4090_v2"
if [[ $# -gt 1 ]]; then echo "Only one optional flag accepted" >&2; exit 2; fi
case "${1:-}" in
  ""|--preflight|--dry-run) ;;
  *) echo "Usage: bash submit_exploring_oft_dino_feature_screen250_cudnnfix_v2.sh [--preflight|--dry-run]" >&2; exit 2 ;;
esac
for script in "${REPAIR_ROOT}"/slurm/*.sbatch "${EXPERIMENT_ROOT}/slurm/04_eval_4090.sbatch"; do bash -n "${script}"; done
"${PYTHON_BIN}" -B "${REPAIR_ROOT}/code/repair_submit.py" --preflight
if [[ ${1:-} == --preflight || ${1:-} == --dry-run ]]; then exit 0; fi
exec "${PYTHON_BIN}" -B "${REPAIR_ROOT}/code/repair_submit.py" --submit

#!/usr/bin/env bash
# USER-RUN ONLY: reuse completed banks/teachers and resume the same experiment.
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
source "${PROJECT_ROOT}/research_runs/wrist_dino_long_final8_v2/slurm/environment.sh"
if [[ $# -gt 1 ]]; then echo "Only one optional flag accepted" >&2; exit 2; fi
case "${1:-}" in
  "") MODE=--submit ;;
  --preflight|--dry-run) MODE=--preflight ;;
  *) echo "Usage: bash submit_exploring_oft_long_final8_2000_resume_v2.sh [--preflight|--dry-run]" >&2; exit 2 ;;
esac
for stage in 07_train_4090 08_validate_cpu 09_eval_3090 10_stats_cpu; do
  bash -n "${EXPERIMENT_ROOT}/slurm/${stage}.sbatch"
done
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/recovery/restart.py" "${MODE}"

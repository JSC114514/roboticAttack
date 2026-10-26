#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT=/mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_closedloop_geometry_v1
case "${1:---check}" in
  --check) source "${EXPERIMENT_ROOT}/slurm/environment.sh" cpu
    exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/submit_run.py" ;;
  --submit|--status|--estimate) source "${EXPERIMENT_ROOT}/slurm/environment.sh" cpu
    exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/submit_run.py" "$1" ;;
  *) echo "usage: bash run.sh [--check|--submit|--status|--estimate]" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# USER-RUN ONLY. --preflight/--dry-run never query/submit Slurm.
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
source "${PROJECT_ROOT}/research_runs/wrist_dino_objectives_screen250_v2/slurm/environment.sh"
if [[ $# -gt 1 ]]; then echo "Only one optional flag accepted" >&2; exit 2; fi
case "${1:-}" in
  ""|--preflight|--dry-run) ;;
  *) echo "Usage: bash submit_exploring_oft_dino_objectives_screen250_v2.sh [--preflight|--dry-run]" >&2; exit 2 ;;
esac
for script in "${EXPERIMENT_ROOT}"/slurm/*.sbatch; do bash -n "${script}"; done
"${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/pipeline.py" --preflight
if [[ ${1:-} == --preflight || ${1:-} == --dry-run ]]; then exit 0; fi
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/pipeline.py" --submit

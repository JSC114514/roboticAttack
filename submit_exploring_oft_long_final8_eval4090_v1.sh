#!/usr/bin/env bash
# USER-RUN ONLY: completed training preserved; 4090 evaluation replaces pending 3090.
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
source "${PROJECT_ROOT}/research_runs/wrist_dino_long_final8_v2/evaluation_4090_v1/slurm/environment.sh"
if [[ $# -gt 1 ]]; then echo "Only one optional flag accepted" >&2; exit 2; fi
case "${1:-}" in
  "") MODE=--submit ;;
  --preflight|--dry-run) MODE=--preflight ;;
  *) echo "Usage: bash submit_exploring_oft_long_final8_eval4090_v1.sh [--preflight|--dry-run]" >&2; exit 2 ;;
esac
for script in "${EXPERIMENT_ROOT}"/slurm/*.sbatch; do bash -n "${script}"; done
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/pipeline.py" "${MODE}"

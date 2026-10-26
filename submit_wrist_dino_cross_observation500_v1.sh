#!/usr/bin/env bash
set -euo pipefail
CROSS_OBS_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_cross_observation500_v1
CROSS_OBS_PYTHON=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
case ${1:-} in
  '') ACTION=submit ;;
  --preflight) ACTION=preflight ;;
  --dry-run) ACTION=dry-run ;;
  --status) ACTION=status ;;
  *) echo 'Usage: bash submit_wrist_dino_cross_observation500_v1.sh [--preflight|--dry-run|--status]' >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo 'Only one action is supported' >&2; exit 2; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec "${CROSS_OBS_PYTHON}" -B "${CROSS_OBS_ROOT}/code/pipeline.py" --action "${ACTION}"

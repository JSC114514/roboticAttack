#!/usr/bin/env bash
set -euo pipefail
FACTORIAL_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_uada_factorial1000_v1
FACTORIAL_PYTHON=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
case ${1:-} in
  '') ACTION=submit ;;
  --preflight) ACTION=preflight ;;
  --dry-run) ACTION=dry-run ;;
  --status) ACTION=status ;;
  --prepare) ACTION=prepare ;;
  *) echo 'Usage: bash submit_wrist_dino_uada_factorial1000_v1.sh [--preflight|--dry-run|--status|--prepare]' >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo 'Only one action is supported' >&2; exit 2; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "${FACTORIAL_PYTHON}" -B "${FACTORIAL_ROOT}/code/pipeline.py" --action "${ACTION}"

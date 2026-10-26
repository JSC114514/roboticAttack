#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
EXPERIMENT_ROOT="${PROJECT_ROOT}/research_runs/wrist_dino_action_control_screen_v1"
source "${EXPERIMENT_ROOT}/slurm/environment.sh"
case "${1:-}" in
  --preflight) action=preflight ;;
  --dry-run) action=dry-run ;;
  --status) action=status ;;
  "") action=submit ;;
  *) echo "Usage: $0 [--preflight|--dry-run|--status]" >&2; exit 2 ;;
esac
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/pipeline.py" --action "${action}"

#!/usr/bin/env bash
set -euo pipefail
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_geometry_action_teacher_v1/slurm/environment.sh
exec "${PYTHON_BIN}" -B "${AT_ROOT}/code/at_cli.py" "$@"

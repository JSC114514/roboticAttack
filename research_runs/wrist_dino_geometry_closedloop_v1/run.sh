#!/usr/bin/env bash
set -euo pipefail
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_geometry_closedloop_v1/slurm/environment.sh
exec "${PYTHON_BIN}" -B "${CLOSED_ROOT}/code/cli.py" "$@"

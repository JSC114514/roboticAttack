#!/usr/bin/env bash
set -euo pipefail
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_spatial_approach_pilot_v1/slurm/environment.sh
exec "${PYTHON_BIN}" -B "${SR_ROOT}/code/sr_cli.py" "$@"

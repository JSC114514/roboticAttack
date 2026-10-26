#!/usr/bin/env bash
set -euo pipefail
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_geometry_rgb2_v1_fix1/slurm/environment.sh
exec "${PYTHON_BIN}" -B "${RGB_ROOT}/code/rgb_cli.py" "$@"

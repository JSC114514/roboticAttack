#!/usr/bin/env bash
set -euo pipefail
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_spatial_b12_diag500_v1/slurm/environment.sh
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/pipeline.py" "$@"

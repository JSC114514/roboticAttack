#!/usr/bin/env bash
set -euo pipefail
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_action_readout_v1/slurm/environment.sh
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/submit_run.py" "$@"

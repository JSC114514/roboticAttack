#!/usr/bin/env bash
set -euo pipefail
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_relation_readout_pilot_v1/slurm/environment.sh
exec "${PYTHON_BIN}" -B "${RR_ROOT}/code/rr_cli.py" "$@"

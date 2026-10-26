#!/usr/bin/env bash
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_spatial_relation_shuffle_v1/slurm/environment.sh "${1:-}"
export RFIX_ROOT=/mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_spatial_relation_shuffle_v1_fix1
export PYTHONPATH="${RFIX_ROOT}/code:${PROJECT_ROOT}/research_tools:${PYTHONPATH}"
export MPLCONFIGDIR="${RFIX_ROOT}/reports/matplotlib"

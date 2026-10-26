#!/usr/bin/env bash
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_geometry_match_v1/slurm/environment.sh "${1:-}"
RR_ROOT="${PROJECT_ROOT}/research_runs/wrist_dino_relation_readout_pilot_v1"
export PYTHONPATH="${RR_ROOT}/code:${PYTHONPATH}" TZ=Asia/Shanghai

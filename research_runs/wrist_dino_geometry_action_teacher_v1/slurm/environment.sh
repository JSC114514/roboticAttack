#!/usr/bin/env bash
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_geometry_match_v1/slurm/environment.sh "${1:-}"
AT_ROOT="${PROJECT_ROOT}/research_runs/wrist_dino_geometry_action_teacher_v1"
export PYTHONPATH="${AT_ROOT}/code:${PYTHONPATH}" TZ=Asia/Shanghai

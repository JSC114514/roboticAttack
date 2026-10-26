#!/usr/bin/env bash
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_geometry_match_v1/slurm/environment.sh "${1:-}"
RGB_ROOT="${PROJECT_ROOT}/research_runs/wrist_dino_geometry_rgb2_v1_fix1"
export PYTHONPATH="${RGB_ROOT}/code:${PYTHONPATH}"

#!/usr/bin/env bash
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_spatial_approach_pilot_v1/slurm/environment.sh "${1:-}"
export FOLLOWUP_ROOT=/mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_spatial_relation_expand_v1
export PYTHONPATH="${PROJECT_ROOT}/research_tools/spatial_followup:${FOLLOWUP_ROOT}/code:${PROJECT_ROOT}/research_runs/wrist_spatial_failure_audit_v1/code:${PROJECT_ROOT}/research_runs/wrist_spatial_relation_rescue_v1/code:${PYTHONPATH}"
export MPLCONFIGDIR="${FOLLOWUP_ROOT}/reports/matplotlib"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}" TF_NUM_INTEROP_THREADS=1

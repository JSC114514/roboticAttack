#!/usr/bin/env bash
source /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_spatial_approach_pilot_v1/slurm/environment.sh "${1:-}"
SFA_ROOT="${PROJECT_ROOT}/research_runs/wrist_spatial_failure_audit_v1"
export PYTHONPATH="${SFA_ROOT}/code:${PYTHONPATH}"
export MPLCONFIGDIR="${SFA_ROOT}/reports/matplotlib"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}" TF_NUM_INTEROP_THREADS=1

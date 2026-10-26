#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
RAMP_ROOT=/home/caijiasheng25/robopatch_defense/oft_ramp_recovery_aware
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/scripts:${RAMP_ROOT}/scripts:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m py_compile \
  scripts/run_exploring_oft_object_decoy500_cross_suite.py \
  scripts/analyze_exploring_oft_object_decoy500_cross_suite.py
bash -n slurm/exploring_oft_dino/42_object_decoy500_cross_suite_3090.sbatch
bash -n slurm/exploring_oft_dino/43_object_decoy500_cross_suite_stats.sbatch
"${PYTHON_BIN}" -B scripts/run_exploring_oft_object_decoy500_cross_suite.py \
  --config configs/exploring_oft_dino/object_decoy500_cross_suite.json --preflight

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_OBJECT_DECOY500_XSUITE_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi

mkdir -p logs/slurm outputs/exploring_oft_dino/object_decoy500_cross_suite
eval_raw=$(sbatch --parsable slurm/exploring_oft_dino/42_object_decoy500_cross_suite_3090.sbatch)
eval_id=${eval_raw%%;*}
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" slurm/exploring_oft_dino/43_object_decoy500_cross_suite_stats.sbatch)
stats_id=${stats_raw%%;*}
echo "EXPLORING_OFT_OBJECT_DECOY500_XSUITE_EVAL=${eval_id}"
echo "EXPLORING_OFT_OBJECT_DECOY500_XSUITE_STATS=${stats_id}"
echo "Three patches, four suites, 1200 new attack rollouts, 400 frozen clean rollouts reused; at most five RTX3090 GPUs."

#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/scripts:/home/caijiasheng25/robopatch_defense/oft_ramp_recovery_aware/scripts:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_scale500_cross_suite.py tests/test_exploring_oft_uada_object2000.py
"${PYTHON_BIN}" -m py_compile scripts/run_exploring_oft_scale500_cross_suite.py \
  scripts/analyze_exploring_oft_scale500_cross_suite.py \
  scripts/preflight_exploring_oft_uada2000_cross_suite.py
bash -n slurm/exploring_oft_dino/13_uada2000_cross_suite.sbatch
bash -n slurm/exploring_oft_dino/14_uada2000_cross_suite_stats.sbatch
"${PYTHON_BIN}" -B scripts/preflight_exploring_oft_uada2000_cross_suite.py \
  --config configs/exploring_oft_dino/uada_object2000_cross_suite.json

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_UADA2000_XSUITE_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi

mkdir -p logs/slurm outputs/exploring_oft_dino/uada_object2000_cross_suite
eval_raw=$(sbatch --parsable slurm/exploring_oft_dino/13_uada2000_cross_suite.sbatch)
eval_id=${eval_raw%%;*}
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" slurm/exploring_oft_dino/14_uada2000_cross_suite_stats.sbatch)
stats_id=${stats_raw%%;*}
echo "EXPLORING_OFT_UADA2000_XSUITE_EVAL=${eval_id}"
echo "EXPLORING_OFT_UADA2000_XSUITE_STATS=${stats_id}"
echo "Four suites, 500/2000 UADA patches paired on frozen clean states, 800 new attack episodes, at most five RTX3090 GPUs."

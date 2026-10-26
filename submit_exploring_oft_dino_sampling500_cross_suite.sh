#!/usr/bin/env bash
# Manual-only evaluation submission; refuses to run until visual review and hashes are frozen.
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
RAMP_ROOT=/home/caijiasheng25/robopatch_defense/oft_ramp_recovery_aware
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${PROJECT_ROOT}/scripts:${RAMP_ROOT}/scripts:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m py_compile \
  scripts/run_exploring_oft_dino_sampling500_cross_suite.py \
  scripts/analyze_exploring_oft_dino_sampling500_cross_suite.py
bash -n slurm/exploring_oft_dino/45_dino_sampling500_cross_suite_3090.sbatch
bash -n slurm/exploring_oft_dino/46_dino_sampling500_cross_suite_stats.sbatch
"${PYTHON_BIN}" -B scripts/run_exploring_oft_dino_sampling500_cross_suite.py \
  --config configs/exploring_oft_dino/dino_sampling500_cross_suite.json --preflight

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_DINO_SAMPLING500_XSUITE_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi
mkdir -p logs/slurm outputs/exploring_oft_dino/dino_sampling500_cross_suite
eval_raw=$(sbatch --parsable slurm/exploring_oft_dino/45_dino_sampling500_cross_suite_3090.sbatch)
eval_id=${eval_raw%%;*}
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" \
  slurm/exploring_oft_dino/46_dino_sampling500_cross_suite_stats.sbatch)
stats_id=${stats_raw%%;*}
echo "EXPLORING_OFT_DINO_SAMPLING500_XSUITE_EVAL=${eval_id}"
echo "EXPLORING_OFT_DINO_SAMPLING500_XSUITE_STATS=${stats_id}"
echo "Four conditions, four suites, 1600 new attack rollouts; clean rollouts reused."

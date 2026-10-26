#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
RAMP_ROOT=/home/caijiasheng25/robopatch_defense/oft_ramp_recovery_aware
export PYTHONPATH="${PROJECT_ROOT}/scripts:${RAMP_ROOT}/scripts:${PYTHONPATH:-}"
export _CHECK_PEFT=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -B -m pytest -q tests/test_exploring_oft_correspondence_cross_suite.py
bash -n slurm/exploring_oft_dino/51_correspondence500_cross_suite_3090.sbatch
bash -n slurm/exploring_oft_dino/52_correspondence500_cross_suite_stats.sbatch
"${PYTHON_BIN}" -B scripts/run_exploring_oft_correspondence500_cross_suite.py \
  --config configs/exploring_oft_dino/object_correspondence500_cross_suite.json --preflight
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "CORRESPONDENCE500_XSUITE_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi
mkdir -p logs/slurm outputs/exploring_oft_dino/object_correspondence500_cross_suite
eval_raw=$(sbatch --parsable slurm/exploring_oft_dino/51_correspondence500_cross_suite_3090.sbatch)
eval_id=${eval_raw%%;*}
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" \
  slurm/exploring_oft_dino/52_correspondence500_cross_suite_stats.sbatch)
stats_id=${stats_raw%%;*}
echo "CORRESPONDENCE500_XSUITE_EVAL=${eval_id}"
echo "CORRESPONDENCE500_XSUITE_STATS=${stats_id}"
echo "Five approved patches, four suites: 2000 new attack episodes; 400 frozen clean; at most five RTX3090 GPUs."

#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${PROJECT_ROOT}"
for script in slurm/exploring_oft_dino/{91_uada_original1000_train_4090,92_uada_original1000_ready_cpu,93_uada_original1000_eval_3090,94_uada_original1000_stats_cpu}.sbatch; do
  bash -n "${script}"
done
"${PYTHON_BIN}" -B -m pytest -q tests/test_exploring_oft_uada_original1000.py tests/test_exploring_oft_uada_object2000.py
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_ORIGINAL1000_wrapper.py --dry-run
"${PYTHON_BIN}" -B scripts/run_exploring_oft_uada_original1000_cross_suite.py --preflight
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo 'UADA_ORIGINAL1000_CHECK_ONLY_PASS; no sbatch submitted'
  exit 0
fi
mkdir -p research_runs/uada_original1000_v1/{logs,training,evaluation/four_suite}
train_raw=$(sbatch --parsable slurm/exploring_oft_dino/91_uada_original1000_train_4090.sbatch)
train_id=${train_raw%%;*}
echo "UADA_ORIGINAL1000_TRAIN=${train_id}"
ready_raw=$(sbatch --parsable --dependency="afterok:${train_id}" slurm/exploring_oft_dino/92_uada_original1000_ready_cpu.sbatch)
ready_id=${ready_raw%%;*}
echo "UADA_ORIGINAL1000_READY=${ready_id}"
eval_raw=$(sbatch --parsable --dependency="afterok:${ready_id}" slurm/exploring_oft_dino/93_uada_original1000_eval_3090.sbatch)
eval_id=${eval_raw%%;*}
echo "UADA_ORIGINAL1000_EVAL=${eval_id}"
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" slurm/exploring_oft_dino/94_uada_original1000_stats_cpu.sbatch)
echo "UADA_ORIGINAL1000_STATS=${stats_raw%%;*}"
echo 'Fresh original Exploring-OFT UADA: 1000 outer/50000 updates; 500/750/1000 checkpoints; 1200 four-suite attack episodes.'

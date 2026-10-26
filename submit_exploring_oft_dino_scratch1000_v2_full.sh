#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export OFT_DINO_SCRATCH_CONFIG="${PROJECT_ROOT}/configs/exploring_oft_dino/object_dino_scratch1000_v2.json"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${PROJECT_ROOT}"
for script in slurm/exploring_oft_dino/{81_dino_scratch_v2_warmup_4090,82_dino_scratch_v2_collect_3090,83_dino_scratch_v2_bank_cpu,84_dino_scratch_v2_train_4090,85_dino_scratch_v2_auto_validate_cpu,86_dino_scratch_v2_eval_3090,87_dino_scratch_v2_stats_cpu}.sbatch; do
  bash -n "${script}"
done
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_SCRATCH_wrapper.py --dry-run
"${PYTHON_BIN}" -B scripts/collect_exploring_oft_dino_scratch.py --preflight
"${PYTHON_BIN}" -B scripts/run_exploring_oft_dino_scratch_cross_suite.py --preflight
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo 'DINO_SCRATCH_V2_CHECK_ONLY_PASS; no sbatch submitted'
  exit 0
fi
mkdir -p research_runs/wrist_dino_scratch_v2/{logs,data,evaluation/four_suite}
prefix_raw=$(sbatch --parsable slurm/exploring_oft_dino/81_dino_scratch_v2_warmup_4090.sbatch)
prefix_id=${prefix_raw%%;*}
echo "DINO_SCRATCH_V2_PREFIX=${prefix_id}"
collect_raw=$(sbatch --parsable --dependency="afterok:${prefix_id}" slurm/exploring_oft_dino/82_dino_scratch_v2_collect_3090.sbatch)
collect_id=${collect_raw%%;*}
echo "DINO_SCRATCH_V2_COLLECT=${collect_id}"
bank_raw=$(sbatch --parsable --dependency="afterok:${collect_id}" slurm/exploring_oft_dino/83_dino_scratch_v2_bank_cpu.sbatch)
bank_id=${bank_raw%%;*}
echo "DINO_SCRATCH_V2_BANK=${bank_id}"
train_raw=$(sbatch --parsable --dependency="afterok:${bank_id}" slurm/exploring_oft_dino/84_dino_scratch_v2_train_4090.sbatch)
train_id=${train_raw%%;*}
echo "DINO_SCRATCH_V2_TRAIN=${train_id}"
validation_raw=$(sbatch --parsable --dependency="afterok:${train_id}" slurm/exploring_oft_dino/85_dino_scratch_v2_auto_validate_cpu.sbatch)
validation_id=${validation_raw%%;*}
echo "DINO_SCRATCH_V2_VALIDATE=${validation_id}"
eval_raw=$(sbatch --parsable --dependency="afterok:${validation_id}" slurm/exploring_oft_dino/86_dino_scratch_v2_eval_3090.sbatch)
eval_id=${eval_raw%%;*}
echo "DINO_SCRATCH_V2_EVAL=${eval_id}"
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" slurm/exploring_oft_dino/87_dino_scratch_v2_stats_cpu.sbatch)
echo "DINO_SCRATCH_V2_STATS=${stats_raw%%;*}"
echo 'From random initialization: 250 common + three 750-outer branches; 9 checkpoints; automatic four-suite evaluation after validation.'

#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${PROJECT_ROOT}"
bash -n slurm/exploring_oft_dino/79_dino_scratch_cross_suite_3090.sbatch
bash -n slurm/exploring_oft_dino/80_dino_scratch_cross_suite_stats_cpu.sbatch
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  "${PYTHON_BIN}" -B scripts/run_exploring_oft_dino_scratch_cross_suite.py --preflight
  echo 'DINO_SCRATCH_EVAL_CHECK_ONLY_DONE; no sbatch submitted'
  exit 0
fi
"${PYTHON_BIN}" -B scripts/run_exploring_oft_dino_scratch_cross_suite.py --require-ready
mkdir -p research_runs/wrist_dino_scratch_v1/logs research_runs/wrist_dino_scratch_v1/evaluation/four_suite
eval_raw=$(sbatch --parsable slurm/exploring_oft_dino/79_dino_scratch_cross_suite_3090.sbatch)
eval_id=${eval_raw%%;*}
echo "DINO_SCRATCH_EVAL=${eval_id}"
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" slurm/exploring_oft_dino/80_dino_scratch_cross_suite_stats_cpu.sbatch)
echo "DINO_SCRATCH_STATS=${stats_raw%%;*}"
echo '3600 attack episodes; up to eight RTX3090 tasks; expected 2-3 hours excluding queues.'

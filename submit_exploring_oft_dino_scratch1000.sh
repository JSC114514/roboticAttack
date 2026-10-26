#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${PROJECT_ROOT}"
bash -n slurm/exploring_oft_dino/75_dino_scratch_warmup_4090.sbatch
bash -n slurm/exploring_oft_dino/76_collect_dino_scratch_3090.sbatch
bash -n slurm/exploring_oft_dino/77_build_dino_scratch_bank_cpu.sbatch
bash -n slurm/exploring_oft_dino/78_train_dino_scratch_4090.sbatch
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_SCRATCH_wrapper.py --dry-run
PREFIX_STATUS=research_runs/wrist_dino_scratch_v1/training/warmup/status.json
if [[ -f ${PREFIX_STATUS} ]]; then
  "${PYTHON_BIN}" -B -c 'from dino_scratch_common import config, prefix_state; prefix_state(config()); print("DINO_SCRATCH_PREFIX_VERIFIED")'
  REUSE_PREFIX=1
else
  REUSE_PREFIX=0
fi
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo 'DINO_SCRATCH_CHECK_ONLY_PASS; no sbatch submitted'
  exit 0
fi
mkdir -p research_runs/wrist_dino_scratch_v1/logs research_runs/wrist_dino_scratch_v1/data
if [[ ${REUSE_PREFIX} == 1 ]]; then
  echo 'DINO_SCRATCH_PREFIX=REUSED_VERIFIED_250'
  collect_raw=$(sbatch --parsable slurm/exploring_oft_dino/76_collect_dino_scratch_3090.sbatch)
else
  prefix_raw=$(sbatch --parsable slurm/exploring_oft_dino/75_dino_scratch_warmup_4090.sbatch)
  prefix_id=${prefix_raw%%;*}
  echo "DINO_SCRATCH_PREFIX=${prefix_id}"
  collect_raw=$(sbatch --parsable --dependency="afterok:${prefix_id}" slurm/exploring_oft_dino/76_collect_dino_scratch_3090.sbatch)
fi
collect_id=${collect_raw%%;*}
echo "DINO_SCRATCH_COLLECT=${collect_id}"
bank_raw=$(sbatch --parsable --dependency="afterok:${collect_id}" slurm/exploring_oft_dino/77_build_dino_scratch_bank_cpu.sbatch)
bank_id=${bank_raw%%;*}
echo "DINO_SCRATCH_BANK=${bank_id}"
train_raw=$(sbatch --parsable --dependency="afterok:${bank_id}" slurm/exploring_oft_dino/78_train_dino_scratch_4090.sbatch)
train_id=${train_raw%%;*}
echo "DINO_SCRATCH_TRAIN=${train_id}"
echo 'Nine PNGs: research_runs/wrist_dino_scratch_v1/training/{uada,uniform,dino}/outer_{0500,0750,1000}.png'
echo 'Expected training + collection: 4-5 hours excluding queues. Evaluation follows user visual review.'

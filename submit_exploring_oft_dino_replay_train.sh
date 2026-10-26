#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:/home/caijiasheng25/robopatch_defense/oft_ramp_recovery_aware/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -B -m pytest -q tests/test_exploring_oft_dino_replay.py
"${PYTHON_BIN}" -B scripts/collect_exploring_oft_dino_replay.py --preflight
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_REPLAY_wrapper.py --dry-run
bash -n slurm/exploring_oft_dino/64_collect_dino_replay_4090.sbatch
bash -n slurm/exploring_oft_dino/65_build_dino_replay_bank_cpu.sbatch
bash -n slurm/exploring_oft_dino/66_train_dino_replay_4090.sbatch
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo 'DINO_REPLAY_CHECK_ONLY_PASS; no sbatch submitted'
  exit 0
fi
mkdir -p research_runs/wrist_dino_replay_v1/logs research_runs/wrist_dino_replay_v1/data
collect_raw=$(sbatch --parsable slurm/exploring_oft_dino/64_collect_dino_replay_4090.sbatch)
collect_id=${collect_raw%%;*}
bank_raw=$(sbatch --parsable --dependency="afterok:${collect_id}" slurm/exploring_oft_dino/65_build_dino_replay_bank_cpu.sbatch)
bank_id=${bank_raw%%;*}
train_raw=$(sbatch --parsable --dependency="afterok:${bank_id}" slurm/exploring_oft_dino/66_train_dino_replay_4090.sbatch)
train_id=${train_raw%%;*}
echo "DINO_REPLAY_COLLECT=${collect_id} BANK=${bank_id} TRAIN=${train_id}"
echo "Four final PNGs: ${PROJECT_ROOT}/research_runs/wrist_dino_replay_v1/training/<method>/outer_0200.png"
echo 'Evaluation is NOT submitted; visually review all four PNGs first.'

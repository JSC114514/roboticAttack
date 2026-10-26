#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -B -m pytest -q tests/test_exploring_oft_dino_window.py
"${PYTHON_BIN}" -B scripts/collect_exploring_oft_dino_window.py --preflight
"${PYTHON_BIN}" -B scripts/build_exploring_oft_dino_window_bank.py --preflight
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_WINDOW_wrapper.py --dry-run
bash -n slurm/exploring_oft_dino/70_collect_dino_window_3090.sbatch
bash -n slurm/exploring_oft_dino/71_build_dino_window_bank_cpu.sbatch
bash -n slurm/exploring_oft_dino/72_train_dino_window_4090.sbatch
grep -Fq 'export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl' slurm/exploring_oft_dino/70_collect_dino_window_3090.sbatch
grep -Fq 'export MUJOCO_EGL_DEVICE_ID=' slurm/exploring_oft_dino/70_collect_dino_window_3090.sbatch
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo 'DINO_WINDOW_CHECK_ONLY_PASS; no sbatch submitted'
  exit 0
fi
mkdir -p research_runs/wrist_dino_window_v1/logs research_runs/wrist_dino_window_v1/data
collect_raw=$(sbatch --parsable slurm/exploring_oft_dino/70_collect_dino_window_3090.sbatch)
collect_id=${collect_raw%%;*}
bank_raw=$(sbatch --parsable --dependency="afterok:${collect_id}" slurm/exploring_oft_dino/71_build_dino_window_bank_cpu.sbatch)
bank_id=${bank_raw%%;*}
train_raw=$(sbatch --parsable --dependency="afterok:${bank_id}" slurm/exploring_oft_dino/72_train_dino_window_4090.sbatch)
train_id=${train_raw%%;*}
echo "DINO_WINDOW_COLLECT=${collect_id} BANK=${bank_id} TRAIN=${train_id}"
echo "Eight PNGs: ${PROJECT_ROOT}/research_runs/wrist_dino_window_v1/training/<method>_seed<seed>/outer_0200.png"
echo 'Four-suite evaluation waits for user visual review of all eight PNGs.'

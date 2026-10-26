#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
ATTRIBUTION_ROOT=/home/caijiasheng25/robopatch_defense/vla_internal_attribution_20260812
OFT_ROOT=/home/caijiasheng25/Desktop/tex3D/openvla-oft
LIBERO_ROOT=/home/caijiasheng25/Desktop/LIBERO
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/VLAAttacker/white_patch:${ATTRIBUTION_ROOT}:${OFT_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
export _CHECK_PEFT=0
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_dino_restart.py tests/test_exploring_oft_full_worst500.py
"${PYTHON_BIN}" -m py_compile VLAAttacker/white_patch/UADA.py VLAAttacker/white_patch/UADA_OFT_DINO.py VLAAttacker/UADA_OFT_DINO_wrapper.py scripts/analyze_exploring_oft_dino_structure.py
"${PYTHON_BIN}" -c 'import sys; sys.path.insert(0, "/home/caijiasheng25/Desktop/tex3D/openvla-oft"); from experiments.robot import openvla_utils, robot_utils; from prismatic.models.action_heads import L1RegressionActionHead; print("OFT_DEPENDENCY_IMPORT_PASS")'
"${PYTHON_BIN}" -c 'import torch; from white_patch.UADA import init_uada_patch, build_uada_adamw, step_uada_patch; from white_patch.UADA_OFT_DINO import ExploringOFTOpenVLAAttacker; patch=init_uada_patch((3, 4, 4), "cpu"); optimizer, scheduler=build_uada_adamw(patch, 0.002, 1, 2); patch.grad=torch.ones_like(patch); step_uada_patch(patch, optimizer, torch.nn.Linear(1, 1)); scheduler.step(); assert patch.grad is None and bool(((patch >= 0) & (patch <= 1)).all()); print("UADA_SHARED_CORE_SMOKE_PASS")'
"${PYTHON_BIN}" VLAAttacker/UADA_OFT_DINO_wrapper.py \
  --config configs/exploring_oft_dino/full_worst500.json \
  --method exploring_oft_full_worst_chunk --dry-run

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_FULL_WORST500_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi

train_raw=$(sbatch --parsable slurm/exploring_oft_dino/07_train_full_worst500.sbatch)
train_id=${train_raw%%;*}
report_raw=$(sbatch --parsable --dependency="afterany:${train_id}" slurm/exploring_oft_dino/08_structure_full_worst500.sbatch)
report_id=${report_raw%%;*}
echo "EXPLORING_OFT_FULL_WORST500_TRAIN=${train_id}"
echo "EXPLORING_OFT_FULL_WORST500_STRUCTURE=${report_id}"
echo "One fresh-start RTX3090 job, 7h time limit; manual visual review required before held-out CASR."

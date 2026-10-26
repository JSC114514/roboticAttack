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

"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_dino_restart.py
"${PYTHON_BIN}" -m py_compile VLAAttacker/white_patch/UADA_OFT_DINO.py VLAAttacker/UADA_OFT_DINO_wrapper.py scripts/analyze_exploring_oft_dino_structure.py
"${PYTHON_BIN}" -c 'import sys; sys.path.insert(0, "/home/caijiasheng25/Desktop/tex3D/openvla-oft"); from experiments.robot import openvla_utils, robot_utils; from prismatic.models.action_heads import L1RegressionActionHead; print("OFT_DEPENDENCY_IMPORT_PASS")'
"${PYTHON_BIN}" VLAAttacker/UADA_OFT_DINO_wrapper.py --config configs/exploring_oft_dino/object_scale500.json --method exploring_oft_uada --dry-run
"${PYTHON_BIN}" VLAAttacker/UADA_OFT_DINO_wrapper.py --config configs/exploring_oft_dino/object_scale500.json --method exploring_oft_dino_cf --dry-run

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_DINO_SCALE500_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi

train_raw=$(sbatch --parsable slurm/exploring_oft_dino/05_train_scale500.sbatch)
train_id=${train_raw%%;*}
report_raw=$(sbatch --parsable --dependency="afterany:${train_id}" slurm/exploring_oft_dino/06_structure_scale500.sbatch)
report_id=${report_raw%%;*}
echo "EXPLORING_OFT_DINO_SCALE500_TRAIN=${train_id}"
echo "EXPLORING_OFT_DINO_SCALE500_STRUCTURE=${report_id}"
echo "Two paired fresh-start jobs; estimated UADA 2.8h and DINO-CF 5.5h. Manual visual review remains required before scale-500 CASR."

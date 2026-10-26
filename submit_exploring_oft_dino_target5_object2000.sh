#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_dino_target2000.py tests/test_exploring_oft_uada_object2000.py
"${PYTHON_BIN}" -m py_compile VLAAttacker/UADA_OFT_DINO_wrapper.py VLAAttacker/white_patch/UADA_OFT_DINO.py scripts/analyze_exploring_oft_dino_structure.py
"${PYTHON_BIN}" VLAAttacker/UADA_OFT_DINO_wrapper.py \
  --config configs/exploring_oft_dino/dino_target5_object2000.json \
  --method exploring_oft_uada --dry-run
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_DINO_TARGET5_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi
train_raw=$(sbatch --parsable slurm/exploring_oft_dino/14_train_dino_target5_object2000.sbatch)
train_id=${train_raw%%;*}
report_raw=$(sbatch --parsable --dependency="afterany:${train_id}" slurm/exploring_oft_dino/15_structure_dino_target5_object2000.sbatch)
report_id=${report_raw%%;*}
echo "EXPLORING_OFT_DINO_TARGET5_TRAIN=${train_id}"
echo "EXPLORING_OFT_DINO_TARGET5_STRUCTURE=${report_id}"
echo "Exploratory DINO-selected action target; same Exploring UADA patch optimizer and 2000x50 schedule. No evaluation is auto-submitted."

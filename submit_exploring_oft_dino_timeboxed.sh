#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_dino_restart.py
"${PYTHON_BIN}" -m py_compile VLAAttacker/white_patch/UADA_OFT_DINO.py VLAAttacker/UADA_OFT_DINO_wrapper.py scripts/analyze_exploring_oft_dino_structure.py
"${PYTHON_BIN}" VLAAttacker/UADA_OFT_DINO_wrapper.py --method exploring_oft_uada --dry-run
"${PYTHON_BIN}" VLAAttacker/UADA_OFT_DINO_wrapper.py --method exploring_oft_dino --dry-run

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_DINO_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi

train_raw=$(sbatch --parsable slurm/exploring_oft_dino/01_train_timeboxed.sbatch)
train_id=${train_raw%%;*}
report_raw=$(sbatch --parsable --dependency="afterany:${train_id}" slurm/exploring_oft_dino/02_structure_report.sbatch)
report_id=${report_raw%%;*}
echo "EXPLORING_OFT_DINO_TRAIN=${train_id}"
echo "EXPLORING_OFT_DINO_STRUCTURE_REPORT=${report_id}"
echo "Two paired RTX3090 jobs; manual patch-image review is mandatory before any CASR evaluation."

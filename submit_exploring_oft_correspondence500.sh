#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
ATTRIBUTION_ROOT=/home/caijiasheng25/robopatch_defense/vla_internal_attribution_20260812
OFT_ROOT=/home/caijiasheng25/Desktop/tex3D/openvla-oft
LIBERO_ROOT=/home/caijiasheng25/Desktop/LIBERO
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/VLAAttacker/white_patch:${ATTRIBUTION_ROOT}:${OFT_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
export _CHECK_PEFT=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -B -m pytest -q tests/test_exploring_oft_correspondence.py
bash -n slurm/exploring_oft_dino/50_train_correspondence500_v1_3090.sbatch
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_CORRESPONDENCE_wrapper.py \
  --config configs/exploring_oft_dino/object_correspondence500_v1.json --dry-run
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "CORRESPONDENCE500_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi
mkdir -p logs/slurm
train_raw=$(sbatch --parsable slurm/exploring_oft_dino/50_train_correspondence500_v1_3090.sbatch)
train_id=${train_raw%%;*}
echo "OFT_CORRESPONDENCE500_TRAIN=${train_id}"
echo "B/C/D/E: four full-RGB Wrist-only patches; 500 x 50 updates; at most four RTX3090 GPUs."
echo "A reuses frozen UADA500. No evaluation is submitted; review final PNGs first."

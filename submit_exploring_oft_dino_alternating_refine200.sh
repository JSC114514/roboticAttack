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
"${PYTHON_BIN}" -B -m pytest -q tests/test_exploring_oft_alternating.py
bash -n slurm/exploring_oft_dino/57_train_dino_alternating_refine200_3090.sbatch
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_ALTERNATING_wrapper.py \
  --config configs/exploring_oft_dino/object_dino_alternating_refine200_v1.json --dry-run
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "DINO_ALTERNATING_REFINE200_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi
mkdir -p logs/slurm
train_raw=$(sbatch --parsable slurm/exploring_oft_dino/57_train_dino_alternating_refine200_3090.sbatch)
train_id=${train_raw%%;*}
echo "OFT_DINO_ALTERNATING_REFINE200_V1_TRAIN=${train_id}"
echo "Two full-RGB Wrist-only patches: same UADA500 start + 200 x (25 UADA + 25 DINO) updates; at most two RTX3090 GPUs."
echo "Array 0=fixed-reference alternating+TV, 1=rolling local-reference alternating+TV."
echo "Independent alternating output directory; completed ILA patches and historical controls remain available."
echo "No evaluation submitted. Review outer_0200.png files before preparing four-suite evaluation."

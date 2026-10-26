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

"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_object_decoy.py
"${PYTHON_BIN}" -m py_compile \
  VLAAttacker/white_patch/UADA.py \
  VLAAttacker/white_patch/UADA_OFT_SOURCE_COPY.py \
  VLAAttacker/white_patch/UADA_OFT_OBJECT_DECOY.py \
  VLAAttacker/UADA_OFT_OBJECT_DECOY_wrapper.py
bash -n slurm/exploring_oft_dino/41_train_object_decoy500_v1_3090.sbatch
for method in exploring_oft_dino_object_decoy_r03 exploring_oft_dino_object_decoy_r10; do
  "${PYTHON_BIN}" VLAAttacker/UADA_OFT_OBJECT_DECOY_wrapper.py \
    --config configs/exploring_oft_dino/object_decoy500_v1.json \
    --method "${method}" --dry-run
done

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_OBJECT_DECOY500_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi

mkdir -p logs/slurm outputs/exploring_oft_dino/object_decoy500_v1
train_raw=$(sbatch --parsable slurm/exploring_oft_dino/41_train_object_decoy500_v1_3090.sbatch)
train_id=${train_raw%%;*}
echo "EXPLORING_OFT_OBJECT_DECOY500_TRAIN=${train_id}"
echo "Two fresh-start RTX3090 jobs; each is 500 outer x 50 inner = 25000 patch updates."
echo "Manual visual review of outer_0500.png remains required before cross-suite CASR."

#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
ATTRIBUTION_ROOT=/home/caijiasheng25/robopatch_defense/vla_internal_attribution_20260812
OFT_ROOT=/home/caijiasheng25/Desktop/tex3D/openvla-oft
LIBERO_ROOT=/home/caijiasheng25/Desktop/LIBERO
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/VLAAttacker/white_patch:${ATTRIBUTION_ROOT}:${OFT_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
export _CHECK_PEFT=0 PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_dino_jacobian.py
"${PYTHON_BIN}" -m py_compile \
  VLAAttacker/white_patch/UADA_OFT_SOURCE_COPY.py \
  VLAAttacker/white_patch/UADA_OFT_DINO_JACOBIAN.py \
  VLAAttacker/UADA_OFT_DINO_JACOBIAN_wrapper.py
bash -n slurm/exploring_oft_dino/47_train_dino_jacobian500_v1_3090.sbatch
for method in exploring_oft_dino_jacobian_dense_r03 exploring_oft_dino_jacobian_dense_r10; do
  "${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_JACOBIAN_wrapper.py \
    --config configs/exploring_oft_dino/object_dino_jacobian500_v1.json \
    --method "${method}" --dry-run
done

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_DINO_JACOBIAN500_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi

mkdir -p logs/slurm outputs/exploring_oft_dino/object_dino_jacobian500_v1
train_raw=$(sbatch --parsable slurm/exploring_oft_dino/47_train_dino_jacobian500_v1_3090.sbatch)
train_id=${train_raw%%;*}
echo "EXPLORING_OFT_DINO_JACOBIAN500_TRAIN=${train_id}"
echo "Two Wrist-only DINO-Jacobian RTX3090 jobs; each is 500 outer x 50 inner = 25000 patch updates."
echo "Manual visual review of both outer_0500.png files remains required before cross-suite CASR."

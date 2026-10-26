#!/usr/bin/env bash
# Manual submission helper. CHECK_ONLY=1 performs checks and never submits.
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
ATTRIBUTION_ROOT=/home/caijiasheng25/robopatch_defense/vla_internal_attribution_20260812
OFT_ROOT=/home/caijiasheng25/Desktop/tex3D/openvla-oft
LIBERO_ROOT=/home/caijiasheng25/Desktop/LIBERO
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/VLAAttacker/white_patch:${ATTRIBUTION_ROOT}:${OFT_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
export _CHECK_PEFT=0 PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_dino_sampling.py
"${PYTHON_BIN}" -m py_compile \
  scripts/plan_dino_guided_placement.py \
  VLAAttacker/white_patch/UADA_OFT_DINO_SAMPLING.py \
  VLAAttacker/UADA_OFT_DINO_SAMPLING_wrapper.py
bash -n slurm/exploring_oft_dino/44_train_dino_sampling500_3090.sbatch
for method in exploring_oft_site_uada exploring_oft_site_dino_sample; do
  "${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_SAMPLING_wrapper.py \
    --config configs/exploring_oft_dino/dino_sampling500_v1.json \
    --method "${method}" --dry-run
done

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_DINO_SAMPLING500_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi
mkdir -p logs/slurm outputs/exploring_oft_dino/dino_sampling500_v1
train_raw=$(sbatch --parsable slurm/exploring_oft_dino/44_train_dino_sampling500_3090.sbatch)
train_id=${train_raw%%;*}
echo "EXPLORING_OFT_DINO_SAMPLING500_TRAIN=${train_id}"
echo "Two matched-position RTX3090 jobs; each 500 outer x 50 inner = 25000 updates."
echo "Manual visual review of both outer_0500.png files required before evaluation."

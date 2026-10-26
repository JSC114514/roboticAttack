#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/caijiasheng25/robopatch_defense/roboticAttack-main}
PYTHON=${PYTHON:-/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python}
CONFIG_PATH=${CONFIG_PATH:-configs/experiments/oft_libero10_gradconf.yaml}
MODEL_PATH=${MODEL_PATH:-/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-10}
DATA_ROOT=${DATA_ROOT:-/home/caijiasheng25/Desktop/roboticAttack}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/when_oft_gradconf}
OUTPUT_STAGE=${OUTPUT_STAGE:-gradconf_oft_multiview}
SUITE_NAME=${SUITE_NAME:-libero_10}
BATCH_SIZE=${BATCH_SIZE:-2}
SEARCH_ROOTS=${SEARCH_ROOTS:-"/home/caijiasheng25/Desktop/roboticAttack /home/caijiasheng25/Desktop/roboticAttack/openvla-main/dataset /home/caijiasheng25/data /home/caijiasheng25/robopatch_defense /home/caijiasheng25/UPA-RFAS /home/caijiasheng25 /mnt/beegfs/home/caijiasheng25/Desktop/roboticAttack /mnt/beegfs/home/caijiasheng25/Desktop/roboticAttack/openvla-main/dataset /mnt/beegfs/home/caijiasheng25/data /mnt/beegfs/home/caijiasheng25/robopatch_defense /mnt/beegfs/home/caijiasheng25"}

cd "${PROJECT_ROOT}"

read -r -a SEARCH_ROOT_ARRAY <<< "${SEARCH_ROOTS}"

echo "[gradconf-validation] D0 libero_10 / LIBERO-Long data check"
"${PYTHON}" scripts/data/check_libero10_oft.py \
  --data_root "${DATA_ROOT}" \
  --dataset_name libero_10 \
  --search_roots "${SEARCH_ROOT_ARRAY[@]}" \
  --dry_run

echo "[gradconf-validation] OFT-libero10 model check"
"${PYTHON}" scripts/models/check_oft_libero10_model.py \
  --model_path "${MODEL_PATH}"

echo "[gradconf-validation] static validation: G0_random_static_validation"
"${PYTHON}" scripts/when/diagnose_oft_multiview_gradients.py \
  --config "${CONFIG_PATH}" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --search_roots "${SEARCH_ROOT_ARRAY[@]}" \
  --output_root "${OUTPUT_ROOT}" \
  --output_stage "${OUTPUT_STAGE}" \
  --suite_name "${SUITE_NAME}" \
  --run_name G0_random_static_validation \
  --mode static \
  --num_batches 2 \
  --batch_size "${BATCH_SIZE}" \
  --dry_run \
  --validation \
  --cleanup_validation_on_success

echo "[gradconf-validation] dynamic validation: G1_dynamic_validation"
"${PYTHON}" scripts/when/diagnose_oft_multiview_gradients.py \
  --config "${CONFIG_PATH}" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --search_roots "${SEARCH_ROOT_ARRAY[@]}" \
  --output_root "${OUTPUT_ROOT}" \
  --output_stage "${OUTPUT_STAGE}" \
  --suite_name "${SUITE_NAME}" \
  --run_name G1_dynamic_validation \
  --mode dynamic \
  --num_batches 2 \
  --batch_size "${BATCH_SIZE}" \
  --dry_run \
  --validation \
  --cleanup_validation_on_success

echo "[gradconf-validation] success; validation run directories were removed by the diagnostic script."

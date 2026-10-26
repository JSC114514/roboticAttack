#!/bin/bash
# Download BridgeData V2 TFDS files in the layout expected by OpenVLA.

set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-/home/caijiasheng25/Desktop/roboticAttack/openvla-main/dataset}
SOURCE_URL=${SOURCE_URL:-https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/}

mkdir -p "${DATASET_ROOT}"
cd "${DATASET_ROOT}"

if [[ -d bridge_orig/1.0.0 ]]; then
  echo "[bridge-download] bridge_orig already exists: ${DATASET_ROOT}/bridge_orig/1.0.0"
  exit 0
fi

echo "[bridge-download] target=${DATASET_ROOT}/bridge_orig"
echo "[bridge-download] source=${SOURCE_URL}"
echo "[bridge-download] expected size is about 124GB; this can take a while."

wget \
  --recursive \
  --no-parent \
  --no-host-directories \
  --cut-dirs=4 \
  --continue \
  --tries=10 \
  --timeout=60 \
  --waitretry=5 \
  --reject="index.html*" \
  "${SOURCE_URL}"

if [[ ! -d bridge_dataset/1.0.0 ]]; then
  echo "[bridge-download][error] expected bridge_dataset/1.0.0 after download" >&2
  exit 1
fi

if [[ -e bridge_orig ]]; then
  echo "[bridge-download][error] bridge_orig exists but bridge_orig/1.0.0 was not found: ${DATASET_ROOT}/bridge_orig" >&2
  exit 1
fi

mv bridge_dataset bridge_orig
echo "[bridge-download] done: ${DATASET_ROOT}/bridge_orig/1.0.0"

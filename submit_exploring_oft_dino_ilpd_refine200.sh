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
"${PYTHON_BIN}" -B -m pytest -q tests/test_exploring_oft_ilpd.py
bash -n slurm/exploring_oft_dino/60_train_dino_ilpd_refine200_3090.sbatch
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_ILPD_wrapper.py \
  --config configs/exploring_oft_dino/object_dino_ilpd_refine200_v1.json --dry-run
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "DINO_ILPD_REFINE200_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi
mkdir -p "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_ilpd_v1/logs" "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_ilpd_v1/reports"
train_raw=$(sbatch --parsable slurm/exploring_oft_dino/60_train_dino_ilpd_refine200_3090.sbatch)
train_id=${train_raw%%;*}
echo "OFT_DINO_ILPD_REFINE200_V1_TRAIN=${train_id}"
echo "Four full-RGB Wrist-only patches: UADA500 + 200 x 50 UADA updates; at most four RTX3090 GPUs."
echo "Array 0=UADA, 1=DINO backward-only x0.5, 2=DINO forward ILPD x0.5, 3=SigLIP forward ILPD x0.5."
echo "Unified files: /home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_ilpd_v1"
echo "PNG: training/<method>/outer_0200.png; logs: logs/ilpd_train_${train_id}_<arrayid>.out/.err"
echo "No evaluation submitted. Review four final PNG files before four-suite evaluation."


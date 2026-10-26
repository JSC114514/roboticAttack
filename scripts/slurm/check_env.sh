#!/bin/bash
set -euo pipefail

echo "[check] cwd=$(pwd)"
test -d VLAAttacker
test -d scripts/experiments
command -v python || true
command -v conda || true
command -v nvidia-smi || true
command -v sbatch || true
command -v squeue || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi -L | grep -i A800 || echo "[warn] 当前节点未看到 A800，提交后请以计算节点为准。"
fi
test -f configs/experiments/M1_oft_train_front_attack_front.yaml
test -f scripts/slurm/submit_M1_oft_train_front_attack_front.sbatch
mkdir -p outputs/multiview_attack logs/slurm
echo "[check] ok"

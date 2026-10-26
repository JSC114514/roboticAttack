#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "用法: bash scripts/slurm/submit_pair.sh M1 M2 dry_run|run" >&2
  exit 2
fi

EXP_A="$1"
EXP_B="$2"
MODE="$3"
if [ "$EXP_A" = "$EXP_B" ]; then
  echo "不允许重复提交同一个实验。" >&2
  exit 2
fi

SUBMIT_SLEEP_SECONDS=${SUBMIT_SLEEP_SECONDS:-30}

bash scripts/slurm/submit_one.sh "$EXP_A" "$MODE"
if [ "$SUBMIT_SLEEP_SECONDS" -gt 0 ]; then
  echo "[submit_pair] sleep ${SUBMIT_SLEEP_SECONDS}s before submitting $EXP_B"
  sleep "$SUBMIT_SLEEP_SECONDS"
fi
bash scripts/slurm/submit_one.sh "$EXP_B" "$MODE"

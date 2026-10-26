#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "用法: bash scripts/slurm/submit_one.sh M1 dry_run|run" >&2
  exit 2
fi

EXP_ID="$1"
MODE="$2"
case "$MODE" in
  dry_run|run) ;;
  *) echo "MODE must be dry_run or run" >&2; exit 2 ;;
esac

case "$EXP_ID" in
  R0|R1|M0|M1|M2|M3|M4|M5|M6|M7|A0|A1) ;;
  *) echo "当前只支持 R0/R1/M0-M7/A0-A1；S/B 系列已跳过。" >&2; exit 2 ;;
esac

check_dep() {
  local source_exp="$1"
  if [ "$MODE" = "run" ]; then
    python scripts/experiments/find_latest_patch.py --source_exp_id "$source_exp" --require_existing_patch >/dev/null
  fi
}

case "$EXP_ID" in
  M3) check_dep M1 ;;
  M4) check_dep M2 ;;
  M6|M7) check_dep M5 ;;
  A1) check_dep M1 ;;
esac

matches=(scripts/slurm/submit_"${EXP_ID}"_*.sbatch)
if [ ! -e "${matches[0]}" ]; then
  echo "找不到实验 ${EXP_ID} 对应 sbatch。" >&2
  exit 2
fi
if [ "${#matches[@]}" -ne 1 ]; then
  echo "实验 ${EXP_ID} 匹配到多个 sbatch。" >&2
  printf '%s\n' "${matches[@]}" >&2
  exit 2
fi

RUN_MODE="$MODE" sbatch "${matches[0]}"

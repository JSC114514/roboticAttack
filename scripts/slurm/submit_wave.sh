#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "用法: bash scripts/slurm/submit_wave.sh W0 dry_run|run" >&2
  exit 2
fi

WAVE="$1"
MODE="$2"
case "$WAVE" in
  W0) exps=(R0 M0) ;;
  W1) exps=(R1 M1) ;;
  W2) exps=(M2 M5) ;;
  W3) exps=(M3 M4) ;;
  W4) exps=(M6 M7) ;;
  WA) exps=(A0 A1) ;;
  *) echo "未知 wave: $WAVE" >&2; exit 2 ;;
esac

if [ "$MODE" = "run" ]; then
  case "$WAVE" in
    W3)
      python scripts/experiments/find_latest_patch.py --source_exp_id M1 --require_existing_patch >/dev/null
      python scripts/experiments/find_latest_patch.py --source_exp_id M2 --require_existing_patch >/dev/null
      ;;
    W4)
      python scripts/experiments/find_latest_patch.py --source_exp_id M5 --require_existing_patch >/dev/null
      ;;
    WA)
      python scripts/experiments/find_latest_patch.py --source_exp_id M1 --require_existing_patch >/dev/null
      ;;
  esac
fi

SUBMIT_SLEEP_SECONDS=${SUBMIT_SLEEP_SECONDS:-30}

for i in "${!exps[@]}"; do
  exp="${exps[$i]}"
  bash scripts/slurm/submit_one.sh "$exp" "$MODE"
  if [ "$i" -lt "$((${#exps[@]} - 1))" ] && [ "$SUBMIT_SLEEP_SECONDS" -gt 0 ]; then
    echo "[submit_wave] sleep ${SUBMIT_SLEEP_SECONDS}s before next submit"
    sleep "$SUBMIT_SLEEP_SECONDS"
  fi
done

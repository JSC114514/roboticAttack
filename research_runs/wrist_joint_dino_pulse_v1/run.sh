#!/usr/bin/env bash
set -euo pipefail
JR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
exec /home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B "${JR_ROOT}/code/dp_cli.py" "$@"

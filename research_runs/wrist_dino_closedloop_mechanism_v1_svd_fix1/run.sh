#!/usr/bin/env bash
set -euo pipefail
PILOT_RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PILOT_RUN_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${PILOT_ROOT}/code/pilot_pipeline.py" "$@"

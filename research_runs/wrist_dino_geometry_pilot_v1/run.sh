#!/usr/bin/env bash
set -euo pipefail
PILOT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${PILOT_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/pilot_pipeline.py" "$@"

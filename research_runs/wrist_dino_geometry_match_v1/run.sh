#!/usr/bin/env bash
set -euo pipefail
GEOM_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${GEOM_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/pipeline.py" "$@"

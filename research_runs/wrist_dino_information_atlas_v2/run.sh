#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/atlas_pipeline.py" "$@"

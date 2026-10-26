#!/usr/bin/env bash
set -euo pipefail
RELATION_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${RELATION_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${EXPERIMENT_ROOT}/code/pipeline.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
RESIDUAL4_ENTRY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RESIDUAL4_ENTRY_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${RESIDUAL4_ROOT}/code/residual4_pipeline.py" "$@"

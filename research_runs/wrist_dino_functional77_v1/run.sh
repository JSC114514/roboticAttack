#!/usr/bin/env bash
set -euo pipefail
FUNCTIONAL77_ENTRY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${FUNCTIONAL77_ENTRY_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${FUNCTIONAL77_ROOT}/code/functional77_pipeline.py" "$@"

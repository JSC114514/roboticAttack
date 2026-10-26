#!/usr/bin/env bash
set -euo pipefail
RR_ENTRY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RR_ENTRY_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${RR_ROOT}/code/rr_cli.py" "$@"

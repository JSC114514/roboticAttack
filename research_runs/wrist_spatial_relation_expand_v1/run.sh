#!/usr/bin/env bash
set -euo pipefail
FOLLOWUP_ENTRY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${FOLLOWUP_ENTRY_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${FOLLOWUP_ROOT}/code/entry.py" "$@"

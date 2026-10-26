#!/usr/bin/env bash
set -euo pipefail
RFIX_ENTRY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RFIX_ENTRY}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${RFIX_ENTRY}/code/rfix_cli.py" "$@"

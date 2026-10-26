#!/usr/bin/env bash
set -euo pipefail
SE_ENTRY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SE_ENTRY}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${SE_ENTRY}/code/se_cli.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
JT_ENTRY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JT_ENTRY}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${JT_ENTRY}/code/jt_cli.py" "$@"

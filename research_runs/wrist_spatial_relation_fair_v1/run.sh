#!/usr/bin/env bash
set -euo pipefail
FC_ENTRY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${FC_ENTRY}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${FC_ENTRY}/code/fc_cli.py" "$@"

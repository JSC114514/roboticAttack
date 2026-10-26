#!/usr/bin/env bash
set -euo pipefail
AUDIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${AUDIT_ROOT}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${AUDIT_ROOT}/code/audit_cli.py" "$@"

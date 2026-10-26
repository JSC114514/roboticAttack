#!/usr/bin/env bash
set -euo pipefail
VA_ENTRY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${VA_ENTRY}/slurm/environment.sh"
exec "${PYTHON_BIN}" -B "${VA_ENTRY}/code/va_cli.py" "$@"

#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/caijiasheng25/robopatch_defense/roboticAttack-main}

cd "${PROJECT_ROOT}"
exec bash scripts/validate/validate_oft_libero10_gradconf.sh "$@"

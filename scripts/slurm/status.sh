#!/bin/bash
set -euo pipefail

OUTPUT_ROOT=${1:-outputs/multiview_attack}
find "$OUTPUT_ROOT" -maxdepth 2 -name status.json -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -20

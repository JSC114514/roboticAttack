#!/usr/bin/env bash
# Object batch was never submitted; retired by the user's Long-source decision.
set -euo pipefail
echo 'Object batch is superseded and will not be submitted.' >&2
echo 'Use: bash submit_exploring_oft_gripper_tma500_long_v1.sh [--preflight|--dry-run]' >&2
exit 2

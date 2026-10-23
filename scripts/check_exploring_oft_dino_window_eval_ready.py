#!/usr/bin/env python3
"""Fail before sbatch unless eight final PNGs have user-approved hashes."""
from __future__ import annotations

import json

from run_exploring_oft_dino_window_cross_suite import load_config, preflight, require

status = preflight(load_config())
require(status["status"] == "PASS", f"Window evaluation not ready: {status['status']}")
print(json.dumps(status, indent=2))

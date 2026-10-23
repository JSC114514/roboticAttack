#!/usr/bin/env python3
from __future__ import annotations

import json
from run_exploring_oft_dino_replay_cross_suite import load_config, preflight, require

status = preflight(load_config())
require(status["status"] == "PASS", f"Evaluation not ready: {status['status']}")
print(json.dumps(status, indent=2))

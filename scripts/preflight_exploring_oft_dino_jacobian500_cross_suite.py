#!/usr/bin/env python3
"""Read-only check of the three frozen patches and paired four-suite grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_exploring_oft_dino_jacobian500_cross_suite import (
    CONDITIONS,
    load_jacobian_patches,
    validate_contract,
)
from run_exploring_oft_scale500_cross_suite import frozen_clean, runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    validate_contract(cfg)
    patches = load_jacobian_patches(cfg, torch.device("cpu"))
    denominators = {}
    for suite in cfg["suites"]:
        clean = frozen_clean(cfg, suite)
        denominators[suite] = int(clean.success.sum())
        runtime = runtime_config(cfg, suite)
        assert Path(runtime["paths"]["checkpoint"], "config.json").is_file()
    assert denominators == {
        "libero_object": 99, "libero_spatial": 90,
        "libero_goal": 98, "libero_10": 95,
    }
    print(json.dumps({
        "status": "DINO_JACOBIAN500_XSUITE_PREFLIGHT_PASS",
        "conditions": list(CONDITIONS),
        "patch_sha256": {name: digest for name, (_patch, digest) in patches.items()},
        "frozen_clean_episodes": 400,
        "new_attack_episodes": 400 * len(CONDITIONS),
        "eligible_attack_rows_expected": sum(denominators.values()) * len(CONDITIONS),
        "clean_success_denominators": denominators,
    }, indent=2))


if __name__ == "__main__":
    main()

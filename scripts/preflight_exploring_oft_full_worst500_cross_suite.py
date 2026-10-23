#!/usr/bin/env python3
"""Read-only preflight for the two-arm held-out temporal-route control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_exploring_oft_scale500_cross_suite import frozen_clean, load_patches, runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    assert cfg["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
    assert cfg["state_indices"] == list(range(40, 50)) and cfg["task_count"] == 10
    assert cfg["patch_xy"] == [26, 137] and cfg["patch_shape"] == [3, 50, 50]
    assert cfg["conditions"] == ["exploring_oft_dino_cf_worst_chunk", "exploring_oft_full_worst_chunk"]
    patches = load_patches(cfg, torch.device("cpu"))
    denominators = {}
    for suite in cfg["suites"]:
        clean = frozen_clean(cfg, suite)
        denominators[suite] = int(clean.success.sum())
        runtime = runtime_config(cfg, suite)
        assert Path(runtime["paths"]["checkpoint"], "config.json").is_file()
    assert denominators == {"libero_object": 99, "libero_spatial": 90,
                            "libero_goal": 98, "libero_10": 95}
    print(json.dumps({"status": "FULL_WORST500_XSUITE_PREFLIGHT_PASS",
                      "conditions": list(patches), "frozen_clean_episodes": 400,
                      "new_attack_episodes": 800,
                      "eligible_attack_rows_expected": 2 * sum(denominators.values()),
                      "clean_success_denominators": denominators}, indent=2))


if __name__ == "__main__":
    main()

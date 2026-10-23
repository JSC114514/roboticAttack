#!/usr/bin/env python3
"""Read-only preflight for frozen five-patch, four-suite evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_exploring_oft_scale500_cross_suite import frozen_clean, load_patches, runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = Path(args.config).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
    assert cfg["task_count"] == 10 and cfg["state_indices"] == list(range(40, 50))
    assert cfg["patch_xy"] == [26, 137] and cfg["patch_shape"] == [3, 50, 50]
    assert len(cfg["conditions"]) == 5 and len(set(cfg["conditions"])) == 5
    patches = load_patches(cfg, torch.device("cpu"))
    assert set(patches) == set(cfg["conditions"])
    denominators = {}
    for suite in cfg["suites"]:
        clean = frozen_clean(cfg, suite)
        denominators[suite] = int(clean.success.sum())
        runtime = runtime_config(cfg, suite)
        assert Path(runtime["paths"]["checkpoint"], "config.json").is_file(), suite
        assert runtime["unnorm_key"] == cfg["unnorm_keys"][suite]
    assert denominators == {
        "libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95
    }, denominators
    print(json.dumps({"status": "SCALE500_XSUITE_PREFLIGHT_PASS", "config": str(path),
                      "conditions": list(patches), "reused_clean_episodes": 400,
                      "new_attack_episodes": 2000,
                      "eligible_attack_rows_expected": 5 * sum(denominators.values()),
                      "clean_success_denominators": denominators,
                      "max_concurrent_gpus": 5}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

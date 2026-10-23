#!/usr/bin/env python3
"""Read-only preflight for paired 500-versus-2000 UADA baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_exploring_oft_scale500_cross_suite import frozen_clean, load_patches, runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preparation-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    assert cfg["analysis_kind"] == "UADA2000_VS500"
    assert cfg["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
    assert cfg["task_count"] == 10 and cfg["state_indices"] == list(range(40, 50))
    assert cfg["patch_xy"] == [26, 137] and cfg["patch_shape"] == [3, 50, 50]
    assert cfg["conditions"] == ["exploring_oft_uada_o0500", "exploring_oft_uada_o2000"]
    assert cfg["required_manual_visual_review"] == "PASS_USER_UADA2000"
    sources = cfg["patch_sources"]
    assert set(sources) == set(cfg["conditions"])
    assert [sources[c]["outer_iterations"] for c in cfg["conditions"]] == [500, 2000]
    old_path = Path(sources["exploring_oft_uada_o0500"]["patch_path"])
    assert old_path.is_file()
    old_payload = torch.load(old_path, map_location="cpu", weights_only=False)
    assert old_payload["experiment_id"] == "EXPLORING-OFT-DINO-SCALE500-01"
    assert old_payload["method"] == "exploring_oft_uada"
    assert old_payload["outer_iterations"] == 500
    denominators = {}
    for suite in cfg["suites"]:
        clean = frozen_clean(cfg, suite)
        denominators[suite] = int(clean.success.sum())
        runtime = runtime_config(cfg, suite)
        assert Path(runtime["paths"]["checkpoint"], "config.json").is_file()
    assert denominators == {"libero_object": 99, "libero_spatial": 90,
                            "libero_goal": 98, "libero_10": 95}
    if not args.preparation_only:
        patches = load_patches(cfg, torch.device("cpu"))
        assert set(patches) == set(cfg["conditions"])
    print(json.dumps({"status": "UADA2000_XSUITE_PREPARATION_PASS" if args.preparation_only
                      else "UADA2000_XSUITE_PREFLIGHT_PASS",
                      "conditions": cfg["conditions"], "reused_clean_episodes": 400,
                      "new_attack_episodes": 800,
                      "eligible_attack_rows_expected": 2 * sum(denominators.values()),
                      "clean_success_denominators": denominators,
                      "new_patch_and_visual_review_checked": not args.preparation_only}, indent=2))


if __name__ == "__main__":
    main()

"""Frozen three-arm 2000-outer patch evaluation contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_exploring_oft_scale500_cross_suite import load_patches, validate  # noqa: E402


CFG_PATH = ROOT / "configs/exploring_oft_dino/three_arm_object2000_cross_suite.json"


def test_three_arm_patch_lineage_and_user_approval():
    cfg = json.loads(CFG_PATH.read_text())
    assert cfg["analysis_kind"] == "UADA2000_TARGET5_VS500"
    assert cfg["manual_visual_review"] == cfg["required_manual_visual_review"] == "PASS_USER_BOTH2000"
    assert cfg["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
    assert cfg["state_indices"] == list(range(40, 50))
    assert [cfg["patch_sources"][c]["outer_iterations"] for c in cfg["conditions"]] == [500, 2000, 2000]
    patches = load_patches(cfg, torch.device("cpu"))
    assert len(patches) == 3 and len({digest for _patch, digest in patches.values()}) == 3


def test_three_arm_rollout_grid_requires_all_conditions_and_paired_seed():
    cfg = json.loads(CFG_PATH.read_text())
    rows = [{"suite": "libero_object", "task_id": 0, "state_index": state,
             "condition": condition, "condition_order": order,
             "rollout_seed": 1000 + state, "frozen_clean_success": 1,
             "renderer_vendor": "NVIDIA"}
            for state in cfg["state_indices"]
            for order, condition in enumerate(cfg["conditions"], start=1)]
    rollouts = pd.DataFrame(rows)
    traces = rollouts[["suite", "task_id", "state_index", "condition"]].copy()
    traces["decision_block"] = 0
    validate(rollouts, traces, "libero_object", 0, cfg)
    rollouts.loc[0, "rollout_seed"] += 1
    with pytest.raises(RuntimeError, match="Paired rollout seed drift"):
        validate(rollouts, traces, "libero_object", 0, cfg)

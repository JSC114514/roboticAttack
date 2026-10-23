from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_exploring_oft_scale500_cross_suite import frozen_clean, load_patches, validate  # noqa: E402
from analyze_exploring_oft_scale500_cross_suite import CONTRASTS  # noqa: E402


CONFIG = ROOT / "configs" / "exploring_oft_dino" / "scale500_cross_suite.json"


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_cross_suite_contract_reuses_clean_and_caps_gpus() -> None:
    cfg = config()
    assert cfg["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
    assert cfg["state_indices"] == list(range(40, 50)) and cfg["task_count"] == 10
    assert cfg["patch_xy"] == [26, 137] and cfg["patch_shape"] == [3, 50, 50]
    assert len(cfg["conditions"]) == 5 and cfg["manual_visual_review"] == "PASS_USER_20260921_SCALE500"
    sbatch = (ROOT / "slurm" / "exploring_oft_dino" / "07_scale500_cross_suite.sbatch").read_text()
    assert "#SBATCH --array=0-39%5" in sbatch
    assert "#SBATCH --gres=gpu:3090:1" in sbatch
    submit = (ROOT / "submit_exploring_oft_scale500_cross_suite.sh").read_text()
    assert "afterok:${eval_id}" in submit and "CHECK_ONLY" in submit


def test_frozen_clean_grids_are_complete_and_unrerun() -> None:
    cfg = config()
    expected = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}
    observed = {suite: int(frozen_clean(cfg, suite).success.sum()) for suite in cfg["suites"]}
    assert observed == expected


def test_five_patches_have_distinct_correct_training_identity() -> None:
    cfg = config()
    patches = load_patches(cfg, torch.device("cpu"))
    assert set(patches) == set(cfg["conditions"])
    assert len({digest for _patch, digest in patches.values()}) == 5
    assert all(tuple(patch.shape) == (3, 50, 50) for patch, _digest in patches.values())


def test_validate_rejects_unpaired_seed() -> None:
    cfg = config()
    rows = []
    for state_index in cfg["state_indices"]:
        for order, condition in enumerate(cfg["conditions"], start=1):
            rows.append({"suite": "libero_object", "task_id": 0, "state_index": state_index,
                         "condition": condition, "condition_order": order,
                         "rollout_seed": 1000 + state_index, "frozen_clean_success": 1,
                         "renderer_vendor": "NVIDIA"})
    rollouts = pd.DataFrame(rows)
    traces = rollouts[["suite", "task_id", "state_index", "condition"]].copy()
    traces["decision_block"] = 0
    validate(rollouts, traces, "libero_object", 0, cfg)
    rollouts.loc[0, "rollout_seed"] += 1
    with pytest.raises(RuntimeError, match="Paired rollout seed drift"):
        validate(rollouts, traces, "libero_object", 0, cfg)


def test_contrasts_are_preregistered_and_long_is_not_pooled() -> None:
    assert [role for role, _left, _right in CONTRASTS] == [
        "DINO_CF_MINUS_UADA", "WORST_CHUNK_MINUS_DINO_CF",
        "DINO_RESTORE_MINUS_UADA", "DINO_CF_MINUS_SIGLIP_CF",
    ]
    source = (ROOT / "scripts" / "analyze_exploring_oft_scale500_cross_suite.py").read_text()
    assert 'for suite_index, suite in enumerate(cfg["suites"]):' in source
    assert "task_episode_bootstrap" in source and "paired_contrast_bootstrap" in source
    assert "SCALE500_XSUITE_PER_SUITE.csv" in source

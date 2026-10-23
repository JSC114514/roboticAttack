from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_exploring_oft_dino_jacobian500_cross_suite import (  # noqa: E402
    CONDITIONS, load_jacobian_patches, validate_contract,
)
from run_exploring_oft_scale500_cross_suite import frozen_clean, validate  # noqa: E402


CONFIG = ROOT / "configs/exploring_oft_dino/object_dino_jacobian500_cross_suite.json"


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_three_patches_are_complete_distinct_and_frozen() -> None:
    cfg = config()
    validate_contract(cfg)
    patches = load_jacobian_patches(cfg, torch.device("cpu"))
    assert tuple(patches) == CONDITIONS
    assert len({digest for _patch, digest in patches.values()}) == 3
    assert all(tuple(patch.shape) == (3, 50, 50) for patch, _digest in patches.values())


def test_cross_suite_grid_reuses_same_frozen_clean_states() -> None:
    cfg = config()
    assert {suite: int(frozen_clean(cfg, suite).success.sum()) for suite in cfg["suites"]} == {
        "libero_object": 99, "libero_spatial": 90,
        "libero_goal": 98, "libero_10": 95,
    }
    assert cfg["conditions"] == list(CONDITIONS)
    assert cfg["patch_xy"] == [26, 137]


def test_contract_rejects_evaluator_drift() -> None:
    cfg = config()
    cfg["patch_xy"] = [27, 137]
    with pytest.raises(RuntimeError, match="Scale-500 evaluator drift: patch_xy"):
        validate_contract(cfg)


def test_shard_validation_rejects_unpaired_seed() -> None:
    cfg = config()
    rows = [
        {"suite": "libero_object", "task_id": 0, "state_index": state,
         "condition": condition, "condition_order": order,
         "rollout_seed": 1000 + state, "frozen_clean_success": 1,
         "renderer_vendor": "NVIDIA"}
        for state in cfg["state_indices"]
        for order, condition in enumerate(cfg["conditions"], start=1)
    ]
    rollouts = pd.DataFrame(rows)
    traces = rollouts[["suite", "task_id", "state_index", "condition"]].copy()
    traces["decision_block"] = 0
    validate(rollouts, traces, "libero_object", 0, cfg)
    rollouts.loc[0, "rollout_seed"] += 1
    with pytest.raises(RuntimeError, match="Paired rollout seed drift"):
        validate(rollouts, traces, "libero_object", 0, cfg)


def test_submission_is_manual_and_analysis_has_three_preregistered_contrasts() -> None:
    submit = (ROOT / "submit_exploring_oft_dino_jacobian500_cross_suite.sh").read_text()
    eval_job = (ROOT / "slurm/exploring_oft_dino/48_dino_jacobian500_cross_suite_3090.sbatch").read_text()
    analysis = (ROOT / "scripts/analyze_exploring_oft_scale500_cross_suite.py").read_text()
    assert "CHECK_ONLY" in submit and "afterok:${eval_id}" in submit
    assert "#SBATCH --array=0-39%5" in eval_job
    assert "#SBATCH --gres=gpu:3090:1" in eval_job
    assert 'prefix = "DINO_JACOBIAN500_XSUITE"' in analysis
    assert "JACOBIAN_R03_MINUS_UADA500" in analysis
    assert "JACOBIAN_R10_MINUS_UADA500" in analysis
    assert "JACOBIAN_R10_MINUS_R03" in analysis

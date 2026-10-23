"""CPU mock checks of counted prefix, optimizer transfer and reproducible resume."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "VLAAttacker")]
from dino_scratch_common import CONFIG, METHODS, draw_record, prefix_state
from white_patch.UADA_OFT_DINO_SCRATCH import ExploringOFTDINOScratchAttacker, validate_state
from test_exploring_oft_ilpd import MockILPD


class MockScratch(ExploringOFTDINOScratchAttacker):
    __init__ = MockILPD.__init__
    _base_inputs = MockILPD._base_inputs
    _forward = MockILPD._forward

    def _validate_runtime(self, *_args):
        return {"status": "PASS", "records": [{"pixel_exact": True, "feature_close": True}], "mock": True}


def setup(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    cfg.update(output_root=str(tmp_path), bootstrap_outer=2, outer_iterations=6,
               inner_loop=4, warmup_outer_iterations=1, checkpoint_outer_iterations=[0, 2, 3, 4, 6],
               resume_every_outer_iterations=1)
    clean = [{"task_id": i % 10, "state_index": 10, "task_name": "object",
              "observation": {"t": 0}} for i in range(300)]
    replay = [{"task_id": task, "state_index": 11, "task_name": "object", "phase_proxy": phase,
               "dino_score": float(i + 1), "dino_weight": float(i + 1),
               "observations": [{"t": 1}, {"t": 2}, {"t": 3}]}
              for task in range(10) for phase in ("early", "middle", "late") for i in (0, 1)]
    return cfg, clean, replay


def train_prefix(cfg, clean):
    MockScratch().train_scratch(clean, [], cfg, "warmup", {}, {})
    return {"prefix_resume": prefix_state(cfg)[1]}


def read_resume(cfg, method):
    return torch.load(Path(cfg["output_root"]) / "training" / method / "resume.pt", weights_only=False)


def test_counted_prefix_preserves_adam_and_cosine_vs_uninterrupted(tmp_path):
    cfg, clean, replay = setup(tmp_path / "split")
    data = train_prefix(cfg, clean)
    MockScratch().train_scratch(clean, replay, cfg, "uada", {}, data)
    split = read_resume(cfg, "uada")
    full_cfg = {**cfg, "output_root": str(tmp_path / "full"), "bootstrap_outer": 6}
    MockScratch().train_scratch(clean, [], full_cfg, "warmup", {}, {})
    full = read_resume(full_cfg, "warmup")
    assert torch.equal(split["patch"], full["patch"])
    assert torch.equal(split["initial"], full["initial"])
    assert split["scheduler"] == full["scheduler"]
    assert [r["learning_rate"] for r in split["rows"]] == [r["learning_rate"] for r in full["rows"]]
    assert len(split["rows"]) == 6 and split["rows"][-1]["optimizer_updates"] == 24
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(split["optimizer"]["state"][0][key], full["optimizer"]["state"][0][key])


@pytest.mark.parametrize("method", METHODS)
def test_each_branch_partial_outer_resume_is_bitwise_equal(tmp_path, method):
    cfg, clean, replay = setup(tmp_path / "interrupted")
    data = train_prefix(cfg, clean)
    with pytest.raises(RuntimeError, match="mock interruption"):
        MockScratch(fail_at=6).train_scratch(clean, replay, cfg, method, {}, data)
    MockScratch().train_scratch(clean, replay, cfg, method, {}, data)
    actual = read_resume(cfg, method)
    full_cfg = {**cfg, "output_root": str(tmp_path / "reference")}
    full_data = train_prefix(full_cfg, clean)
    MockScratch().train_scratch(clean, replay, full_cfg, method, {}, full_data)
    expected = read_resume(full_cfg, method)
    assert torch.equal(actual["patch"], expected["patch"])
    assert actual["scheduler"] == expected["scheduler"]
    assert actual["optimizer"]["state"][0]["step"] == 24
    assert actual["rows"][:2] == read_resume(cfg, "warmup")["rows"]
    assert [r["origin"] for r in actual["rows"]] == (
        ["clean"] * 6 if method == "uada" else ["clean", "clean", "clean", "replay", "clean", "replay"])
    for step in (3, 4, 6):
        payload = torch.load(Path(cfg["output_root"]) / "training" / method / f"outer_{step:04d}.pt",
                             weights_only=False)
        assert payload["total_outer"] == step and payload["total_optimizer_updates"] == step * 4


def test_reject_reset_adam_or_scheduler(tmp_path):
    cfg, clean, _ = setup(tmp_path)
    train_prefix(cfg, clean)
    saved = read_resume(cfg, "warmup")
    saved["optimizer"]["state"][0]["step"] = 0
    with pytest.raises(RuntimeError, match="Adam"):
        validate_state(saved, cfg)
    saved = read_resume(cfg, "warmup")
    saved["scheduler"]["last_epoch"] = 0
    with pytest.raises(RuntimeError, match="scheduler"):
        validate_state(saved, cfg)


def test_sampling_extends_to_1000_and_matches_task_phase_and_clean_order(tmp_path):
    cfg, clean, replay = setup(tmp_path)
    cfg.update(bootstrap_outer=250, outer_iterations=1000)
    differences = 0
    for outer in range(251, 1001):
        u, origin_u, _, base_u = draw_record(clean, replay, "uada", outer, cfg)
        a, origin_a, index_a, base_a = draw_record(clean, replay, "uniform", outer, cfg)
        b, origin_b, index_b, base_b = draw_record(clean, replay, "dino", outer, cfg)
        assert base_u == base_a == base_b and origin_u == "clean" and origin_a == origin_b
        assert u["task_id"] == a["task_id"] == b["task_id"]
        if origin_a == "replay":
            assert a["phase_proxy"] == b["phase_proxy"]
            differences += index_a != index_b
        else:
            assert a is u and b is u
    assert differences > 0


def test_evaluation_conditions_compare_equal_budgets():
    from run_exploring_oft_dino_scratch_cross_suite import CONDITIONS
    from analyze_exploring_oft_dino_scratch_cross_suite import CONTRASTS
    assert len(CONDITIONS) == 9
    assert set(CONDITIONS) == {f"{method}_{step}" for method in METHODS for step in (500, 750, 1000)}
    for role, left, right in CONTRASTS:
        assert left in CONDITIONS and right in CONDITIONS
        if "1000_minus_500" not in role:
            assert left.rsplit("_", 1)[1] == right.rsplit("_", 1)[1]

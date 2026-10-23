from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
sys.path.insert(0, str(ROOT))
from white_patch.UADA_OFT_ALTERNATING import METHODS, ExploringOFTAlternatingAttacker, source_manifest
from white_patch.UADA_OFT_ILA import ExploringOFTILAAttacker
from white_patch.oft_alternating_objectives import alternating_projection
from white_patch.oft_ila_objectives import dense_objectives
from VLAAttacker.UADA_OFT_ALTERNATING_wrapper import CONFIG, validate_config
from test_exploring_oft_ila import MockAttacker, mock_setup


def setup(tmp_path):
    cfg, records = mock_setup(tmp_path)
    cfg.update(experiment_id="MOCK-ALTERNATING", methods=list(METHODS),
               inner_loop=4, action_steps_per_outer=2, feature_steps_per_outer=2)
    return cfg, records


class MockAlternating(ExploringOFTAlternatingAttacker, MockAttacker):
    def __init__(self, fail_at=None):
        MockAttacker.__init__(self, fail_at)
        self.events = []
        self.ready = False
        self.in_full = False

    def _validate_runtime(self, record, reference, cfg):
        value = MockAttacker._validate_runtime(self, record, reference, cfg)
        self.ready = True
        return value

    def _features(self, observation, patch, geometry):
        if self.ready and patch is not None:
            marker = (hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                      hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            self.events.append({"patch": patch.detach().clone(), "grad": patch.requires_grad,
                                "grad_enabled": torch.is_grad_enabled(), "full": self.in_full, "rng": marker})
        return MockAttacker._features(self, observation, patch, geometry)

    def _forward(self, inputs, proprio, observation, patch, geometry):
        self.in_full = True
        try:
            return MockAttacker._forward(self, inputs, proprio, observation, patch, geometry)
        finally:
            self.in_full = False


def example():
    torch.manual_seed(23)
    anchor = torch.randn(1, 16, 8)
    endpoint = anchor + torch.randn_like(anchor) / 5
    return anchor, endpoint, torch.zeros(16)


def test_local_projection_teacher_one_gradient_direction_and_detach():
    anchor, endpoint, support = example()
    anchor.requires_grad_()
    endpoint.requires_grad_()
    current = endpoint.detach().clone().requires_grad_()
    loss, stats = alternating_projection(current, anchor, endpoint, support, support, support)
    assert stats["projection_score"] == pytest.approx(1.)
    loss.backward()
    assert anchor.grad is None and endpoint.grad is None
    assert current.grad is not None and current.grad.norm() > 0
    better, _ = alternating_projection(current.detach() - 0.01 * current.grad, anchor, endpoint,
                                        support, support, support)
    assert better < loss.detach()
    scaled, _ = alternating_projection(3 * current.detach(), anchor, endpoint, support, support, support)
    torch.testing.assert_close(loss.detach(), scaled)
    wrong_origin, _ = alternating_projection(current.detach(), torch.zeros_like(anchor), endpoint,
                                             support, support, support)
    assert not torch.isclose(better, wrong_origin)


def test_three_paste_supports_are_excluded_from_local_gradient():
    anchor, endpoint, _ = example()
    current = endpoint.detach().clone().requires_grad_()
    covers = [torch.zeros(16) for _ in range(3)]
    for idx, support in enumerate(covers):
        support[idx] = 1.
    loss, stats = alternating_projection(current, anchor, endpoint, *covers)
    assert stats["visible_tokens"] == 13
    loss.backward()
    assert current.grad[:, :3].eq(0).all()
    assert current.grad[:, 3:].abs().sum() > 0
    with pytest.raises(ValueError, match="Invalid anchor coverage"):
        alternating_projection(current, anchor, endpoint, covers[0], torch.full((16,), -1.), covers[2])


def test_fixed_projection_matches_audited_ila_and_weak_direction_skips():
    anchor, endpoint, support = example()
    new, _ = alternating_projection(endpoint, anchor, endpoint, support, None, support)
    old, _, _ = dense_objectives(endpoint, anchor, endpoint, support, support)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    skip, stats = alternating_projection(anchor, anchor, anchor, support, support, support)
    assert skip is None and stats["feature_active"] == 0


@pytest.mark.parametrize("method", METHODS)
def test_phase_counts_eot_pairing_frozen_snapshots_and_shared_contract(tmp_path, method):
    cfg, records = setup(tmp_path)
    attacker = MockAlternating()
    attacker.train(records, cfg, method, {}, {})
    folder = Path(cfg["output_root"]) / method
    rows = pd.read_parquet(folder / "training.parquet")
    status = json.loads((folder / "status.json").read_text())
    assert status["optimizer_updates"] == 12
    assert status["action_optimizer_updates"] == status["feature_optimizer_updates"] == 6
    assert status["total_optimizer_updates"] == (500 + 3) * 4
    assert rows.action_steps.eq(2).all() and rows.feature_steps.eq(2).all()
    assert rows.reference_refreshes.eq(int(method == METHODS[1])).all()
    assert rows.feature_grad_active.between(0, 1).all() and rows.feature_grad_active.sum() > 0
    assert attacker.full_calls == 2 + 3 * (1 + 2)
    events = [event for event in attacker.events if not event["full"]]
    stride = 2 if method == METHODS[0] else 3
    assert len(events) == 3 * 2 * stride
    for index in range(0, len(events), stride):
        group = events[index:index + stride]
        assert len({event["rng"] for event in group}) == 1
        assert all(not event["grad"] and not event["grad_enabled"] for event in group[:-1])
        assert group[-1]["grad"] and group[-1]["grad_enabled"]
    baseline = torch.load(cfg["baseline_patch"], weights_only=False)["patch"]
    if method == METHODS[0]:
        assert all(torch.equal(event["patch"], baseline) for event in events[::stride])
    else:
        for outer in range(3):
            group = events[outer * 2 * stride:(outer + 1) * 2 * stride]
            assert torch.equal(group[0]["patch"], group[3]["patch"])
            assert torch.equal(group[1]["patch"], group[4]["patch"])
            assert torch.equal(group[1]["patch"], group[2]["patch"])
            assert not torch.equal(group[0]["patch"], group[1]["patch"])
        assert not torch.equal(events[1]["patch"], events[7]["patch"])
    assert ExploringOFTAlternatingAttacker._wrist_pixels is ExploringOFTILAAttacker._wrist_pixels
    assert ExploringOFTAlternatingAttacker._validate_runtime is ExploringOFTILAAttacker._validate_runtime


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fail_at", [5, 7])
def test_action_or_feature_interruption_restores_exact_outer_and_optimizer(tmp_path, method, fail_at):
    cfg, records = setup(tmp_path)
    MockAlternating().train(records, cfg, method, {}, {}, model_load_s=2.)
    full_folder = Path(cfg["output_root"]) / method
    expected = torch.load(full_folder / "outer_0003.pt", weights_only=False)["patch"]
    expected_rows = pd.read_parquet(full_folder / "training.parquet")
    cfg["output_root"] = str(tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="mock interruption"):
        MockAlternating(fail_at).train(records, cfg, method, {}, {}, model_load_s=2.)
    folder = Path(cfg["output_root"]) / method
    resume = torch.load(folder / "resume.pt", weights_only=False)
    assert resume["outer"] == 1 and len(resume["rows"]) == 1
    failed = json.loads((folder / "status.json").read_text())
    assert failed["optimizer_updates"] == 4
    MockAlternating().train(records, cfg, method, {}, {}, model_load_s=3.)
    actual = torch.load(folder / "outer_0003.pt", weights_only=False)["patch"]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    pd.testing.assert_frame_equal(expected_rows.drop(columns="elapsed_s"),
                                 pd.read_parquet(folder / "training.parquet").drop(columns="elapsed_s"))
    status = json.loads((folder / "status.json").read_text())
    assert status["costs"]["model_load_s"] == 5. and status["elapsed_s"] >= failed["elapsed_s"]
    with pytest.raises(RuntimeError, match="completed training"):
        MockAlternating().train(records, cfg, method, {}, {})


def test_config_identity_phase_budget_and_source_drift_are_rejected(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    validate_config(cfg)
    for key, value in [("action_steps_per_outer", 24), ("feature_steps_per_outer", 26),
                       ("seed", 43), ("maskidx", [1]), ("tv_weight", 2.)]:
        changed = copy.deepcopy(cfg)
        changed[key] = value
        with pytest.raises(RuntimeError, match="drift"):
            validate_config(changed)
    cfg, records = setup(tmp_path)
    with pytest.raises(RuntimeError, match="mock interruption"):
        MockAlternating(1).train(records, cfg, METHODS[1], {"source": "first"}, {})
    with pytest.raises(RuntimeError, match="signature changed"):
        MockAlternating().train(records, cfg, METHODS[1], {"source": "changed"}, {})


def test_manual_submission_gate_array_budget_and_source_snapshot():
    cfg = json.loads(CONFIG.read_text())
    sources = source_manifest(cfg)
    trainer = ROOT / "VLAAttacker/white_patch/UADA_OFT_ALTERNATING.py"
    assert str(trainer.resolve()) in sources
    submit = (ROOT / "submit_exploring_oft_dino_alternating_refine200.sh").read_text()
    assert submit.index("CHECK_ONLY") < submit.index("sbatch --parsable")
    assert "afterok" not in submit
    job = (ROOT / "slurm/exploring_oft_dino/57_train_dino_alternating_refine200_3090.sbatch").read_text()
    assert "#SBATCH --array=0-1%2" in job and "#SBATCH --gres=gpu:3090:1" in job
    assert "METHODS=(alternating_fixed_tv alternating_rolling_tv)" in job

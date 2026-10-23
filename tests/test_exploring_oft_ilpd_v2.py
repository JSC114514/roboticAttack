"""Engineering regression tests; no real VLA or CUDA workload on the login node."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
sys.path.insert(0, str(ROOT))

from test_exploring_oft_ilpd import MockILPD, mock_setup
from white_patch.UADA_OFT_ILPD import ExploringOFTILPDAttacker
from white_patch.UADA_OFT_ILPD_V2 import (
    ExploringOFTILPDV2Attacker, calibrated_allowance, gradient_diagnostics,
    require_gradient_stability, source_manifest,
)
from white_patch.oft_ilpd_features import METHODS
from VLAAttacker.UADA_OFT_ILPD_V2_wrapper import CONFIG, validate_config


class MockV2(ExploringOFTILPDV2Attacker, MockILPD):
    def __init__(self, fail_at=None, jitters=None):
        MockILPD.__init__(self, fail_at)
        self.jitters = jitters
        self.validation_backwards = 0

    def _validate_ilpd_runtime(self, *args):
        self.validating = True
        self.validation_backwards = 0
        try:
            return super()._validate_ilpd_runtime(*args)
        finally:
            self.validating = False

    def _forward(self, inputs, proprio, observation, patch, geometry, *, capture_siglip=False):
        if self.validating and self.jitters is not None and patch is not None and patch.requires_grad:
            owner = self
            class SimulatedCUDABackward(torch.autograd.Function):
                @staticmethod
                def forward(ctx, value):
                    return value.clone()

                @staticmethod
                def backward(ctx, gradient):
                    index = owner.validation_backwards
                    owner.validation_backwards += 1
                    # Amplify the toy's otherwise tiny gradients so the legacy
                    # absolute tolerance also fails on small relative jitter.
                    return gradient * (1000. * (1. + owner.jitters[index]))
            patch = SimulatedCUDABackward.apply(patch)
        return MockILPD._forward(self, inputs, proprio, observation, patch, geometry,
                                 capture_siglip=capture_siglip)


def setup(tmp_path):
    cfg, records = mock_setup(tmp_path)
    cfg["experiment_id"] = "OFT-DINO-ILPD-REFINE200-V2"
    return cfg, records


def test_relative_diagnostics_preserve_small_gradients_and_reject_bad_values():
    x = torch.linspace(.1, 1., 7500).reshape(3, 50, 50)
    metrics = gradient_diagnostics(x * 1.0002, x)
    assert not metrics["legacy_elementwise_close"]
    assert metrics["relative_l2"] == pytest.approx(.0002 / 1.0002, rel=1e-3)
    assert metrics["cosine"] == pytest.approx(1.)
    limit = calibrated_allowance([metrics])
    require_gradient_stability(metrics, limit)
    for bad in (x * .1, -x, x * 1.2):
        with pytest.raises(RuntimeError, match="hard engineering"):
            require_gradient_stability(gradient_diagnostics(bad, x), limit)
    with pytest.raises(RuntimeError, match="measured ordinary"):
        require_gradient_stability(gradient_diagnostics(x * 1.02, x), limit)
    for bad in (x * float("nan"), torch.zeros_like(x)):
        with pytest.raises(RuntimeError):
            gradient_diagnostics(bad, x)
    with pytest.raises(RuntimeError, match="shape"):
        gradient_diagnostics(x.flatten(), x)
    with pytest.raises(RuntimeError, match="missing"):
        calibrated_allowance([])


@pytest.mark.parametrize("method", METHODS)
def test_small_backward_jitter_passes_but_forward_is_still_exact(tmp_path, method):
    cfg, records = setup(tmp_path)
    a = MockV2(jitters=[0., .0002, -.0002, .0001, 0.])
    a.train(records, cfg, method, {}, {})
    folder = Path(cfg["output_root"]) / method
    runtime = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
    assert runtime["status"] == "PASS" and runtime["engineering_revision"] == 2
    assert runtime["full_forward_calls"] == 6
    audit = runtime["ilpd"]
    assert len(audit["normal_repeat_comparisons"]) == 3
    assert any(not m["legacy_elementwise_close"] for m in audit["normal_repeat_comparisons"])
    assert len(audit["checks"]) == 2 and audit["phase"] == "complete"
    assert all(c["formula_exact"] and c["front_and_other_encoder_unchanged"] for c in audit["checks"])
    assert audit["checks"][0]["identity_tensor_graph_unchanged"]
    if method != METHODS[0]:
        assert audit["checks"][1]["local_gradient"]["passed"]
    assert a.validation_backwards == 5 and a.train_calls == 12
    assert not a.wrapper.vla.vision_backbone.featurizer._forward_hooks


@pytest.mark.parametrize("bad, message", [(.3, "hard engineering"), (.02, "measured ordinary")])
def test_identity_corruption_is_rejected_with_metrics_saved_before_failure(tmp_path, bad, message):
    cfg, records = setup(tmp_path)
    method = METHODS[2]
    a = MockV2(jitters=[0., .0002, -.0002, bad, 0.])
    with pytest.raises(RuntimeError, match=message):
        a.train(records, cfg, method, {}, {})
    folder = Path(cfg["output_root"]) / method
    runtime = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
    assert runtime["status"] == "FAILED"
    comparisons = runtime["ilpd"]["checks"][0]["identity_gradient_comparisons"]
    assert len(comparisons) == 3 and all(c["relative_l2"] > .01 for c in comparisons)
    state = json.loads((folder / "status.json").read_text())
    assert state["outer"] == 0 and state["optimizer_updates"] == 0
    assert not a.wrapper.vla.vision_backbone.featurizer._forward_hooks


def test_unstable_ordinary_gradient_does_not_widen_tolerance(tmp_path):
    cfg, records = setup(tmp_path)
    a = MockV2(jitters=[0., .2, 0., 0., 0.])
    with pytest.raises(RuntimeError, match="hard engineering"):
        a.train(records, cfg, METHODS[0], {}, {})
    runtime = json.loads((Path(cfg["output_root"]) / METHODS[0] / "RUNTIME_ALIGNMENT.json").read_text())
    assert runtime["ilpd"]["phase"] == "ordinary_gradient_calibration"
    assert runtime["ilpd"]["normal_repeat_comparisons"][0]["relative_l2"] > .05
    assert "identity_relative_allowance" not in runtime["ilpd"]


def test_wrong_identity_graph_and_wrong_local_derivative_are_not_hidden(tmp_path, monkeypatch):
    import white_patch.oft_ilpd_features as features
    original = features.transform_dense
    cfg, records = setup(tmp_path)
    def wrong_identity(value, anchor, alpha, backward_only=False):
        if alpha == 1:
            return value.detach() + .25 * (value - value.detach())
        return original(value, anchor, alpha, backward_only)
    monkeypatch.setattr(features, "transform_dense", wrong_identity)
    with pytest.raises(RuntimeError, match="Tensor/graph"):
        MockV2().train(records, cfg, METHODS[2], {}, {})
    monkeypatch.setattr(features, "transform_dense", original)
    cfg["output_root"] = str(tmp_path / "wrong_local")
    def wrong_local(value, anchor, alpha, backward_only=False):
        mixed = original(value, anchor, alpha, backward_only)
        if alpha != 1:
            return mixed.detach() + .25 * (mixed - mixed.detach())
        return mixed
    monkeypatch.setattr(features, "transform_dense", wrong_local)
    with pytest.raises(RuntimeError, match="local gradient"):
        MockV2().train(records, cfg, METHODS[2], {}, {})
    runtime = json.loads((Path(cfg["output_root"]) / METHODS[2] / "RUNTIME_ALIGNMENT.json").read_text())
    assert not runtime["ilpd"]["checks"][-1]["local_gradient"]["passed"]


def test_wrong_forward_action_still_fails(tmp_path, monkeypatch):
    cfg, records = setup(tmp_path)
    a = MockV2()
    original = a._modified_forward
    def corrupt(*args):
        (action, dino, siglip), capture = original(*args)
        if args[-3] == METHODS[2]:
            action = action + .01
        return (action, dino, siglip), capture
    monkeypatch.setattr(a, "_modified_forward", corrupt)
    with pytest.raises(RuntimeError, match="changed actions"):
        a.train(records, cfg, METHODS[2], {}, {})
    assert not a.wrapper.vla.vision_backbone.featurizer._forward_hooks


@pytest.mark.parametrize("method", METHODS)
def test_v2_training_loop_is_unchanged_and_resume_bitwise(tmp_path, method):
    cfg, records = setup(tmp_path)
    a = MockV2()
    a.train(records, cfg, method, {}, {}, model_load_s=2.)
    full_dir = Path(cfg["output_root"]) / method
    full = torch.load(full_dir / "resume.pt", weights_only=False)
    full_rows = pd.read_parquet(full_dir / "training.parquet")
    assert a.full_calls == 21 and a.train_calls == 12
    cfg["output_root"] = str(tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="mock interruption"):
        MockV2(fail_at=6).train(records, cfg, method, {}, {}, model_load_s=2.)
    MockV2().train(records, cfg, method, {}, {}, model_load_s=3.)
    resumed_dir = Path(cfg["output_root"]) / method
    resumed = torch.load(resumed_dir / "resume.pt", weights_only=False)
    assert torch.equal(full["patch"], resumed["patch"])
    assert torch.equal(full["torch_rng"], resumed["torch_rng"])
    assert full["scheduler"] == resumed["scheduler"]
    for parameter, state in full["optimizer"]["state"].items():
        for key, value in state.items():
            other = resumed["optimizer"]["state"][parameter][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    pd.testing.assert_frame_equal(full_rows.drop(columns="elapsed_s"),
                                  pd.read_parquet(resumed_dir / "training.parquet").drop(columns="elapsed_s"))
    with pytest.raises(RuntimeError, match="completed training"):
        MockV2().train(records, cfg, method, {}, {})
    assert ExploringOFTILPDV2Attacker.train is ExploringOFTILPDAttacker.train
    assert ExploringOFTILPDV2Attacker._modified_forward is ExploringOFTILPDAttacker._modified_forward


def test_frozen_scientific_contract_new_directory_and_manual_helper():
    cfg = json.loads(CONFIG.read_text())
    validate_config(cfg)
    old = json.loads((ROOT / "configs/exploring_oft_dino/object_dino_ilpd_refine200_v1.json").read_text())
    assert {k: v for k, v in cfg.items() if k not in ("experiment_id", "output_root")} == {
        k: v for k, v in old.items() if k not in ("experiment_id", "output_root")}
    assert Path(cfg["output_root"]).name == "training_v2"
    for key, value in (("feature_alpha", .4), ("outer_iterations", 300), ("maskidx", [1]),
                       ("rgb_radius", .1), ("output_root", old["output_root"]), ("experiment_id", old["experiment_id"])):
        with pytest.raises(RuntimeError):
            validate_config(dict(cfg, **{key: value}))
    sources = source_manifest(cfg)
    assert str((ROOT / "VLAAttacker/white_patch/UADA_OFT_ILPD_V2.py").resolve()) in sources
    assert str((ROOT / "VLAAttacker/white_patch/UADA_OFT_ILPD.py").resolve()) in sources
    helper = (ROOT / "submit_exploring_oft_dino_ilpd_refine200.sh").read_text()
    assert helper.index("CHECK_ONLY") < helper.index("sbatch --parsable")
    assert "61_train_dino_ilpd_refine200_v2_3090.sbatch" in helper
    assert "UADA_OFT_ILPD_V2_wrapper.py" in helper
    assert helper.count("sbatch --parsable") == 1
    slurm = (ROOT / "slurm/exploring_oft_dino/61_train_dino_ilpd_refine200_v2_3090.sbatch").read_text()
    assert "#SBATCH --array=0-3%4" in slurm and "#SBATCH --gres=gpu:3090:1" in slurm
    assert "ilpd_v2_train_%A_%a.out" in slurm and "sbatch " not in slurm

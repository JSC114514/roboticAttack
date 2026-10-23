from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
sys.path.insert(0, str(ROOT))

from white_patch.UADA_OFT_ILPD import ExploringOFTILPDAttacker, source_manifest
from white_patch.UADA_OFT_ILA import ExploringOFTILAAttacker
from white_patch.UADA_OFT_DINO import sha256_file
from white_patch.oft_ilpd_features import METHODS, box_metrics, project_rgb_, transform_dense, wrist_feature_transform
from VLAAttacker.UADA_OFT_ILPD_wrapper import CONFIG, validate_config


def test_forward_formula_anchor_detached_native_dtype_and_identity():
    x = torch.randn(1, 256, 4, requires_grad=True)
    z0 = torch.randn_like(x, requires_grad=True)
    mixed = transform_dense(x, z0, .5)
    torch.testing.assert_close(mixed, z0.detach() + .5 * (x - z0.detach()))
    gx, g0 = torch.autograd.grad(mixed.sum(), (x, z0), allow_unused=True)
    assert torch.equal(gx, torch.full_like(x, .5)) and g0 is None
    assert transform_dense(x, z0, 1.) is x
    assert torch.equal(transform_dense(x, z0, 0.), z0.detach())
    for dtype in (torch.float32, torch.bfloat16):
        assert transform_dense(x.to(dtype), z0, .5).dtype == dtype
    for wrong in (None, z0[:, :3], z0 * float("nan")):
        with pytest.raises(RuntimeError):
            transform_dense(x, wrong, .5)
    with pytest.raises(RuntimeError):
        transform_dense(x, z0, 2.)


def test_backward_only_is_not_forward_decay_for_nonlinear_action_loss():
    x = torch.full((1, 256, 1), .8, requires_grad=True)
    clean = torch.full_like(x, .1)
    backward = transform_dense(x, None, .5, True)
    assert torch.equal(backward, x)
    normal_loss = x.square().mean().square()
    grad_normal = torch.autograd.grad(normal_loss, x)[0]
    gb = torch.autograd.grad(backward.square().mean().square(), x)[0]
    torch.testing.assert_close(gb, grad_normal * .5)
    mixed = transform_dense(x, clean, .5)
    gm = torch.autograd.grad(mixed.square().mean().square(), x)[0]
    assert not torch.allclose(gm, gb)


def backbone():
    m = torch.nn.Module()
    m.featurizer = torch.nn.Identity()
    m.fused_featurizer = torch.nn.Identity()
    return m


@pytest.mark.parametrize("method", METHODS)
def test_temporary_hooks_only_modify_wrist_of_correct_encoder_and_cleanup(method):
    m = backbone()
    front, wrist = torch.ones(1, 256, 4), torch.full((1, 256, 4), 3., requires_grad=True)
    anchors = {b: torch.zeros_like(wrist) for b in ("dino", "siglip")}
    with wrist_feature_transform(m, method, anchors, .5) as capture:
        df, sf = m.featurizer(front), m.fused_featurizer(front)
        dw, sw = m.featurizer(wrist), m.fused_featurizer(wrist)
    assert df is front and sf is front
    assert torch.equal(dw, wrist * .5 if method == METHODS[2] else wrist)
    assert torch.equal(sw, wrist * .5 if method == METHODS[3] else wrist)
    assert len(capture["dino"]) == len(capture["siglip"]) == 2
    assert not m.featurizer._forward_hooks and not m.fused_featurizer._forward_hooks
    with pytest.raises(ValueError, match="model failure"):
        with wrist_feature_transform(m, method, anchors, .5):
            m.featurizer(front)
            raise ValueError("model failure")
    assert not m.featurizer._forward_hooks and not m.fused_featurizer._forward_hooks
    with pytest.raises(RuntimeError, match="exactly two"):
        with wrist_feature_transform(m, method, anchors, .5):
            m.featurizer(front)
    assert not m.featurizer._forward_hooks and not m.fused_featurizer._forward_hooks
    with pytest.raises(RuntimeError, match="call count"):
        with wrist_feature_transform(m, method, anchors, .5):
            for _ in range(3):
                m.featurizer(front)
    assert not m.featurizer._forward_hooks


def test_full_rgb_projection_box_no_reparameterization():
    torch.manual_seed(7)
    initial = torch.rand(3, 50, 50)
    patch = (initial + torch.randn_like(initial)).requires_grad_(True)
    identity = id(patch)
    project_rgb_(patch, initial, 16 / 255)
    assert id(patch) == identity and patch.is_leaf and patch.requires_grad and patch.numel() == 7500
    metrics = box_metrics(patch, initial, 16 / 255)
    assert metrics["rgb_max_abs_delta"] <= 16 / 255 + 1e-7
    assert 0 < metrics["box_boundary_fraction"] <= 1
    assert bool((patch >= 0).all() and (patch <= 1).all())
    with pytest.raises(RuntimeError, match="appearance box"):
        box_metrics(initial + .1, initial, 16 / 255)


class MockILPD(ExploringOFTILPDAttacker):
    """Tiny differentiable two-view/two-encoder module, not a real CPU VLA."""
    def __init__(self, fail_at=None):
        vla = torch.nn.Module()
        vla.vision_backbone = backbone()
        self.wrapper = SimpleNamespace(device="cpu", vla=vla)
        self.full_calls, self.train_calls = 0, 0
        self.fail_at = fail_at
        self.events = []
        self.validating = False

    def _base_inputs(self, record):
        return {}, None

    def _validate_runtime(self, record, reference, cfg):
        self.runtime_alignment = {"status": "PASS", "mock": True, "full_forward_calls": 0}
        return self.runtime_alignment

    def _validate_ilpd_runtime(self, *args):
        self.validating = True
        try:
            return super()._validate_ilpd_runtime(*args)
        finally:
            self.validating = False

    def _forward(self, inputs, proprio, observation, patch, geometry, *, capture_siglip=False):
        self.full_calls += 1
        clean = torch.sin(torch.arange(1024).reshape(1, 256, 4).float() / 20) * .1 + .3
        wrist = clean
        if patch is not None:
            scale = .9 + random.random() / 10 + float(torch.rand(())) / 10 + float(np.random.rand()) / 10
            wrist = clean + scale * (.1 * patch.mean() + .2 * patch.flatten()[:1024].reshape(1, 256, 4))
            if patch.requires_grad and torch.is_grad_enabled() and not self.validating:
                self.train_calls += 1
                self.events.append(scale)
                if self.train_calls == self.fail_at:
                    raise RuntimeError("mock interruption")
        m = self.wrapper.vla.vision_backbone
        df, sf = m.featurizer(clean + .2), m.fused_featurizer(clean + .4)
        dw, sw = m.featurizer(wrist), m.fused_featurizer(wrist + .3)
        action = torch.tanh(.2 * dw.square().mean() + .3 * sw.square().mean() + .04 * df.mean() + .01 * sf.mean()).expand(1, 8, 7)
        return action, dw.float(), sw if capture_siglip else None


def mock_setup(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    torch.manual_seed(3)
    path = tmp_path / "accepted.pt"
    torch.save({"patch": torch.rand(3, 50, 50)}, path)
    cfg.update(baseline_patch=str(path), baseline_patch_sha256=sha256_file(path),
               output_root=str(tmp_path / "full"), outer_iterations=3, inner_loop=4,
               warmup_outer_iterations=0, resume_every_outer_iterations=1,
               checkpoint_outer_iterations=[0, 1, 3])
    records = [{"task_id": i, "observation": {}} for i in range(3)]
    return cfg, records


@pytest.mark.parametrize("method", METHODS)
def test_all_arms_action_updates_bound_runtime_and_bitwise_resume(tmp_path, method):
    cfg, records = mock_setup(tmp_path)
    full = MockILPD()
    full.train(records, cfg, method, {}, {}, model_load_s=2.)
    folder = Path(cfg["output_root"]) / method
    original = torch.load(cfg["baseline_patch"], weights_only=False)["patch"]
    start = torch.load(folder / "outer_0000.pt", weights_only=False)["patch"]
    final = torch.load(folder / "outer_0003.pt", weights_only=False)["patch"]
    assert torch.equal(start, original) and not torch.equal(final, start)
    assert full.train_calls == 12 and full.full_calls == 19
    box_metrics(final, original, cfg["rgb_radius"])
    rows = pd.read_parquet(folder / "training.parquet")
    assert (rows.action_steps == 4).all() and (rows.feature_steps == 0).all()
    assert (rows.rgb_max_abs_delta <= cfg["rgb_radius"] + 1e-7).all()
    runtime = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
    assert runtime["status"] == "PASS" and runtime["full_forward_calls"] == 4
    assert not full.wrapper.vla.vision_backbone.featurizer._forward_hooks
    cfg["output_root"] = str(tmp_path / "resumed")
    interrupted = MockILPD(fail_at=6)
    with pytest.raises(RuntimeError, match="mock interruption"):
        interrupted.train(records, cfg, method, {}, {}, model_load_s=2.)
    resumed_dir = Path(cfg["output_root"]) / method
    failed = json.loads((resumed_dir / "status.json").read_text())
    assert failed["outer"] == 1 and failed["optimizer_updates"] == 4
    assert not interrupted.wrapper.vla.vision_backbone.featurizer._forward_hooks
    resumed_attacker = MockILPD()
    resumed_attacker.train(records, cfg, method, {}, {}, model_load_s=3.)
    resumed = torch.load(resumed_dir / "outer_0003.pt", weights_only=False)["patch"]
    assert torch.equal(final, resumed)
    resumed_rows = pd.read_parquet(resumed_dir / "training.parquet")
    pd.testing.assert_frame_equal(rows.drop(columns="elapsed_s"), resumed_rows.drop(columns="elapsed_s"))
    status = json.loads((resumed_dir / "status.json").read_text())
    assert status["optimizer_updates"] == status["action_optimizer_updates"] == 12
    assert status["feature_optimizer_updates"] == 0 and status["total_optimizer_updates"] == 2012
    assert status["costs"]["model_load_s"] == 5. and status["elapsed_s"] >= failed["elapsed_s"]
    assert status["costs"]["training_full_forward_attempts"] > 15
    assert sha256_file(cfg["baseline_patch"]) == cfg["baseline_patch_sha256"]
    with pytest.raises(RuntimeError, match="completed training"):
        MockILPD().train(records, cfg, method, {}, {})


def test_four_arms_use_identical_state_and_eot_schedules(tmp_path):
    cfg, records = mock_setup(tmp_path)
    events, indices = [], []
    for method in METHODS:
        m = MockILPD()
        m.train(records, cfg, method, {}, {})
        events.append(m.events)
        rows = pd.read_parquet(Path(cfg["output_root"]) / method / "training.parquet")
        indices.append(rows.record_index.tolist())
    assert all(e == events[0] for e in events) and all(i == indices[0] for i in indices)


def test_signature_early_failure_resume_box_and_original_inputs(tmp_path):
    cfg, records = mock_setup(tmp_path)
    method = METHODS[2]
    with pytest.raises(RuntimeError, match="mock interruption"):
        MockILPD(fail_at=1).train(records, cfg, method, {}, {})
    folder = Path(cfg["output_root"]) / method
    resume = torch.load(folder / "resume.pt", weights_only=False)
    assert resume["outer"] == 0 and not resume["rows"]
    with pytest.raises(RuntimeError, match="signature changed"):
        MockILPD().train(records, dict(cfg, feature_alpha=.3), method, {}, {})
    with pytest.raises(RuntimeError, match="signature changed"):
        MockILPD().train(records, cfg, method, {str(Path(__file__)): "changed"}, {})
    resume["patch"] = (resume["initial"] + .1).clamp(0, 1)
    torch.save(resume, folder / "resume.pt")
    with pytest.raises(RuntimeError, match="appearance box"):
        MockILPD().train(records, cfg, method, {}, {})
    for name in ("_patched_pixels", "_paste_with_support", "_forward", "_wrist_pixels", "_validate_runtime"):
        assert getattr(ExploringOFTILPDAttacker, name) is getattr(ExploringOFTILAAttacker, name)


def test_real_config_frozen_source_and_manual_submission_contract():
    cfg = json.loads(CONFIG.read_text())
    validate_config(cfg)
    assert cfg["rgb_radius"] == 16 / 255 and cfg["feature_alpha"] == .5
    assert (cfg["base_outer_iterations"] + cfg["outer_iterations"]) * cfg["inner_loop"] == 35000
    for key, value in (("outer_iterations", 500), ("rgb_radius", .5), ("feature_alpha", .3),
                       ("baseline_patch_sha256", "wrong"), ("resize_patch", True), ("output_root", "/tmp/overlap")):
        with pytest.raises(RuntimeError):
            validate_config(dict(cfg, **{key: value}))
    script = (ROOT / "slurm/exploring_oft_dino/60_train_dino_ilpd_refine200_3090.sbatch").read_text()
    assert "#SBATCH --array=0-3%4" in script and "#SBATCH --gres=gpu:3090:1" in script
    assert "research_runs/wrist_dino_ilpd_v1/logs/" in script and "sbatch " not in script
    assert "METHODS=(" + " ".join(METHODS) + ")" in script
    helper = (ROOT / "submit_exploring_oft_dino_ilpd_refine200.sh").read_text()
    assert helper.index("CHECK_ONLY") < helper.index("sbatch --parsable")
    assert helper.index("mkdir -p") < helper.index("sbatch --parsable")
    assert helper.count("sbatch --parsable") == 1 and "cross_suite.sh" not in helper
    sources = source_manifest(cfg)
    assert str((ROOT / "VLAAttacker/white_patch/UADA_OFT_ILPD.py").resolve()) in sources
    assert str((ROOT / "VLAAttacker/white_patch/oft_ilpd_features.py").resolve()) in sources

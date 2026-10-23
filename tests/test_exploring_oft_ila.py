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

from white_patch.UADA_OFT_ILA import METHODS, ExploringOFTILAAttacker, seed_transform
from white_patch.UADA_OFT_DINO import sha256_file
from white_patch.oft_ila_objectives import dense_objectives
from VLAAttacker.UADA_OFT_ILA_wrapper import CONFIG, validate_config


def feature_example():
    torch.manual_seed(2)
    clean = torch.randn(1, 16, 8)
    reference = clean + 0.2 * torch.randn_like(clean)
    return clean, reference, torch.zeros(16)


def test_projection_uses_realized_direction_and_cannot_win_by_norm_scaling():
    clean, reference, support = feature_example()
    same, _, stats = dense_objectives(reference, clean, reference, support, support)
    torch.testing.assert_close(same, torch.tensor(-1.))
    assert stats["projection_score"] == pytest.approx(1.)
    opposite = 2 * clean - reference
    bad, _, _ = dense_objectives(opposite, clean, reference, support, support)
    assert bad > same
    scaled, _, _ = dense_objectives(reference * 10, clean * 3, reference * 7, support, support)
    torch.testing.assert_close(same, scaled)


def test_teacher_detached_union_excluded_and_both_losses_have_correct_gradients():
    clean, reference, coverage = feature_example()
    clean.requires_grad_(True)
    reference.requires_grad_(True)
    current = reference.detach().clone().requires_grad_(True)
    teacher_coverage = coverage.clone()
    coverage[0], teacher_coverage[1] = 1., 1.
    directional, disruptive, stats = dense_objectives(current, clean, reference, coverage, teacher_coverage)
    assert stats["visible_tokens"] == 14 and stats["support_disagreement_tokens"] == 2
    for loss in (directional, disruptive):
        g, c, t = torch.autograd.grad(loss, (current, clean, reference), retain_graph=True, allow_unused=True)
        assert g[:, :2].abs().sum() == 0 and g[:, 2:].norm() > 0
        assert c is None and t is None
    g = torch.autograd.grad(directional, current)[0]
    better, _, _ = dense_objectives(current.detach() - 0.01 * g, clean, reference, coverage, teacher_coverage)
    assert better < directional.detach()


def test_invalid_or_tiny_direction_does_not_create_fake_guidance():
    clean, reference, coverage = feature_example()
    assert dense_objectives(reference, clean, reference, torch.ones(16), coverage)[:2] == (None, None)
    assert dense_objectives(reference, clean, clean, coverage, coverage)[:2] == (None, None)
    with pytest.raises(ValueError, match="shape"):
        dense_objectives(reference, clean, reference, coverage[:3], coverage)
    with pytest.raises(ValueError, match="Nonfinite"):
        dense_objectives(reference * float("nan"), clean, reference, coverage, coverage)


def test_real_rgb_eot_is_unchanged_and_teacher_student_receive_same_transform():
    wrapper = SimpleNamespace(device="cpu", vla=torch.nn.Linear(1, 1))
    attacker = ExploringOFTILAAttacker(wrapper, "/unused")
    rgb = np.full((224, 224, 3), 170, dtype=np.uint8)
    p = torch.linspace(0.1, 0.9, 7500).reshape(3, 50, 50).requires_grad_()
    for geometry in (False, True):
        for seed in range(3):
            seed_transform(seed)
            old = attacker.randomPatchTransform._paste_patch_tensor(rgb, p, geometry)
            old_gradient = torch.autograd.grad(old.sum(), p)[0]
            seed_transform(seed)
            new, footprint = attacker._paste_with_support(rgb, p, geometry)
            new_gradient = torch.autograd.grad(new.sum(), p)[0]
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            torch.testing.assert_close(old_gradient, new_gradient, rtol=0, atol=0)
            seed_transform(seed)
            teacher_pixels, teacher_support = attacker._paste_with_support(rgb, p.detach(), geometry)
            torch.testing.assert_close(new.detach(), teacher_pixels, rtol=0, atol=0)
            torch.testing.assert_close(footprint, teacher_support, rtol=0, atol=0)


def test_wrist_direct_call_receives_primary_branch_pixels_and_matches_full_hook(monkeypatch):
    import VLAAttacker.multiview.openvla_oft_experiments as processing
    inputs = []
    def normalize(x, processor, center_crop):
        assert center_crop
        return torch.cat([x + 1., x + 9.], dim=1)
    def featurizer(x):
        inputs.append(x.detach().clone())
        return x.mean((-1, -2))[:, None].expand(1, 256, 3)
    monkeypatch.setattr(processing, "_processor_normalize_tensor", normalize)
    wrapper = SimpleNamespace(device="cpu", dtype=torch.float32,
                              processor=SimpleNamespace(image_processor=None),
                              vla=SimpleNamespace(eval=lambda: wrapper.vla,
                                                  vision_backbone=SimpleNamespace(featurizer=featurizer)))
    attacker = ExploringOFTILAAttacker(wrapper, "/unused")
    obs = {"full_image": np.full((224, 224, 3), 250, np.uint8),
           "wrist_image": np.full((224, 224, 3), 51, np.uint8)}
    direct, coverage = attacker._features(obs, None, False)
    # Full OFT layout: Front primary, Front fused, Wrist primary, Wrist fused.
    pixels = attacker._patched_pixels(obs, None, False)
    hooked = featurizer(pixels[:, 6:9])
    torch.testing.assert_close(direct, hooked)
    assert coverage.sum() == 0
    assert inputs[0].mean() == pytest.approx(1.2)
    assert pixels[:, :3].mean() > inputs[0].mean()
    assert pixels[:, 9:].mean() > inputs[0].mean()


def test_config_and_submission_contract_rejects_budget_or_reference_drift():
    cfg = json.loads(CONFIG.read_text())
    validate_config(cfg)
    assert (cfg["base_outer_iterations"] + cfg["outer_iterations"]) * cfg["inner_loop"] == 35000
    for key, value in [("outer_iterations", 500), ("baseline_patch_sha256", "wrong"), ("resize_patch", True)]:
        changed = dict(cfg, **{key: value})
        with pytest.raises(RuntimeError):
            validate_config(changed)
    script = (ROOT / "slurm/exploring_oft_dino/53_train_dino_ila_refine200_3090.sbatch").read_text()
    assert "#SBATCH --array=0-2%3" in script and "#SBATCH --gres=gpu:3090:1" in script
    assert "sbatch " not in script
    helper = (ROOT / "submit_exploring_oft_dino_ila_refine200.sh").read_text()
    assert helper.index("CHECK_ONLY") < helper.index("sbatch --parsable")


class MockAttacker(ExploringOFTILAAttacker):
    """Differentiable tiny feature/action tensors, never a real CPU VLA."""
    def __init__(self, fail_at=None):
        self.wrapper = SimpleNamespace(device="cpu", vla=torch.nn.Linear(1, 1))
        self.full_calls = 0
        self.train_calls = 0
        self.fail_at = fail_at

    def _base_inputs(self, record):
        return {}, None

    def _features(self, observation, patch, geometry):
        clean = torch.sin(torch.arange(256 * 8).reshape(1, 256, 8).float() / 17)
        support = torch.zeros(256)
        if patch is None:
            return clean, support
        if patch.requires_grad and torch.is_grad_enabled():
            self.train_calls += 1
            if self.train_calls == self.fail_at:
                raise RuntimeError("mock interruption")
        scale = 0.9 + random.random() / 10 + float(torch.rand(())) / 10
        offset = random.randint(0, 200)
        support[offset:offset + 4] = 1.
        delta = (patch.mean() + patch.flatten()[:256][None, :, None]) * torch.arange(8)[None, None] / 10
        return clean + scale * delta, support

    def _forward(self, inputs, proprio, observation, patch, geometry):
        self.full_calls += 1
        features, support = self._features(observation, patch, geometry)
        self.patch_tokens = support
        action = torch.zeros(1, 8, 7) if patch is None else patch.mean().expand(1, 8, 7)
        return action, features, None


def mock_setup(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    torch.manual_seed(3)
    path = tmp_path / "accepted_baseline.pt"
    torch.save({"patch": torch.rand(3, 50, 50)}, path)
    cfg.update(baseline_patch=str(path), baseline_patch_sha256=sha256_file(path),
               output_root=str(tmp_path / "full"), outer_iterations=3, inner_loop=2,
               warmup_outer_iterations=0, resume_every_outer_iterations=1,
               checkpoint_outer_iterations=[0, 1, 3])
    records = [{"task_id": i, "observation": {}} for i in range(3)]
    return cfg, records


@pytest.mark.parametrize("method", METHODS)
def test_three_arms_warm_start_resume_and_only_control_runs_full_model(tmp_path, method):
    cfg, records = mock_setup(tmp_path)
    full_attacker = MockAttacker()
    full_attacker.train(records, cfg, method, {}, {}, model_load_s=2.)
    root = Path(cfg["output_root"]) / method
    full = torch.load(root / "outer_0003.pt", weights_only=False)
    initial = torch.load(root / "outer_0000.pt", weights_only=False)["patch"]
    baseline = torch.load(cfg["baseline_patch"], weights_only=False)["patch"]
    torch.testing.assert_close(initial, baseline, rtol=0, atol=0)
    assert initial.numel() == 7500
    assert not torch.equal(full["patch"], initial)
    assert full_attacker.full_calls == (11 if method == "uada_continue_tv" else 2)
    full_rows = pd.read_parquet(root / "training.parquet")
    cfg["output_root"] = str(tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="mock interruption"):
        MockAttacker(fail_at=3).train(records, cfg, method, {}, {}, model_load_s=2.)
    folder = Path(cfg["output_root"]) / method
    failed = json.loads((folder / "status.json").read_text())
    assert failed["outer"] == 1 and failed["optimizer_updates"] == 2
    MockAttacker().train(records, cfg, method, {}, {}, model_load_s=3.)
    resumed = torch.load(folder / "outer_0003.pt", weights_only=False)
    torch.testing.assert_close(full["patch"], resumed["patch"], rtol=0, atol=0)
    resumed_rows = pd.read_parquet(folder / "training.parquet")
    pd.testing.assert_frame_equal(full_rows.drop(columns="elapsed_s"), resumed_rows.drop(columns="elapsed_s"))
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "COMPLETE" and status["optimizer_updates"] == 6
    assert status["total_optimizer_updates"] == 1006 and status["total_outer"] == 503
    assert status["costs"]["model_load_s"] == 5.
    assert status["elapsed_s"] >= failed["elapsed_s"]
    assert sha256_file(cfg["baseline_patch"]) == cfg["baseline_patch_sha256"]
    with pytest.raises(RuntimeError, match="completed training"):
        MockAttacker().train(records, cfg, method, {}, {})


def test_early_failure_and_signature_drift_can_not_corrupt_reference(tmp_path):
    cfg, records = mock_setup(tmp_path)
    with pytest.raises(RuntimeError, match="mock interruption"):
        MockAttacker(fail_at=1).train(records, cfg, "dino_ila_tv", {}, {})
    folder = Path(cfg["output_root"]) / "dino_ila_tv"
    state = torch.load(folder / "resume.pt", weights_only=False)
    assert state["outer"] == 0 and state["rows"] == []
    with pytest.raises(RuntimeError, match="signature changed"):
        MockAttacker().train(records, dict(cfg, learning_rate=0.01), "dino_ila_tv", {}, {})
    MockAttacker().train(records, cfg, "dino_ila_tv", {}, {})
    assert json.loads((folder / "status.json").read_text())["status"] == "COMPLETE"

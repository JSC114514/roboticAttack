from __future__ import annotations

import ast
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
sys.path.insert(0, str(ROOT))

from white_patch.oft_correspondence_objectives import (
    appearance_loss, calibrated_gradient, correspondence_loss, feature_loss, make_teacher,
    shift_right, token_coverage, total_variation, visible_teacher,
)
from white_patch.UADA_OFT_CORRESPONDENCE import (
    METHODS, ExploringOFTCorrespondenceAttacker, run_signature, validate_resume,
)
from VLAAttacker.UADA_OFT_CORRESPONDENCE_wrapper import validate_config

CONFIG = ROOT / "configs/exploring_oft_dino/object_correspondence500_v1.json"


def example():
    clean = torch.eye(16)[None]
    mask = torch.zeros(16)
    mask[5] = 1
    teacher = make_teacher(clean, mask, grid=4, radius=1, temperature=0.07)
    return clean, teacher


def test_spatial_shift_is_rightward_and_does_not_wrap():
    x = torch.zeros(2, 16)
    x[0, 5], x[1, 7] = 1, 1
    shifted = shift_right(x, 4)
    assert shifted[0, 6] == 1 and shifted[1].sum() == 0
    with pytest.raises(ValueError):
        shift_right(x, 4, 4)


def test_correspondence_rewards_wrong_location_not_large_feature_norm():
    clean, teacher = example()
    changed = clean.clone()
    changed[:, [5, 6]] = changed[:, [6, 5]]
    zero = torch.zeros(16)
    base, _ = correspondence_loss(clean, teacher, zero)
    good, stats = correspondence_loss(changed, teacher, zero)
    scaled, _ = correspondence_loss(clean * 100, teacher, zero)
    assert good < base
    assert stats["matching_dx"] > 0.9
    torch.testing.assert_close(base, scaled)


def test_patch_covered_queries_and_candidates_cannot_satisfy_target():
    clean, teacher = example()
    footprint = torch.zeros(16)
    footprint[5] = 1
    assert correspondence_loss(clean, teacher, footprint)[0] is None
    assert feature_loss(clean, clean, teacher, footprint) is None
    assert visible_teacher(teacher, torch.ones(16)) is None
    footprint.zero_()
    footprint[1] = 1
    attacked = (clean + 0.03).requires_grad_()
    loss, _ = correspondence_loss(attacked, teacher, footprint)
    grad = torch.autograd.grad(loss, attacked)[0]
    assert torch.count_nonzero(grad[0, 1]) == 0
    assert grad.norm() > 0


def test_feature_control_uses_same_support_and_disruptive_sign():
    clean, teacher = example()
    footprint = torch.zeros(16)
    base = feature_loss(clean, clean, teacher, footprint)
    reversed_features = feature_loss(-clean, clean, teacher, footprint)
    assert reversed_features < base
    assert visible_teacher(teacher, footprint)[0].sum() == 9


def test_full_rgb_tv_is_inactive_below_threshold_and_has_descent_gradient():
    flat = torch.full((3, 50, 50), 0.5, requires_grad=True)
    assert flat.numel() == 7500
    assert appearance_loss(flat, 0.4).item() == 0
    assert torch.autograd.grad(appearance_loss(flat, 0.4), flat)[0].abs().sum() == 0
    torch.manual_seed(7)
    noisy = torch.rand(3, 50, 50, requires_grad=True)
    loss = appearance_loss(noisy, 0.4)
    grad = torch.autograd.grad(loss, noisy)[0]
    assert total_variation(noisy - 0.1 * grad) < total_variation(noisy)
    with pytest.raises(ValueError):
        appearance_loss(noisy, 0)


def test_aux_weight_is_frozen_and_zero_gradient_is_safe():
    main, aux = torch.ones(3), torch.tensor([1., 2., 3.])
    _, weight, stats = calibrated_gradient(main, aux, None, 0.3)
    assert stats["aux_ratio"] == pytest.approx(0.3)
    _, again, stats = calibrated_gradient(main, aux * 2, weight, 0.3)
    assert again == weight and stats["aux_ratio"] == pytest.approx(0.6)
    result, unchanged, _ = calibrated_gradient(main, aux * 0, None, 0.3)
    assert unchanged is None
    torch.testing.assert_close(result, main)


def test_mask_crop_matches_actual_rgb_crop_grid():
    path = ROOT / "VLAAttacker/multiview/openvla_oft_experiments.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_center_crop_resize_tensor")
    namespace = {"Any": object, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    mask = torch.zeros(224, 224)
    mask[12:100, 81:170] = 1
    cropped = namespace[node.name](mask[None, None], output_size=(224, 224))
    actual = torch.nn.functional.avg_pool2d(cropped, 14).flatten()
    torch.testing.assert_close(token_coverage(mask), actual, rtol=0, atol=0)


@pytest.mark.parametrize("geometry", [False, True])
def test_new_support_does_not_change_legacy_rgb_or_gradients(geometry):
    wrapper = SimpleNamespace(device="cpu", vla=torch.nn.Linear(1, 1))
    attacker = ExploringOFTCorrespondenceAttacker(wrapper, "/unused")
    image = np.full((224, 224, 3), 173, dtype=np.uint8)
    for seed in range(4):
        patch = torch.full((3, 50, 50), 0.8, requires_grad=True)
        random.seed(seed)
        expected = attacker.randomPatchTransform._paste_patch_tensor(image, patch, geometry)
        eg = torch.autograd.grad(expected.sum(), patch)[0]
        random.seed(seed)
        actual, support = attacker._paste_with_support(image, patch, geometry)
        ag = torch.autograd.grad(actual.sum(), patch)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(ag, eg, rtol=0, atol=0)
        changed = (actual.detach() != 173 / 255).any(0)
        assert not bool((changed & ~support.bool()).any())


def test_config_contract_and_manual_submission():
    cfg = json.loads(CONFIG.read_text())
    validate_config(cfg)
    assert tuple(cfg["methods"]) == METHODS
    assert cfg["outer_iterations"] * cfg["inner_loop"] == 25000
    assert cfg["patch_shape"] == [3, 50, 50]
    sbatch = (ROOT / "slurm/exploring_oft_dino/50_train_correspondence500_v1_3090.sbatch").read_text()
    assert "#SBATCH --array=0-3%4" in sbatch and "#SBATCH --gres=gpu:3090:1" in sbatch
    assert "sbatch " not in sbatch
    submit = (ROOT / "submit_exploring_oft_correspondence500.sh").read_text()
    assert submit.index("CHECK_ONLY") < submit.index("sbatch --parsable")


def test_resume_signature_covers_source_data_and_configuration():
    a = run_signature({"lr": 1}, "a", {"bank": "x"}, {"trainer": "y"})
    for signature in (run_signature({"lr": 2}, "a", {"bank": "x"}, {"trainer": "y"}),
                      run_signature({"lr": 1}, "a", {"bank": "z"}, {"trainer": "y"}),
                      run_signature({"lr": 1}, "a", {"bank": "x"}, {"trainer": "z"})):
        assert signature != a
        with pytest.raises(RuntimeError, match="signature changed"):
            validate_resume({"signature": a, "method": "a"}, signature, "a", 50)


class MockAttacker(ExploringOFTCorrespondenceAttacker):
    """Tiny differentiable tensors only; no real VLA model on CPU."""
    def __init__(self, fail_at=None):
        self.wrapper = SimpleNamespace(device="cpu", vla=torch.nn.Linear(1, 1))
        self.calls = 0
        self.fail_at = fail_at

    def _base_inputs(self, record):
        return {}, None

    def _forward(self, inputs, proprio, observation, patch, geometry):
        features = torch.sin(torch.arange(256 * 8).reshape(1, 256, 8).float() / 17)
        self.patch_tokens = torch.zeros(256)
        if patch is None:
            return torch.zeros(1, 8, 7), features, None
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("mock interruption")
        action = patch.mean().expand(1, 8, 7)
        features = features + patch[:, :16, :16].mean(0).flatten()[None, :, None] * torch.arange(8)
        return action, features, None


@pytest.mark.parametrize("method", METHODS)
def test_mock_training_and_resume_match_uninterrupted(tmp_path, method):
    cfg = json.loads(CONFIG.read_text())
    cfg.update(output_root=str(tmp_path / "full"), outer_iterations=2, inner_loop=2,
               warmup_outer_iterations=0, resume_every_outer_iterations=1,
               checkpoint_outer_iterations=[0, 1, 2])
    records = [{"task_id": 0, "trajectory_rank": 0, "decision_index": 0, "observation": {}}]
    mask = torch.zeros(256)
    mask[5 * 16 + 5] = 1
    masks = {(0, 0, 0): mask}
    MockAttacker().train(records, masks, cfg, method, {}, {})
    full = torch.load(Path(cfg["output_root"]) / method / "outer_0002.pt", weights_only=False)
    cfg["output_root"] = str(tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="mock interruption"):
        MockAttacker(fail_at=3).train(records, masks, cfg, method, {}, {})
    MockAttacker().train(records, masks, cfg, method, {}, {})
    folder = Path(cfg["output_root"]) / method
    resumed = torch.load(folder / "outer_0002.pt", weights_only=False)
    torch.testing.assert_close(full["patch"], resumed["patch"], rtol=0, atol=0)
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "COMPLETE" and status["optimizer_updates"] == 4
    with pytest.raises(RuntimeError, match="completed training"):
        MockAttacker().train(records, masks, cfg, method, {}, {})

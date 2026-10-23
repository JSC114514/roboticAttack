"""CPU tests for fixed Clear RGB feasibility and current-score replay weights."""
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
from white_patch.oft_dino_sampling_clear_objectives import (
    HF_INTERNAL_BUDGET, HF_THRESHOLD, TV_INTERNAL_BUDGET, TV_THRESHOLD,
    apply_score_weights, constraint_metrics, enforce_constraints, gaussian_blur,
    high_frequency_max, soft_tv_loss, stable_score, total_variation,
)


def _random_patch(seed, dtype=torch.float32):
    return torch.rand(3, 50, 50, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _rounded_png_tensor(patch, path):
    pixels = torch.round(patch.detach() * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(pixels, mode="RGB").save(path)
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).to(patch.dtype) / 255


@pytest.mark.parametrize("seed", [0, 1, 39, 42, 1337])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_random_feasibility_and_real_png_rounding(seed, dtype, tmp_path):
    patch = (_random_patch(seed, dtype) * 1.4 - .2).requires_grad_(True)
    before = patch.detach().clone()
    fixed, info = enforce_constraints(patch)
    assert torch.equal(patch.detach(), before) and patch.grad is None
    assert not fixed.requires_grad and fixed.shape == (3, 50, 50)
    assert constraint_metrics(fixed)["feasible_internal"]
    assert info["correction_applied"] and 0 <= info["correction_passes"] <= 3
    assert info["correction_distance"] > 0 and 0 < info["shrink_scale"] <= 1
    png = _rounded_png_tensor(fixed, tmp_path / f"patch_{seed}.png")
    metrics = constraint_metrics(png)
    assert metrics["rgb_min"] >= 0 and metrics["rgb_max"] <= 1
    assert metrics["tv"] <= TV_THRESHOLD and metrics["hf_max"] <= HF_THRESHOLD
    # Quantization perturbs each difference by <=1/255 and each Gaussian
    # residual by <=1/255, giving the frozen two-axis TV and HF margins.
    continuous = constraint_metrics(fixed)
    assert metrics["tv"] <= continuous["tv"] + 2 / 255 + 2e-7
    assert metrics["hf_max"] <= continuous["hf_max"] + 1 / 255 + 2e-7


def test_feasible_input_is_bitwise_unchanged_without_storage_aliasing():
    patch = torch.stack([torch.full((50, 50), value) for value in (.1, .5, .9)])
    patch[:, 12:32, 10:20] += .01
    patch.requires_grad_(True)
    fixed, info = enforce_constraints(patch)
    assert torch.equal(fixed, patch)
    assert fixed.data_ptr() != patch.data_ptr() and not fixed.requires_grad
    assert not info["correction_applied"] and info["correction_passes"] == 0
    assert info["shrink_scale"] == 1 and info["correction_distance"] == 0
    second, _ = enforce_constraints(fixed)
    assert torch.equal(second, fixed)


def test_float32_shrink_seed39_meets_strict_internal_hf_boundary():
    # The original convolution implementation exceeded 15/255 on this seed
    # by 5.6e-8 with a 1e-6 numerical shrink margin.
    fixed, info = enforce_constraints(_random_patch(39))
    assert info["correction_passes"] == 3 and info["shrink_scale"] < 1
    assert constraint_metrics(fixed)["hf_max"] <= HF_INTERNAL_BUDGET


def test_inference_first_blur_still_allows_later_rgb_gradients():
    patch = _random_patch(24)
    with torch.inference_mode():
        gaussian_blur(patch, sigma=.7)
        fixed, _ = enforce_constraints(patch)
        assert constraint_metrics(fixed)["feasible_internal"]
    live = patch.clone().requires_grad_(True)
    gradient = torch.autograd.grad(gaussian_blur(live, sigma=.7).sum(), live)[0]
    assert bool((gradient > 0).all())


@pytest.mark.parametrize("kind", ["isolated", "edge", "checkerboard", "all_zero", "all_one"])
def test_spikes_edges_and_extreme_constants_are_jointly_feasible(kind):
    patch = torch.zeros(3, 50, 50)
    if kind == "isolated":
        patch[:, 25, 25] = 1
    elif kind == "edge":
        patch[:, :, 25:] = 1
    elif kind == "checkerboard":
        patch[:, ::2, ::2] = 1
        patch[:, 1::2, 1::2] = 1
    elif kind == "all_one":
        patch.fill_(1)
    fixed, info = enforce_constraints(patch)
    assert constraint_metrics(fixed)["feasible_internal"]
    assert constraint_metrics(torch.round(fixed * 255) / 255)["feasible_scientific"]
    assert info["correction_passes"] <= 3
    if kind in ("all_zero", "all_one"):
        assert torch.equal(patch, fixed) and not info["correction_applied"]
    else:
        assert info["correction_applied"] and info["correction_passes"] > 0


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-7), (torch.float64, 1e-14)])
def test_gaussian_kernel_reflect_padding_and_constant_preservation(dtype, tolerance):
    values = torch.exp(-torch.arange(-1, 2, dtype=torch.float64).square() / (2 * .5**2))
    values /= values.sum()
    kernel = (values[:, None] * values[None]).to(dtype)
    kernel = (kernel / kernel.sum()).expand(3, 1, 3, 3)
    patch = _random_patch(4, dtype)
    expected = F.conv2d(F.pad(patch[None], (1, 1, 1, 1), mode="reflect"), kernel, groups=3)[0]
    assert torch.allclose(gaussian_blur(patch), expected, atol=tolerance, rtol=0)
    constant = torch.tensor([.0, .372, 1.], dtype=dtype)[:, None, None].expand(3, 7, 9)
    assert torch.allclose(gaussian_blur(constant), constant, atol=tolerance, rtol=0)
    assert float(total_variation(constant)) == 0 and float(high_frequency_max(constant)) < tolerance
    fixed, _ = enforce_constraints(constant)
    assert torch.equal(fixed, constant)


def test_gaussian_does_not_use_convolution_backend(monkeypatch):
    def forbidden_convolution(*args, **kwargs):
        raise AssertionError("HF blur must not dispatch through a convolution backend")
    monkeypatch.setattr(F, "conv2d", forbidden_convolution)
    live = _random_patch(65).requires_grad_(True)
    first, second = gaussian_blur(live), gaussian_blur(live)
    assert torch.equal(first, second)
    gradient = torch.autograd.grad(first.sum(), live)[0]
    assert bool((gradient > 0).all())


def test_positive_homogeneity_justifies_final_mean_preserving_shrink():
    patch = _random_patch(10, torch.float64)
    mean = patch.mean(dim=(-2, -1), keepdim=True)
    for scale in (.03, .31, .9):
        shrunk = mean + scale * (patch - mean)
        assert torch.allclose(total_variation(shrunk), scale * total_variation(patch), atol=1e-14)
        assert torch.allclose(high_frequency_max(shrunk), scale * high_frequency_max(patch), atol=1e-14)
        assert torch.allclose(shrunk.mean(dim=(-2, -1)), patch.mean(dim=(-2, -1)), atol=1e-14)


def test_tv_only_violation_uses_global_shrink_and_keeps_all_rgb_parameters():
    patch = torch.full((3, 50, 50), .5 - .086, dtype=torch.float64)
    patch[:, ::2, ::2] += .172
    patch[:, 1::2, 1::2] += .172
    before = constraint_metrics(patch)
    assert before["tv"] > TV_INTERNAL_BUDGET and before["hf_max"] < HF_INTERNAL_BUDGET
    # No dimensions or independent RGB variables are removed by feasibility restoration.
    fixed, info = enforce_constraints(patch)
    assert info["correction_passes"] == 0 and info["shrink_scale"] < 1
    assert torch.allclose(fixed.mean(dim=(-2, -1)), patch.mean(dim=(-2, -1)), atol=1e-14)
    assert fixed.numel() == 7500 and fixed.shape == patch.shape
    assert constraint_metrics(fixed)["feasible_internal"]
    live = fixed.clone().requires_grad_(True)
    coefficients = torch.arange(1, 7501, dtype=live.dtype).reshape_as(live)
    gradient = torch.autograd.grad((live * coefficients).sum(), live)[0]
    assert bool((gradient != 0).all())


def test_soft_tv_fixed_hinge_matches_formula_and_has_full_rgb_gradient():
    patch = _random_patch(22).requires_grad_(True)
    expected = F.relu((total_variation(patch) - .3) / .4).square()
    assert torch.equal(soft_tv_loss(patch), expected)
    gradient = torch.autograd.grad(soft_tv_loss(patch) + patch.square().mean(), patch)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs().sum(dim=(1, 2)) > 0).all()
    smooth = torch.full((3, 50, 50), .5, requires_grad=True)
    assert soft_tv_loss(smooth) == 0
    assert torch.equal(torch.autograd.grad(soft_tv_loss(smooth), smooth)[0], torch.zeros_like(smooth))
    blur_patch = _random_patch(21, torch.float64).requires_grad_(True)
    assert torch.autograd.gradcheck(gaussian_blur, (blur_patch,), fast_mode=True)


@pytest.mark.parametrize("deltas,expected", [((2, 3), 2), ((2, -3), 0), ((-2, 3), 0),
                                           ((-2, -3), 0), ((0, 4), 0)])
def test_stable_score_keeps_raw_negative_views(deltas, expected):
    assert stable_score(*deltas) == expected


def _records():
    rows = []
    for task in (0, 4):
        for phase in ("early", "middle", "late"):
            scores = [4., 3., 2., -1., -2.] if phase != "late" else [-1.] * 5
            for state, score in enumerate(scores):
                rows.append({"task_id": task, "phase_proxy": phase, "state_index": state,
                             "decision_index": state + 10, "dino_score": score})
    return rows


def test_score_weights_match_legacy_full_group_upper_half_and_ties():
    records = _records()
    result = apply_score_weights(records, 42, 500)
    assert all("dino_weight" not in row for row in records)
    groups = defaultdict(list)
    for row in result:
        groups[(row["task_id"], row["phase_proxy"])].append(row)
    for (_, phase), group in groups.items():
        expected = [2., 2., 2., 1., 1.] if phase != "late" else [1.] * 5
        assert [row["dino_weight"] for row in group] == expected
        assert sorted(row["shuffled_weight"] for row in group) == sorted(expected)
    # One positive among five ranks is still upweighted; it is not half positives.
    records = [dict(row, dino_score=1. if i == 0 else -1.) for i, row in enumerate(_records()[:5])]
    assert [row["dino_weight"] for row in apply_score_weights(records, 42, 250)] == [2., 1., 1., 1., 1.]
    ties = [dict(row, dino_score=1., state_index=4-i, decision_index=i)
            for i, row in enumerate(_records()[:5])]
    assert [row["dino_weight"] for row in apply_score_weights(ties, 42, 250)] == [1., 1., 2., 2., 2.]


def test_shuffle_uses_own_group_weights_is_reproducible_and_leaves_rng_unchanged():
    records = _records()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    first = apply_score_weights(records, 42, 500)
    second = apply_score_weights(records, 42, 500)
    enforce_constraints(_random_patch(73))
    assert first == second
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and np.array_equal(after_numpy[1], numpy_rng[1])
    assert after_numpy[2:] == numpy_rng[2:]
    # Refresh changes the fixed permutation stream, not the marginal weights.
    changed = apply_score_weights(records, 42, 1000)
    assert [row["dino_weight"] for row in changed] == [row["dino_weight"] for row in first]
    assert [row["shuffled_weight"] for row in changed] != [row["shuffled_weight"] for row in first]
    assert any(row["dino_weight"] != row["shuffled_weight"] for row in first)
    stable = [dict(row, dino_score=stable_score(row["dino_score"], row["dino_score"])) for row in records]
    assert apply_score_weights(stable, 42, 500, stable=True)


def test_invalid_inputs_and_frozen_contract_drift_are_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        enforce_constraints(torch.full((3, 4, 4), float("nan")))
    with pytest.raises(ValueError, match="float32 or float64"):
        enforce_constraints(torch.zeros(3, 4, 4, dtype=torch.float16))
    with pytest.raises(ValueError, match="Frozen"):
        enforce_constraints(_random_patch(5), {"tv_threshold": .4})
    with pytest.raises(ValueError, match="positive"):
        gaussian_blur(_random_patch(5), sigma=0)
    with pytest.raises(ValueError, match="finite"):
        stable_score(float("nan"), 1)
    with pytest.raises(ValueError, match="Empty"):
        apply_score_weights([], 42, 250)
    with pytest.raises(ValueError, match="clamped"):
        apply_score_weights(_records(), 42, 250, stable=True)

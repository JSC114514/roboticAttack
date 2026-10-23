"""CPU tensor tests: action ascent, best-zero guard and feasible random control."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
from white_patch.oft_dino_action_counter_objectives import (
    action_counter_eta, bounded_tv_loss, eta_metrics, feasible_permute_eta)
from white_patch.oft_correspondence_objectives import appearance_loss


def test_fixed_scale_hinge_equals_legacy_and_has_inactive_budget():
    torch.manual_seed(42)
    patch = torch.rand(3, 50, 50, requires_grad=True)
    assert torch.allclose(bounded_tv_loss(patch, .4), appearance_loss(patch, .4), atol=2e-7)
    actual = torch.autograd.grad(bounded_tv_loss(patch, .4), patch, retain_graph=True)[0]
    legacy = torch.autograd.grad(appearance_loss(patch, .4), patch)[0]
    assert torch.allclose(actual, legacy, atol=2e-7)
    smooth = torch.full((3, 50, 50), .5, requires_grad=True)
    assert bounded_tv_loss(smooth, .3) == 0
    assert torch.equal(torch.autograd.grad(bounded_tv_loss(smooth, .3), smooth)[0], torch.zeros_like(smooth))


def test_counter_ascends_wrong_target_mse_without_touching_live_patch_graph():
    patch = torch.tensor(.2, requires_grad=True)
    rgb = torch.ones(3, 8, 8) * patch
    original = rgb.detach().clone()
    support = torch.zeros(8, 8)
    support[2:4, 2:4] = 1
    eta, stats = action_counter_eta(rgb, support, lambda value: ((value - .9)**2).mean())
    assert stats["counter_loss_after"] >= stats["counter_loss_before"]
    assert stats["counter_loss_delta"] > 0 and stats["counter_effective"] == 1
    assert len(stats["counter_candidate_losses"]) == 3
    assert stats["counter_forward_calls"] == 3 and stats["counter_backward_calls"] == 2
    assert not eta.requires_grad and patch.grad is None
    assert not eta[:, 2:4, 2:4].any()
    assert eta.abs().max() <= 4/255 + 1e-7
    assert torch.equal(rgb.detach(), original)
    # Eta is stop-gradient; the later normal outer RGB still differentiates P.
    outer = ((rgb + eta).clamp(0, 1) - .9).square().mean()
    assert torch.autograd.grad(outer, patch)[0].abs() > 0


def test_best_zero_candidate_guards_misleading_partial_gradient():
    rgb, support = torch.full((3, 4, 4), .5), torch.zeros(4, 4)
    def misleading(value):
        actual = -((value-.5)**2).mean()
        # Correct forward scalar, intentionally wrong nonzero gradient.
        return actual.detach() + (value-value.detach()).mean()
    eta, stats = action_counter_eta(rgb, support, misleading)
    assert not eta.any() and stats["counter_effective"] == 0
    assert stats["counter_loss_after"] == stats["counter_loss_before"] == 0
    assert stats["counter_candidate_losses"][1] < 0


def test_zero_gradient_is_legal_but_disconnected_loss_is_not():
    rgb, support = torch.full((3, 4, 4), .5), torch.zeros(4, 4)
    eta, stats = action_counter_eta(rgb, support, lambda value: value.sum()*0)
    assert not eta.any() and stats["counter_loss_delta"] == 0
    with pytest.raises(RuntimeError, match="disconnected"):
        action_counter_eta(rgb, support, lambda value: torch.tensor(0.))
    with pytest.raises(RuntimeError, match="finite"):
        action_counter_eta(rgb, support, lambda value: value.mean()*float("nan"))


def test_rgb_projection_and_partial_support_exclusion():
    rgb, support = torch.zeros(3, 4, 4), torch.zeros(4, 4)
    support[0, 0] = .0001  # Affine support is not necessarily binary.
    eta, _ = action_counter_eta(rgb, support, lambda value: -value.mean())
    assert not eta.any()  # Requested negative step at the RGB bound is infeasible.
    eta, _ = action_counter_eta(rgb, support, lambda value: value.mean())
    assert not eta[:, 0, 0].any()
    assert eta[:, 1:, :].min() > 0


def test_feasible_random_preserves_absolute_multiset_and_all_budgets():
    torch.manual_seed(123)
    rgb = torch.rand(3, 16, 16)
    rgb[:, :2] = 0
    rgb[:, -2:] = 1
    support = torch.zeros(16, 16)
    support[5:9, 5:9] = 1
    proposed = torch.rand_like(rgb)*8/255 - 4/255
    eta = ((rgb + proposed).clamp(0, 1) - rgb) * (support <= 0)[None]
    generator = torch.Generator().manual_seed(321)
    result, stats = feasible_permute_eta(rgb, support, eta, generator)
    assert not torch.equal(result, eta)
    assert torch.equal(result.abs().flatten().sort().values, eta.abs().flatten().sort().values)
    assert not result[:, 5:9, 5:9].any()
    assert (rgb+result).min() >= 0 and (rgb+result).max() <= 1
    assert not result.requires_grad
    assert all(abs(eta_metrics(result)[key]-eta_metrics(eta)[key]) < 1e-10 for key in eta_metrics(eta))
    assert max(stats[key] for key in stats if key.startswith("random_match_")) < 1e-10


def test_random_zero_is_matched_and_uniform_direction_can_be_destroyed():
    rgb, support = torch.full((3, 8, 8), .5), torch.zeros(8, 8)
    result, stats = feasible_permute_eta(rgb, support, torch.zeros_like(rgb))
    assert not result.any() and stats["random_zero_matched"] == 1
    eta = torch.full_like(rgb, 4/255)
    result, _ = feasible_permute_eta(rgb, support, eta, torch.Generator().manual_seed(3))
    assert (result < 0).any() and (result > 0).any()
    assert torch.equal(result.abs(), eta)


def test_random_permutation_updates_real_noncontiguous_renderer_tensor():
    torch.manual_seed(17)
    rgb = torch.rand(12, 12, 3).permute(2, 0, 1)
    eta = torch.full((12, 12, 3), 2/255).permute(2, 0, 1)
    support = torch.zeros(12, 12)
    eta = ((rgb + eta).clamp(0, 1) - rgb)
    assert not rgb.is_contiguous() and not eta.is_contiguous()
    result, stats = feasible_permute_eta(rgb, support, eta, torch.Generator().manual_seed(10))
    assert not torch.equal(result, eta) and stats["random_changed_fraction"] > 0
    assert torch.equal(result.abs().flatten().sort().values, eta.abs().flatten().sort().values)
    assert (rgb+result).min() >= 0 and (rgb+result).max() <= 1


def test_random_does_not_silently_fallback_when_bounds_fix_direction():
    rgb, support = torch.zeros(3, 2, 2), torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match="no guided fallback"):
        feasible_permute_eta(rgb, support, torch.full_like(rgb, 4/255))


def test_invalid_hyperparameters_and_support_rejected():
    patch = torch.rand(3, 5, 5)
    with pytest.raises(ValueError):
        bounded_tv_loss(patch, .3, 0)
    with pytest.raises(ValueError):
        action_counter_eta(patch, torch.zeros(4, 4), lambda x: x.mean())
    with pytest.raises(ValueError):
        action_counter_eta(patch, torch.zeros(5, 5), lambda x: x.mean(), steps=-1)

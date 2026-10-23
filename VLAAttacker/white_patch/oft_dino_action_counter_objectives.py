"""Model-free objectives for the independent action-counter experiment.

The inner objective is supplied by the trainer: true OFT action MSE with
Wrist-SigLIP values unchanged and only its inner input gradient detached.
No original Exploring source or old experiment is modified here.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from white_patch.oft_correspondence_objectives import total_variation


def bounded_tv_loss(patch, threshold, scale=.4):
    """Fixed-scale hinge: threshold=.4 is mathematically the legacy penalty."""
    if not (math.isfinite(threshold) and threshold > 0
            and math.isfinite(scale) and scale > 0):
        raise ValueError("TV budget and scale must be finite and positive")
    return F.relu((total_variation(patch) - threshold) / scale).square()


def _validate_rgb(rgb, support):
    if rgb.ndim != 3 or rgb.shape[0] != 3 or tuple(support.shape) != tuple(rgb.shape[-2:]):
        raise ValueError("Expected one 3-channel RGB image and its actual 2D support")
    if not rgb.is_floating_point() or not bool(torch.isfinite(rgb).all()):
        raise ValueError("Nonfinite/nonfloating RGB")
    if not bool(torch.isfinite(support).all()) or bool(((support < 0) | (support > 1)).any()):
        raise ValueError("Invalid actual patch support")
    if bool(((rgb < 0) | (rgb > 1)).any()):
        raise ValueError("RGB must be within [0,1] before the inner loop")
    if support.device != rgb.device:
        raise ValueError("RGB/support device mismatch")


def eta_metrics(eta):
    # Float64 reductions make permutation checks independent of reduction order.
    value = eta.detach().double()
    return {"linf": float(value.abs().max()), "rms": float(value.square().mean().sqrt()),
            "mean_abs": float(value.abs().mean())}


def action_counter_eta(rgb, support, loss_fn, *, steps=2, epsilon=4/255, step_size=2/255):
    """Projected ascent of the SAME wrong-target MSE minimized by the patch.

    Include eta=0 in best-candidate selection. Only eta is differentiated; the
    caller's live patch RGB graph is detached before any inner computation.
    loss_fn must itself arrange the temporary DINO-guided gradient restriction.
    """
    _validate_rgb(rgb, support)
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("Counter steps must be a nonnegative integer")
    if not (math.isfinite(epsilon) and epsilon >= 0
            and math.isfinite(step_size) and step_size > 0):
        raise ValueError("Invalid counter amplitude/step size")
    frozen = rgb.detach()
    permitted = (support.detach() <= 0).to(rgb.dtype)[None]
    eta = torch.zeros_like(frozen)
    best_eta, best_loss, initial_loss = eta.clone(), -math.inf, None
    candidate_losses = []
    for step in range(steps + 1):
        eta = eta.detach().requires_grad_(step < steps)
        context = torch.enable_grad() if step < steps else torch.no_grad()
        with context:
            loss = loss_fn((frozen + eta).clamp(0, 1))
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
            raise RuntimeError("Counter objective must be one finite scalar Tensor")
        scalar = float(loss.detach())
        candidate_losses.append(scalar)
        if initial_loss is None:
            initial_loss = scalar
        if scalar > best_loss:
            best_loss, best_eta = scalar, eta.detach().clone()
        if step < steps:
            if not loss.requires_grad:
                raise RuntimeError("Counter objective disconnected from input RGB")
            gradient = torch.autograd.grad(loss, eta)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Nonfinite counter input gradient")
            candidate = (eta.detach() + step_size * gradient.sign()).clamp(-epsilon, epsilon) * permitted
            # Store the ACTUAL post-clamp RGB perturbation, not an unobservable proposal.
            eta = ((frozen + candidate).clamp(0, 1) - frozen) * permitted
            if float(eta.abs().max()) > epsilon + 2e-7 or bool((eta * (1-permitted)).any()):
                raise RuntimeError("Counter escaped actual support/amplitude constraints")
    actual = ((frozen + best_eta).clamp(0, 1) - frozen).detach()
    metrics = eta_metrics(actual)
    return actual, {"counter_loss_before": initial_loss, "counter_loss_after": best_loss,
        "counter_loss_delta": best_loss-initial_loss,
        "counter_effective": float(best_loss > initial_loss),
        "counter_eta_linf": metrics["linf"], "counter_eta_rms": metrics["rms"],
        "counter_eta_mean_abs": metrics["mean_abs"], "counter_steps": steps,
        "counter_forward_calls": steps+1, "counter_backward_calls": steps,
        "counter_candidate_losses": candidate_losses}


@torch.no_grad()
def feasible_permute_eta(rgb, support, eta, generator=None):
    """Destroy guidance while matching actual amplitudes, zeros and RGB feasibility.

    Disjoint pair swaps are accepted only when BOTH resulting pixels are legal.
    Feasible random signs then preserve every absolute magnitude. At RGB bounds
    only legal signs can be chosen. No clipping changes the amplitude budget.
    """
    _validate_rgb(rgb, support)
    if eta.shape != rgb.shape or not bool(torch.isfinite(eta).all()):
        raise ValueError("Invalid eta shape/values")
    allowed = (support <= 0)[None].expand_as(rgb)
    if bool(eta[~allowed].any()) or bool(((rgb + eta < 0) | (rgb + eta > 1)).any()):
        raise ValueError("Input eta is not an actual feasible patch-exterior perturbation")
    original = eta.detach()
    if not bool(original.any()):
        result = torch.zeros_like(original)
    else:
        # Renderer tensors often arrive as permuted HWC->CHW views. A contiguous
        # result is required: flatten() on a noncontiguous clone would be a COPY
        # and randomization would silently modify only that discarded copy.
        result = original.contiguous().clone()
        eligible = torch.nonzero(allowed.flatten(), as_tuple=False).flatten()
        flat_rgb = rgb.detach().flatten()
        for attempt in range(3):
            flat = result.flatten()
            for _ in range(3):
                perm = eligible[torch.randperm(eligible.numel(), device=rgb.device, generator=generator)]
                count = (perm.numel() // 2) * 2
                left, right = perm[:count:2], perm[1:count:2]
                lv, rv = flat[left].clone(), flat[right].clone()
                lrgb, rrgb = flat_rgb[left] + rv, flat_rgb[right] + lv
                accept = (lrgb >= 0) & (lrgb <= 1) & (rrgb >= 0) & (rrgb <= 1)
                flat[left[accept]], flat[right[accept]] = rv[accept], lv[accept]
            amplitude = flat[eligible].abs()
            positive = flat_rgb[eligible] + amplitude <= 1
            negative = flat_rgb[eligible] - amplitude >= 0
            if not bool((positive | negative).all()):
                raise RuntimeError("Permutation destroyed RGB feasibility")
            coin = torch.rand(amplitude.shape, device=rgb.device, generator=generator) < .5
            signs = torch.where(positive & (~negative | coin), 1., -1.).to(rgb.dtype)
            flat[eligible] = amplitude * signs
            if not torch.equal(result, original):
                break
        if torch.equal(result, original):
            raise RuntimeError("Cannot randomize nonzero eta under matched feasible constraints; no guided fallback")
    if bool(result[~allowed].any()) or bool(((rgb + result < 0) | (rgb + result > 1)).any()):
        raise RuntimeError("Random eta escaped support/RGB bounds")
    before, after = eta_metrics(original), eta_metrics(result)
    errors = {key: abs(after[key]-before[key]) for key in before}
    if any(error > 1e-10 for error in errors.values()):
        raise RuntimeError("Random eta actual amplitude matching failed")
    return result.detach(), {"random_changed_fraction": float((result != original).float().mean()),
        "random_zero_matched": float(bool(result.any()) == bool(original.any())),
        "random_zero_eta": float(not bool(result.any())),
        **{f"random_eta_{key}": value for key, value in after.items()},
        **{f"random_match_{key}_error": value for key, value in errors.items()}}

"""Pure RGB appearance and sample-specific DINO repair objectives."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def charbonnier_tv(patch):
    horizontal = patch[..., :, 1:] - patch[..., :, :-1]
    vertical = patch[..., 1:, :] - patch[..., :-1, :]
    return ((horizontal.square() + 1e-6).sqrt() - 1e-3).mean() + (
        (vertical.square() + 1e-6).sqrt() - 1e-3).mean()


def high_frequency_energy(patch, sigma=.8, kernel_size=5):
    if sigma <= 0 or kernel_size < 3 or kernel_size % 2 != 1:
        raise ValueError("Invalid Gaussian appearance kernel")
    radius = kernel_size // 2
    axis = torch.arange(-radius, radius + 1, device=patch.device, dtype=patch.dtype)
    kernel = torch.exp(-axis.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel2 = (kernel[:, None] * kernel[None, :])[None, None].expand(patch.shape[0], 1, -1, -1)
    smooth = F.conv2d(F.pad(patch[None], (radius,) * 4, mode="reflect"),
                      kernel2, groups=patch.shape[0])[0]
    return (patch - smooth).square().mean()


def outside_dense_error(current, clean, coverage):
    """Per-token unit-normalized DINO discrepancy outside ANY covered token."""
    if current.shape != clean.shape or current.ndim != 3 or current.shape[0] != 1:
        raise ValueError("Dense DINO reference shape drift")
    visible = coverage.detach().flatten().to(current.device) <= 0
    if visible.numel() != current.shape[1] or not bool(visible.any()):
        raise ValueError("No uncovered dense DINO token")
    current = F.normalize(current.float(), dim=-1)
    clean = F.normalize(clean.detach().float(), dim=-1)
    return (current[:, visible] - clean[:, visible]).square().mean()


def conservative_token_coverage(support, grid=16):
    """Transport actual RGB support through the deployed crop before pooling.

    The patch renderer currently operates on 224 RGB. If another input size is
    supplied, dilating before bicubic resize conservatively includes its
    interpolation footprint rather than accidentally declaring tokens clean.
    """
    if support.ndim != 2:
        raise ValueError("Expected a two-dimensional RGB support")
    value = (support.detach() > 0).float()[None, None]
    if value.shape[-2:] != (224, 224):
        value = F.max_pool2d(value, 5, stride=1, padding=2)
        value = F.interpolate(value, (224, 224), mode="bicubic", align_corners=False,
                              antialias=True).abs().clamp(0, 1)
    axis = torch.linspace(-math.sqrt(.9), math.sqrt(.9), 224,
                          device=value.device, dtype=value.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    value = F.grid_sample(value, torch.stack((xx, yy), -1)[None], mode="bilinear",
                          padding_mode="border", align_corners=True)
    if 224 % grid:
        raise ValueError("Dense token grid must divide image size")
    return F.max_pool2d(value, 224 // grid).flatten()


def repair_rgb(rgb, support, clean, feature_fn, *, steps=2, epsilon=4/255,
               step_size=2/255):
    """Sign-PGD minimizing DINO error; the best iterate includes eta=0.

    `rgb` is detached before PGD, freezing the sampled patch. Its live graph is
    retained by the caller for the outer update; no EOT is sampled here.
    """
    if rgb.ndim != 3 or tuple(support.shape) != tuple(rgb.shape[-2:]):
        raise ValueError("RGB/support shape drift")
    if steps < 0 or epsilon < 0 or step_size < 0:
        raise ValueError("Invalid repair budget")
    frozen = rgb.detach()
    permitted = (support.detach() <= 0).to(rgb.dtype)[None]
    coverage = conservative_token_coverage(support)
    eta = torch.zeros_like(frozen)
    best_eta, best_loss, first_loss = eta.clone(), float("inf"), None
    errors, forwards, pgd_updates = [], 0, 0
    for step in range(steps + 1):
        eta = eta.detach().requires_grad_(step < steps)
        with torch.set_grad_enabled(step < steps):
            repaired = (frozen + eta).clamp(0, 1)
            loss = outside_dense_error(feature_fn(repaired), clean, coverage)
        forwards += 1
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite DINO repair objective")
        scalar = float(loss.detach())
        errors.append(scalar)
        if first_loss is None:
            first_loss = scalar
        if scalar < best_loss:
            best_loss, best_eta = scalar, eta.detach().clone()
        if step < steps:
            gradient = torch.autograd.grad(loss, eta)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Nonfinite DINO repair RGB gradient")
            candidate = (eta - step_size * gradient.sign()).clamp(-epsilon, epsilon) * permitted
            # Store actual RGB changes, so clamp cannot hide budget violations.
            eta = ((frozen + candidate).clamp(0, 1) - frozen) * permitted
            if float(eta.abs().max()) > epsilon + 2e-7 or bool((eta * (1-permitted)).any()):
                raise RuntimeError("Repair escaped RGB support/amplitude constraint")
            pgd_updates += 1
    actual = ((frozen + best_eta).clamp(0, 1) - frozen).detach()
    return actual, {"repair_error_before": first_loss, "repair_error_after": best_loss,
                   "repair_error_path": errors, "repair_effective": float(best_loss < first_loss),
                   "repair_eta_linf": float(actual.abs().max()),
                   "repair_eta_mean_abs": float(actual.abs().mean()),
                   "repair_visible_tokens": int((coverage <= 0).sum()),
                   "repair_dino_forward_calls": forwards, "repair_pgd_updates": pgd_updates}

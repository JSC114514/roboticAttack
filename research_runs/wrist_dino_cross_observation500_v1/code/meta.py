"""The approved first-order update, with no model, optimizer or queue side effects."""
from __future__ import annotations
import math
import torch
from common import require

def cosine(a, b):
    if a is None or b is None:
        return None
    n = float(a.norm() * b.norm())
    return float((a.flatten() @ b.flatten()) / n) if n > 0 else 0.

def split_rgb_gradient(loss, dino, siglip, patch):
    """One upstream native VJP, then the two encoder VJPs; all outputs detached.

    The full policy forward is untouched. The upstream graph is released before
    either encoder VJP; only the shared RGB preprocessing graph is retained once.
    """
    d, s = torch.autograd.grad(loss, (dino, siglip))
    gd = torch.autograd.grad(dino, patch, grad_outputs=d, retain_graph=True)[0].detach()
    gs = torch.autograd.grad(siglip, patch, grad_outputs=s)[0].detach()
    require(torch.isfinite(gd).all() and torch.isfinite(gs).all(), "Nonfinite native visual VJP")
    return gd + gs, gd

@torch.no_grad()
def preview_adamw(patch, gradient, optimizer):
    """Exact installed Transformers AdamW equations; clone moments, never step."""
    require(len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]["params"]) == 1
            and optimizer.param_groups[0]["params"][0] is patch, "Only one RGB optimizer parameter allowed")
    group = optimizer.param_groups[0]
    require("correct_bias" in group and group["weight_decay"] == 0, "Expected frozen Transformers AdamW, no decay")
    state = optimizer.state.get(patch, {})
    m = state["exp_avg"].clone() if state else torch.zeros_like(patch)
    v = state["exp_avg_sq"].clone() if state else torch.zeros_like(patch)
    step = int(state["step"]) + 1 if state else 1
    b1, b2 = group["betas"]
    m.mul_(b1).add_(gradient, alpha=1-b1)
    v.mul_(b2).addcmul_(gradient, gradient, value=1-b2)
    denom = v.sqrt().add_(group["eps"])
    rate = group["lr"]
    if group["correct_bias"]:
        rate = rate * math.sqrt(1-b2**step) / (1-b1**step)
    result = patch.detach().clone()
    result.addcdiv_(m, denom, value=-rate)
    return result.clamp_(0, 1)

@torch.no_grad()
def virtual_patch(patch, dino_gradient, target_norm):
    """Match the clipped physical RGB norm; explicitly return infeasible cases."""
    p = patch.detach(); direction = -dino_gradient.detach()
    require(math.isfinite(target_norm) and target_norm >= 0 and torch.isfinite(direction).all(), "Invalid virtual RGB direction/norm")
    if target_norm == 0:
        return p.clone(), {"virtual_status": "ZERO_ADAM_STEP", "virtual_alpha": 0., "virtual_rgb_l2": 0., "virtual_target_l2": 0.}
    if float(direction.norm()) == 0:
        return None, {"virtual_status": "ZERO_DINO_GRADIENT", "virtual_alpha": None, "virtual_rgb_l2": None, "virtual_target_l2": target_norm}
    endpoint = torch.where(direction > 0, torch.ones_like(p), torch.where(direction < 0, torch.zeros_like(p), p))
    tolerance = max(1e-8, target_norm * 1e-5)
    if float((endpoint - p).norm()) + tolerance < target_norm:
        return None, {"virtual_status": "UNATTAINABLE_CLIPPED_NORM", "virtual_alpha": None, "virtual_rgb_l2": None, "virtual_target_l2": target_norm}
    lo, hi = 0., target_norm / float(direction.norm())
    for _ in range(60):
        if float(((p + hi*direction).clamp(0, 1)-p).norm()) >= target_norm - tolerance:
            break
        hi *= 2
    else:
        return None, {"virtual_status": "SCALAR_SEARCH_FAILED", "virtual_alpha": None, "virtual_rgb_l2": None, "virtual_target_l2": target_norm}
    for _ in range(36):
        mid = (lo+hi)/2
        if float(((p+mid*direction).clamp(0, 1)-p).norm()) < target_norm:
            lo = mid
        else:
            hi = mid
    alpha = (lo+hi)/2
    value = (p+alpha*direction).clamp(0, 1)
    norm = float((value-p).norm())
    require(abs(norm-target_norm) <= max(5e-8, target_norm*2e-5), "Clipped virtual step failed numeric norm matching")
    return value, {"virtual_status": "MATCHED", "virtual_alpha": alpha, "virtual_rgb_l2": norm, "virtual_target_l2": target_norm}

def update(engine, record, successor, patch, optimizer, seed):
    """Return detached monitoring tensors. Advance the real Adam exactly once."""
    optimizer.zero_grad(set_to_none=True)
    current = engine.gradients(record, patch, seed, split=True)
    full, gd = current["full"], current["dino"]
    require(torch.isfinite(full).all() and torch.isfinite(gd).all(), "Nonfinite current RGB gradient")
    before = patch.detach().clone()
    virtual, query = None, None
    vm = {"virtual_status": "NO_EXPERT_SUCCESSOR", "virtual_alpha": None,
          "virtual_rgb_l2": None, "virtual_target_l2": None}
    if successor is not None:
        preview = preview_adamw(patch, full, optimizer)
        virtual, vm = virtual_patch(patch, gd, float((preview-before).norm()))
        if virtual is not None:
            leaf = virtual.detach().requires_grad_(True)
            query = engine.gradients(successor, leaf, seed, split=False)
            require(torch.isfinite(query["full"]).all(), "Nonfinite successor RGB gradient")
    q = None if query is None else query["full"]
    combined = full if q is None else (full+q)*.5
    patch.grad = combined.detach().clone()
    optimizer.step()
    with torch.no_grad():
        patch.clamp_(0, 1)
    after = patch.detach().clone(); delta = after-before
    row = {"current_uada_loss": float(current["loss"]), "query_virtual_uada_loss": None if query is None else float(query["loss"]),
           "used_query": query is not None, "current_gradient_l2": float(full.norm()), "dino_gradient_l2": float(gd.norm()),
           "query_gradient_l2": None if q is None else float(q.norm()), "combined_gradient_l2": float(combined.norm()),
           "current_dino_cosine": cosine(full, gd), "current_query_cosine": cosine(full, q), "dino_query_cosine": cosine(gd, q),
           "rgb_update_l2": float(delta.norm()), "rgb_update_mae": float(delta.abs().mean()),
           "current_gradient_dot_update": float((full*delta).sum()),
           "query_gradient_dot_update": None if q is None else float((q*delta).sum()),
           "combined_gradient_dot_update": float((combined*delta).sum()),
           "actual_step_current_descent_cosine": cosine(delta, -full),
           "quantized_rgb_changed_fraction": float((after.mul(255).round() != before.mul(255).round()).float().mean()),
           "virtual_quantized_changed_fraction": None if virtual is None else float((virtual.mul(255).round() != before.mul(255).round()).float().mean()),
           **vm}
    optimizer.zero_grad(set_to_none=True)
    tensors = {"before": before, "after": after, "virtual": virtual, "current_gradient": full,
               "dino_gradient": gd, "query_gradient": q, "combined_gradient": combined}
    return row, tensors

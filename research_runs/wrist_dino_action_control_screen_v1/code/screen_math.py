"""Pure tensor operations used by the bounded intervention screen."""
from __future__ import annotations
import hashlib
import torch
import torch.nn.functional as F


def group_dims(headwise: bool):
    return (-2, -1) if headwise else tuple()


def clean_scale(value: torch.Tensor, headwise: bool) -> torch.Tensor:
    if headwise:
        return value.float().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    return value.float().square().mean().sqrt().clamp_min(1e-6).reshape(*([1] * value.ndim))


def group_rms(value: torch.Tensor, headwise: bool) -> torch.Tensor:
    if headwise:
        return value.float().square().mean((-2, -1), keepdim=True).sqrt()
    return value.float().square().mean().sqrt().reshape(*([1] * value.ndim))


def project_relative_rms_(value: torch.Tensor, limit: float, headwise: bool) -> None:
    with torch.no_grad():
        rms = group_rms(value, headwise)
        value.mul_((limit / rms.clamp_min(1e-12)).clamp_max(1))


def matched_random(reference: torch.Tensor, seed: int, headwise: bool) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(reference.shape, generator=generator, dtype=torch.float32).to(reference.device)
    target = group_rms(reference.detach(), headwise)
    return random / group_rms(random, headwise).clamp_min(1e-12) * target


def normalized_descent_direction(gradient: torch.Tensor, limit: float, headwise: bool) -> torch.Tensor:
    denominator = group_rms(gradient, headwise)
    active = denominator > 1e-12
    return torch.where(active, -gradient.float() / denominator.clamp_min(1e-12) * limit,
                       torch.zeros_like(gradient, dtype=torch.float32))


def stable_seed(*parts) -> int:
    value = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**31 - 1)


def endpoint_target(record, endpoint: str, device) -> torch.Tensor | None:
    if endpoint == "wrong_open":
        return None
    if endpoint == "uada_opposite":
        expert = record["expert_actions"].to(device=device, dtype=torch.float32)
        return torch.where(expert >= 0, -torch.ones_like(expert), torch.ones_like(expert)).detach()
    raise RuntimeError(f"Unknown endpoint: {endpoint}")


def primary_loss(action: torch.Tensor, record, endpoint: str, margin: float, target=None) -> torch.Tensor:
    if endpoint == "wrong_open":
        mask = record["close_slots"].to(action.device)
        return (margin - action.reshape(8, 7)[mask, 6].float()).clamp_min(0).mean()
    if endpoint == "uada_opposite":
        return F.mse_loss(action[..., 0].float(), target[..., 0].float())
    raise RuntimeError(f"Unknown endpoint: {endpoint}")


def action_metrics(action: torch.Tensor, clean: torch.Tensor, record, endpoint: str, margin: float, target=None):
    action = action.detach().float().reshape(8, 7)
    clean = clean.detach().float().reshape(8, 7)
    delta = action - clean
    result = {"action_delta_l2": float(delta.norm()), "action_delta_rms": float(delta.square().mean().sqrt())}
    for axis in range(7):
        result[f"axis_{axis}_mean_abs_delta"] = float(delta[:, axis].abs().mean())
    mask = record["close_slots"].to(action.device)
    gripper = action[mask, 6]
    result.update(minimum_target_gripper=float(gripper.min()), mean_target_gripper=float(gripper.mean()),
                  wrong_open_slot_fraction=float((gripper > .5).float().mean()),
                  any_wrong_open=bool((gripper > .5).any()), all_wrong_open=bool((gripper > .5).all()),
                  all_open_margin=bool((gripper >= margin).all()))
    if endpoint == "uada_opposite":
        target = target.detach().float().reshape(8, 7)
        vector = target[:, 0] - clean[:, 0]
        result["target_direction_progress"] = float((delta[:, 0] * vector).sum() / vector.square().sum().clamp_min(1e-12))
    return result


def consensus(sum_gradient, sum_abs_gradient, norm_sum: float):
    if sum_gradient is None or norm_sum <= 0:
        return {"cancellation_ratio": 0., "element_sign_agreement": 0., "aggregate_norm": 0., "individual_norm_sum": norm_sum}
    aggregate = float(sum_gradient.float().norm())
    agreement = float((sum_gradient.abs() / sum_abs_gradient.clamp_min(1e-12)).mean())
    return {"cancellation_ratio": aggregate / norm_sum, "element_sign_agreement": agreement,
            "aggregate_norm": aggregate, "individual_norm_sum": norm_sum}


def cosine(a, b):
    if a is None or b is None:
        return 0.
    denominator = a.float().norm() * b.float().norm()
    return float((a.float() * b.float()).sum() / denominator.clamp_min(1e-12))

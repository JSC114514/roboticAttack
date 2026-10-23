"""Target geometry and transactional Adam. No simulator or native-policy imports."""
from __future__ import annotations
import copy
import torch
from common import require

def close_slots(expert, clean):
    return (expert.reshape(8, 7)[:, 6] < .5) & (clean.reshape(8, 7)[:, 6] < .5)

def open_loss(action, mask, target=.6):
    require(bool(mask.any()), "No dual-close slots")
    return (target - action.reshape(8, 7)[mask, 6].float()).clamp_min(0).max()

def action_metrics(action, mask):
    g = action.reshape(8, 7)[mask, 6].float()
    return {"minimum_target_gripper": float(g.min()), "wrong_open_slot_fraction": float((g > .5).float().mean()),
            "all_slots_wrong_open": bool((g > .5).all()), "all_slots_margin": bool((g >= .6).all())}

def project_rms(z, limit=.2):
    with torch.no_grad():
        rms = z.square().mean((-2, -1), keepdim=True).sqrt()
        z.mul_((limit / rms.clamp_min(1e-12)).clamp_max(1))

def feature_loss(current, clean, target, scale, method):
    x, t = (current.float()-clean)/scale, (target-clean)/scale
    energy = t.square().mean(-1, keepdim=True)
    require(float(energy.sum()) > 1e-12, "No nonzero target")
    weight = energy / energy.sum()
    if method == "key_point": return (weight * (x-t).square()).sum() / x.shape[-1]
    require(method == "key_direction", "Unknown Key loss")
    progress = (weight*x*t).sum() / (weight*t.square()).sum().clamp_min(1e-12)
    return (1-progress).clamp_min(0).square()

def transactional_step(patch, optimizer, evaluate, before_key, before_open, scales=(1., .5, .25)):
    """One moment update if accepted; rejected trials never accumulate moments or pixels."""
    before = patch.detach().clone(); state = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    delta = patch.detach().clone()-before
    trials = []
    for alpha in scales:
        with torch.no_grad(): patch.copy_((before + alpha*delta).clamp(0, 1))
        key, action = evaluate(patch)
        require(torch.isfinite(torch.tensor([key, action])).all(), "Nonfinite native acceptance loss")
        trials.append({"alpha": alpha, "key_loss": key, "open_loss": action})
        if key < before_key and action <= before_open:
            return {"accepted": True, "alpha": alpha, "trials": trials}
    with torch.no_grad(): patch.copy_(before)
    optimizer.load_state_dict(state)
    return {"accepted": False, "alpha": 0., "trials": trials}

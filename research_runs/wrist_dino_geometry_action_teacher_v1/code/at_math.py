"""Loss definitions and budget policy, independent of CUDA and the VLA loader."""
import math
import torch

def far_boundary(clean):
    low=-torch.ones_like(clean); high=torch.ones_like(clean)
    low[...,6]=0.
    # Farther endpoint; upper wins ties. Gripper retains native [0,1].
    return torch.where((clean-low).abs() > (high-clean).abs(),low,high).detach()

def objective(action, teacher, method):
    if method=='geometry_teacher': return (action[...,:3]-teacher['target'].to(action)[...,:3]).square().mean()
    if method=='clean_far_boundary': return (action-far_boundary(teacher['clean'].to(action))).square().mean()
    raise ValueError(method)

def learning_rate(step, total, warmup, peak):
    if step<=warmup: return peak*step/warmup
    return peak*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))

def paired_checkpoint(available_a, available_b):
    common=set(available_a)&set(available_b)
    return max(common) if common else None

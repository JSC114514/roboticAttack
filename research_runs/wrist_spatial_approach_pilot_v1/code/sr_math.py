"""Geometric command targets; commands are never called realized motion."""
import math
import numpy as np

def retreat_target(clean, unit, lower, upper, margin):
    """Euclidean projection onto box intersected with u.dot(v)<=-margin."""
    v=np.asarray(clean,dtype=np.float64);u=np.asarray(unit,dtype=np.float64)
    lo=np.asarray(lower,dtype=np.float64);hi=np.asarray(upper,dtype=np.float64)
    if v.shape[-1]!=3 or u.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid command/direction')
    if not np.isclose(np.linalg.norm(u),1,atol=1e-7) or not (np.all(lo<hi) and margin>0):raise ValueError('Invalid bounds')
    if np.sum(np.where(u>=0,lo,hi)*u)>-margin:raise ValueError('Infeasible retreat')
    def project(lam):return np.clip(v-lam[...,None]*u,lo,hi)
    a=np.zeros(v.shape[:-1]);b=np.ones_like(a)
    while np.any(np.sum(project(b)*u,axis=-1)>-margin):b*=2
    for _ in range(64):
        mid=(a+b)*.5;bad=np.sum(project(mid)*u,axis=-1)>-margin
        a=np.where(bad,mid,a);b=np.where(bad,b,mid)
    original=np.clip(v,lo,hi)
    return np.where((np.sum(original*u,axis=-1)<=-margin)[...,None],original,project(b)).astype(np.float32)

def far_target(clean,lower,upper):
    v=np.asarray(clean);lo=np.asarray(lower);hi=np.asarray(upper)
    return np.where(abs(v-lo)>abs(hi-v),lo,hi).astype(np.float32)

def affine_actions(action,low,high,mask):
    import torch
    lo=torch.as_tensor(low,device=action.device,dtype=action.dtype)
    hi=torch.as_tensor(high,device=action.device,dtype=action.dtype)
    mask=torch.as_tensor(mask,device=action.device,dtype=torch.bool)
    return torch.where(mask,.5*(action+1)*(hi-lo+1e-8)+lo,action)

def objective(command,teacher,method):
    target=teacher['targets'][method].to(command)
    scale=teacher['command_scale'].to(command)
    return ((command-target)/scale).square().mean()

def learning_rate(step,total,warmup,peak):
    if step<=warmup:return peak*step/warmup
    return peak*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))

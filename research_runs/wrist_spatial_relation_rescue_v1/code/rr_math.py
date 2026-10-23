"""Restoration acts on native executed controller inputs, not normalized heads."""
import numpy as np

def radial_projection(patch,clean,unit,scale=.05):
    """Closest point in [-scale,scale]^3 with u.v == u.clean (per slot)."""
    original=np.clip(np.asarray(patch,dtype=np.float64),-1,1)
    p=original*scale
    c=np.clip(np.asarray(clean,dtype=np.float64),-1,1)*scale
    u=np.asarray(unit,dtype=np.float64)
    if p.shape!=c.shape or p.shape[-1]!=3 or u.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(c).all():raise ValueError('Invalid XYZ')
    if not np.isfinite(u).all() or abs(np.linalg.norm(u)-1)>1e-7:raise ValueError('Invalid direction')
    target=c@u
    def point(lam):return np.clip(p+lam[...,None]*u,-scale,scale)
    lo=np.full(p.shape[:-1],-1.);hi=-lo
    while np.any(point(lo)@u>target):lo*=2
    while np.any(point(hi)@u<target):hi*=2
    for _ in range(80):
        mid=(lo+hi)/2;bad=point(mid)@u<target;lo=np.where(bad,mid,lo);hi=np.where(bad,hi,mid)
    result=point((lo+hi)/2)
    # Preserve exact zero change when the constraint already holds.
    return np.where((np.abs(p@u-target)<1e-14)[...,None],original,result/scale)

def restore(patch,clean,unit,method,epsilon=1e-8):
    p=np.asarray(patch,dtype=np.float64);c=np.asarray(clean,dtype=np.float64)
    if p.shape!=(8,7) or c.shape!=(8,7) or not np.isfinite(p).all() or not np.isfinite(c).all():raise ValueError('Expected finite native 8x7 actions')
    out=p.copy();u=np.asarray(unit,dtype=float);degenerate=np.linalg.norm(u)<=epsilon
    meta={'degenerate_direction':bool(degenerate),'method':method,'radial_error_m':None,'tangential_change_m':None}
    if method=='restore_radial' and not degenerate:
        u=u/np.linalg.norm(u);out[:,:3]=radial_projection(p[:,:3],c[:,:3],u)
        delta=.05*(out[:,:3]-np.clip(p[:,:3],-1,1))
        meta.update(radial_error_m=float(np.max(np.abs((np.clip(out[:,:3],-1,1)-np.clip(c[:,:3],-1,1))@u))*.05),
                    tangential_change_m=float(np.linalg.norm(delta-np.outer(delta@u,u),axis=1).max()))
    elif method=='restore_xyz':out[:,:3]=np.clip(c[:,:3],-1,1)
    elif method=='restore_pose_grip':out[:,3:]=c[:,3:]
    elif method!='restore_radial':raise ValueError('Unknown restoration')
    # Solve in float64; deployment still passes through the native float32 processor.
    return out,meta

def executed_radial_error(environment_action,clean_native,unit,solver_tolerance=1e-9):
    u=np.asarray(unit,dtype=float);u=u/np.linalg.norm(u)
    a=np.clip(np.asarray(environment_action,dtype=np.float64)[:3],-1,1)
    c=np.clip(np.asarray(clean_native,dtype=np.float64)[:3],-1,1)
    error=float(abs((a-c)@u)*.05)
    # Conservative IEEE float32 roundoff bound for valid normalized controller inputs.
    bound=float(.05*np.finfo(np.float32).eps*np.abs(u).sum()+solver_tolerance)
    return error,bound

def orientation_error_radians(goal,actual):
    relative=np.asarray(goal)@np.asarray(actual).T
    return float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))

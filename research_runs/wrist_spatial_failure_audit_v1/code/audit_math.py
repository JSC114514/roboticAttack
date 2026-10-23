"""Pure diagnostic math, with no simulator or policy import."""
import numpy as np
from PIL import Image

def decode_actions(value):
    a=np.stack([np.asarray(x,dtype=np.float32) for x in value])
    if a.shape!=(8,7) or not np.isfinite(a).all(): raise ValueError('Expected finite 8x7 native action chunk')
    return a

def visibility(mask, patched, xy=(26,137)):
    """Categorical masks: NN 256->224; exact normalized 0.9 crop sampling grid.

    Visibility is a segmentation estimate, not JPEG color resampling. This
    avoids manufacturing/interpolating instance identities at object edges.
    """
    m=np.asarray(Image.fromarray(np.asarray(mask,dtype=np.uint8)*255).resize((224,224),Image.Resampling.NEAREST))>0
    keep=m.copy(); x,y=xy
    if patched: keep[y:y+50,x:x+50]=False
    side=np.float32(np.sqrt(np.float32(.9))); lo=np.float32((1-side)/2)
    indices=np.clip(np.rint(np.linspace(lo,lo+side,224,dtype=np.float32)*223).astype(int),0,223)
    fov=m[np.ix_(indices,indices)]; remaining=keep[np.ix_(indices,indices)]
    return {'raw_visible_pixels':int(np.asarray(mask).sum()),'policy_fov_pixels':int(fov.sum()),
            'policy_uncovered_pixels':int(remaining.sum()),'patch_covered_pixels':int(fov.sum()-remaining.sum())}

def event_summary(frame):
    """Descriptive timestamps only. Ordering does not establish causality."""
    def first(column):
        v=frame[frame[column].fillna(False).astype(bool)]
        return int(v.iloc[0]['step']) if len(v) else None
    valid=frame[frame.visibility_verified.astype(bool)]
    def first_zero(column):
        g=valid[valid[column]==0]; return int(g.iloc[0]['step']) if len(g) else None
    patch_only=valid[(valid.policy_fov_pixels>0)&(valid.policy_uncovered_pixels==0)]
    return {'first_contact_step':first('contact'),'first_grasp_step':first('grasp'),
        'first_not_visible_in_policy_fov_step':first_zero('policy_fov_pixels'),
        'first_fully_hidden_after_overlay_step':first_zero('policy_uncovered_pixels'),
        'first_fully_occluded_by_patch_step':int(patch_only.iloc[0]['step']) if len(patch_only) else None,
        'visibility_verified_frames':int(len(valid)),
        'uncovered_visibility_fraction':float((valid.policy_uncovered_pixels>0).mean()) if len(valid) else None,
        'minimum_distance_m':float(frame.distance_m.min()),'final_distance_m':float(frame.iloc[-1].distance_m)}

def paired_interval(pairs, draws=5000, seed=42):
    if not pairs: return None
    groups={}
    for p in pairs: groups.setdefault((p['suite'],p['task_id']),[]).append(p['difference'])
    tasks=sorted(groups); rng=np.random.default_rng(seed); values=[]
    for _ in range(draws):
        sample=[]
        for j in rng.integers(0,len(tasks),len(tasks)):
            g=groups[tasks[j]]; sample.extend(rng.choice(g,len(g),replace=True))
        values.append(float(np.mean(sample)))
    return np.quantile(values,[.025,.975]).tolist()

def frame_alignment(current, expected, tolerance):
    """Check actual recorded movement, not controller goal displacement."""
    a=np.asarray(current,float); b=np.asarray(expected,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Malformed physical coordinates')
    error=float(np.max(np.abs(a-b)))
    if error>tolerance: raise RuntimeError(f'Replay physical mismatch: {error:.3g} m > {tolerance:.3g}')
    return error

"""Pure label geometry, native crop alignment and trajectory-aware statistics."""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from atlas_common import require


def rgb_error(reference,candidate):
    a,b=np.asarray(reference),np.asarray(candidate)
    require(a.shape==b.shape==(224,224,3) and a.dtype==b.dtype==np.uint8,'RGB schema changed')
    d=(a.astype(np.float64)-b.astype(np.float64))/255.
    return {'normalized_rms':float(np.sqrt(np.mean(d*d))),
            'max_abs_255':int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max()),
            'changed_elements':int(np.count_nonzero(d))}


def rgb_to_lab(rgb):
    x=np.asarray(rgb,dtype=float)
    linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    xyz=linear@np.array([[.4124564,.2126729,.0193339],
                         [.3575761,.7151522,.1191920],[.1804375,.0721750,.9503041]])
    xyz/=np.array([.95047,1.,1.08883]); delta=6/29
    f=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta**2)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)


def crop_map(array,nearest=False):
    """Same TF crop-and-resize sample coordinates as native sqrt(.9) crop."""
    from scipy.ndimage import map_coordinates
    a=np.asarray(array); size=a.shape[0]
    require(a.shape[:2]==(size,size),'Expected square label map')
    low=(1-math.sqrt(.9))*(size-1)/2
    axis=np.linspace(low,size-1-low,224)
    yy,xx=np.meshgrid(axis,axis,indexing='ij')
    if a.ndim==2: return map_coordinates(a,[yy,xx],order=0 if nearest else 1,mode='nearest')
    return np.stack([map_coordinates(a[:,:,k],[yy,xx],order=0 if nearest else 1,mode='nearest')
                     for k in range(a.shape[2])],-1)


def patch_reduce(array):
    a=np.asarray(array)
    require(a.shape[:2]==(224,224),'Expected processor-space map')
    return a.reshape(16,14,16,14,*a.shape[2:]).mean((1,3)).reshape(256,*a.shape[2:])


def patch_majority(labels,classes):
    counts=np.stack([patch_reduce((np.asarray(labels)==k).astype(np.float32)) for k in range(classes)],-1)
    return counts.argmax(-1).astype(np.int16),counts.max(-1)


def pooled_grid(tokens,grid=4):
    import torch.nn.functional as F
    x=tokens.detach().float().reshape(16,16,-1).permute(2,0,1)[None]
    return F.adaptive_avg_pool2d(x,(grid,grid)).permute(0,2,3,1).flatten().cpu().numpy().astype(np.float16)


def grouped_folds(rows,folds=5):
    assignment={}
    require(len({r['episode_id'] for r in rows})>=2,'Need at least two training trajectories')
    for t in sorted({r['task_id'] for r in rows}):
        episodes=sorted({r['episode_id'] for r in rows if r['task_id']==t})
        assignment.update({name:(i+2*t)%folds for i,name in enumerate(episodes)})
    return np.asarray([assignment[r['episode_id']] for r in rows])


def permuted_targets(y,rows,seed):
    """Move whole five-state trajectories; never shuffle individual adjacent frames."""
    rng=np.random.default_rng(seed); out=np.array(y,copy=True)
    names=sorted({r['episode_id'] for r in rows})
    lookup={n:sorted([i for i,r in enumerate(rows) if r['episode_id']==n],key=lambda i:rows[i]['step_id']) for n in names}
    order=rng.permutation(names)
    for dst,src in zip(names,order):
        require(len(lookup[dst])==len(lookup[src]),'Trajectory length mismatch')
        out[lookup[dst]]=y[lookup[src]]
    return out


def angular_error(truth,prediction):
    a=np.asarray(truth); b=np.asarray(prediction)
    valid=np.isfinite(a).all(-1)&np.isfinite(b).all(-1)&(np.linalg.norm(a,axis=-1)>1e-12)&(np.linalg.norm(b,axis=-1)>1e-12)
    a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-12)
    return np.where(valid,np.degrees(np.arccos(np.clip((a*b).sum(-1),-1,1))),np.nan)


def rotation_error(z_true,x_true,z_pred,x_pred):
    def matrix(z,x):
        zn=np.linalg.norm(z,axis=-1,keepdims=True); z=z/np.maximum(zn,1e-12)
        x=x-(x*z).sum(-1,keepdims=True)*z; xn=np.linalg.norm(x,axis=-1,keepdims=True)
        x=x/np.maximum(xn,1e-12); y=np.cross(z,x)
        return np.stack([x,y,z],-1),(zn[:,0]>1e-12)&(xn[:,0]>1e-12)
    a,va=matrix(z_true,x_true); b,vb=matrix(z_pred,x_pred)
    cosine=((a*b).sum((1,2))-1)/2
    return np.where(va&vb,np.degrees(np.arccos(np.clip(cosine,-1,1))),np.nan)


def balanced_token_indices(labels,limit,seed):
    rng=np.random.default_rng(seed); labels=np.asarray(labels)
    order=[rng.permutation(np.where(labels==k)[0]).tolist() for k in np.unique(labels) if k>=0]
    result=[]
    while len(result)<limit and any(order):
        for group in order:
            if group and len(result)<limit: result.append(group.pop())
    return np.asarray(result,dtype=int)


def cluster_interval(values,rows,draws=2000,seed=42):
    """Task -> independent episode bootstrap of paired per-frame quantities."""
    rng=np.random.default_rng(seed); values=np.asarray(values,dtype=float)
    grouped={}
    for task in sorted({r['task_id'] for r in rows}):
        group=[]
        for ep in sorted({r['episode_id'] for r in rows if r['task_id']==task}):
            v=values[[i for i,r in enumerate(rows) if r['episode_id']==ep]]
            if np.isfinite(v).any(): group.append(float(np.nanmean(v)))
        if group: grouped[task]=np.asarray(group)
    if not grouped: return {'mean':None,'ci95_low':None,'ci95_high':None,'tasks':0,'episodes':0}
    keys=list(grouped)
    # Bootstrap episode means within each task, then resample tasks.
    task_draw=rng.integers(0,len(keys),size=(draws,len(keys)))
    sampled=np.empty_like(task_draw,dtype=float)
    for i,v in enumerate(grouped.values()):
        positions=np.where(task_draw==i)
        sampled[positions]=v[rng.integers(0,len(v),size=(len(positions[0]),len(v)))].mean(1)
    samples=sampled.mean(1)
    return {'mean':float(np.mean([v.mean() for v in grouped.values()])),
            'ci95_low':float(np.quantile(samples,.025)),'ci95_high':float(np.quantile(samples,.975)),
            'tasks':len(keys),'episodes':sum(map(len,grouped.values()))}


@lru_cache(maxsize=8)
def _binary_weights(key,draws,seed):
    """Weights for task -> independent episode draws; frames retain within-episode means."""
    rng=np.random.default_rng(seed); episodes=sorted(set(key)); tasks=sorted({t for t,e in key})
    lookup={v:i for i,v in enumerate(episodes)}; owner=np.asarray([lookup[k] for k in key])
    counts=np.bincount(owner,minlength=len(episodes)); w=np.zeros((draws,len(episodes)),np.float32)
    point=np.zeros(len(episodes)); td=rng.integers(0,len(tasks),size=(draws,len(tasks)))
    for i,t in enumerate(tasks):
        ids=np.asarray([j for j,(task,e) in enumerate(episodes) if task==t]); where=np.where(td==i)
        sampled=ids[rng.integers(0,len(ids),size=(len(where[0]),len(ids)))]
        np.add.at(w,(np.repeat(where[0],len(ids)),sampled.ravel()),1/(len(tasks)*len(ids)))
        point[ids]=1/(len(tasks)*len(ids))
    return np.vstack([point[owner]/counts[owner],w[:,owner]/counts[owner]])


def binary_intervals(y,p,rows,draws=2000,seed=42):
    """Hierarchical intervals for ranking and binary decisions, including tied scores."""
    y=np.asarray(y); p=np.asarray(p); keep=np.isfinite(y)&np.isfinite(p)
    y=y[keep]; p=p[keep]
    empty={k:None for k in ('auroc_cluster_mean','auroc_ci95_low','auroc_ci95_high',
                            'balanced_accuracy_cluster_mean','balanced_accuracy_ci95_low','balanced_accuracy_ci95_high')}
    if len(np.unique(y))<2: return empty
    key=tuple((r['task_id'],r['episode_id']) for r,k in zip(rows,keep) if k)
    weights=_binary_weights(key,draws,seed); pos=y==1; neg=~pos
    order=np.argsort(p,kind='stable'); starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    positive=np.add.reduceat(weights[:,order]*pos[order],starts,axis=1)
    negative=np.add.reduceat(weights[:,order]*neg[order],starts,axis=1)
    npos=positive.sum(1); nneg=negative.sum(1)
    denom=npos*nneg; numerator=(positive*(np.cumsum(negative,axis=1)-negative/2)).sum(1)
    auc=np.divide(numerator,denom,out=np.full(len(weights),np.nan),where=denom>0)
    tp=(weights*((p>=.5)&pos)).sum(1); tn=(weights*((p<.5)&neg)).sum(1)
    ba=np.full(len(weights),np.nan); valid=(npos>0)&(nneg>0); ba[valid]=(tp[valid]/npos[valid]+tn[valid]/nneg[valid])/2
    result={}
    for name,value in [('auroc',auc),('balanced_accuracy',ba)]:
        samples=value[1:]; samples=samples[np.isfinite(samples)]
        result[name+'_cluster_mean']=float(value[0]) if np.isfinite(value[0]) else None
        result[name+'_ci95_low']=float(np.quantile(samples,.025)) if len(samples) else None
        result[name+'_ci95_high']=float(np.quantile(samples,.975)) if len(samples) else None
        result[name+'_valid_bootstrap_draws']=len(samples)
    return result

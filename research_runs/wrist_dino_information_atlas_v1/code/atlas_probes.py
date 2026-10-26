"""Small frozen-feature readouts. All fitting and tuning use training episodes only."""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from atlas_common import config,require
from atlas_math import grouped_folds


def _standardize(x,other):
    x=np.asarray(x,dtype=np.float64); other=np.asarray(other,dtype=np.float64)
    mean=x.mean(0); scale=x.std(0); scale=np.where(scale>1e-6,scale,1.)
    return (x-mean)/scale,(other-mean)/scale


def ridge_path(x,y,test,alphas):
    """Dual ridge path, sharing eigensystems among target columns with identical missingness."""
    x,test=_standardize(x,test); y=np.asarray(y,dtype=float)
    alphas=np.asarray(alphas); out=np.full((len(test),y.shape[1],len(alphas)),np.nan)
    gram=x@x.T/max(x.shape[1],1)
    cross=test@x.T/max(x.shape[1],1)
    masks={}
    for col in range(y.shape[1]):
        mask=np.isfinite(y[:,col]); masks.setdefault(mask.tobytes(),[mask,[]])[1].append(col)
    for mask,columns in masks.values():
        if mask.sum()<2: continue
        target=y[np.ix_(mask,columns)]; mean=target.mean(0); scale=target.std(0)
        scale=np.where(scale>1e-8,scale,1.); target=(target-mean)/scale
        if x.shape[1]==0:
            out[:,columns,:]=mean[None,:,None]; continue
        # Normalize by feature count to keep alpha's scale consistent across sites.
        g=gram[np.ix_(mask,mask)]; c=cross[:,mask]
        gm=g.mean(0); grand=float(gm.mean())
        g=g-gm[None,:]-gm[:,None]+grand
        c=c-c.mean(1,keepdims=True)-gm[None,:]+grand
        eigen,vectors=eigh(g,check_finite=False)
        rotated=vectors.T@target; transfer=c@vectors
        for i,alpha in enumerate(alphas):
            pred=transfer@(rotated/(np.maximum(eigen,0)[:,None]+alpha))
            out[:,columns,i]=pred*scale+mean
    return out


def fit_ridge(x,y,test,rows,alphas=None):
    alphas=config()['probe']['ridge_alphas'] if alphas is None else alphas
    folds=grouped_folds(rows,config()['probe']['folds']); y=np.asarray(y,dtype=float)
    losses=np.zeros((y.shape[1],len(alphas))); count=np.zeros(y.shape[1])
    for fold in sorted(set(folds)):
        train=folds!=fold; valid=~train
        pred=ridge_path(x[train],y[train],x[valid],alphas)
        mean=np.nanmean(y[train],0); std=np.nanstd(y[train],0); std=np.where(std>1e-8,std,1.)
        error=((pred-y[valid,:,None])/std[None,:,None])**2
        losses+=np.nansum(error,axis=0); count+=np.isfinite(y[valid]).sum(0)
    scores=losses/np.maximum(count[:,None],1)
    best=scores.argmin(1)
    paths=ridge_path(x,y,test,alphas)
    pred=paths[:,np.arange(y.shape[1]),best]
    return pred,{'alpha':[float(alphas[i]) for i in best],
                 'cv_normalized_mse':scores[np.arange(len(best)),best].tolist(),
                 'cv_episode_grouped':True,'training_episodes':len({r['episode_id'] for r in rows})}


def local_ridge(x,y,test,alphas,folds,weights=None):
    """Primal ridge for many native tokens, grouped by source trajectory in CV."""
    def path(a,b,c,w):
        mean_x=a.mean(0); scale_x=a.std(0); scale_x=np.where(scale_x>1e-6,scale_x,1.)
        aa=(a-mean_x)/scale_x/np.sqrt(a.shape[1])
        mean=b.mean(0); std=b.std(0); std=np.where(std>1e-8,std,1.)
        w=np.ones(len(a)) if w is None else np.asarray(w)
        # Center after weighting, including an unpenalized intercept.
        xm=np.average(aa,axis=0,weights=w); ym=np.average((b-mean)/std,axis=0,weights=w)
        aa-=xm; yy=(b-mean)/std-ym
        gram=(aa.T*w)@aa; cross=(aa.T*w)@yy
        e,v=eigh(gram,check_finite=False); rotated=v.T@cross
        coefficients=[v@(rotated/(np.maximum(e,0)[:,None]+alpha)) for alpha in alphas]
        result=np.empty((len(c),b.shape[1],len(alphas)))
        for start in range(0,len(c),1024):
            cc=(np.asarray(c[start:start+1024],dtype=float)-mean_x)/scale_x/np.sqrt(a.shape[1])-xm
            result[start:start+1024]=np.stack([(cc@coef+ym)*std+mean for coef in coefficients],-1)
        return result
    losses=np.zeros((y.shape[1],len(alphas))); counts=0
    for f in sorted(set(folds)):
        keep=folds!=f
        prediction=path(x[keep],y[keep],x[~keep],None if weights is None else weights[keep])
        scale=y[keep].std(0); scale=np.where(scale>1e-8,scale,1.)
        losses+=(((prediction-y[~keep,:,None])/scale[None,:,None])**2).sum(0)
        counts+=(~keep).sum()
    losses/=max(counts,1); best=losses.argmin(1)
    all_pred=path(x,y,test,weights)
    return all_pred[:,np.arange(y.shape[1]),best],{'alpha':[alphas[i] for i in best],
                                               'cv_normalized_mse':losses[np.arange(len(best)),best].tolist()}


def fit_mlp(x,y,test,rows,seeds=None):
    """Fixed tiny nonlinear check on the consumed output, not on selected best sites."""
    import torch
    from torch import nn
    cfg=config()['probe']['mlp']; folds=grouped_folds(rows,5)
    # Reserve an episode-grouped fifth of Discovery for early stopping; Validation is untouched.
    train=folds!=0; valid=~train
    mean_x=x[train].mean(0); scale_x=x[train].std(0); scale_x=np.where(scale_x>1e-6,scale_x,1.)
    x=(x-mean_x)/scale_x; test=(test-mean_x)/scale_x
    supported=np.isfinite(y[train]).sum(0)>=2
    means=np.nan_to_num(np.nanmean(y[train],0)); scales=np.nanstd(y[train],0); scales=np.where(scales>1e-8,scales,1.)
    target=(y-means)/scales; mask=np.isfinite(target)
    target=np.nan_to_num(target)
    xx=torch.as_tensor(x,dtype=torch.float32); tt=torch.as_tensor(test,dtype=torch.float32)
    yy=torch.as_tensor(target,dtype=torch.float32); mm=torch.as_tensor(mask,dtype=torch.float32)
    train_t=torch.as_tensor(train); valid_t=~train_t
    output=[]; details=[]
    for seed in (cfg['seeds'] if seeds is None else seeds):
        torch.manual_seed(seed)
        net=nn.Sequential(nn.Linear(x.shape[1],cfg['width']),nn.ReLU(),nn.Linear(cfg['width'],y.shape[1]))
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        best=float('inf'); saved=None; stale=0; epoch=0
        for epoch in range(cfg['max_epochs']):
            net.train(); opt.zero_grad(set_to_none=True); prediction=net(xx[train_t])
            loss=((prediction-yy[train_t]).square()*mm[train_t]).sum()/mm[train_t].sum().clamp_min(1)
            loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                score=float(((net(xx[valid_t])-yy[valid_t]).square()*mm[valid_t]).sum()/mm[valid_t].sum().clamp_min(1))
            if score<best:
                best=score; saved={k:v.detach().clone() for k,v in net.state_dict().items()}; stale=0
            else: stale+=1
            if stale>=cfg['patience']: break
        require(saved is not None,'MLP never produced a finite training-fold score')
        net.load_state_dict(saved); net.eval()
        with torch.no_grad():
            p=net(tt).numpy()*scales+means; p[:,~supported]=np.nan; output.append(p)
        details.append({'seed':seed,'epochs':epoch+1,'training_holdout_mse':best})
    return np.stack(output),details


def metrics(y,p,binary=False):
    y=np.asarray(y); p=np.asarray(p)
    keep=np.isfinite(y)&np.isfinite(p); y=np.asarray(y)[keep]; p=np.asarray(p)[keep]
    if not len(y): return {'n':0,'mae':None,'r2':None,'auroc':None,'balanced_accuracy':None}
    denom=float(((y-y.mean())**2).sum())
    result={'n':len(y),'mae':float(abs(y-p).mean()),'mse':float(((y-p)**2).mean()),
            'r2':float(1-((y-p)**2).sum()/denom) if denom>0 else None}
    if binary:
        from scipy.stats import rankdata
        pos=y==1; npos=int(pos.sum()); nneg=len(y)-npos
        result['auroc']=float((rankdata(p)[pos].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else None
        recalls=[np.mean((p[y==v]>=.5)==bool(v)) for v in np.unique(y)]
        result['balanced_accuracy']=float(np.mean(recalls)) if npos and nneg else None
    return result

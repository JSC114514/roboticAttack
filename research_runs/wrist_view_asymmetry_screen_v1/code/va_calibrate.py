"""Train-trajectory-only PCA/autoregression; validation never chooses anything."""
import numpy as np
from scipy.ndimage import laplace
from va_common import ROOT,SOURCE,config,read,atomic,finish,complete,require

def transitions(meta,allowed):
    allowed=set(allowed);result=[];groups={}
    for i in allowed:groups.setdefault(meta[i]['episode_id'],[]).append(i)
    for indices in groups.values():
        indices.sort(key=lambda i:meta[i]['decision_block'])
        for a,b in zip(indices,indices[1:]):
            if meta[b]['decision_block']!=meta[a]['decision_block']+1:raise ValueError('Nonconsecutive cached transition')
            result.append((a,b))
    return np.asarray(result,dtype=int)

def fit_model(raw,proprio,meta,allowed,alpha):
    allowed=np.asarray(allowed,int);pairs=transitions(meta,allowed)
    pca_mean=raw[allowed].mean(0)
    _,_,components=np.linalg.svd(raw[allowed]-pca_mean,full_matrices=False)
    components=components[:config()['pca_components']]
    require(len(components)==config()['pca_components'],'Too few fitting observations for PCA8')
    # Deterministic PCA signs; same eigenbasis convention in final and CV fits.
    pivots=np.argmax(abs(components),axis=1);signs=np.sign(components[np.arange(len(components)),pivots]);components=components*signs[:,None]
    z=(raw-pca_mean)@components.T;x=np.c_[z[pairs[:,0]],proprio[pairs[:,1]]-proprio[pairs[:,0]]];y=z[pairs[:,1]]
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
    xn=(x-mean)/scale;ymean=y.mean(0)
    coef=np.linalg.solve(xn.T@xn+alpha*np.eye(xn.shape[1]),xn.T@(y-ymean))
    return dict(pca_mean=pca_mean,components=components,xmean=mean,xscale=scale,coef=coef,ymean=ymean)

def group_folds(groups,folds=5):
    names=sorted(set(groups));require(len(names)>=folds,'Too few fitting trajectories for five-fold CV')
    assignment={g:i%folds for i,g in enumerate(names)}
    for fold in range(folds):
        valid=np.array([assignment[g]==fold for g in groups]);yield np.flatnonzero(~valid),np.flatnonzero(valid)

def prediction_errors(raw,proprio,pairs,model):
    z=(raw-model['pca_mean'])@model['components'].T
    x=np.c_[z[pairs[:,0]],proprio[pairs[:,1]]-proprio[pairs[:,0]]]
    return np.linalg.norm(z[pairs[:,1]]-(((x-model['xmean'])/model['xscale'])@model['coef']+model['ymean']),axis=1)

def run(monitor):
    if complete(ROOT/'reports/CALIBRATION_COMPLETE.json'):return
    require(complete(ROOT/'reports/SHARED_COMPLETE.json'),'Shared capture incomplete')
    meta=read(ROOT/'data/FEATURE_ROWS.json');fit=np.array([i for i,r in enumerate(meta) if r['cohort']=='fit'])
    groups=np.array([meta[i]['episode_id'] for i in fit]);require(len(fit)==58 and len(set(groups))==12,'Wrong calibration split')
    with np.load(ROOT/'data/FEATURES.npz',allow_pickle=False) as z:features=z['features'];proprio=z['proprio']
    output={};report={'fit_state_ids':[meta[i]['state_id'] for i in fit], 'fit_trajectories':sorted(set(groups)),
                      'validation_used':False,'thresholds':'clean out-of-fold residual quantiles; 90/95% are reporting operating points, not scientific pass gates'}
    for vi,view in enumerate(('front','wrist')):
        raw=features[:,vi];scores={};errors_by_alpha={}
        for alpha in config()['ridge_alphas']:
            errors=[]
            for train,valid in group_folds(groups,config()['folds']):
                # PCA and normalization are refit inside EACH training fold.
                m=fit_model(raw,proprio,meta,fit[train],alpha)
                pairs=transitions(meta,fit[valid]);errors.extend(prediction_errors(raw,proprio,pairs,m).tolist())
            scores[str(alpha)]=float(np.mean(np.square(errors)));errors_by_alpha[alpha]=np.asarray(errors)
        chosen=min(config()['ridge_alphas'],key=lambda a:(scores[str(a)],a))
        model=fit_model(raw,proprio,meta,fit,chosen);errors=errors_by_alpha[chosen];scale=max(float(np.median(errors)),1e-8)
        model['residual_scale']=np.full(config()['pca_components'],scale)
        for k,v in model.items():output[view+'__'+k]=v
        pairs=transitions(meta,fit);drift=np.linalg.norm(raw[pairs[:,1]]-raw[pairs[:,0]],axis=1)
        report[view]={'alpha':chosen,'group_cv_mse':scores,'oof_transitions':len(errors),
                      'residual_threshold_90':float(np.quantile(errors/scale,.90)),
                      'residual_threshold_95':float(np.quantile(errors/scale,.95)),
                      'drift_threshold_90':float(np.quantile(drift,.90)),
                      'drift_threshold_95':float(np.quantile(drift,.95))}
        monitor.update(phase='train_trajectory_group_cv',completed=vi+1,total=2)
    # Sharpness calibration uses the same 58 clean fitting images only.
    import torch
    sharpness={'front':[],'wrist':[]}
    for e in read(ROOT/'data/MANIFEST.json')['episodes']:
        if e['cohort']!='fit':continue
        rows=torch.load(SOURCE/'data/cache'/(e['episode_id']+'.pt'),map_location='cpu',weights_only=False)['records']
        for row in rows:
            for view,key in (('front','full_image'),('wrist','wrist_image')):
                sharpness[view].append(float(laplace(row['observation'][key].astype(float).mean(2)).var()))
    for view in sharpness:
        report[view]['sharpness_lower_05']=float(np.quantile(sharpness[view],.05))
        report[view]['sharpness_lower_10']=float(np.quantile(sharpness[view],.10))
    np.savez_compressed(ROOT/'data/CALIBRATION.npz',**output);atomic(report,ROOT/'data/CALIBRATION.json')
    finish(ROOT/'reports/CALIBRATION_COMPLETE.json',[ROOT/'data/CALIBRATION.npz',ROOT/'data/CALIBRATION.json'])

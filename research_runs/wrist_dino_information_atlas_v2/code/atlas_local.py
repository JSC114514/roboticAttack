"""Shared native-token evaluation and matched controls for classes, depth and parts."""
from __future__ import annotations
import numpy as np
import time
from atlas_common import atomic_json,config,progress,require,resources
from atlas_math import cluster_interval,grouped_folds,permuted_targets
from atlas_probes import local_ridge


def token_metrics(prediction,truth,valid,rows,class_names,depth=None):
    n=len(rows); classes=len(class_names)
    predicted=prediction[:,:classes].argmax(1).reshape(n,256)
    correct=np.where(valid,predicted==truth,np.nan)
    denominator=valid.sum(1)
    accuracy=np.divide(np.nansum(correct,1),denominator,out=np.full(n,np.nan),where=denominator>0)
    confusion=np.zeros((classes,classes),np.int64)
    np.add.at(confusion,(truth[valid].astype(int),predicted[valid].astype(int)),1)
    per_class=[]
    for c,name in enumerate(class_names):
        union=(valid&((truth==c)|(predicted==c))).sum(1)
        intersection=(valid&(truth==c)&(predicted==c)).sum(1)
        iou=np.divide(intersection,union,out=np.full(n,np.nan),where=union>0)
        total=confusion[c].sum()+confusion[:,c].sum()-confusion[c,c]
        per_class.append({'class':name,'pooled_iou':float(confusion[c,c]/total) if total else None,
                          'true_tokens':int(confusion[c].sum()),'image_iou':cluster_interval(iou,rows)})
    values={'accuracy':accuracy}; summary={'token_accuracy':cluster_interval(accuracy,rows),'per_class_iou':per_class}
    if depth is not None:
        predicted_depth=prediction[:,-1].reshape(n,256)
        values['depth_mae_m']=abs(predicted_depth-depth).mean(1)
        summary['depth_mae_m']=cluster_interval(values['depth_mae_m'],rows)
    return summary,values,predicted,confusion


def audit_tokens(folder,key,x,full_targets,owner,token_ids,all_x,train,test,truth,valid,class_names,weights,depth=None):
    """No masks/coordinates enter x. Controls are separate fits, all tuned in Discovery."""
    started=time.monotonic(); classes=len(class_names); alphas=config()['probe']['ridge_alphas']
    y=full_targets[owner,token_ids]; folds=grouped_folds(train)[owner]
    require(len(np.unique(folds))>=2,'Too few nonempty episode folds for native-token readout')
    prediction,fit=local_ridge(x,y,all_x,alphas,folds,weights)
    main,values,predicted,conf=token_metrics(prediction,truth,valid,test,class_names,depth)
    coords=np.stack(np.meshgrid(np.linspace(0,1,16),np.linspace(0,1,16),indexing='ij'),-1).reshape(256,2)
    cp,cf=local_ridge(coords[token_ids],y,np.tile(coords,(len(test),1)),alphas,folds,weights)
    coordinate,cvalues,cc,_=token_metrics(cp,truth,valid,test,class_names,depth)
    # Majority/mean baseline is estimated from observed training targets only.
    observed=full_targets[:,:,:classes].sum(-1)>0
    count=full_targets[:,:,:classes][observed].sum(0)
    majority=np.zeros_like(prediction); majority[:,int(count.argmax())]=1.
    if depth is not None: majority[:,-1]=full_targets[:,:,-1].mean()
    constant,mvalues,mc,_=token_metrics(majority,truth,valid,test,class_names,depth)
    arrays={'class_prediction':predicted,'coordinate_class_prediction':cc,'majority_class_prediction':mc,
            'truth_class':truth,'valid_tokens':valid,'confusion':conf,
            'state_ids':np.asarray([r['state_id'] for r in test])}
    if depth is not None:
        arrays.update(depth_prediction=prediction[:,-1].reshape(len(test),256),truth_depth=depth,
                      coordinate_depth_prediction=cp[:,-1].reshape(len(test),256))
    loto=np.full_like(prediction,np.nan); loto_details=[]
    for task in sorted({r['task_id'] for r in test}):
        keep=np.asarray([train[i]['task_id']!=task for i in owner])
        image_keep=np.asarray([r['task_id']==task for r in test]); indices=np.flatnonzero(image_keep)
        test_indices=(indices[:,None]*256+np.arange(256)).ravel()
        if not keep.any() or len(np.unique(folds[keep]))<2:
            loto_details.append({'task':task,'status':'INSUFFICIENT_TRAINING_FOLDS'}); continue
        progress('token_leave_one_task_out',site=key,task=task)
        p,detail=local_ridge(x[keep],y[keep],all_x[test_indices],alphas,folds[keep],weights[keep])
        loto[test_indices]=p; loto_details.append({'task':task,'fit':detail})
    # Missing LOTO fits are explicitly marked; don't argmax all-NaN rows into class zero.
    loto_valid=valid&np.isfinite(loto[:,:classes]).all(1).reshape(len(test),256)
    lsummary,_,lc,_=token_metrics(loto,truth,loto_valid,test,class_names,depth)
    arrays['loto_class_prediction']=lc
    if depth is not None: arrays['loto_depth_prediction']=loto[:,-1].reshape(len(test),256)
    shuffle=[]; shuffle_classes=[]; shuffle_depth=[]
    for repeat in range(config()['probe']['shuffle_repeats']):
        progress('token_episode_shuffle',site=key,repeat=repeat)
        yy=permuted_targets(full_targets,train,config()['seed']+repeat)[owner,token_ids]
        known=yy[:,:classes].sum(1)>0
        if not known.any() or len(np.unique(folds[known]))<2:
            shuffle.append({'repeat':repeat,'status':'INSUFFICIENT_PERMUTED_LABELS'})
            shuffle_classes.append(np.full_like(predicted,-1))
            if depth is not None: shuffle_depth.append(np.full((len(test),256),np.nan))
            continue
        p,detail=local_ridge(x[known],yy[known],all_x,alphas,folds[known],weights[known])
        s,_,c,_=token_metrics(p,truth,valid,test,class_names,depth)
        shuffle.append({'repeat':repeat,'fit':detail,**s}); shuffle_classes.append(c)
        if depth is not None: shuffle_depth.append(p[:,-1].reshape(len(test),256))
    arrays['shuffle_class_prediction']=np.stack(shuffle_classes)
    if depth is not None: arrays['shuffle_depth_prediction']=np.stack(shuffle_depth)
    np.savez_compressed(folder/'PREDICTIONS.npz',**arrays)
    atomic_json({'fit':fit,'coordinates_fit':cf,'main':main,'coordinates':coordinate,'majority':constant,
                 'leave_one_task_out':{'per_task_fit':loto_details,**lsummary},'episode_shuffled_labels':shuffle,
                 'train_images':len(train),'test_images':len(test),'valid_test_images':int(valid.any(1).sum()),
                 'train_labelled_tokens':int(len(x)),'test_labelled_tokens':int(valid.sum()),
                 'training_class_tokens':dict(zip(class_names,full_targets[:,:,:classes].sum((0,1)).astype(int).tolist())),
                 'truth_regions_not_inputs':True},folder/'FIT.json')
    summary=[{'site':key,'metric':metric,**v} for metric,v in main.items() if metric!='per_class_iou']
    summary += [{'site':key,'metric':'accuracy_gain_over_coordinates',
                 **cluster_interval(values['accuracy']-cvalues['accuracy'],test)},
                {'site':key,'metric':'accuracy_gain_over_majority',
                 **cluster_interval(values['accuracy']-mvalues['accuracy'],test)}]
    for condition in ('common_visible','front_only_visible','wrist_only_visible','front_occlusion_proxy','wrist_occlusion_proxy',
                      'contact','no_contact','grasp','no_grasp'):
        from atlas_analysis import condition_mask
        keep=condition_mask(test,condition)
        for metric,v in values.items():
            summary.append({'site':key,'metric':metric,'condition':condition,
                            **cluster_interval(np.where(keep,v,np.nan),test)})
    atomic_json({'elapsed_s':time.monotonic()-started,'peak_scope':'analysis_process_since_start',**resources()},folder/'COST.json')
    return summary

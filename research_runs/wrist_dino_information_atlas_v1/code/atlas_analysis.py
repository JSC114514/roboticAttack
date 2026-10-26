"""Resumeable CPU readouts, baselines, paired intervals and evidence reports."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from atlas_common import (ROOT,atomic_json,atomic_text,commit,config,digest,load_json,prepared,
                          progress,receipt_valid,require,resources,sha256_file)
from atlas_labels import label_specs
from atlas_math import angular_error,balanced_token_indices,binary_intervals,cluster_interval,grouped_folds,permuted_targets,rotation_error
from atlas_probes import fit_mlp,fit_ridge,local_ridge,metrics


def records_and_schema(prep):
    rows=[]; schema=None
    for task in range(10):
        folder=ROOT/f'data/task_{task:02d}'
        require(receipt_valid(folder/'COMPLETE.json',prep['signature']),f'Missing capture task {task}')
        records=load_json(folder/'RECORDS.json')
        require(len(records)==120 and len({r['state_id'] for r in records})==120,'Bad task record count')
        sc=load_json(folder/'SCHEMA.json')
        require(schema is None or schema==sc,'Feature schema differs across tasks')
        schema=sc; rows.extend(records)
    require({r['state_id'] for r in rows}=={r['state_id'] for r in prep['states']},'Source-state grid mismatch')
    require(all(r['alignment']['pass'] for r in rows),'Unaligned source images')
    return rows,schema


def load_feature(rows,key,kind='global'):
    arrays={}
    for t in sorted({r['task_id'] for r in rows}):
        arrays[t]=np.load(ROOT/f'data/task_{t:02d}'/kind/(key+'.npy'),mmap_mode='r',allow_pickle=False)
    return np.stack([np.asarray(arrays[r['task_id']][r['row_index']],dtype=np.float32) for r in rows])


def metadata_features(rows,kind):
    if kind=='constant': return np.zeros((len(rows),0),np.float32)
    if kind=='task_progress':
        return np.asarray([np.r_[np.eye(10)[r['task_id']],r['normalized_progress']] for r in rows],np.float32)
    if kind=='proprio': return np.asarray([r['proprio'] for r in rows],np.float32)
    raise ValueError(kind)


def target_matrix(rows,specs):
    return np.asarray([[np.nan if r['truth'].get(s['name']) is None else r['truth'][s['name']] for s in specs]
                       for r in rows],dtype=float)


def condition_mask(rows,name):
    if name=='all': return np.ones(len(rows),bool)
    if name=='common_visible':
        return np.asarray([r['visibility']['front']['subject_visible'] and r['visibility']['wrist']['subject_visible'] for r in rows])
    if name=='front_only_visible':
        return np.asarray([r['visibility']['front']['subject_visible'] and not r['visibility']['wrist']['subject_visible'] for r in rows])
    if name=='wrist_only_visible':
        return np.asarray([r['visibility']['wrist']['subject_visible'] and not r['visibility']['front']['subject_visible'] for r in rows])
    if name in ('contact','no_contact','grasp','no_grasp'):
        key=name.removeprefix('no_'); positive=not name.startswith('no_')
        return np.asarray([bool(r['truth'][key])==positive for r in rows])
    if name.endswith('_occlusion_proxy'):
        view=name.split('_')[0]
        return np.asarray([r['visibility'][view]['subject_occlusion_proxy'] for r in rows],dtype=bool)
    raise ValueError(name)


def summarize_prediction(key,truth,pred,test_rows,specs,baseline=None):
    output=[]; draws=config()['probe']['bootstrap_draws']
    for col,spec in enumerate(specs):
        for condition in ('all','common_visible','front_only_visible','wrist_only_visible','front_occlusion_proxy',
                          'wrist_occlusion_proxy','contact','no_contact','grasp','no_grasp'):
            keep=condition_mask(test_rows,condition)
            eligible=np.isfinite(truth[:,col])&np.isfinite(pred[:,col])&keep
            error=np.where(eligible,abs(truth[:,col]-pred[:,col]),np.nan)
            info=metrics(truth[keep,col],pred[keep,col],spec['kind']=='binary')
            row={'site':key,'label':spec['name'],'category':spec['kind'],'unit':spec['unit'],'condition':condition,**info,
                 **{'mae_'+k:v for k,v in cluster_interval(error,test_rows,draws).items()}}
            if spec['kind']=='binary':
                row.update(binary_intervals(truth[keep,col],pred[keep,col],[r for r,k in zip(test_rows,keep) if k],draws))
            if baseline is not None:
                gain=np.where(eligible,abs(truth[:,col]-baseline[:,col])-error,np.nan)
                row.update({'baseline_gain_'+k:v for k,v in cluster_interval(gain,test_rows,draws).items()})
            output.append(row)
    # Rotation-symmetry-aware angles supplement coordinate losses.
    for group in ('eef_normal','eef_x_axis','object_normal','relative_normal'):
        cols=[i for i,s in enumerate(specs) if s['group']==group]
        require(len(cols)==3,'Normal label group changed')
        angle=angular_error(truth[:,cols],pred[:,cols])
        output.append({'site':key,'label':group+'_angle','category':'normal_angle','unit':'degree','condition':'all',
                       **cluster_interval(angle,test_rows,draws)})
    zn=[i for i,s in enumerate(specs) if s['group']=='eef_normal']; xn=[i for i,s in enumerate(specs) if s['group']=='eef_x_axis']
    angle=rotation_error(truth[:,zn],truth[:,xn],pred[:,zn],pred[:,xn])
    output.append({'site':key,'label':'eef_rotation_angle','category':'rotation_angle','unit':'degree','condition':'all',
                   **cluster_interval(angle,test_rows,draws)})
    return output


def panel_paths(key,tag='global'):
    folder=ROOT/'readouts'/tag/key; folder.mkdir(parents=True,exist_ok=True)
    return folder


def global_readouts(rows,schema,signature):
    specs=label_specs(); train=[r for r in rows if r['split']=='discovery']; test=[r for r in rows if r['split']=='validation']
    y=target_matrix(train,specs); truth=target_matrix(test,specs)
    keys=['baseline__'+k for k in ('constant','task_progress','proprio')]+sorted(schema['global'])
    results=[]; baseline_predictions=[]; baseline_scores=[]
    for key in keys:
        folder=panel_paths(key)
        if receipt_valid(folder/'COMPLETE.json',signature):
            results.extend(load_json(folder/'SUMMARY.json'))
            if key.startswith('baseline__'):
                with np.load(folder/'PREDICTIONS.npz',allow_pickle=False) as z: baseline_predictions.append(z['prediction'].copy())
                baseline_scores.append(load_json(folder/'FIT.json')['cv_normalized_mse'])
            continue
        started=time.monotonic()
        progress('global_readout',site=key,completed_panels=len({r['site'] for r in results}),planned_panels=len(keys))
        if key.startswith('baseline__'):
            kind=key.split('__')[1]; x=metadata_features(train,kind); z=metadata_features(test,kind)
        else: x=load_feature(train,key); z=load_feature(test,key)
        prediction,fit=fit_ridge(x,y,z,train)
        baseline=None
        if not key.startswith('baseline__'):
            selected=np.asarray(baseline_scores).argmin(0)
            base=np.asarray(baseline_predictions)
            baseline=base[selected,np.arange(len(test))[:,None],np.arange(len(specs))[None,:]]
        summary=summarize_prediction(key,truth,prediction,test,specs,baseline)
        # Evaluate unseen tasks without fitting on any trajectory of that task.
        loto=[]; loto_prediction=np.full_like(prediction,np.nan)
        for task in range(10):
            tr=np.asarray([r['task_id']!=task for r in train]); te=np.asarray([r['task_id']==task for r in test])
            p,detail=fit_ridge(x[tr],y[tr],z[te],[r for r,k in zip(train,tr) if k])
            loto_prediction[te]=p
            loto.append({'task':task,'fit':detail,'metrics':{
                s['name']:metrics(truth[te,i],p[:,i],s['kind']=='binary') for i,s in enumerate(specs)}})
        shuffled=[]; shuffle_predictions=[]
        if not key.startswith('baseline__'):
            for repeat in range(config()['probe']['shuffle_repeats']):
                yy=permuted_targets(y,train,config()['seed']+repeat)
                p,_=fit_ridge(x,yy,z,train)
                shuffle_predictions.append(p)
                shuffled.append([metrics(truth[:,i],p[:,i],s['kind']=='binary') for i,s in enumerate(specs)])
        if shuffle_predictions: shuffled_array=np.stack(shuffle_predictions)
        else: shuffled_array=np.empty((0,*prediction.shape))
        np.savez_compressed(folder/'PREDICTIONS.npz',prediction=prediction,truth=truth,loto_prediction=loto_prediction,
                            shuffle_prediction=shuffled_array,
                            state_ids=np.asarray([r['state_id'] for r in test]))
        for entry in summary:
            if entry['condition']!='all' or entry['label'] not in {s['name'] for s in specs}: continue
            i=next(i for i,s in enumerate(specs) if s['name']==entry['label'])
            m=metrics(truth[:,i],loto_prediction[:,i],specs[i]['kind']=='binary')
            entry.update({'loto_'+k:v for k,v in m.items()})
            if shuffled:
                entry['shuffle_mae_mean']=float(np.mean([s[i]['mae'] for s in shuffled if s[i]['mae'] is not None])) if any(s[i]['mae'] is not None for s in shuffled) else None
        atomic_json(fit,folder/'FIT.json'); atomic_json(summary,folder/'SUMMARY.json')
        atomic_json({'leave_one_task_out':loto,'episode_shuffled_labels':shuffled},folder/'ROBUSTNESS.json')
        if key.endswith('__consumed'):
            nonlinear,detail=fit_mlp(x,y,z,train)
            np.savez_compressed(folder/'MLP_PREDICTIONS.npz',prediction=nonlinear,truth=truth)
            atomic_json({'seeds':detail,'seed_metrics':[
                {s['name']:metrics(truth[:,i],p[:,i],s['kind']=='binary') for i,s in enumerate(specs)} for p in nonlinear]},
                folder/'MLP_SUMMARY.json')
            atomic_json([summarize_prediction(key+'__mlp_seed'+str(seed),truth,p,test,specs,baseline)
                         for seed,p in zip(config()['probe']['mlp']['seeds'],nonlinear)],folder/'MLP_CONDITIONS.json')
            mlp_shuffle=[]; mlp_controls=[]
            for repeat in range(config()['probe']['shuffle_repeats']):
                yy=permuted_targets(y,train,config()['seed']+repeat)
                control,detail=fit_mlp(x,yy,z,train,seeds=config()['probe']['mlp']['seeds'][:1])
                mlp_shuffle.append(control[0]); mlp_controls.append({'repeat':repeat,'seeds':detail,
                    'metrics':{s['name']:metrics(truth[:,i],control[0,:,i],s['kind']=='binary') for i,s in enumerate(specs)}})
            np.savez_compressed(folder/'MLP_SHUFFLE_PREDICTIONS.npz',prediction=np.stack(mlp_shuffle),truth=truth)
            atomic_json({'seed_scope':'first_fixed_seed_for_each_of_five_permutations','controls':mlp_controls},folder/'MLP_SHUFFLE.json')
        atomic_json({'elapsed_s':time.monotonic()-started,'peak_scope':'analysis_process_since_start',**resources()},folder/'COST.json')
        commit(folder/'COMPLETE.json',[p for p in sorted(folder.glob('*')) if p.is_file()],signature,site=key)
        results.extend(summary)
        if key.startswith('baseline__'):
            baseline_predictions.append(prediction); baseline_scores.append(fit['cv_normalized_mse'])
    # Paired view contrasts use the same labels/physical states, not two independent samples.
    paired=[]
    for key in sorted(schema['global']):
        if not key.startswith('front__'): continue
        wrist=key.replace('front__','wrist__',1)
        with np.load(panel_paths(key)/'PREDICTIONS.npz',allow_pickle=False) as z: front=z['prediction'].copy()
        with np.load(panel_paths(wrist)/'PREDICTIONS.npz',allow_pickle=False) as z: wp=z['prediction'].copy()
        for i,spec in enumerate(specs):
            if spec['kind'] in ('view_geometry','appearance'): continue
            for condition in ('all','common_visible'):
                mask=condition_mask(test,condition)
                delta=np.where(mask,abs(truth[:,i]-front[:,i])-abs(truth[:,i]-wp[:,i]),np.nan)
                paired.append({'site':key.removeprefix('front__'),'label':spec['name'],'condition':condition,
                               'positive_means':'wrist_lower_error',**cluster_interval(delta,test,config()['probe']['bootstrap_draws'])})
    return results,paired


def local_readouts(rows,signature):
    from atlas_local import audit_tokens
    train=[r for r in rows if r['split']=='discovery']; test=[r for r in rows if r['split']=='validation']
    output=[]; classes=len(config()['class_names'])
    for view in config()['views']:
        for block in config()['detail_blocks']:
            key=f'{view}__b{block:02d}'; folder=panel_paths(key,'local')
            if receipt_valid(folder/'COMPLETE.json',signature):
                output.extend(load_json(folder/'SUMMARY.json')); continue
            progress('local_readout',site=key)
            x=[]; targets=[]; weights=[]; owner=[]; token_ids=[]; cache={}
            for i,row in enumerate(train):
                task=row['task_id']
                if task not in cache:
                    cache[task]=np.load(ROOT/f'data/task_{task:02d}/dense/{view}__dense_b{block:02d}.npy',mmap_mode='r')
                with np.load(ROOT/f"data/task_{task:02d}/states/{row['state_id']}.npz",allow_pickle=False) as z:
                    labels=z[view+'_class_tokens']; depth=z[view+'_depth_tokens']; confidence=z[view+'_class_confidence']
                ids=balanced_token_indices(labels,config()['probe']['local_tokens_per_image'],42+i)
                x.append(np.asarray(cache[task][row['row_index']][ids],np.float32))
                targets.append(np.c_[np.eye(classes)[labels],depth])
                weights.extend(confidence[ids]); owner.extend([i]*len(ids)); token_ids.extend(ids)
            truth=[]; depth=[]
            for row in test:
                with np.load(ROOT/f"data/task_{row['task_id']:02d}/states/{row['state_id']}.npz",allow_pickle=False) as z:
                    truth.append(z[view+'_class_tokens']); depth.append(z[view+'_depth_tokens'])
            truth=np.asarray(truth); depth=np.asarray(depth)
            all_x=load_feature(test,f'{view}__dense_b{block:02d}','dense').reshape(-1,1024)
            summary=audit_tokens(folder,key,np.concatenate(x),np.asarray(targets),np.asarray(owner),np.asarray(token_ids),
                                 all_x,train,test,truth,np.ones_like(truth,dtype=bool),config()['class_names'],
                                 np.asarray(weights),depth)
            atomic_json(summary,folder/'SUMMARY.json')
            commit(folder/'COMPLETE.json',[p for p in folder.glob('*') if p.is_file()],signature,site=key)
            output.extend(summary)
            del x,targets,all_x,cache
    return output


def analyze():
    import pandas as pd
    from atlas_parts import export_review,review_status,parts_readouts
    from atlas_report import write_report
    prep=prepared(); rows,schema=records_and_schema(prep)
    export_review(prep,rows)
    global_rows,paired=global_readouts(rows,schema,prep['signature'])
    local_rows=local_readouts(rows,prep['signature'])
    part_rows=parts_readouts(rows,prep['signature'])
    for name,values in [('INFORMATION_ATLAS',global_rows),('VIEW_CONTRASTS',paired),
                        ('LOCAL_READOUTS',local_rows),('PART_READOUTS',part_rows)]:
        pd.DataFrame(values).to_csv(ROOT/'results'/(name+'.csv'),index=False)
    inventory=[]
    for spec in label_specs():
        for split in ('discovery','validation'):
            selected=[r for r in rows if r['split']==split]
            values=[r['truth'].get(spec['name']) for r in selected]
            valid=[v for v in values if v is not None and np.isfinite(v)]
            source=('same_model_rgb_region_statistics' if spec['kind']=='appearance' else
                    'same_model_depth_segmentation_camera' if spec['kind']=='view_geometry' or spec['name'].startswith(('front_','wrist_')) else
                    'same_model_contacts_and_grasp_predicate' if spec['name'] in ('contact','grasp') else
                    'same_model_physics')
            inventory.append({**spec,'split':split,'source':source,
                              'valid_states':len(valid),'missing_states':len(values)-len(valid),'unique_values':len(set(valid)),
                              'reliability':'replay_aligned_proxy' if 'occlusion_proxy' in spec['name'] else 'replay_aligned',
                              'standard_deviation':float(np.std(valid)) if valid else None})
    coverage=[]
    for view in config()['views']:
        for split in ('discovery','validation'):
            counts=np.zeros(len(config()['class_names']),np.int64); images=np.zeros_like(counts)
            for r in rows:
                if r['split']!=split: continue
                with np.load(ROOT/f"data/task_{r['task_id']:02d}/states/{r['state_id']}.npz",allow_pickle=False) as z:
                    c=np.bincount(z[view+'_class_tokens'],minlength=len(counts)); counts+=c; images+=c>0
            coverage += [{'view':view,'split':split,'class':name,'native_tokens':int(n),'images':int(k)}
                         for name,n,k in zip(config()['class_names'],counts,images)]
    atomic_json({'labels':inventory,'scene_class_coverage':coverage,
                 'instance_identity_limit':'Identical-looking bowls are one visual class, simulator IDs are audit metadata only.',
                 'occlusion_limit':'Center-in-cropped-frustum and zero visible pixels proxy; partial occlusion is not labelled.',
                 'old_label_discrepancies':[
        {'state_id':r['state_id'],'differences':r['old_label_deltas']} for r in rows],
                 'parts':review_status(prep),'physical_labels_are_replay_aligned':True},ROOT/'results/INVENTORY.json')
    write_report(rows,global_rows,paired,local_rows,part_rows)
    parts=review_status(prep)
    status='COMPLETE' if parts['status']=='REVIEWED_80' else 'CORE_COMPLETE_PARTS_PENDING'
    atomic_json({'status':status,'signature':prep['signature'],'states':1200,'episodes':240,
                 'parts':parts,'gpu_model_training':False,'causal_intervention_performed':False,
                 'resources':resources(),'artifacts':{str(p.relative_to(ROOT/'results')):sha256_file(p) for p in (ROOT/'results').rglob('*')
                                                     if p.is_file() and p.name!='STATUS.json'}},ROOT/'results/STATUS.json')

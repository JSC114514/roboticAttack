"""Only saved PNGs, full native RGB forwards, shared paired EOT and baselines."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from pilot_common import (ROOT,PROJECT,atomic_json,commit,complete,config,load_json,observation,
                          progress,require,sha256)
from pilot_geometry import action_metrics
from pilot_mechanism import features_file
from pilot_pixels import compose,initial_patch,load_png,save_png,verify_initial_png
from pilot_preprocessing import deployment_uint8
from pilot_readouts import monitor


def draws(): return [('fixed',0,False)]+[(f'eot{i+1}',seed,True) for i,seed in enumerate(config()['evaluation_eot_seeds'])]


def feature_distance(value,clean,target):
    energy=np.mean((np.asarray(target,dtype=float)-clean)**2,axis=-1)
    if energy.sum()==0: return None
    weight=energy/energy.sum()
    denominator=max(float((weight*energy).sum()),np.finfo(np.float32).tiny)
    return float((weight*np.mean((np.asarray(value,dtype=float)-target)**2,axis=-1)).sum()/denominator)


@lru_cache(maxsize=40)
def metadata_file(path): return load_json(path)


def clean_metadata(row): return metadata_file(str(features_file(row).parent/'METADATA.json'))


@lru_cache(maxsize=32)
def feature_reference(path,condition,layer):
    with np.load(path,allow_pickle=False) as z:
        return z[f'clean__dino_b{layer}'].copy(),z[f'{condition}__dino_b{layer}'].copy()


def execute_png(row,patch,engine,draw,condition=None,layer=None,restore=False):
    obs,language=observation(row); name,seed,diverse=draw
    rgb,eot=compose(obs['wrist_image'],patch,seed,diverse)
    modified={**obs,'wrist_image':deployment_uint8(rgb)}
    action,features=engine.forward(modified,language)
    meta=clean_metadata(row); clean=np.asarray(meta['actions']['clean'])
    base={k:row[k] for k in ('state_id','episode_id','task_id','split')}
    result={**base,'draw':name,'eot':eot,'actions':action.tolist(),
            'monitor_predictions_m':{str(site):monitor(site,features[f'dino_b{site}']) for site in config()['layers']},
            'first_translation_l2':float(np.linalg.norm((action-clean)[0,:3])),
            'native_rgb_sha256':__import__('hashlib').sha256(modified['wrist_image'].tobytes()).hexdigest()}
    if condition is not None:
        result.update(action_metrics(action,clean,condition))
        key=f'dino_b{layer}'
        clean_features,target_features=feature_reference(str(features_file(row)),condition['condition_id'],layer)
        result['feature_target_distance']=feature_distance(features[key],clean_features,target_features)
        with np.load(features_file(row),allow_pickle=False) as z:
            if restore:
                restored={}
                for branch,overrides in [('restore_dino',{'dino_b22':z['clean__dino_b22']}),
                                         ('restore_siglip',{'siglip':z['clean__siglip']})]:
                    other,_=engine.forward(modified,language,overrides,capture=False)
                    restored[branch]={'actions':other.tolist(),**action_metrics(other,clean,condition)}
                result['branch_restorations']=restored
    else:
        result['target_feature_distances']={}
        for target in ('direction_pos','direction_neg','distance_near','distance_far'):
            for site in config()['layers']:
                a,b=feature_reference(str(features_file(row)),target+'_large',site)
                result['target_feature_distances'][f'{target}|{site}']=feature_distance(features[f'dino_b{site}'],a,b)
    return result


def evaluate_baselines(validation,engine,sid):
    folder=ROOT/'evaluation/baselines'; receipt=folder/'COMPLETE.json'
    if complete(receipt,sid): return
    folder.mkdir(parents=True,exist_ok=True)
    specifications=[('uada500_b1',PROJECT/config()['uada_png'])]
    for seed in config()['seeds']:
        path=folder/f'initial_s{seed}.png'
        if not path.exists(): save_png(initial_patch(seed),path)
        verify_initial_png(path,seed)
        specifications.append((f'initial_s{seed}',path))
    files=[path for _,path in specifications if path.is_relative_to(ROOT)]
    for index,row in enumerate(validation):
        state_folder=folder/row['state_id']; state_receipt=state_folder/'COMPLETE.json'
        if not complete(state_receipt,sid):
            state_folder.mkdir(parents=True,exist_ok=True); rows=[]
            meta=clean_metadata(row)
            rows.append({**{k:row[k] for k in ('state_id','episode_id','task_id','split')},
                         'baseline':'clean','draw':'fixed','actions':meta['actions']['clean'],
                         'monitor_predictions_m':meta['monitor_predictions_m']['clean'],'first_translation_l2':0.})
            for name,path in specifications:
                patch=load_png(path,engine.wrapper.device)
                for draw in draws():
                    result=execute_png(row,patch,engine,draw)
                    rows.append({**result,'baseline':name,'png_sha256':sha256(path)})
            path=state_folder/'ROWS.json'; atomic_json(rows,path); commit(state_receipt,[path],sid)
        files.append(state_receipt)
        progress('baseline_evaluation',completed=index+1,total=20)
    commit(receipt,files,sid,validation_states=20,reused_uada_png_sha256=sha256(PROJECT/config()['uada_png']))


def evaluate_candidate(candidate,validation,engine,sid):
    name=candidate['candidate_id']; folder=ROOT/'evaluation'/name; receipt=folder/'COMPLETE.json'
    if complete(receipt,sid): return
    training=ROOT/'training'/name
    require(complete(training/'COMPLETE.json',sid),'Candidate training not sealed')
    png=training/f'step_{config()["updates"]:04d}.png'; patch=load_png(png,engine.wrapper.device)
    folder.mkdir(parents=True,exist_ok=True); files=[]
    for index,row in enumerate(validation):
        state_folder=folder/row['state_id']; state_receipt=state_folder/'COMPLETE.json'
        if not complete(state_receipt,sid):
            state_folder.mkdir(parents=True,exist_ok=True)
            condition=next(c for c in row['conditions'] if c['condition_id']==candidate['target']+'_large')
            results=[]
            for draw in draws():
                result=execute_png(row,patch,engine,draw,condition,candidate['layer'],restore=draw[0]=='fixed')
                results.append({**result,**candidate,'png_sha256':sha256(png),'checkpoint':config()['updates'],
                                'cf_subject_visible':load_json(ROOT/'data/renders'/row['state_id']/'METADATA.json')['conditions'][condition['condition_id']]['subject_visible']})
            path=state_folder/'ROWS.json'; atomic_json(results,path); commit(state_receipt,[path],sid)
        files.append(state_receipt)
        if (index+1)%5==0: progress('png_evaluation',candidate=name,completed=index+1,total=20)
    commit(receipt,files,sid,candidate=candidate,validation_states=20,draws=5,checkpoint_selection='fixed_final_only')

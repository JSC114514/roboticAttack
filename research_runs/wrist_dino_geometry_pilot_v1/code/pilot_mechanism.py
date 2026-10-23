"""Native RGB/branch interventions on a fixed 40-episode repeated-measures cohort."""
from __future__ import annotations
import time
import numpy as np
from pilot_common import (ROOT,atomic_json,atomic_npz,commit,complete,config,load_json,
                          observation,progress,require,resources)
from pilot_geometry import action_metrics
from pilot_readouts import monitor


def features_file(row): return ROOT/'data/features'/row['state_id']/'FEATURES.npz'
def mechanism_file(row): return ROOT/'data/mechanism'/row['state_id']/'ROWS.json'


def feature_state(row,engine,signature):
    folder=features_file(row).parent
    if complete(folder/'COMPLETE.json',signature): return
    folder.mkdir(parents=True,exist_ok=True)
    obs,language=observation(row)
    arrays={}; actions={}; geometries={}; qa={}
    render=ROOT/'data/renders'/row['state_id']
    require(complete(render/'COMPLETE.json',signature),'Missing verified render shard')
    with np.load(render/'FRAMES.npz',allow_pickle=False) as frames:
        for name in ['clean','zero_render']+[c['condition_id'] for c in row['conditions']]:
            current=obs if name=='clean' else {**obs,'wrist_image':frames[('clean' if name=='zero_render' else name)+'__rgb'].copy()}
            action,features=engine.forward(current,language)
            actions[name]=action.tolist()
            for key,value in features.items(): arrays[name+'__'+key]=value
            geometries[name]={str(layer):monitor(layer,features[f'dino_b{layer}']) for layer in config()['layers']}
        # Exact same tensors must be identity interventions at both sites.
        clean=np.asarray(actions['clean'])
        for layer in config()['layers']:
            action,_=engine.forward(obs,language,{f'dino_b{layer}':arrays[f'clean__dino_b{layer}'],
                                                'siglip':arrays['clean__siglip']},capture=False)
            error=float(np.max(abs(action-clean)))
            require(error<=1e-5,'Clean identity intervention changed native actions')
            qa[f'identity_b{layer}_max_abs']=error
        cf=row['conditions'][0]['condition_id']
        action,_=engine.forward(obs,language,{'dino_b22':arrays[cf+'__dino_b22'],
                                            'siglip':arrays[cf+'__siglip']},capture=False)
        error=float(np.max(abs(action-np.asarray(actions[cf]))))
        require(error<=1e-5,'block22 both-branch replacement differs from native CF RGB')
        qa['b22_both_native_cf_max_abs']=error
    atomic_npz(features_file(row),**arrays)
    meta=folder/'METADATA.json'; atomic_json({'actions':actions,'monitor_predictions_m':geometries,'qa':qa},meta)
    commit(folder/'COMPLETE.json',[features_file(row),meta],signature,state_id=row['state_id'])


def mechanism_state(row,engine,signature):
    from pilot_engine import random_matched
    folder=mechanism_file(row).parent
    if complete(folder/'COMPLETE.json',signature): return
    require(complete(features_file(row).parent/'COMPLETE.json',signature),'Missing native feature shard')
    folder.mkdir(parents=True,exist_ok=True)
    obs,language=observation(row); meta=load_json(features_file(row).parent/'METADATA.json')
    visibility=load_json(ROOT/'data/renders'/row['state_id']/'METADATA.json')['conditions']
    clean_action=np.asarray(meta['actions']['clean']); rows=[]
    base={k:row[k] for k in ('state_id','episode_id','task_id','split')}
    def add(condition,branch,layer,action):
        rows.append({**base,'condition_id':condition['condition_id'],'target':condition['target'],
                     'strength':condition['strength'],'sign':condition['sign'],'branch':branch,'layer':layer,
                     'cf_subject_visible':visibility[condition['condition_id']]['subject_visible'],
                     'clean_subject_visible':visibility['clean']['subject_visible'],
                     'displacement_m':condition['displacement_m'],'angle_degrees':condition['angle_degrees'],
                     'actions':np.asarray(action).tolist(),'clean_actions':clean_action.tolist(),
                     **action_metrics(action,clean_action,condition)})
    with np.load(features_file(row),allow_pickle=False) as z:
        for ci,condition in enumerate(row['conditions']):
            name=condition['condition_id']
            add(condition,'native_cf_rgb',22,meta['actions'][name])
            add(condition,'zero',22,clean_action)
            add(condition,'zero_render_rgb',22,meta['actions']['zero_render'])
            siglip=z[name+'__siglip']
            action,_=engine.forward(obs,language,{'siglip':siglip},capture=False)
            add(condition,'siglip_only',None,action)
            for layer in config()['layers']:
                key=f'dino_b{layer}'; target=z[name+'__'+key]; clean=z['clean__'+key]
                for branch,overrides in [
                    ('dino_only',{key:target}),
                    ('both',{key:target,'siglip':siglip}),
                    ('random_dino',{key:random_matched(clean,target,config()['random_control_seed']+ci+100*layer+10000*row['task_id']+row['step_id'])})]:
                    action,_=engine.forward(obs,language,overrides,capture=False)
                    if layer==22 and branch=='both':
                        require(np.max(abs(action-np.asarray(meta['actions'][name])))<=1e-5,
                                'Serialized block22 branches no longer reproduce native counterfactual')
                    add(condition,branch,layer,action)
    atomic_json(rows,mechanism_file(row))
    commit(folder/'COMPLETE.json',[mechanism_file(row)],signature,state_id=row['state_id'],rows=len(rows))


def mechanism(prep):
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json',prep['signature']),'CPU readout preparation incomplete')
    from pilot_render import render_all
    render_all(prep)
    from pilot_engine import Engine
    progress('loading_frozen_vla',renders_complete=40)
    engine=Engine()
    first=prep['states'][0]; obs,language=observation(first)
    qa=engine.validate_rgb(obs,language)
    atomic_json(qa,ROOT/'reports/GPU_RGB_INTERFACE.json')
    started=time.monotonic()
    for index,row in enumerate(prep['states']):
        feature_state(row,engine,prep['signature'])
        mechanism_state(row,engine,prep['signature'])
        elapsed=time.monotonic()-started
        progress('mechanism',completed=index+1,total=40,state_id=row['state_id'],elapsed_s=elapsed,
                 estimated_remaining_s=elapsed/(index+1)*(39-index))
    files=[ROOT/'reports/GPU_RGB_INTERFACE.json',ROOT/'reports/RENDERS_COMPLETE.json']
    for row in prep['states']:
        files.extend([features_file(row).parent/'COMPLETE.json',mechanism_file(row).parent/'COMPLETE.json'])
    commit(ROOT/'reports/MECHANISM_COMPLETE.json',files,prep['signature'],states=40,**resources())

"""Native PNG diagnostics and paired LIBERO rollouts; no test-time optimization."""
from __future__ import annotations
import copy,gc,time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT,PROJECT,cfg,origin_run,require,require_gpu,complete,finish,atomic_parquet,
                    load_wrapper,load_png,paste_numpy,runtime_module,runtime_config,read_json,
                    decode_action_chunk,resources,sha,signature)
from data import compact_pilot,iter_records
from engine import Engine
from train import make_engine,objective

def array(path):
    with Image.open(path) as im:
        require(im.mode=='RGB' and im.size==(50,50),'Invalid deployment PNG')
        return np.array(im,copy=True)

def metrics(feature,minus,plus):
    v=torch.as_tensor(np.array(feature,copy=True)).float()
    result=objective(v,minus,plus,'full')
    if result is None:return {'q':None,'projection_loss':None,'orthogonal':None,'target_distance':None}
    _,m=result
    require(all(np.isfinite(m[k]) for k in ('loss','q','orthogonal','projection_loss')),'Nonfinite native feature diagnostics')
    return {'q':m['q'],'projection_loss':m['projection_loss'],'orthogonal':m['orthogonal'],'target_distance':m['loss']}

def action_values(action,clean,direction):
    a=np.asarray(action,dtype=np.float32).reshape(8,7);c=np.asarray(clean,dtype=np.float32).reshape(8,7)
    b=np.asarray(direction,dtype=np.float32).reshape(8,7);delta=a-c
    n=float(np.linalg.norm(delta[0,:3]));bn=float(np.linalg.norm(b[0,:3]));dot=float(delta[0,:3]@b[0,:3])
    return {'first_translation_l2':n,'first_rotation_l2':float(np.linalg.norm(delta[0,3:6])),
            'first_gripper_delta':float(delta[0,6]),'direction_dot':dot,
            'direction_cos':dot/(n*bn) if n*bn>1e-12 else None,
            'direction_positive':float(dot>0) if bn>1e-12 else None,
            'ideal_direction_l2':bn,'chunk_translation_l2':float(np.linalg.norm(delta[:,:3],axis=1).mean()),
            'actions':a.tolist(),'clean_actions':c.tolist()}

def identity(row,cohort,condition):
    return {k:row[k] for k in ('state_id','episode_id','suite','task_id','state_index','frame_rank')} | {
        'cohort':cohort,'condition':condition,'no_target':bool(row['geometry'].get('no_target',False))}

def oracle_record(engine,row,cohort):
    obs,language=row['observation'],row['language'];clean,features=engine.forward(obs,language)
    targets=row['target_features'];positive=np.asarray(targets['translation_plus'].float())
    negative=np.asarray(targets['translation_minus'].float())
    p,_=engine.forward(obs,language,{'dino':positive});m,_=engine.forward(obs,language,{'dino':negative})
    middle=features['dino']+.5*(positive.reshape(features['dino'].shape)-features['dino'])
    half,_=engine.forward(obs,language,{'dino':middle})
    rgbp,_=engine.forward({**obs,'wrist_image':row['counterfactual_wrist']['translation_plus']},language)
    rgbm,_=engine.forward({**obs,'wrist_image':row['counterfactual_wrist']['translation_minus']},language)
    initial,_=engine.forward(paste_numpy(obs,array(ROOT/'data/initial_s42.png')),language)
    return {**identity(row,cohort,'oracle'),'clean_actions':clean.tolist(),
            'dino_plus_actions':p.tolist(),'dino_minus_actions':m.tolist(),'dino_half_actions':half.tolist(),
            'rgb_plus_actions':rgbp.tolist(),'rgb_minus_actions':rgbm.tolist(),'initial_actions':initial.tolist(),
            'ideal_direction_l2':float(np.linalg.norm((p-m)[0,:3])),
            'dino_plus_change_l2':float(np.linalg.norm((p-clean)[0,:3])),
            'dino_half_change_l2':float(np.linalg.norm((half-clean)[0,:3])),
            'native_rgb_direction_l2':float(np.linalg.norm((rgbp-rgbm)[0,:3])),
            **{'clean_'+k:v for k,v in metrics(features['b22'],negative,positive).items()}}

def run_oracle(monitor):
    require_gpu('oracle');folder=ROOT/'evaluation/pilot';receipt=folder/'ORACLE_COMPLETE.json'
    if complete(receipt):return
    engine=make_engine();cohorts=compact_pilot();rows=[]
    for cohort,records in cohorts.items():
        for record in records:
            rows.append(oracle_record(engine,record,cohort))
            monitor.update(phase='ideal_dino_and_native_rgb',completed=len(rows),total=96)
    path=folder/'ORACLE.parquet';atomic_parquet(pd.DataFrame(rows),path)
    finish(receipt,[path],observations=len(rows),checkpoint='libero_spatial',**resources())

def patch_record(engine,row,cohort,condition,path,reference,restore=True):
    obs,language=row['observation'],row['language'];clean=decode_action_chunk(reference['clean_actions'])
    plus=decode_action_chunk(reference['dino_plus_actions']);minus=decode_action_chunk(reference['dino_minus_actions'])
    current=paste_numpy(obs,array(path));action,features=engine.forward(current,language)
    targets=row['target_features'];m=metrics(features['b22'],targets['translation_minus'],targets['translation_plus'])
    output={**identity(row,cohort,condition),**m,**action_values(action,clean,plus-minus),'png_sha256':sha(path)}
    output['q_delta']=None if m['q'] is None else m['q']-reference['clean_q']
    if restore:
        _,clean_features=engine.forward(obs,language)
        restored_dino,_=engine.forward(current,language,{'dino':clean_features['dino']})
        restored_siglip,_=engine.forward(current,language,{'siglip':clean_features['siglip']})
        output.update(restore_dino_translation_l2=float(np.linalg.norm((restored_dino-clean)[0,:3])),
                      restore_siglip_translation_l2=float(np.linalg.norm((restored_siglip-clean)[0,:3])))
    return output

def run_pilot_check(monitor):
    require_gpu('pilot_check');folder=ROOT/'evaluation/pilot';receipt=folder/'CHECK_COMPLETE.json'
    if complete(receipt):return
    for kind in ('individual','projection','full'):
        require(complete(ROOT/f'training/pilot/{kind}/COMPLETE.json'),'Pilot training incomplete: '+kind)
    require(complete(folder/'ORACLE_COMPLETE.json'),'Oracle incomplete')
    refs=pd.read_parquet(folder/'ORACLE.parquet').set_index('state_id').to_dict('index')
    engine=make_engine();cohorts=compact_pilot();rows=[];total=48*5+48*3
    for cohort,records in cohorts.items():
        for record in records:
            conditions={'initial':ROOT/'data/initial_s42.png',
                        'shared_projection':ROOT/'training/pilot/projection/final.png',
                        'shared_full':ROOT/'training/pilot/full/final.png'}
            if cohort=='fit':
                for loss in ('projection','full'):
                    conditions['individual_'+loss]=ROOT/f'training/pilot/individual/{loss}/{record["state_id"]}/final.png'
            for condition,path in conditions.items():
                rows.append(patch_record(engine,record,cohort,condition,path,refs[record['state_id']],restore=condition.startswith('shared')))
                monitor.update(phase='native_png_check',completed=len(rows),total=total)
    path=folder/'PNG_CHECK.parquet';atomic_parquet(pd.DataFrame(rows),path)
    finish(receipt,[path],records=len(rows),individual_generalization_claimed=False,**resources())

def night_paths():
    paths={'initial_s42':ROOT/'data/initial_s42.png','initial_s43':ROOT/'data/initial_s43.png',
           'projection_s42':origin_run()/'training/translation/epoch_50.png','uada':PROJECT/cfg()['uada_png']}
    for candidate in cfg()['night_candidates']:
        require(complete(ROOT/f'training/night/{candidate}/COMPLETE.json'),'Night candidate incomplete: '+candidate)
        paths[candidate]=ROOT/f'training/night/{candidate}/final.png'
    return paths

def offline_night(engine,suite,checkpoint,paths,monitor):
    sealed=ROOT/f'evaluation/night/{suite}_{checkpoint}_offline_COMPLETE.json'
    path=ROOT/f'evaluation/night/{suite}_{checkpoint}_offline.parquet'
    if complete(sealed):return sealed
    output=[];records=list(iter_records('validation',suite))
    for i,row in enumerate(records):
        # Validation shards originally contain images only; extract targets under this actual checkpoint.
        targets={}
        for sign in ('minus','plus'):
            targets['translation_'+sign]=engine.spatial_observation({**row['observation'],
                'wrist_image':row['counterfactual_wrist']['translation_'+sign]})
        row={**row,'target_features':targets};reference=oracle_record(engine,row,'validation')
        for condition,path in paths.items():
            item=patch_record(engine,row,'validation',condition,path,reference,restore=condition in ('full_s42','projection_s42'))
            output.append({**item,'checkpoint':checkpoint})
        monitor.update(phase='offline_'+checkpoint,completed=i+1,total=len(records))
    atomic_parquet(pd.DataFrame(output),path)
    finish(sealed,[path],checkpoint=checkpoint,observations=len(records));return sealed

def episode(env,wrapper,task,initial,seed,patch):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    runtime=runtime_module();raw=runtime.reset_wait(env,initial,seed,runtime_config()['num_steps_wait'])
    step=0;decision=0;success=False;trace=[]
    limit=runtime_config()['max_steps_by_suite'][wrapper.gen_cfg.task_suite_name]
    while step<limit and not success:
        obs=runtime.observation(raw,runtime_config())
        if patch is not None:obs=paste_numpy(obs,patch)
        with torch.no_grad():actions=np.asarray(wrapper.predict_action(obs,task.language),dtype=np.float32).reshape(8,7)
        require(np.isfinite(actions).all(),'Nonfinite rollout actions')
        used=0
        for slot in range(min(8,limit-step)):
            raw,_reward,done,_info=env.step(process_libero_action(actions[slot]).tolist());step+=1;used+=1
            if done:success=True;break
        trace.append({'decision_block':decision,'executed_slots':used,'step_end':step,
                      'predicted_environment_action_chunk':actions.tolist()});decision+=1
    return {'success':int(success),'failure':int(not success),'executed_steps':step,'decision_blocks':decision},trace

def rollout_suite(wrapper,suite,paths,monitor):
    c=cfg();runtime=runtime_module();patches={'clean':None,**{k:array(v) for k,v in paths.items()}}
    output=[];traces=[];artifacts=[]
    tasks=[(split,t) for split in ('validation','heldout') for t in c['task_splits'][suite][split]]
    source=read_json(ROOT/'data/SOURCE_MANIFEST.json')['init_files']
    for index,(split,task_id) in enumerate(tasks):
        part=ROOT/f'evaluation/night/episodes/{suite}_{split}_t{task_id:02d}.json'
        if complete(part):
            output.extend(pd.read_parquet(part.with_suffix('.parquet')).to_dict('records'))
            traces.extend(pd.read_parquet(part.with_name(part.stem+'_traces.parquet')).to_dict('records'))
            artifacts.append(part);continue
        entry=source[f'{suite}/{task_id}'];require(sha(entry['path'])==entry['sha256'],'LIBERO initial states changed')
        env,task,initial=runtime.make_env(runtime_config(),suite,task_id);local=[];local_traces=[]
        try:
            require(__import__('pathlib').Path(entry['path']).stem==task.name,'Runtime benchmark task mapping changed')
            states=[48] if split=='validation' else c['fresh_heldout_states']
            for state in states:
                require(state<len(initial),'Initial state index unavailable')
                seed=c['rollout_seed_base']+c['suites'].index(suite)*10000+task_id*100+state
                for condition,patch in patches.items():
                    meta={'split':split,'suite':suite,'task_id':task_id,'state_index':state,
                          'rollout_seed':seed,'condition':condition,'checkpoint':suite,'reused_clean':False,
                          'png_sha256':sha(paths[condition]) if condition!='clean' else None}
                    result,trace=episode(env,wrapper,task,initial[state],seed,patch)
                    local.append({**meta,**result});local_traces.extend({**meta,**x} for x in trace)
                    monitor.update(phase='closedloop_'+split,completed=len(output)+len(local),total=48,
                                   task_id=task_id,state_index=state,condition=condition)
        finally:runtime.close_environment(env)
        roll_path=part.with_suffix('.parquet');trace_path=part.with_name(part.stem+'_traces.parquet')
        atomic_parquet(pd.DataFrame(local),roll_path);atomic_parquet(pd.DataFrame(local_traces),trace_path)
        finish(part,[roll_path,trace_path],suite=suite,split=split,task_id=task_id)
        artifacts.append(part);output.extend(local);traces.extend(local_traces)
    path=ROOT/f'evaluation/night/{suite}_rollouts.parquet';atomic_parquet(pd.DataFrame(output),path)
    return [path,*artifacts]

def run_night_eval(suite,monitor):
    require_gpu('evaluate_'+suite);receipt=ROOT/f'evaluation/night/{suite}_COMPLETE.json'
    if complete(receipt):return
    paths=night_paths();artifacts=[]
    # Separate models are loaded serially in each worker; training never uses suite-specific checkpoints.
    wrapper=load_wrapper('libero_spatial');engine=Engine(wrapper)
    engine.verify_patch_path(next(iter_records('fit')))
    artifacts.append(offline_night(engine,suite,'fixed_spatial',paths,monitor))
    if suite!='libero_spatial':
        del engine,wrapper;gc.collect();torch.cuda.empty_cache()
        wrapper=load_wrapper(suite);engine=Engine(wrapper)
        engine.verify_patch_path(next(iter_records('validation',suite)))
    artifacts.append(offline_night(engine,suite,'native_suite',paths,monitor))
    artifacts.extend(rollout_suite(wrapper,suite,paths,monitor))
    finish(receipt,artifacts,suite=suite,primary_candidate=cfg()['primary_candidate'],**resources())

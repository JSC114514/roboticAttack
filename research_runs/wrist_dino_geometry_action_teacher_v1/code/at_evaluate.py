"""Native PNG evaluation, fixed Spatial checkpoint for all four environment suites."""
import copy
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from at_common import (ROOT,config,native,source,read_json,require,sha,complete,finish,records,
    make_engine,atomic_json,atomic_parquet,verify_inputs)

def conditions():
    selected=read_json(ROOT/'data/SELECTED.json')['candidates']
    paths={'clean':None,**{f'initial_s{s}':ROOT/f'data/initial_s{s}.png' for s in config()['seeds']},
        'uada500':Path(read_json(ROOT/'data/MANIFEST.json')['uada_png'])}
    for name,v in selected.items():
        require(sha(v['path'])==v['png_sha256'],'Selected PNG changed')
        paths[name]=Path(v['path'])
    return paths

def png(path):
    if path is None:return None
    with Image.open(path) as im:
        require(im.mode=='RGB' and im.size==(50,50),'Deployment PNG shape/mode changed')
        return np.array(im,copy=True)

def chunks(x):return np.asarray(x,dtype=np.float32).reshape(8,7)

def offline(index,engine,paths,monitor):
    selected=read_json(ROOT/'data/SELECTED.json')['candidates']
    for n,row in enumerate(records('verify',with_teacher=True)):
        if n%4!=index:continue
        receipt=ROOT/f'evaluation/offline/{row["state_id"]}.json'
        if complete(receipt):continue
        with torch.no_grad():
            clean,clean_caps=engine.action(row['observation'],row['language'])
            target=row['teacher']['target'].to(clean);desired=(target-clean)[0,:,:3]
            clean_physical=engine.physical(clean);values=[]
            clean_feature=clean_caps['dino'].detach()
            for condition,path in paths.items():
                obs=row['observation'] if path is None else native.paste_numpy(row['observation'],png(path))
                action,caps=engine.action(obs,row['language'])
                delta=(action-clean)[0,:,:3];dot=float((delta*desired).sum())
                den=float(delta.norm()*desired.norm());target_norm=float(desired.norm())
                m={'suite':row['suite'],'task_id':row['task_id'],'state_id':row['state_id'],
                    'episode_id':row['episode_id'],'state_index':row['state_index'],'frame_rank':row['frame_rank'],
                    'condition':condition,'no_target':row['teacher']['no_target'],
                    'no_target_reason':row['teacher']['no_target_reason'],
                    'target_mse':float((action[...,:3]-target[...,:3]).square().mean()) if not row['teacher']['no_target'] else None,
                    'translation_delta_l2':float(delta.norm(dim=-1).mean()),
                    'direction_cos':dot/den if den>1e-12 and not row['teacher']['no_target'] else None,
                    'direction_positive':float(dot>0) if target_norm>1e-12 and not row['teacher']['no_target'] else None,
                    'actions':engine.physical(action).tolist(),'normalized_actions':action[0].cpu().tolist(),
                    'clean_actions':clean_physical.tolist(),'teacher_normalized_actions':target[0].cpu().tolist()}
                result=source.module('at_geometry_monitor',source.MATCH/'code/engine.py').signed_projection(
                    caps['dino'],row['target_features']['translation_minus'],row['target_features']['translation_plus'])
                if result is not None:m.update(q=float(result[1]),orthogonal=float(result[2]),target_distance=float(result[0]+result[2]))
                if condition in selected:
                    restored_dino,_=engine.action(obs,row['language'],replacements={'dino':clean_feature})
                    restored_siglip,_=engine.action(obs,row['language'],replacements={'siglip':clean_caps['siglip']})
                    m.update(restore_clean_dino_delta_l2=float((restored_dino-clean)[...,:3].norm(dim=-1).mean()),
                             restore_clean_siglip_delta_l2=float((restored_siglip-clean)[...,:3].norm(dim=-1).mean()))
                values.append(m)
        path=receipt.with_suffix('.parquet');atomic_parquet(pd.DataFrame(values),path)
        finish(receipt,[path],state_id=row['state_id'],conditions=list(paths))
        monitor.update(phase='offline_native_png',completed=n//4+1,total=60,state_id=row['state_id'])

def start_evidence(env,raw,runtime):
    from oracle_camera import fingerprint
    obs=runtime.observation(raw,native.runtime_config())
    return {'sim_state':np.array(env.get_sim_state(),copy=True),'front':obs['full_image'],
            'wrist':obs['wrist_image'],'proprio':obs['state']},fingerprint(env)

def episode(env,task,initial,seed,suite,patch,engine,folder,start_folder,monitor):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    runtime=native.runtime_module();raw=runtime.reset_wait(env,initial,seed,native.runtime_config()['num_steps_wait'])
    arrays,fp=start_evidence(env,raw,runtime);proof={}
    own=folder/'START.npz';np.savez_compressed(own,**arrays)
    if not (start_folder/'START.json').exists():
        require(patch is None,'Clean must establish each paired start first')
        start_folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(start_folder/'START.npz',**arrays)
        atomic_json({'fingerprint':fp,'rollout_seed':seed},start_folder/'START.json')
    before=read_json(start_folder/'START.json')
    require(before['rollout_seed']==seed and before['fingerprint']==fp,'Paired reset physics/cameras/clocks changed')
    with np.load(start_folder/'START.npz',allow_pickle=False) as reference:
        require(np.array_equal(arrays['sim_state'],reference['sim_state']) and
            np.array_equal(arrays['proprio'],reference['proprio']),'Paired reset state changed')
        for view in ('front','wrist'):
            proof[view+'_rgb_nrmse']=float(np.sqrt(np.mean(((arrays[view].astype(float)-reference[view].astype(float))/255.)**2)))
    proof['exact_reset_fingerprint']=True
    # Exact state/time/camera/observable proof is the already audited RGB2 repair route.
    # RGB differences are retained, never hidden by increasing the old tolerance.
    atomic_json(proof,folder/'PAIRING.json')
    limit=native.runtime_config()['max_steps_by_suite'][suite]
    step=decision=0;success=False;traces=[];images=[]
    while step<limit and not success:
        obs=runtime.observation(raw,native.runtime_config())
        deployed=obs if patch is None else native.paste_numpy(obs,patch)
        with torch.no_grad():
            actions=chunks(engine.wrapper.predict_action(copy.deepcopy(deployed),task.language))
            clean=actions if patch is None else chunks(engine.wrapper.predict_action(copy.deepcopy(obs),task.language))
        require(np.isfinite(actions).all() and np.isfinite(clean).all(),'Nonfinite native rollout actions')
        if decision in (0,4,12):
            path=folder/f'frame_{decision:03d}.png'
            Image.fromarray(np.concatenate([deployed['full_image'],deployed['wrist_image']],axis=1)).save(path);images.append(path)
        start=step;used=0;eef=np.asarray(raw['robot0_eef_pos']).copy()
        for slot in range(min(8,limit-step)):
            raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist());step+=1;used+=1
            if done:success=True;break
        delta=actions-clean
        traces.append({'decision_block':decision,'step_start':start,'step_end':step,'executed_slots':used,
            'predicted_actions':actions.tolist(),'same_observation_clean_actions':clean.tolist(),
            'translation_delta_l2':float(np.linalg.norm(delta[:used,:3],axis=1).mean()),
            'rotation_delta_l2':float(np.linalg.norm(delta[:used,3:6],axis=1).mean()),
            'gripper_disagreement':float(((actions[:used,6]>.5)!=(clean[:used,6]>.5)).mean()),
            'eef_before':eef.tolist(),'eef_after':np.asarray(raw['robot0_eef_pos']).tolist(),
            'gripper_qpos_after':np.asarray(raw['robot0_gripper_qpos']).tolist(),'success_after_chunk':success})
        decision+=1;monitor.update(executed_steps=step,decision_blocks=decision)
    obs=runtime.observation(raw,native.runtime_config());deployed=obs if patch is None else native.paste_numpy(obs,patch)
    path=folder/'frame_final.png';Image.fromarray(np.concatenate([deployed['full_image'],deployed['wrist_image']],axis=1)).save(path);images.append(path)
    return {'success':int(success),'failure':int(not success),'executed_steps':step,'decision_blocks':decision,
            'max_steps':limit},traces,[own,folder/'PAIRING.json',*images]

def rollout(index,engine,paths,monitor):
    c=config();runtime=native.runtime_module();manifest=read_json(ROOT/'data/MANIFEST.json')
    grid=[(s,t,i) for s in c['suites'] for t in range(10) for i in c['eval_states']][index::4]
    patches={k:png(p) for k,p in paths.items()}
    for n,(suite,task_id,state) in enumerate(grid):
        root=ROOT/f'evaluation/episodes/{suite}_t{task_id:02d}_s{state:02d}'
        if all(complete(root/k/'COMPLETE.json') for k in paths):continue
        entry=manifest['initial_states'][f'{suite}/{task_id}']
        require(sha(entry['path'])==manifest['inputs'][entry['path']],'Initial state file changed')
        env,task,initial=runtime.make_env(native.runtime_config(),suite,task_id,
            on_phase=lambda phase,**kw:monitor.update(phase=phase,suite=suite,task_id=task_id,**kw))
        require(task.name==entry['task_name'],'Runtime task mapping changed')
        try:
            seed=c['rollout_seed_base']+c['suites'].index(suite)*10000+task_id*100+state
            for condition,path in paths.items():
                folder=root/condition;receipt=folder/'COMPLETE.json'
                if complete(receipt):continue
                attempt=folder/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True,exist_ok=True)
                monitor.update(phase='native_png_closedloop',suite=suite,task_id=task_id,state_index=state,
                    condition=condition,completed=n*len(paths),total=len(grid)*len(paths))
                try:
                    result,traces,artifacts=episode(env,task,initial[state],seed,suite,patches[condition],engine,attempt,root/'paired_start',monitor)
                except BaseException as error:
                    atomic_json({'status':'ENGINEERING_FAILURE','suite':suite,'task_id':task_id,'state_index':state,
                        'condition':condition,'error':repr(error)},attempt/'FAILURE.json')
                    raise
                meta={'suite':suite,'task_id':task_id,'state_index':state,'rollout_seed':seed,'condition':condition,
                    'checkpoint':'fixed_spatial','png_sha256':sha(path) if path else None,'reused_baseline':False,**result}
                metadata=folder/'EPISODE.json';trace=folder/'TRACE.parquet'
                atomic_json(meta,metadata);atomic_parquet(pd.DataFrame(traces),trace)
                finish(receipt,[metadata,trace,root/'paired_start/START.json',root/'paired_start/START.npz',*artifacts])
            # Read-only aggregation of completed starts, shared writer uses a file lock.
            from at_report import run as report
            report(None,final=False)
        finally:runtime.close_environment(env)
    monitor.update(phase='closedloop_finished',completed=len(grid)*len(paths),total=len(grid)*len(paths))

def run(index,monitor):
    require(complete(ROOT/'reports/SELECT_COMPLETE.json'),'Selection unavailable: refuse model load')
    if complete(ROOT/f'evaluation/worker_{index}_COMPLETE.json'):return
    native.require_gpu('action_teacher_evaluate')
    verify_inputs();paths=conditions();engine=make_engine()
    # Closed-loop outcomes first: they are the primary morning deliverable.
    rollout(index,engine,paths,monitor)
    offline(index,engine,paths,monitor)
    assigned=[(s,t,i) for s in config()['suites'] for t in range(10) for i in config()['eval_states']][index::4]
    artifacts=[ROOT/f'evaluation/episodes/{s}_t{t:02d}_s{i:02d}/{k}/COMPLETE.json'
               for s,t,i in assigned for k in paths]
    artifacts += [ROOT/f'evaluation/offline/{r["state_id"]}.json'
                  for n,r in enumerate(records('verify')) if n%4==index]
    require(all(complete(p) for p in artifacts),'Evaluation artifacts incomplete')
    finish(ROOT/f'evaluation/worker_{index}_COMPLETE.json',artifacts,worker=index,checkpoint='fixed_spatial')

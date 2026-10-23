"""Native PNG deployments; one frozen Goal policy for all environment suites."""
import copy
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sr_common import (ROOT,config,native,read_json,require,sha,complete,finish,records,
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

def start_evidence(env,raw,runtime):
    from oracle_camera import fingerprint
    obs=runtime.observation(raw,native.runtime_config())
    return {'sim_state':np.array(env.get_sim_state(),copy=True),'front':obs['full_image'],
            'wrist':obs['wrist_image'],'proprio':obs['state']},fingerprint(env)

def episode(env,task,initial,seed,suite,patch,engine,folder,start_folder,monitor,target_anchor=None):
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
    step=decision=0;success=False;traces=[];images=[];steps=[]
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
            before_geometry=truth(env,target_anchor) if target_anchor is not None else None
            before_eef=np.asarray(raw['robot0_eef_pos']).copy()
            raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist());step+=1;used+=1
            after_geometry=truth(env,target_anchor) if target_anchor is not None else None
            actual_delta=np.asarray(raw['robot0_eef_pos'])-before_eef
            relative=(np.asarray(before_geometry['target_xyz'])-before_eef) if before_geometry is not None else None
            unit=relative/max(float(np.linalg.norm(relative)),1e-9) if relative is not None else None
            steps.append({'step':step,'decision_block':decision,'executed_slot':slot,'eef_before':before_eef.tolist(),
                'eef_after':np.asarray(raw['robot0_eef_pos']).tolist(),'geometry_before':before_geometry,'geometry_after':after_geometry,
                'radial_eef_displacement_m':float(actual_delta@unit) if unit is not None else None,
                'distance_change_m':after_geometry['distance_m']-before_geometry['distance_m'] if before_geometry is not None else None,
                'target_displacement_m':float(np.linalg.norm(np.asarray(after_geometry['target_xyz'])-before_geometry['target_xyz'])) if before_geometry is not None else None,
                'pre_contact':not(before_geometry['contact'] or before_geometry['grasp']) if before_geometry is not None else None,
                'deployed_action':actions[slot].tolist(),'same_observation_clean_chunk_action':clean[slot].tolist()})
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
    steps_path=folder/'STEPS.parquet';atomic_parquet(pd.DataFrame(steps),steps_path)
    return {'success':int(success),'failure':int(not success),'executed_steps':step,'decision_blocks':decision,
            'max_steps':limit},traces,[own,folder/'PAIRING.json',steps_path,*images]


from sr_geometry import anchor,truth

def offline(index,engine,paths,monitor):
    # One predetermined final pre-contact sample per verification trajectory.
    rows={}
    for r in records('verify'):rows[r['episode_id']]=r
    for n,row in enumerate(rows.values()):
        if n%2!=index:continue
        receipt=ROOT/f'evaluation/offline/{row["state_id"]}.json'
        if complete(receipt):continue
        values=[]
        with torch.no_grad():
            clean,caps=engine.action(row['observation'],row['language']);unit=torch.as_tensor(row['teacher']['unit'],device=clean.device)
            for condition,path in paths.items():
                obs=row['observation'] if path is None else native.paste_numpy(row['observation'],png(path))
                action,_=engine.action(obs,row['language']);goal=engine.command(action,clipped=True)
                target=row['teacher']['targets']['radial_reverse'].to(goal)
                values.append({'suite':row['suite'],'task_id':row['task_id'],'episode_id':row['episode_id'],'state_id':row['state_id'],
                    'condition':condition,'eligible':not row['teacher']['no_target'],'no_target_reason':row['teacher']['no_target_reason'],
                    'target_controller_goal_mse':float((goal-target).square().mean()),
                    'first_slot_radial_goal_m':float((goal[0,0]*unit).sum()),
                    'first_slot_clean_radial_goal_m':row['teacher']['clean_first_slot_radial_goal_m'],
                    'native_action':engine.physical(action).tolist(),'normalized_action':action[0].cpu().tolist()})
        p=receipt.with_suffix('.parquet');atomic_parquet(pd.DataFrame(values),p);finish(receipt,[p],conditions=list(paths))
        monitor.update(phase='offline_native_PNG',completed=n//2+1,total=(len(rows)+1)//2)

def run(index,monitor):
    if complete(ROOT/f'evaluation/worker_{index}_COMPLETE.json'):return
    native.require_gpu('spatial_approach_evaluate');manifest=verify_inputs();paths=conditions();engine=make_engine()
    runtime=native.runtime_module();c=config()
    grid=[(s,t,i) for s in c['suites'] for t in range(10) for i in c['eval_states']]
    grid.sort(key=lambda v:(v[0]!='libero_goal',v))
    assigned=grid[index::2];patches={k:png(v) for k,v in paths.items()};receipts=[]
    for n,(suite,task_id,state) in enumerate(assigned):
        root=ROOT/f'evaluation/episodes/{suite}_t{task_id:02d}_s{state:02d}'
        clean_receipt=root/'clean/COMPLETE.json'
        if complete(root/'SKIPPED_CLEAN_FAILURE.json'):
            receipts.append(root/'SKIPPED_CLEAN_FAILURE.json');continue
        if all(complete(root/k/'COMPLETE.json') for k in paths):
            receipts.extend(root/k/'COMPLETE.json' for k in paths);continue
        entry=manifest['initial_states'][f'{suite}/{task_id}'];env,task,initial=runtime.make_env(native.runtime_config(),suite,task_id)
        try:
            require(task.name==entry['task_name'],'Task mapping changed')
            pinned=torch.load(entry['path'],map_location='cpu',weights_only=False)
            require(np.array_equal(np.asarray(initial[state]),np.asarray(pinned[state])),'Initial file differs from benchmark')
            try:a=anchor(env,suite,task_id);anchor_error=None
            except (ValueError,RuntimeError,KeyError) as error:a=None;anchor_error=repr(error)
            seed=c['rollout_seed_base']+c['suites'].index(suite)*10000+task_id*100+state
            for condition,path in paths.items():
                folder=root/condition;receipt=folder/'COMPLETE.json'
                if condition!='clean' and not read_json(root/'clean/EPISODE.json')['success']:
                    finish(root/'SKIPPED_CLEAN_FAILURE.json',[clean_receipt],reason='clean_Goal_policy_failed_not_an_attack_success');break
                if complete(receipt):continue
                attempt=folder/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True,exist_ok=True)
                monitor.update(phase='native_PNG_closedloop',suite=suite,task_id=task_id,state_index=state,condition=condition,completed=n,total=len(assigned))
                try:result,traces,artifacts=episode(env,task,initial[state],seed,suite,patches[condition],engine,attempt,root/'paired_start',monitor,target_anchor=a)
                except BaseException as error:
                    atomic_json({'status':'ENGINEERING_FAILURE','error':repr(error)},attempt/'FAILURE.json');raise
                meta={'suite':suite,'task_id':task_id,'state_index':state,'rollout_seed':seed,'condition':condition,
                    'checkpoint':c['checkpoint'],'png_sha256':sha(path) if path else None,'geometry_tracking_error':anchor_error,**result}
                metadata=folder/'EPISODE.json';trace=folder/'TRACE.parquet';atomic_json(meta,metadata);atomic_parquet(pd.DataFrame(traces),trace)
                finish(receipt,[metadata,trace,root/'paired_start/START.json',root/'paired_start/START.npz',*artifacts])
            from sr_report import run as report
            report(None,final=False)
        finally:runtime.close_environment(env)
        if complete(root/'SKIPPED_CLEAN_FAILURE.json'):receipts.append(root/'SKIPPED_CLEAN_FAILURE.json')
        else:receipts.extend(root/k/'COMPLETE.json' for k in paths)
    offline(index,engine,paths,monitor)
    offline_rows={}
    for r in records('verify'):offline_rows[r['episode_id']]=r
    receipts.extend(ROOT/f'evaluation/offline/{r["state_id"]}.json' for n,r in enumerate(offline_rows.values()) if n%2==index)
    require(all(complete(p) for p in receipts),'Incomplete evaluation')
    finish(ROOT/f'evaluation/worker_{index}_COMPLETE.json',receipts,checkpoint='fixed_Goal')

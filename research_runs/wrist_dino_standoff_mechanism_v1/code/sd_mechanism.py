"""Paired native actions and short physical rollouts, not an RGB patch experiment."""
import hashlib
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw
from sd_common import (ROOT,config,runtime,runtime_config,require,read_json,atomic_json,atomic_parquet,
    complete,finish,key,assigned,scan_path,scan_receipt,case_root,require_gpu,resources,cost)
from sd_geometry import truth
from sd_model import load_engine,verify_isolation
from sd_math import trajectory_metrics
from sd_replay import restore_case,verify_start

def cf_observation(raw,payload,sign):
    changed={**raw,'robot0_eye_in_hand_image':payload['images'][sign][::-1,::-1].copy()}
    return runtime().observation(changed,runtime_config())

def substitute(engine,obs,language,clean,donor):
    action,features=engine.forward(obs,language,{'dino':donor})
    verify_isolation(clean,features)
    expected=torch.as_tensor(donor).to(engine.wrapper.dtype).float().numpy().reshape(features['dino_1'].shape)
    require(np.array_equal(features['dino_1'],expected),'Spatial replacement not exactly realized in native dtype')
    return action,features

def action_row(name,action,clean_action,clean_feature,feature,valid=True,**extra):
    a=np.asarray(action);delta=a-clean_action
    return {'condition':name,'actions':a.tolist(),'valid_target':valid,
        'first_translation_delta':delta[0,:3].tolist(),'first_translation_command_norm':float(np.linalg.norm(delta[0,:3])),
        'all_slot_translation_command_norm_mean':float(np.linalg.norm(delta[:,:3],axis=1).mean()),
        'feature_change_norm':float(np.linalg.norm(feature-clean_feature)),
        'action_units':'native_controller_commands_not_realized_metres',**extra}

def offline(env,initial,episode,case,engine,monitor):
    folder=case_root(case)/'offline';receipt=folder/'COMPLETE.json'
    if complete(receipt):return
    started=time.monotonic()
    with restore_case(env,initial,episode,case,full=True) as (raw,camera):
        verify_start(env,raw,camera,case,folder);payload=camera.take(raw,case['decision_block'])
        obs=runtime().observation(raw,runtime_config());a,f=engine.forward(obs,case['language'])
        identity,identity_f=substitute(engine,obs,case['language'],f,f['dino_1'])
        require(np.array_equal(a,identity),'Identity spatial replacement changed native actions')
        rows=[action_row('clean',a,a,f['dino_1'],f['dino_1']),
              action_row('identity',identity,a,f['dino_1'],identity_f['dino_1'])]
        zero_a,zero_f=engine.forward(cf_observation(raw,payload,'zero'),case['language'])
        rows.append(action_row('native_zero_rgb',zero_a,a,f['dino_1'],zero_f['dino_1'],
            zero_rgb_nrmse=payload['zero_rgb_nrmse']))
        changed_branch_checks=0;near_donor=None;near_norm=0.
        for sign in ('near','far'):
            info=payload['conditions'][sign]
            if info['valid']:
                natural,cf_f=engine.forward(cf_observation(raw,payload,sign),case['language'])
                encoded=engine.encode_wrist(cf_observation(raw,payload,sign))
                require(np.array_equal(encoded,cf_f['dino_1']),'Standalone DINO differs from consumed native CF feature')
                both,both_f=engine.forward(obs,case['language'],{'dino':cf_f['dino_1'],'siglip':cf_f['siglip_1']})
                require(np.array_equal(natural,both),'Dual-branch swap differs from native CF RGB')
                changed,changed_f=substitute(engine,obs,case['language'],f,cf_f['dino_1']);changed_branch_checks+=1
                if sign=='near':near_donor=changed_f['dino_1'];near_norm=float(np.linalg.norm(near_donor-f['dino_1']))
                rows.append(action_row('dino_'+sign,changed,a,f['dino_1'],changed_f['dino_1'],geometry=info))
                rows.append(action_row('native_'+sign+'_rgb',natural,a,f['dino_1'],cf_f['dino_1'],geometry=info))
            else:
                for name in ('dino_'+sign,'native_'+sign+'_rgb'):
                    rows.append(action_row(name,a,a,f['dino_1'],f['dino_1'],False,geometry=info,
                        fallback='clean_no_target_switch'))
        state_seed=config()['seed']+int(hashlib.sha256(case['case_id'].encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(state_seed)
        for i in range(config()['random_directions']):
            if near_donor is None or near_norm<=1e-12:
                rows.append(action_row(f'random_{i}',a,a,f['dino_1'],f['dino_1'],False,reason='no_valid_nonzero_near_DINO_donor'));continue
            direction=rng.normal(size=f['dino_1'].shape).astype(np.float32);direction/=np.linalg.norm(direction);scale=near_norm
            # Match the norm AFTER native BF16 quantization, not just in FP32.
            for _ in range(5):
                donor=torch.as_tensor(f['dino_1']+scale*direction).to(torch.bfloat16).float().numpy()
                actual=float(np.linalg.norm(donor-f['dino_1']));scale*=near_norm/max(actual,1e-12)
            actual=float(np.linalg.norm(donor-f['dino_1']))
            require(abs(actual/near_norm-1)<.01,'Random native feature norm mismatch >1%')
            changed,cf_f=substitute(engine,obs,case['language'],f,donor)
            rows.append(action_row(f'random_{i}',changed,a,f['dino_1'],cf_f['dino_1'],
                seed=state_seed,norm_ratio=actual/near_norm))
        engine.frozen()
        output={'case_id':case['case_id'],'episode_id':case['episode_id'],'cohort':case['cohort'],'suite':case['suite'],
            'task_id':case['task_id'],'bin':case['bin'],'truth':truth(env,camera.anchor),'rows':rows,
            'geometry':payload['conditions'],'sample_fingerprint':payload['sample_fingerprint'],
            'startup_checks':{'identity_actions_equal':True,'model_frozen':True,'changed_branch_checks':changed_branch_checks,
                              'dino_only_isolation_checked':changed_branch_checks>0},
            'elapsed_s':time.monotonic()-started,**resources()}
        atomic_json(output,folder/'OFFLINE.json')
    finish(receipt,[folder/'OFFLINE.json',folder/'START.npz',folder/'PAIRING.json'])
    monitor.update(phase='offline_checked',case_id=case['case_id'],changed_branch_checks=changed_branch_checks)

def frame(obs,text):
    im=Image.fromarray(np.concatenate([obs['full_image'],obs['wrist_image']],axis=1))
    ImageDraw.Draw(im).text((4,4),text,fill='red');return im

def rollout(env,initial,episode,case,condition,engine,monitor,attempt):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    traces=[];decisions=[];images=[];step=decision=0;rt=runtime();started=time.monotonic();stop_reason='max_steps'
    with restore_case(env,initial,episode,case,full=condition!='clean') as (raw,camera):
        verify_start(env,raw,camera,case,attempt)
        camera.signs=(condition.removeprefix('dino_'),) if condition!='clean' else ()
        first=camera.take(raw,case['decision_block']);current_key=case['decision_block']
        t=truth(env,camera.anchor)
        require(not t['contact'] and not t['grasp'],'Selected start is not contact-free')
        traces.append({**t,'step':0,'visible':first['visible'],'visibility_sample_time_s':first['truth']['physics_time_s'],
            'applied':False,'slot':None,'decision_block':None,'action':None,'same_observation_clean_action':None})
        images.append(frame(rt.observation(raw,runtime_config()),f'{condition} start d={t["distance_m"]*100:.2f}cm'))
        while step<config()['max_steps']:
            payload=camera.take(raw,current_key);obs=rt.observation(raw,runtime_config())
            clean,features=engine.forward(obs,case['language']);action=clean;applied=False
            sign=condition.removeprefix('dino_') if condition!='clean' else None
            info=payload['conditions'].get(sign) if sign else None
            feature_norm=0.
            if sign and info['valid']:
                donor=engine.encode_wrist(cf_observation(raw,payload,sign))
                action,changed=substitute(engine,obs,case['language'],features,donor);applied=True
                feature_norm=float(np.linalg.norm(changed['dino_1']-features['dino_1']))
            used=0;start_step=step;start_eef=np.asarray(t['eef_xyz']);stop=False
            for slot in range(min(8,config()['max_steps']-step)):
                final_slot=slot==min(8,config()['max_steps']-start_step)-1
                next_key=f'rollout_decision_{decision+1}' if final_slot else f'control_step_{step+1}'
                camera.arm(next_key,full=final_slot and condition!='clean')
                environment_action=process_libero_action(action[slot].copy()).tolist()
                raw,_,done,_=env.step(environment_action);step+=1;used+=1
                sample=camera.take(raw,next_key);t=truth(env,camera.anchor)
                traces.append({**t,'step':step,'visible':sample['visible'],
                    'visibility_sample_time_s':sample['truth']['physics_time_s'],'applied':applied,'slot':slot,
                    'decision_block':decision,'action':action[slot].tolist(),'environment_action':environment_action,
                    'same_observation_clean_action':clean[slot].tolist()})
                if t['grasp'] or t['contact'] or done:
                    stop_reason='first_grasp' if t['grasp'] else ('first_target_contact' if t['contact'] else 'task_done');stop=True;break
            delta=action-clean
            decisions.append({'decision_block':decision,'step_start':start_step,'step_end':step,'executed_slots':used,
                'actions':action.tolist(),'same_observation_clean_actions':clean.tolist(),
                'intervention_applied':applied,'geometry':info,'visible_before':payload['visible'],
                'feature_change_norm':feature_norm,'first_translation_command_delta':delta[0,:3].tolist(),
                'first_eef_displacement_m':(np.asarray(traces[start_step+1]['eef_xyz'])-start_eef).tolist(),
                'sample_time_s':payload['truth']['physics_time_s']})
            if decision%2==0 or stop or step==config()['max_steps']:
                images.append(frame(rt.observation(raw,runtime_config()),f'{condition} step={step} d={t["distance_m"]*100:.2f}cm applied={applied}'))
            decision+=1;current_key=next_key
            monitor.update(phase='short_closedloop',case_id=case['case_id'],condition=condition,executed_steps=step,
                distance_cm=t['distance_m']*100,applied=applied,invalid_target_reason=info['reasons'] if info and not info['valid'] else None)
            if stop:break
        engine.frozen()
    metrics=trajectory_metrics(traces,config()['dwell_band_m'],config()['distance_m'])
    dt=float(np.median(np.diff([r['physics_time_s'] for r in traces]))) if len(traces)>1 else 0.
    images[0].save(attempt/'rollout.gif',save_all=True,append_images=images[1:],duration=max(1,round(dt*8*2*1000)),loop=0)
    images[-1].save(attempt/'final.png')
    steps=attempt/'STEPS.parquet';chunks=attempt/'DECISIONS.parquet'
    atomic_parquet(pd.DataFrame(traces),steps);atomic_parquet(pd.DataFrame(decisions),chunks)
    meta={k:case[k] for k in ('case_id','episode_id','suite','task_id','state_index','cohort','bin','decision_block')}
    meta.update(condition=condition,checkpoint_suite='libero_spatial',scope='internal_DINO_dynamic_mechanism_not_PNG',
        stop_reason=stop_reason,**metrics,longest_band_dwell_seconds=metrics['longest_band_dwell_steps']*dt,
        applied_steps=sum(r['applied'] for r in traces[1:]),
        intervention_step_fraction=sum(r['applied'] for r in traces[1:])/max(step,1),
        elapsed_s=time.monotonic()-started,attempt=str(attempt.relative_to(ROOT)),**resources())
    atomic_json(meta,attempt/'EPISODE.json')
    return [attempt/p for p in ('EPISODE.json','STEPS.parquet','DECISIONS.parquet','START.npz','PAIRING.json','rollout.gif','final.png')]

def run(index,monitor):
    episodes=read_json(ROOT/'data/MANIFEST.json')['episodes'];tasks=assigned(index)
    entries=[e for e in episodes if any(key(e)==key(t) for t in tasks)]
    for e in entries:require(complete(scan_receipt(e,e['state_index'])),'Capture incomplete; refusing to load VLA')
    loaded={e['episode_id']:torch.load(scan_path(e,e['state_index']),map_location='cpu',weights_only=False) for e in entries}
    cases=[c for x in loaded.values() for c in x['cases']]
    paths=[case_root(c)/'offline/COMPLETE.json' for c in cases]
    paths += [case_root(c)/condition/'COMPLETE.json' for c in cases for condition in config()['conditions']]
    done=sum(complete(p) for p in paths);monitor.update(completed=done,resume_completed=done,total=len(paths),selected_states=len(cases))
    if done<len(paths):
        require_gpu();rt=runtime();rt.prepare_graphics_backend();engine=load_engine()
        for task in tasks:
            rows=[c for c in cases if key(c)==key(task)]
            if not rows:continue
            env,actual,initial=rt.make_env(runtime_config(),task['suite'],task['task_id'])
            try:
                require(actual.name==read_json(ROOT/'data/MANIFEST.json')['initial_states'][key(task)]['task_name'],'Runtime task mapping changed')
                for case in rows:
                    case['language']=actual.language;episode=next(e for e in entries if e['episode_id']==case['episode_id'])
                    before=complete(case_root(case)/'offline/COMPLETE.json')
                    with cost('offline_'+case['case_id']):offline(env,initial[case['state_index']],episode,case,engine,monitor)
                    if not before:done+=1;monitor.update(completed=done,total=len(paths))
                    order=list(config()['conditions'])
                    random.Random(config()['seed']+int(hashlib.sha256(case['case_id'].encode()).hexdigest()[:8],16)).shuffle(order)
                    for condition in order:
                        receipt=case_root(case)/condition/'COMPLETE.json'
                        if complete(receipt):continue
                        attempt=receipt.parent/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True,exist_ok=False)
                        with cost(condition+'_'+case['case_id']):
                            artifacts=rollout(env,initial[case['state_index']],episode,case,condition,engine,monitor,attempt)
                            finish(receipt,artifacts,attempt=str(attempt.relative_to(ROOT)))
                        done+=1;monitor.update(completed=done,total=len(paths),first_batch_measured=done>=16)
            finally:rt.close_environment(env)
    finish(ROOT/f'reports/MECHANISM_{index}_COMPLETE.json',paths,selected_states=len(cases),**resources())

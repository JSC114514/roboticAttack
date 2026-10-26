"""Saved-action replay, contact-free selection and exact physical start pairing."""
from contextlib import contextmanager
from functools import lru_cache
import gc
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw
from sd_common import (ROOT,config,runtime,runtime_config,require,read_json,atomic_json,atomic_torch,
    atomic_parquet,complete,finish,key,assigned,scan_receipt,scan_path,source_chunks,require_gpu,resources)
from sd_geometry import Camera,truth,anchor_public
from sd_math import choose,paired_arrays

def decode(value):
    a=np.stack([np.asarray(v,dtype=np.float32) for v in value]);require(a.shape==(8,7) and np.isfinite(a).all(),'Invalid saved action chunk');return a

@lru_cache(maxsize=4)
def expected_initials(task_key):
    path=read_json(ROOT/'data/MANIFEST.json')['initial_states'][task_key]['path']
    return torch.load(path,map_location='cpu',weights_only=False)

@contextmanager
def reset_camera(env,initial,episode,full=False,target_decision=0):
    rt=runtime()
    from VLAAttacker.multiview.openvla_oft_experiments import get_libero_dummy_action
    require(np.array_equal(np.asarray(initial),np.asarray(expected_initials(key(episode))[episode['state_index']])),
        'Benchmark initial state differs from pinned source file')
    env.seed(episode['rollout_seed']);rt.bind_environment_context(env);rt.reset_environment(env);gc.collect();rt.renderer_metadata(env)
    raw=env.set_init_state(initial)
    require(np.array_equal(np.asarray(env.get_sim_state()),np.asarray(initial)),'Original initial-state reset mismatch')
    with Camera(env,episode,full=full) as camera:
        for i in range(runtime_config()['num_steps_wait']):
            camera.arm(0 if target_decision==0 and i==runtime_config()['num_steps_wait']-1 else None)
            raw,_,done,_=env.step(get_libero_dummy_action());require(not done,'Task ended during fixed wait')
        rt.renderer_metadata(env);rt.validate_camera_frames(raw)
        yield raw,camera

def start_arrays(env,raw):
    obs=runtime().observation(raw,runtime_config());out={'sim_state':np.array(env.get_sim_state(),copy=True),
        'front':obs['full_image'].copy(),'wrist':obs['wrist_image'].copy(),'proprio':obs['state'].copy()}
    for k in ('body_xpos','body_xquat','geom_xpos','geom_xmat','site_xpos','site_xmat','cam_xpos','cam_xmat'):
        out[k]=np.array(getattr(env.sim.data,k),copy=True)
    for k in ('body_pos','body_quat'):out['model_'+k]=np.array(getattr(env.sim.model,k),copy=True)
    return out

def verify_start(env,raw,camera,case,folder):
    current=start_arrays(env,raw);folder.mkdir(parents=True,exist_ok=True)
    p=folder/'START.npz';np.savez_compressed(p,**current)
    details=paired_arrays(current,case['start_arrays'],config()['rgb_nrmse_limit'])
    payload=camera.take(raw,case['decision_block'])
    sample_equal=payload['sample_fingerprint']==case['sample_fingerprint']
    atomic_json({'arrays':details,'sample_fingerprint_equal':sample_equal,
        'reference_decision':case['decision_block'],'physical_arrays_strict':True,
        'rgb_nrmse_limit':config()['rgb_nrmse_limit']},folder/'PAIRING.json')
    require(all(v['pass'] for v in details.values()) and sample_equal,
        'Paired start mismatch BEFORE new actions; inspect '+str(folder/'PAIRING.json'))
    return p

def restore_case(env,initial,episode,case,full):
    """Context manager yields a deterministic prefix replay, never set_state at a camera sample."""
    return replay_prefix(env,initial,episode,case,full)

@contextmanager
def replay_prefix(env,initial,episode,case,full):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    with reset_camera(env,initial,episode,full,case['decision_block']) as (raw,camera):
        for row in source_chunks(episode).itertuples(index=False):
            decision=int(row.decision_block)
            if decision==case['decision_block']:break
            actions=decode(row.predicted_environment_action_chunk)
            for slot in range(int(row.executed_slots)):
                camera.arm(case['decision_block'] if decision+1==case['decision_block'] and slot==int(row.executed_slots)-1 else None)
                raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist())
                require(not done,'Saved replay ended before selected decision')
        camera.take(raw,case['decision_block']);yield raw,camera

def overlay(raw,payload,path):
    images=[]
    for view,k in (('agentview','agentview_image'),('robot0_eye_in_hand','robot0_eye_in_hand_image')):
        im=Image.fromarray(np.asarray(raw[k])[::-1,::-1].copy());draw=ImageDraw.Draw(im)
        p=payload['projection'][view]
        if p is not None:
            x,y=p;draw.ellipse((x-4,y-4,x+4,y+4),outline='red',width=2)
        draw.text((5,5),payload['anchor']['entity'],fill='red');images.append(im)
    out=Image.new('RGB',(512,256));out.paste(images[0],(0,0));out.paste(images[1],(256,0));out.save(path)

def capture_episode(env,initial,episode,monitor):
    receipt=scan_receipt(episode,episode['state_index'])
    if complete(receipt):return
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    candidates=[];best={};best_distance={};audits=[];contact_seen=False;end_reason='source_actions_exhausted'
    folder=receipt.parent;folder.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    with reset_camera(env,initial,episode,full=False) as (raw,camera):
        for row in source_chunks(episode).itertuples(index=False):
            decision=int(row.decision_block);payload=camera.take(raw,decision);t=truth(env,camera.anchor)
            candidate={**t,'visible':payload['visible'],'visible_pixels':payload['visible_pixels'],
                'decision_block':decision,'task_done':False,'contact_seen':contact_seen,
                'sample_distance_m':payload['truth']['distance_m'],'zero_rgb_nrmse':payload['zero_rgb_nrmse']}
            # "Not yet in contact" means before the first contact, not a later
            # contact-free frame after release. No action-response selection.
            candidate['contact']=candidate['contact'] or contact_seen;candidates.append(candidate)
            selected,_=choose([candidate],config()['bins'])
            for name in selected:
                order=(abs(t['distance_m']-config()['bins'][name]['preferred']),decision)
                if name not in best_distance or order<best_distance[name]:
                    case={**episode,**candidate,'bin':name,'case_id':episode['episode_id']+'_'+name,
                        'start_arrays':start_arrays(env,raw),'sample_fingerprint':payload['sample_fingerprint'],
                        'anchor':anchor_public(camera.anchor)}
                    best[name]=case;best_distance[name]=order
                    overlay(raw,payload,folder/f'{name}_anchor.png')
            actions=decode(row.predicted_environment_action_chunk);used=0
            for slot in range(int(row.executed_slots)):
                camera.arm(decision+1 if slot==int(row.executed_slots)-1 else None)
                raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist());used+=1
                now=truth(env,camera.anchor)
                if now['contact'] or now['grasp'] or done:
                    contact_seen=now['contact'] or now['grasp'];end_reason='first_target_contact_or_grasp' if contact_seen else 'task_done';break
            audits.append({'decision_block':decision,'executed_slots':used,'source_executed_slots':int(row.executed_slots)})
            monitor.update(phase='replay_select',episode_id=episode['episode_id'],source_decision=decision,
                selected_bins=list(best),distance_cm=t['distance_m']*100)
            if contact_seen or done:break
    expected,missing=choose(candidates,config()['bins'])
    require(set(expected)==set(best),'Online selection disagrees with full candidate rule')
    data=scan_path(episode,episode['state_index']);atomic_torch({'episode':episode,'cases':list(best.values()),
        'missing_bins':missing,'anchor_definition':'collision_AABB_center_not_exact_grasp_point'},data)
    candidate_path=folder/'CANDIDATES.parquet';atomic_parquet(pd.DataFrame(candidates),candidate_path)
    summary=folder/'SELECTION.json';atomic_json({'episode_id':episode['episode_id'],'cohort':episode['cohort'],
        'selected':[{k:v for k,v in r.items() if k not in ('start_arrays','sample_fingerprint')} for r in best.values()],
        'missing_bins':missing,'end_reason':end_reason,'audits':audits,'policy_called':False,
        'elapsed_s':time.monotonic()-started,**resources()},summary)
    finish(receipt,[data,candidate_path,summary,*[folder/f'{b}_anchor.png' for b in best]],selected_states=len(best))

def run(index,monitor):
    tasks=assigned(index);episodes=read_json(ROOT/'data/MANIFEST.json')['episodes']
    todo=[e for e in episodes if any(key(e)==key(t) for t in tasks)]
    done=sum(complete(scan_receipt(e,e['state_index'])) for e in todo)
    monitor.update(completed=done,resume_completed=done,total=len(todo))
    if done<len(todo):
        require_gpu();rt=runtime();rt.prepare_graphics_backend()
        for task in tasks:
            rows=[e for e in todo if key(e)==key(task)]
            if all(complete(scan_receipt(e,e['state_index'])) for e in rows):continue
            env,actual,initial=rt.make_env(runtime_config(),task['suite'],task['task_id'])
            try:
                require(actual.name==read_json(ROOT/'data/MANIFEST.json')['initial_states'][key(task)]['task_name'],'Runtime task mapping changed')
                for e in rows:
                    if complete(scan_receipt(e,e['state_index'])):continue
                    capture_episode(env,initial[e['state_index']],e,monitor);done+=1
                    monitor.update(completed=done,total=len(todo))
            finally:rt.close_environment(env)
    finish(ROOT/f'reports/CAPTURE_{index}_COMPLETE.json',[scan_receipt(e,e['state_index']) for e in todo],**resources())

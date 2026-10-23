"""Replay saved native commands; collect pre-contact clean RGB and labels."""
from collections import deque,Counter
import gc
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw
from sr_common import (ROOT,native,config,episodes,read_json,require,cache_receipt,complete,finish,
    atomic_json,atomic_torch,make_engine,signature)
from sr_geometry import anchor,truth,Capture,controller
from sr_math import retreat_target,far_target

def replay(env,initial,e,traces,monitor):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action,get_libero_dummy_action
    rt=native.runtime_module();env.seed(e['rollout_seed']);rt.bind_environment_context(env);rt.reset_environment(env);gc.collect();rt.renderer_metadata(env)
    raw=env.set_init_state(initial);require(np.array_equal(np.asarray(env.get_sim_state()),np.asarray(initial)),'Replay reset differs')
    a=anchor(env,e['suite'],e['task_id']);scale=controller(env);ring=deque(maxlen=config()['frames_before_first_contact'])
    contact_seen=False;first_contact=None;executed=0
    with Capture(env,a) as camera:
        for i in range(native.runtime_config()['num_steps_wait']):
            camera.arm(0 if i==native.runtime_config()['num_steps_wait']-1 else None)
            raw,_,done,_=env.step(get_libero_dummy_action());require(not done,'Task ended during fixed wait')
        for saved in traces.itertuples(index=False):
            decision=int(saved.decision_block);payload=camera.take(raw,decision);t=payload['truth']
            if t['contact'] or t['grasp']:contact_seen=True;first_contact=executed;break
            obs=rt.observation(raw,native.runtime_config())
            ring.append({**e,'decision_block':decision,'state_id':e['episode_id']+f'_d{decision:04d}',
                'observation':obs,'geometry':t,'sample_fingerprint':payload['fingerprint'],
                'visible_pixels':payload['visible_pixels'],'zero_rgb_nrmse':payload['zero_rgb_nrmse']})
            actions=native.decode_action_chunk(saved.predicted_environment_action_chunk,e['episode_id']);used=int(saved.executed_slots)
            require(1<=used<=8,'Invalid native executed slots')
            for slot in range(used):
                camera.arm(decision+1 if slot==used-1 else None)
                raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist());executed+=1
                now=truth(env,a)
                if now['contact'] or now['grasp']:contact_seen=True;first_contact=executed;break
                if done:break
            monitor.update(phase='native_saved_action_replay',episode_id=e['episode_id'],decision=decision,replayed_steps=executed)
            if contact_seen or done:break
            require(executed==int(saved.step_end),'Source replay boundary changed')
    return list(ring),{'source_first_contact_seen':contact_seen,'first_contact_step':first_contact,'executed_source_steps':executed,
        'anchor':{k:v for k,v in a.items() if k!='gripper'},'controller':scale}

def teach(engine,row,source_contact):
    with torch.no_grad():
        action,_=engine.action(row['observation'],row['language']);clean=engine.command(action,clipped=True).cpu().numpy()
    g=row['geometry'];r=np.asarray(g['target_xyz'])-g['eef_xyz'];d=float(np.linalg.norm(r))
    reason=None
    if not source_contact:reason='source_never_contacted_fixed_target'
    elif not row['visible_pixels']:reason='fixed_target_not_visible'
    elif d<=1e-9:reason='undefined_direction'
    unit=r/max(d,1e-9);radial=float(clean[0,0]@unit)
    if reason is None and radial<=0:reason='Goal_clean_first_slot_not_approaching'
    reverse=retreat_target(clean,unit,np.full(3,-.05),np.full(3,.05),config()['retreat_goal_m']) if d>1e-9 else clean.copy()
    far=far_target(clean,np.full(3,-.05),np.full(3,.05))
    return {'clean':action.cpu(),'targets':{'radial_reverse':torch.as_tensor(reverse),'translation_far_boundary':torch.as_tensor(far)},
        'unit':unit.tolist(),'command_scale':torch.full((3,),.05),'no_target':reason is not None,'no_target_reason':reason,
        'clean_first_slot_radial_goal_m':radial,'target_first_slot_radial_goal_m':float(reverse[0,0]@unit),
        'units':'controller_goal_delta_m_not_realized_displacement'}

def run(index,monitor):
    native.require_gpu('spatial_approach_capture');rt=native.runtime_module();rt.prepare_graphics_backend()
    manifest=read_json(ROOT/'data/MANIFEST.json');assigned=episodes()[index::2]
    done=sum(complete(cache_receipt(e['episode_id'])) for e in assigned)
    if done==len(assigned):return
    traces=pd.read_parquet(manifest['trace_path'],filters=[('condition','==','clean')]);engine=None
    for suite,task_id in sorted({(e['suite'],e['task_id']) for e in assigned}):
        todo=[e for e in assigned if e['suite']==suite and e['task_id']==task_id]
        if all(complete(cache_receipt(e['episode_id'])) for e in todo):continue
        env,task,initial=rt.make_env(native.runtime_config(),suite,task_id,on_phase=lambda phase,**kw:monitor.update(phase=phase,**kw))
        try:
            entry=manifest['initial_states'][f'{suite}/{task_id}'];require(task.name==entry['task_name'],'Task mapping changed')
            pinned=torch.load(entry['path'],map_location='cpu',weights_only=False)
            for e in todo:
                receipt=cache_receipt(e['episode_id'])
                if complete(receipt):continue
                require(np.array_equal(np.asarray(initial[e['state_index']]),np.asarray(pinned[e['state_index']])),'Runtime initial differs from pinned file')
                current=traces[(traces.suite==suite)&(traces.task_id==task_id)&(traces.state_index==e['state_index'])&(traces.rollout_seed==e['rollout_seed'])].sort_values('decision_block')
                require(len(current)==e['decision_count'] and list(current.decision_block)==list(range(len(current))),'Saved source actions incomplete')
                rows,audit=replay(env,initial[e['state_index']],e,current,monitor)
                if engine is None and rows:engine=make_engine()
                for rank,row in enumerate(rows):
                    row['language']=task.language;row['frame_rank']=rank;row['teacher']=teach(engine,row,audit['source_first_contact_seen'])
                path=receipt.with_suffix('.pt');atomic_torch({'signature':signature(),'episode_id':e['episode_id'],'records':rows,'audit':audit},path)
                audit_path=receipt.with_name(receipt.stem+'_AUDIT.json')
                atomic_json({**audit,'episode_id':e['episode_id'],'cohort':e['cohort'],'frames':len(rows),
                    'eligible':sum(not r['teacher']['no_target'] for r in rows),
                    'ineligible_reasons':dict(Counter(r['teacher']['no_target_reason'] for r in rows if r['teacher']['no_target'])),
                    'reference_point':'fixed_functional_collision_AABB_center_not_exact_grasp_point',
                    'source_policy':'per_suite_native_source_not_current_Goal_policy','model':config()['checkpoint']},audit_path)
                artifacts=[path,audit_path]
                if rows:
                    last=rows[-1];obs=last['observation'];im=Image.fromarray(np.concatenate([obs['full_image'],obs['wrist_image']],axis=1))
                    ImageDraw.Draw(im).text((3,3),audit['anchor']['entity']+' '+audit['anchor']['kind'],fill='red')
                    preview=receipt.with_suffix('.png');im.save(preview);artifacts.append(preview)
                finish(receipt,artifacts,cohort=e['cohort'],frames=len(rows),eligible=sum(not r['teacher']['no_target'] for r in rows))
                done+=1;monitor.update(phase='cache_complete',completed=done,total=len(assigned))
        finally:rt.close_environment(env)
    finish(ROOT/f'reports/CACHE_{index}_COMPLETE.json',[cache_receipt(e['episode_id']) for e in assigned])

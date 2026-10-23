"""Recorded-action references, then actual frozen-policy restoration episodes."""
import copy
from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image
from rr_common import ROOT,SOURCE,AUDIT,config,adapters,require,read,atomic,complete,finish
from rr_math import restore,executed_radial_error
from rr_logger import PhysicsLogger
from audit_math import decode_actions,visibility,frame_alignment,event_summary
from audit_replay import SensorCapture,frame,fingerprint,rgb_check

def folder_for(case,method):return ROOT/'episodes'/case['start']/method

def validate_frozen(wrapper):
    modules=(wrapper.vla,wrapper.action_head,wrapper.proprio_projector,wrapper.noisy_action_projector)
    require(all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in modules if m is not None),'Victim must remain frozen/eval')

def episode(env,task,initial,case,method,attempt,wrapper,monitor):
    import torch
    import imageio.v2 as imageio
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    from sr_geometry import truth,anchor,controller
    native=adapters().native;runtime=native.runtime_module();cfg=native.runtime_config();c=config()
    reference=method=='reference_replay';raw=runtime.reset_wait(env,initial,case['episode']['rollout_seed'],cfg['num_steps_wait'])
    obs=runtime.observation(raw,cfg);old_start=read(Path(case['paired_start'])/'START.json')
    require(fingerprint(env)==old_start['fingerprint'] and old_start['rollout_seed']==case['episode']['rollout_seed'],'Original paired reset fingerprint changed')
    with np.load(Path(case['attempt'])/'START.npz',allow_pickle=False) as old:
        require(np.array_equal(env.get_sim_state(),old['sim_state']) and np.array_equal(obs['state'],old['proprio']),'Initial physical arrays/proprio differ')
        reset_rgb={'front':rgb_check(obs['full_image'],old['front']),'wrist':rgb_check(obs['wrist_image'],old['wrist'])}
    np.savez_compressed(attempt/'START.npz',sim_state=np.array(env.get_sim_state(),copy=True),proprio=obs['state'],front=obs['full_image'],wrist=obs['wrist_image'])
    target=anchor(env,'libero_goal',case['episode']['task_id']);controller_evidence=controller(env)
    for field in ('position_limits','orientation_limits'):
        value=getattr(env.env.robots[0].controller,field)
        controller_evidence[field]=None if value is None else np.asarray(value).tolist()
    old_trace=pd.read_parquet(case['trace']).sort_values('decision_block');old_steps=pd.read_parquet(case['steps']).sort_values('step')
    require(list(old_trace.decision_block)==list(range(len(old_trace))),'Historical trace has gaps')
    with Image.open(case['patch']) as im:patch=np.array(im,copy=True)
    require(patch.shape==(50,50,3) and patch.dtype==np.uint8,'Unexpected fixed patch')
    fps=1/float(env.env.control_timestep);geo=truth(env,target);events=[];steps=[];traces=[];step=decision=0;success=False;max_error=0.;keyframes={}
    limit=case['episode']['executed_steps'] if reference else c['max_steps']
    gate=None;degenerate_count=0
    def event(step,geometry,captured=None,physical=None):
        vis=visibility(captured['mask'],True,c['patch_xy']) if captured else None
        r={'step':step,'time_s':step/fps,'sensor_time_s':captured['sample_time_s'] if captured else None,
            'visibility_verified':captured is not None,'zero_rgb_nrmse':captured['zero_rgb_nrmse'] if captured else None,
            'distance_m':geometry['distance_m'],'contact':geometry['contact'],'grasp':geometry['grasp'],
            'camera_xyz':captured['camera_xyz'] if captured else None,'camera_rotation_matrix':captured['camera_rotation_matrix'] if captured else None,
            **(vis or {k:None for k in ('raw_visible_pixels','policy_fov_pixels','policy_uncovered_pixels','patch_covered_pixels')})}
        for k in ('tracking_error_final_m','tracking_error_peak_m','orientation_error_final_rad','orientation_error_peak_rad',
                  'torque_clip_peak_nm','actuator_bound_samples','contact_pair_count','peak_contact_force_n','peak_robot_contact_force_n'):
            r[k]=physical[k] if physical else None
        events.append(r);return vis
    video=attempt/'video.mp4'
    with imageio.get_writer(video,fps=fps,codec='libx264',quality=7,macro_block_size=16,
            ffmpeg_params=['-threads','2']) as writer, SensorCapture(env,target) as camera,PhysicsLogger(env) as physics:
        event(0,geo);writer.append_data(frame(native.paste_numpy(obs,patch),method,0,fps,geo,None))
        while step<limit and not success:
            obs=runtime.observation(raw,cfg);patched=native.paste_numpy(obs,patch)
            unit=np.asarray(geo['target_xyz'])-np.asarray(geo['eef_xyz']);norm=float(np.linalg.norm(unit))
            if reference:
                old=old_trace.iloc[decision];require(int(old.step_start)==step,'Reference block start changed')
                ap=decode_actions(old.predicted_actions);ac=decode_actions(old.same_observation_clean_actions);executed=ap.copy();meta={'degenerate_direction':norm<=c['direction_epsilon_m']}
                slots=int(old.executed_slots)
                if decision in (0,4,12):
                    with Image.open(Path(case['attempt'])/f'frame_{decision:03d}.png') as original:
                        keyframes[str(decision)]=rgb_check(np.concatenate([patched['full_image'],patched['wrist_image']],axis=1),original)
            else:
                with torch.no_grad():
                    ap=decode_actions(wrapper.predict_action(copy.deepcopy(patched),task.language))
                    ac=decode_actions(wrapper.predict_action(copy.deepcopy(obs),task.language))
                if decision==0:
                    saved_ap=decode_actions(old_trace.iloc[0].predicted_actions);saved_ac=decode_actions(old_trace.iloc[0].same_observation_clean_actions)
                    gate={'patch_max_abs_error':float(np.max(np.abs(ap-saved_ap))),'clean_max_abs_error':float(np.max(np.abs(ac-saved_ac)))}
                    require(max(gate.values())<=c['action_tolerance'],'Native first-block forward changed: '+repr(gate))
                    validate_frozen(wrapper)
                executed,meta=restore(ap,ac,unit,method,c['direction_epsilon_m']);slots=min(8,limit-step)
                if meta['radial_error_m'] is not None:require(meta['radial_error_m']<=c['radial_tolerance_m'],'Radial equality violated')
            degenerate_count+=int(meta['degenerate_direction']);start=step
            for slot in range(slots):
                before=geo;before_eef=np.asarray(raw['robot0_eef_pos']).copy()
                if reference:
                    expected=old_steps.iloc[step]
                    require(int(expected.step)==step+1 and int(expected.executed_slot)==slot and int(expected.decision_block)==decision,'Reference slot changed')
                    require(np.array_equal(ap[slot],np.asarray(expected.deployed_action,dtype=np.float32)),'Reference action changed')
                    frame_alignment(before_eef,expected.eef_before,c['position_tolerance_m'])
                camera.arm(step+1);physics.arm()
                actual_input=process_libero_action(executed[slot].copy())
                actual_radial_error=actual_radial_bound=None
                if method=='restore_radial' and norm>c['direction_epsilon_m']:
                    actual_radial_error,actual_radial_bound=executed_radial_error(actual_input,ac[slot],unit,c['radial_tolerance_m'])
                    require(actual_radial_error<=actual_radial_bound,'Native float32 processing changed radial command beyond its roundoff bound')
                raw,_,done,_=env.step(actual_input.tolist());step+=1
                sampled=camera.take(raw,step);physical=physics.take();geo=truth(env,target)
                after_eef=np.asarray(raw['robot0_eef_pos']).copy();delta=after_eef-before_eef
                if reference:
                    original=expected.geometry_after
                    max_error=max(max_error,frame_alignment(after_eef,expected.eef_after,c['position_tolerance_m']),
                        frame_alignment(geo['target_xyz'],original['target_xyz'],c['position_tolerance_m']),
                        frame_alignment(geo['eef_xyz'],original['eef_xyz'],c['position_tolerance_m']))
                    require(geo['contact']==original['contact'] and geo['grasp']==original['grasp'],'Reference target contact/grasp changed')
                    require(not done or step==limit,'Reference ended early')
                actual_relative=np.asarray(before['target_xyz'])-np.asarray(before['eef_xyz']);actual_unit=actual_relative/max(np.linalg.norm(actual_relative),1e-12)
                steps.append({'step':step,'decision_block':decision,'executed_slot':slot,'eef_before':before_eef.tolist(),'eef_after':after_eef.tolist(),
                    'geometry_before':before,'geometry_after':geo,'patch_native_action':ap[slot].tolist(),'clean_same_observation_native_action':ac[slot].tolist(),
                    'restored_native_action':executed[slot].tolist(),'environment_action':actual_input.tolist(),
                    'executed_radial_error_m':actual_radial_error,'executed_radial_roundoff_bound_m':actual_radial_bound,
                    'decision_target_unit':(unit/norm).tolist() if norm>c['direction_epsilon_m'] else None,
                    'actual_eef_displacement_m':delta.tolist(),'radial_eef_displacement_m':float(delta@actual_unit),'controller':physical})
                obs=runtime.observation(raw,cfg);vis=event(step,geo,sampled,physical)
                writer.append_data(frame(native.paste_numpy(obs,patch),method,step,fps,geo,vis))
                monitor.update(current_start=case['start'],condition=method,current_step=step,current_limit=limit)
                success=bool(done)
                if success:break
            if reference:require(int(old.step_end)==step and bool(old.success_after_chunk)==success,'Reference block outcome changed')
            traces.append({'decision_block':decision,'step_start':start,'step_end':step,'executed_slots':step-start,
                'patch_actions':ap.tolist(),'clean_same_observation_actions':ac.tolist(),'restored_actions':executed.tolist(),
                'restoration':meta,'success_after_chunk':success})
            decision+=1
        if reference:
            require(step==case['episode']['executed_steps'] and success==bool(case['episode']['success']),'Reference outcome changed')
            final=native.paste_numpy(obs,patch)
            with Image.open(Path(case['attempt'])/'frame_final.png') as original:
                keyframes['final']=rgb_check(np.concatenate([final['full_image'],final['wrist_image']],axis=1),original)
    if wrapper is not None:validate_frozen(wrapper)
    ef=pd.DataFrame(events);ef.to_parquet(attempt/'EVENTS.parquet',index=False);pd.DataFrame(steps).to_parquet(attempt/'STEPS.parquet',index=False)
    pd.DataFrame(traces).to_parquet(attempt/'TRACE.parquet',index=False)
    result={'start':case['start'],'suite':'libero_goal','task_id':case['episode']['task_id'],'state_index':case['episode']['state_index'],
        'condition':method,'success':int(success),'executed_steps':step,'decision_blocks':decision,'fps':fps,
        'new_policy_episode':not reference,'privileged_diagnosis_only':True,'seed':case['episode']['rollout_seed'],
        'reset_rgb_nrmse':reset_rgb,'native_first_block_gate':gate,'controller_evidence':controller_evidence,
        'target_annotation':{k:target[k] for k in ('entity','kind','contact_ids','render_ids','hand_ids','local_center')},
        'reference_max_position_error_m':max_error if reference else None,'reference_keyframe_rgb_nrmse':keyframes if reference else None,
        'degenerate_direction_blocks':degenerate_count,'events':event_summary(ef),'hooks_restored':True}
    atomic(result,attempt/'EPISODE.json')
    finish(attempt/'COMPLETE.json',[attempt/n for n in ('video.mp4','EVENTS.parquet','STEPS.parquet','TRACE.parquet','START.npz','EPISODE.json')])

def run(manifest,monitor):
    import os
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'),'User-submitted one-GPU allocation required')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import torch
    s=adapters();native=s.native;runtime=native.runtime_module();cfg=native.runtime_config()
    env=None;identity=None;wrapper=None;engine=None;receipts=[];global_start=time.monotonic()
    try:
        for phase,methods in (('reference_replay',['reference_replay']),('restoration',config()['methods'])):
            work=[]
            for case in manifest['cases']:
                order=methods if phase=='reference_replay' else case['order']
                for method in order:
                    receipt=folder_for(case,method)/'COMPLETE.json'
                    if complete(receipt):receipts.append(receipt)
                    else:work.append((case,method))
            if not work:continue
            if phase=='restoration':
                require(all(complete(folder_for(e,'reference_replay')/'COMPLETE.json') for e in manifest['cases']),'References must pass before policy episodes')
                monitor.update(phase='load_frozen_Goal_model',completed_episodes=len(receipts),total_episodes=72)
                native.require_gpu('spatial_relation_rescue_evaluate');engine=s.make_engine();wrapper=engine.wrapper;validate_frozen(wrapper)
            phase_start=time.monotonic();phase_steps=0;done_count=0
            for case,method in work:
                next_identity=('libero_goal',case['episode']['task_id'])
                if next_identity!=identity:
                    if env is not None:runtime.close_environment(env);env=None
                    env,task,states=runtime.make_env(cfg,*next_identity);identity=next_identity
                require(task.name==case['task_name'],'Benchmark task mapping changed')
                saved=torch.load(case['initial_file'],map_location='cpu',weights_only=False);initial=states[case['episode']['state_index']]
                require(np.array_equal(initial,saved[case['episode']['state_index']]),'Pinned initial state differs from benchmark')
                folder=folder_for(case,method);folder.mkdir(parents=True,exist_ok=True);attempt=folder/f'attempt_{time.time_ns()}';attempt.mkdir()
                monitor.update(phase=phase,current_start=case['start'],condition=method,completed_episodes=len(receipts),total_episodes=72)
                started=time.monotonic()
                try:episode(env,task,initial,case,method,attempt,wrapper if phase=='restoration' else None,monitor)
                except BaseException as error:
                    atomic({'status':'ENGINEERING_FAILURE','error':repr(error),'start':case['start'],'condition':method,'elapsed_s':time.monotonic()-started},attempt/'FAILURE.json');raise
                artifacts=[attempt/n for n in ('COMPLETE.json','video.mp4','EVENTS.parquet','STEPS.parquet','TRACE.parquet','START.npz','EPISODE.json')]
                finish(folder/'COMPLETE.json',artifacts);receipts.append(folder/'COMPLETE.json');done_count+=1
                result=read(attempt/'EPISODE.json');phase_steps+=result['executed_steps'];elapsed=time.monotonic()-phase_start
                remaining=(len(work)-done_count)*elapsed/done_count
                if phase=='reference_replay':remaining+=54*40  # Unmeasured policy/video budget, replaced after native episodes.
                monitor.update(completed_episodes=len(receipts),estimated_remaining_s=remaining,
                    phase_steps_per_second=phase_steps/max(elapsed,1e-9),elapsed_total_s=time.monotonic()-global_start)
                if done_count==(2 if phase=='reference_replay' else 3):
                    atomic({'phase':phase,'new_completed_episodes':done_count,'elapsed_s':elapsed,'native_steps':phase_steps,
                        'estimated_remaining_s':remaining,'note':'Includes native rendering/encoding; reference estimate includes unmeasured policy budget'},
                        ROOT/f'reports/TIMING_{phase}.json')
    finally:
        if env is not None:runtime.close_environment(env)
    require(len(receipts)==72,'Missing restoration/reference receipts')
    finish(ROOT/'reports/EVALUATE_COMPLETE.json',receipts)

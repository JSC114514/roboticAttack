"""Saved-prefix replay and paired 64-control-step diagnostics; never train VLA."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
from va_common import ROOT,SOURCE,config,require,adapters,read,atomic,save_torch,finish,complete,branch_receipt,branch_files,reference_path
from va_math import condition_spec,blur
from va_vision import pooled
from va_camera import Sensors

def rng_state():return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
def restore_rng(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state(state[3])

@lru_cache(maxsize=1)
def cached_rows(path):return torch.load(path,map_location='cpu',weights_only=False)['records']

def source_rows(case):return cached_rows(case['cache'])

@lru_cache(maxsize=1)
def cached_trace(task,state,seed,count):
    trace=pd.read_parquet(read(SOURCE/'data/MANIFEST.json')['trace_path'],filters=[('condition','==','clean'),('suite','==','libero_goal'),('task_id','==',task),('state_index','==',state)])
    trace=trace[trace.rollout_seed==seed].sort_values('decision_block')
    require(list(trace.decision_block)==list(range(count)),'Source action prefix missing')
    return trace

@contextmanager
def replay(case,constructor_rng,monitor):
    from sr_geometry import anchor
    from VLAAttacker.multiview.openvla_oft_experiments import get_libero_dummy_action,process_libero_action
    source,native=adapters();rt=native.runtime_module();cfg=native.runtime_config()
    restore_rng(constructor_rng);env=None
    try:
        env,task,states=rt.make_env(cfg,'libero_goal',case['task_id'])
        require(task.name==case['task_name'],'Task mapping differs')
        initial=torch.load(case['initial_file'],map_location='cpu',weights_only=False)[case['state_index']]
        require(np.array_equal(initial,states[case['state_index']]),'Initial state differs')
        env.seed(case['rollout_seed']);rt.bind_environment_context(env);rt.reset_environment(env)
        __import__('gc').collect();rt.renderer_metadata(env)
        raw=env.set_init_state(initial);require(np.array_equal(env.get_sim_state(),initial),'Reset array mismatch')
        target=anchor(env,'libero_goal',case['task_id']);history=[]
        with Sensors(env,target) as sensors:
            control=-cfg['num_steps_wait']
            for _ in range(cfg['num_steps_wait']):
                control+=1;sensors.arm(control)
                raw,_,done,_=env.step(get_libero_dummy_action());require(not done,'Task ended in fixed warmup')
                sensors.take(raw);history.append(copy.deepcopy(rt.observation(raw,cfg)))
            trace=cached_trace(case['task_id'],case['state_index'],case['rollout_seed'],case['decision_count'])
            for saved in trace.itertuples():
                if int(saved.decision_block)>=case['first_decision']:break
                actions=native.decode_action_chunk(saved.predicted_environment_action_chunk,case['episode_id'])
                for slot in range(int(saved.executed_slots)):
                    control+=1;sensors.arm(control)
                    raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist())
                    require(not done,'Saved prefix ended before selected point')
                    sensors.take(raw);history.append(copy.deepcopy(rt.observation(raw,cfg)))
                    history=history[-10:]
            row=min(source_rows(case),key=lambda r:r['decision_block']);obs=rt.observation(raw,cfg)
            require(sensors.latest['wrist']['fingerprint']==row['sample_fingerprint'],'Cached sensor-time physical/camera/clock mismatch')
            for k in ('full_image','wrist_image','state'):require(np.array_equal(obs[k],row['observation'][k]),'Source cached observation differs: '+k)
            require(len(history)>=9,'No real frame eight controls before selected point')
            monitor.update(current_start=case['episode_id'],phase='strict_saved_prefix_reproduced')
            yield env,task,raw,sensors,target,history,row
    finally:
        if env is not None:rt.close_environment(env)

class Videos:
    def __init__(self,folder,fps):self.folder=folder;self.fps=fps;self.writers={}
    def __enter__(self):
        import imageio.v2 as imageio
        try:
            for view in ('front','wrist'):
                self.writers[view]=imageio.get_writer(self.folder/(view+'.mp4'),fps=self.fps,codec='libx264',quality=7,
                    macro_block_size=16,ffmpeg_params=['-threads','1','-pix_fmt','yuv420p','-profile:v','baseline','-movflags','+faststart'])
        except BaseException:self.__exit__();raise
        return self
    def append(self,observation):
        for view,key in (('front','full_image'),('wrist','wrist_image')):self.writers[view].append_data(observation[key])
    def __exit__(self,*_):
        for w in self.writers.values():w.close()

def deployed_observation(raw,sensors,history,spec,step,patch):
    native=adapters()[1];obs=copy.deepcopy(sensors.observation(raw))
    if spec.get('fault'):
        key={'front':'full_image','wrist':'wrist_image'}[spec['damaged']]
        if spec['fault']=='blur':obs[key]=blur(obs[key],config()['blur_sigma'],config()['blur_kernel'])
        else:
            require(len(history)>=9,'Delay history absent')
            obs[key]=np.array(history[-9][key],copy=True)
    if patch is not None and (not spec['first_chunk'] or step<8):obs=native.paste_numpy(obs,patch)
    return obs

def run_episode(case,condition,vision,means,monitor,constructor_rng,reference=None):
    from sr_geometry import truth
    from audit_math import visibility
    from audit_replay import fingerprint
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    native=adapters()[1];rt=native.runtime_module();cfg=native.runtime_config()
    receipt=branch_receipt(case,condition)
    if complete(receipt):return read(branch_files(receipt)['DETAIL.json'])
    spec=condition_spec(condition);attempt=receipt.parent/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True)
    patches=read(ROOT/'data/MANIFEST.json')['patches'];patch=None
    if spec['patch']:
        from PIL import Image
        patch=np.array(Image.open(patches[spec['patch']]).convert('RGB'),copy=True)
    horizon=config()['max_steps'] if reference is None else reference['horizon']
    amplitudes=None;amplitude_valid_until=horizon
    if spec['camera']=='random':
        donor=branch_receipt(case,'b_'+spec['camera_source']+'_replay')
        require(complete(donor),'Replay camera amplitudes must precede matched random control')
        detail=read(branch_files(donor)['DETAIL.json']);amplitudes={int(k):v for k,v in detail['camera_amplitudes'].items()}
        amplitude_valid_until=max(amplitudes)
        # A terminal donor has no later counterfactual samples. Both random arms
        # hold its terminal magnitude; causal contrasts use the recorded common prefix.
        amplitudes.update({i:amplitudes[amplitude_valid_until] for i in range(amplitude_valid_until+1,horizon+1)})
    with replay(case,constructor_rng,monitor) as (env,task,raw,sensors,target,history,row):
        start=dict(fingerprint=fingerprint(env),sensor_fingerprint=sensors.latest['wrist']['fingerprint'])
        arrays=dict(sim_state=np.array(env.get_sim_state(),copy=True),**{k:np.array(v,copy=True) for k,v in row['observation'].items()})
        if reference is not None:
            require(start==reference['start'],'Paired control/sensor start differs')
            for k,v in arrays.items():require(np.array_equal(v,reference['start_arrays'][k]),'Paired start array differs: '+k)
        expected=vision.wrapper.vla._unnormalize_actions(row['teacher']['clean'].numpy().reshape(8,7),vision.wrapper.gen_cfg.unnorm_key)
        native_first,_=vision.action(row['observation'],task.language)
        action_error=float(np.max(abs(expected-native_first)));require(action_error<=config()['action_tolerance'],'First native action differs')
        seed=config()['seed']+case['task_id']*100+case['state_index']*2+(spec.get('camera_source')=='uada7')
        sensors.activate(spec['camera'],reference['camera_path'] if reference else {},amplitudes,seed)
        if condition=='b_clean_replay':
            require(np.sqrt(np.mean(((sensors.latest['wrist']['pixels'].astype(float)-sensors.latest['wrist']['alternate_pixels'])/255.)**2))<=config()['rgb_tolerance'],'Initial clean camera replay not identity')
        fps=1/float(env.env.control_timestep);events=[];decisions=[];features=[];proprios=[];camera_path={};camera_amplitudes={}
        previous=copy.deepcopy(history[-9]);previous_features=pooled(vision.features(previous,task.language))
        previous_proprio=np.array(previous['state'],copy=True);success=False;step=0
        fixed_target=np.asarray(row['geometry']['target_xyz']);initial_distance=float(np.linalg.norm(fixed_target-row['geometry']['eef_xyz']))
        def record():
            w=sensors.latest['wrist'];g=truth(env,target);camera_path[step]={'position':w['position'].tolist(),'rotation':w['rotation'].tolist(),'sample_time_s':w['sample_time_s']}
            camera_amplitudes[step]={'translation_m':w['translation_m'],'rotation_rad':w['rotation_rad']}
            obs=rt.observation(raw,cfg)
            events.append(dict(step=step,time_s=step/fps,sensor_time_s=w['sample_time_s'],contact=g['contact'],grasp=g['grasp'],
                distance_m=g['distance_m'],distance_to_start_target_m=float(np.linalg.norm(fixed_target-g['eef_xyz'])),
                eef_xyz=g['eef_xyz'],target_xyz=g['target_xyz'],
                actual_eef_displacement_m=(np.asarray(g['eef_xyz'])-np.asarray(events[-1]['eef_xyz'])).tolist() if events else [0.,0.,0.],
                front_visible_pixels=int(sensors.latest['front']['mask'].sum()),
                **visibility(w['mask'],patch is not None and (not spec['first_chunk'] or step<8)),
                camera_translation_m=w['translation_m'],camera_rotation_rad=w['rotation_rad']))
            return obs
        obs=record();deployed=deployed_observation(raw,sensors,history,spec,step,patch)
        gate=None
        if spec.get('gate') in ('attenuate_damaged','attenuate_other'):
            gate=spec['damaged'] if spec['gate']=='attenuate_damaged' else ('wrist' if spec['damaged']=='front' else 'front')
        with Videos(attempt,fps) as videos:
            videos.append(deployed)
            while step<horizon and not success:
                actions,caps=vision.action(deployed,task.language,gate,means)
                if condition=='clean' and step==0:require(np.max(abs(actions-native_first))<=config()['action_tolerance'],'Clean branch action mismatch')
                if condition=='b_clean_replay':
                    require(np.max(abs(actions-reference['actions'][step]))<=config()['action_tolerance'],'Clean camera replay changed native action')
                current_features=pooled(caps);features.append(current_features);proprios.append(np.array(deployed['state'],copy=True))
                from scipy.ndimage import laplace
                decisions.append(dict(step=step,actions=actions.tolist(),gate_view=gate,
                    lag_source_control=step-8 if spec.get('fault')=='lag' else None,
                    **{v+'_sharpness':float(laplace(np.asarray(deployed[k],float).mean(2)).var()) for v,k in (('front','full_image'),('wrist','wrist_image'))}))
                for slot in range(min(8,horizon-step)):
                    sensors.arm(step+1)
                    raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist());step+=1
                    sensors.take(raw);history.append(copy.deepcopy(rt.observation(raw,cfg)));history=history[-10:]
                    obs=record()
                    if condition=='b_clean_replay':
                        require(np.sqrt(np.mean(((sensors.latest['wrist']['pixels'].astype(float)-sensors.latest['wrist']['alternate_pixels'])/255.)**2))<=config()['rgb_tolerance'],'Clean camera replay differs at a native sample')
                    deployed=deployed_observation(raw,sensors,history,spec,step,patch);videos.append(deployed)
                    success=bool(done)
                    if done:break
                monitor.update(current_start=case['episode_id'],condition=condition,executed_steps=step,horizon=horizon)
        require(len(events)==step+1,'Control-time events missing')
        pd.DataFrame(events).to_parquet(attempt/'EVENTS.parquet',index=False)
        pd.DataFrame(decisions).to_parquet(attempt/'DECISIONS.parquet',index=False)
        np.savez_compressed(attempt/'FEATURES.npz',features=np.asarray(features,dtype=np.float32),proprio=np.asarray(proprios,dtype=np.float32),
                            previous_features=previous_features,previous_proprio=previous_proprio)
        first_contact=next((e['step'] for e in events if e['contact']),None)
        detail=dict(start_id=case['episode_id'],condition=condition,route=spec['route'],task_id=case['task_id'],state_index=case['state_index'],
            executed_steps=step,planned_horizon=horizon,task_success_within_window=success,first_contact_step=first_contact,
            first_grasp_step=next((e['step'] for e in events if e['grasp']),None),initial_distance_m=initial_distance,
            final_distance_m=events[-1]['distance_m'],actual_progress_to_start_target_m=initial_distance-events[-1]['distance_to_start_target_m'],
            camera_path=camera_path,camera_amplitudes=camera_amplitudes,random_matched_prefix_steps=amplitude_valid_until,
            random_terminal_amplitude_held=amplitude_valid_until<horizon,native_first_action_error=action_error,
            shadow_zero_checks=sensors.zero_checks,shadow_zero_max_nrmse=sensors.zero_max,
            privileged_camera_intervention=spec['camera']!='native',privileged_feature_intervention=gate is not None,
            video_semantics='single-view policy RGB; feature attenuation is internal and not visible in RGB',
            no_CASR_claim=True)
        atomic(detail,attempt/'DETAIL.json');np.savez_compressed(attempt/'START.npz',**arrays)
        atomic(dict(detail,start=case['episode_id'],suite='libero_goal',checkpoint='libero_goal',
                    success=success,success_semantics='task ended successfully within this short diagnostic window; not full-episode failure'),attempt/'EPISODE.json')
        if condition=='clean':
            reference=dict(start=start,start_arrays=arrays,constructor_rng=constructor_rng,horizon=step,camera_path=camera_path,
                           previous_observation=previous,previous_features=previous_features,previous_proprio=previous_proprio,
                           actions={r['step']:np.asarray(r['actions'],np.float32) for r in decisions})
            save_torch(reference,reference_path(case))
        files=[attempt/n for n in ('DETAIL.json','EPISODE.json','EVENTS.parquet','DECISIONS.parquet','FEATURES.npz','START.npz','front.mp4','wrist.mp4')]
        if condition=='clean':files.append(reference_path(case))
        finish(receipt,files)
        return detail

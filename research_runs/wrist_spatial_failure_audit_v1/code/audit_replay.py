"""Replay only recorded executed actions. No policy inference or new training."""
from functools import wraps
from pathlib import Path
import hashlib
import os
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from audit_common import ROOT, SOURCE, config, require, source_adapter, read, atomic, complete, finish
from audit_math import decode_actions, visibility, frame_alignment, event_summary

def fingerprint(env):
    arrays={'sim_state':np.asarray(env.get_sim_state())}
    for k in ('body_xpos','body_xquat','geom_xpos','geom_xmat','site_xpos','site_xmat','cam_xpos','cam_xmat'):
        arrays[k]=np.asarray(getattr(env.sim.data,k))
    clocks={name:{k:v.item() if isinstance(v,np.generic) else v for k,v in vars(obs).items()
             if k!='_sensor' and isinstance(v,(bool,int,float,str,np.generic))}
            for name,obs in env.env._observables.items()}
    return {'arrays':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()},
            'time':float(env.sim.data.time),'observable_clocks':clocks}

class SensorCapture:
    """Segmentation sampled inside the unchanged native wrist camera callback."""
    def __init__(self,env,target): self.env=env; self.target=target; self.key=None; self.latest=None
    def arm(self,key): self.key=key; self.latest=None
    def __enter__(self):
        from robosuite import macros
        self.observable=self.env.env._observables['robot0_eye_in_hand_image']; self.original=self.observable._sensor
        @wraps(self.original)
        def sensor(cache):
            pixels=self.original(cache)
            if self.key is not None:
                before=fingerprint(self.env)
                zero=self.env.sim.render(camera_name='robot0_eye_in_hand',width=256,height=256)
                seg=self.env.sim.render(camera_name='robot0_eye_in_hand',width=256,height=256,segmentation=True)
                sign=1 if macros.IMAGE_CONVENTION=='opengl' else -1
                zero=np.asarray(zero)[::sign]; seg=np.asarray(seg)[::sign][::-1,::-1]
                error=float(np.sqrt(np.mean(((zero.astype(float)-np.asarray(pixels).astype(float))/255.)**2)))
                require(error<=config()['rgb_nrmse_tolerance'],'Sensor-time RGB zero-render mismatch')
                mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],self.target['render_ids'])
                require(fingerprint(self.env)==before,'Segmentation rendering changed physics/cameras/clocks')
                cam=int(self.env.sim.model.camera_name2id('robot0_eye_in_hand'))
                self.latest={'key':self.key,'pixels':np.array(pixels,copy=True),'mask':mask,
                             'camera_xyz':np.array(self.env.sim.data.cam_xpos[cam],copy=True).tolist(),
                             'camera_rotation_matrix':np.array(self.env.sim.data.cam_xmat[cam],copy=True).tolist(),
                             'zero_rgb_nrmse':error,'sample_time_s':float(self.env.sim.data.time)}
            return pixels
        self.observable._sensor=sensor; return self
    def take(self,raw,key):
        require(self.latest is not None and self.latest['key']==key,'Native sensor did not sample; refuse forced refresh')
        require(np.array_equal(raw['robot0_eye_in_hand_image'],self.latest['pixels']),'Mask and cached native frame timestamps differ')
        return self.latest
    def __exit__(self,*args): self.observable._sensor=self.original

def frame(obs, condition, step, fps, geo, vis):
    canvas=Image.new('RGB',(448,288),(20,20,20)); canvas.paste(Image.fromarray(obs['full_image']),(0,64))
    canvas.paste(Image.fromarray(obs['wrist_image']),(224,64)); draw=ImageDraw.Draw(canvas)
    draw.text((5,3),condition,fill='white')
    draw.text((5,17),f't={step/fps:.2f}s step={step}  d={geo["distance_m"]*100:.1f}cm',fill='white')
    text='visibility pending native sample' if vis is None else f'FOV px={vis["policy_fov_pixels"]} uncovered={vis["policy_uncovered_pixels"]}'
    draw.text((5,31),text,fill='white')
    draw.text((5,45),f'contact={int(geo["contact"])} grasp={int(geo["grasp"])} | front / wrist',fill='white')
    return np.asarray(canvas)

def rgb_check(current, reference):
    error=float(np.sqrt(np.mean(((np.asarray(current,dtype=float)-np.asarray(reference,dtype=float))/255.)**2)))
    require(error<=config()['rgb_nrmse_tolerance'],f'Historical keyframe RGB mismatch: {error}')
    return error

def run_episode(env, task, initial, entry, folder, monitor, base_steps, total_steps):
    import imageio.v2 as imageio
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    from sr_geometry import anchor, truth
    s=source_adapter(); native=s.native; runtime=native.runtime_module(); cfg=native.runtime_config()
    meta=entry['episode']; raw=runtime.reset_wait(env,initial,meta['rollout_seed'],cfg['num_steps_wait'])
    obs=runtime.observation(raw,cfg)
    paired=read(Path(entry['paired_start'])/'START.json')
    require(paired['rollout_seed']==meta['rollout_seed'] and fingerprint(env)==paired['fingerprint'],
            'Historical paired physics/camera/clock reset differs')
    with np.load(Path(entry['attempt'])/'START.npz',allow_pickle=False) as old:
        require(np.array_equal(env.get_sim_state(),old['sim_state']) and np.array_equal(obs['state'],old['proprio']),
                'Historical start physical arrays must match strictly')
        reset_rgb={k:rgb_check(obs[k],old[name]) for k,name in (('full_image','front'),('wrist_image','wrist'))}
    target=anchor(env,meta['suite'],meta['task_id']); geo=truth(env,target)
    patch=None
    if entry['patch']:
        with Image.open(entry['patch']) as im:
            require(im.mode=='RGB' and im.size==(50,50),'Unexpected pinned PNG'); patch=np.array(im,copy=True)
    def deployed(observation): return observation if patch is None else native.paste_numpy(observation,patch)
    fps=1/float(env.env.control_timestep); require(0<fps<=120,'Invalid native control rate')
    trace=pd.read_parquet(entry['trace']).sort_values('decision_block'); old_steps=pd.read_parquet(entry['steps']).sort_values('step')
    require(list(old_steps.step)==list(range(1,meta['executed_steps']+1)), 'Saved step sequence has gaps')
    video=folder/'video.mp4'; events=[]; step=0; max_error=0.; keyframes={}; final_done=False
    def record(step,geo,captured=None):
        vis=None if captured is None else visibility(captured['mask'],patch is not None,config()['patch_xy'])
        events.append({'step':step,'time_s':step/fps,'sensor_time_s':captured['sample_time_s'] if captured else None,
            'visibility_verified':captured is not None,'zero_rgb_nrmse':captured['zero_rgb_nrmse'] if captured else None,
            'camera_xyz':captured['camera_xyz'] if captured else None,
            'camera_rotation_matrix':captured['camera_rotation_matrix'] if captured else None,
            'distance_m':geo['distance_m'],'contact':geo['contact'],'grasp':geo['grasp'],
            **(vis or {k:None for k in ('raw_visible_pixels','policy_fov_pixels','policy_uncovered_pixels','patch_covered_pixels')})})
        return vis
    with imageio.get_writer(video,fps=fps,codec='libx264',quality=7,macro_block_size=16,
                            ffmpeg_params=['-threads','2','-pix_fmt','yuv420p']) as writer, SensorCapture(env,target) as camera:
        record(0,geo); writer.append_data(frame(deployed(obs),meta['condition'],0,fps,geo,None))
        for decision,row in trace.iterrows():
            require(int(row.step_start)==step,'Trace chunk start changed')
            actions=decode_actions(row.predicted_actions)
            if int(row.decision_block) in (0,4,12):
                p=Path(entry['attempt'])/f'frame_{int(row.decision_block):03d}.png'
                current=deployed(obs)
                with Image.open(p) as old: keyframes[str(p.name)]=rgb_check(np.concatenate([current['full_image'],current['wrist_image']],axis=1),old)
            for slot in range(int(row.executed_slots)):
                expected=old_steps.iloc[step]; require(int(expected.executed_slot)==slot and int(expected.decision_block)==int(row.decision_block),'Executed slot changed')
                require(np.array_equal(actions[slot],np.asarray(expected.deployed_action,dtype=np.float32)), 'Executed action differs from saved trace')
                frame_alignment(raw['robot0_eef_pos'],expected.eef_before,config()['position_tolerance_m'])
                camera.arm(step+1)
                raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist()); step+=1
                captured=camera.take(raw,step); geo=truth(env,target); original=expected.geometry_after
                error=max(frame_alignment(raw['robot0_eef_pos'],expected.eef_after,config()['position_tolerance_m']),
                          frame_alignment(geo['target_xyz'],original['target_xyz'],config()['position_tolerance_m']),
                          frame_alignment(geo['eef_xyz'],original['eef_xyz'],config()['position_tolerance_m']))
                max_error=max(max_error,error)
                require(geo['contact']==original['contact'] and geo['grasp']==original['grasp'], 'Replay contact/grasp state diverged')
                require(not done or step==meta['executed_steps'],'Replay ended before saved endpoint')
                obs=runtime.observation(raw,cfg); vis=record(step,geo,captured)
                writer.append_data(frame(deployed(obs),meta['condition'],step,fps,geo,vis))
                final_done=bool(done)
                monitor.update(current_start=entry['start'],condition=meta['condition'],completed_steps=base_steps+step,total_steps=total_steps)
            require(int(row.step_end)==step and bool(row.success_after_chunk)==final_done,'Saved action-block outcome changed')
        require(step==meta['executed_steps'] and final_done==bool(meta['success']), 'Historical outcome differs; reconstruction invalid')
        current=deployed(obs)
        with Image.open(Path(entry['attempt'])/'frame_final.png') as old:
            keyframes['frame_final.png']=rgb_check(np.concatenate([current['full_image'],current['wrist_image']],axis=1),old)
    output=pd.DataFrame(events); output.to_parquet(folder/'EVENTS.parquet',index=False)
    result={'key':entry['key'],'historical_success':meta['success'],'executed_steps':step,'fps':fps,
            'max_position_error_m':max_error,'start_rgb_nrmse':reset_rgb,'keyframe_rgb_nrmse':keyframes,
            'strict_initial_physics':True,'sample_zero_unchanged':True,'hook_restored':True,
            'reconstruction_only':True,'no_policy_inference':True,'events':event_summary(output)}
    atomic(result,folder/'REPLAY.json')
    finish(folder/'COMPLETE.json',[video,folder/'EVENTS.parquet',folder/'REPLAY.json'])

def run(manifest,monitor):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'), 'User-submitted single-GPU allocation required')
    # TensorFlow is used only by the original JPEG/crop preprocessing, on CPU.
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    s=source_adapter(); runtime=s.native.runtime_module(); cfg=s.native.runtime_config()
    # This module is imported for its simulator/image helpers. load_wrapper is never called.
    receipts=[]; pending=[]; completed=0
    for entry in manifest['entries']:
        receipt=ROOT/'replays'/entry['key']/'COMPLETE.json'
        if complete(receipt): receipts.append(receipt); completed+=entry['episode']['executed_steps']
        else: pending.append(entry)
    monitor.update(resumed_steps=completed,completed_steps=completed,total_steps=manifest['total_steps'])
    env=None; active_task=None; started=time.monotonic(); newly_completed=0
    try:
        for index,entry in enumerate(pending):
            destination=ROOT/'replays'/entry['key']; receipt=destination/'COMPLETE.json'
            destination.mkdir(parents=True,exist_ok=True)
            identity=(entry['episode']['suite'],entry['episode']['task_id'])
            if active_task!=identity:
                if env is not None: runtime.close_environment(env); env=None
                monitor.update(phase='create_environment',current_start=entry['start'])
                env,task,states=runtime.make_env(cfg,*identity); active_task=identity
                require(task.name==entry['task_name'],'Task mapping changed')
            import torch
            pinned=torch.load(entry['initial_file'],map_location='cpu',weights_only=False)
            initial=states[entry['episode']['state_index']]
            require(np.array_equal(initial,pinned[entry['episode']['state_index']]),'Benchmark initial state differs from pinned source')
            attempt=destination/f'attempt_{time.time_ns()}'; attempt.mkdir()
            try:
                monitor.update(phase='saved_action_video_replay',replays_completed=len(receipts),replays_total=len(manifest['entries']))
                run_episode(env,task,initial,entry,attempt,monitor,completed,manifest['total_steps'])
                finish(receipt,[attempt/'COMPLETE.json',attempt/'video.mp4',attempt/'EVENTS.parquet',attempt/'REPLAY.json'])
            except BaseException as error:
                atomic({'status':'INVALID_RECONSTRUCTION','error':repr(error),'key':entry['key']},attempt/'FAILURE.json')
                raise  # No automatic retry or rerender after mismatch; release GPU.
            completed+=entry['episode']['executed_steps']; newly_completed+=entry['episode']['executed_steps']; receipts.append(receipt)
            if index==1:
                elapsed=time.monotonic()-started
                atomic({'measured_replays':2,'elapsed_s':elapsed,'new_steps':newly_completed,'resumed_run':len(pending)!=len(manifest['entries']),
                    'estimated_remaining_s':(manifest['total_steps']-completed)*elapsed/max(newly_completed,1),
                    'includes_first_environment_and_encoder_startup':True},ROOT/'reports/FIRST_PAIR_TIMING.json')
    finally:
        if env is not None: runtime.close_environment(env)
    finish(ROOT/'reports/REPLAY_COMPLETE.json',receipts)

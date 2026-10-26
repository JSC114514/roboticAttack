"""Native full-task screening with timestamped real frame history; no learned intervention."""
from collections import deque
from contextlib import contextmanager
from functools import wraps
import gc
import time
import numpy as np
from sc_common import (ROOT,config,read,atomic,require,finish,complete,verify,runtime,runtime_config,
    native,geometry,episode_receipt,episode_files,resources)
from sc_math import KEYS,condition,deploy,schedule,pose_error,action_difference


class Samples:
    """Passive native camera callbacks. No render(), forward(), or sensor refresh."""
    def __init__(self,env): self.env=env;self.original={};self.latest={};self.count={'front':0,'wrist':0}
    def __enter__(self):
        try:
            for view,name,camera in [('front','agentview_image','agentview'),('wrist','robot0_eye_in_hand_image','robot0_eye_in_hand')]:
                observable=self.env.env._observables[name];original=observable._sensor
                self.original[view]=(observable,original)
                camera_id=int(self.env.sim.model.camera_name2id(camera))
                def wrap(view,original,camera_id):
                    @wraps(original)
                    def sensor(cache):
                        image=original(cache);self.count[view]+=1
                        self.latest[view]=dict(frame_id=self.count[view],sensor_timestamp_s=float(self.env.sim.data.time),
                            xyz=np.array(self.env.sim.data.cam_xpos[camera_id],copy=True).tolist(),
                            rotation=np.array(self.env.sim.data.cam_xmat[camera_id],copy=True).tolist(),
                            image=np.array(image,copy=True))
                        return image
                    return sensor
                observable._sensor=wrap(view,original,camera_id)
        except BaseException:
            self.__exit__();raise
        return self
    def take(self,raw):
        out={}
        for view,key in [('front','agentview_image'),('wrist','robot0_eye_in_hand_image')]:
            require(view in self.latest,'Native camera has not sampled: '+view)
            value=self.latest[view]
            require(np.array_equal(value['image'],raw[key]),'Native frame/sample timestamp mismatch: '+view)
            require(value['sensor_timestamp_s']<=float(self.env.sim.data.time)+1e-9,'Future camera sample')
            out[view]={k:v for k,v in value.items() if k!='image'}
        return out
    def __exit__(self,*_):
        for observable,original in self.original.values(): observable._sensor=original


def snapshot(raw,samples,env,step):
    return dict(control_step=step,time_s=float(env.sim.data.time),sensors=samples.take(raw),
        observation=runtime().observation(raw,runtime_config()))


@contextmanager
def warm_history(env,initial,seed):
    from VLAAttacker.multiview.openvla_oft_experiments import get_libero_dummy_action
    r=runtime();cfg=runtime_config();env.seed(seed);r.bind_environment_context(env);r.reset_environment(env)
    gc.collect();r.renderer_metadata(env)
    raw=env.set_init_state(initial);require(np.array_equal(env.get_sim_state(),initial),'Native initial array mismatch')
    history=deque(maxlen=3)
    with Samples(env) as samples:
        for i in range(cfg['num_steps_wait']):
            raw,_,done,_=env.step(get_libero_dummy_action());require(not done,'Task ended during fixed warmup')
            history.append(snapshot(raw,samples,env,i+1-cfg['num_steps_wait']))
        require(len(history)==3 and history[-1]['control_step']==0,'Warmup delay history incomplete')
        r.renderer_metadata(env);r.validate_camera_frames(raw)
        yield raw,history,samples


def start_arrays(env,obs):
    return dict(sim_state=np.array(env.get_sim_state(),copy=True),proprio=obs['state'],front=obs['full_image'],wrist=obs['wrist_image'])


def create(case):
    import torch
    utils=native();utils.seed_all(case['rollout_seed']);r=runtime()
    env,task,states=r.make_env(runtime_config(),case['suite'],case['task_id'])
    try:
        require(task.name==case['task_name'],'Task identity changed')
        initial=torch.load(case['initial_file'],map_location='cpu',weights_only=False)[case['state_index']]
        require(np.array_equal(initial,states[case['state_index']]),'Initial-state file differs')
        return env,task,initial
    except BaseException:
        r.close_environment(env);raise


def native_start_reference(case):
    """An unmodified native reset, used to verify the passive sampler per clean start."""
    env=None;r=runtime();utils=native()
    try:
        env,task,initial=create(case)
        raw=r.reset_wait(env,initial,case['rollout_seed'],runtime_config()['num_steps_wait'])
        obs=r.observation(raw,runtime_config())
        return start_arrays(env,obs),dict(fingerprint=utils.fingerprint(env),rollout_seed=case['rollout_seed'])
    finally:
        if env is not None:r.close_environment(env)


def sample_metrics(history,source,spec,step):
    now=history[-1];out={}
    for view in ('front','wrist'):
        selected=source if spec['view']==view else now
        a=now['sensors'][view];b=selected['sensors'][view];shift,angle=pose_error(a,b)
        out.update({view+'_frame_id':b['frame_id'],view+'_fresh_frame_id':a['frame_id'],
            view+'_source_control_step':selected['control_step'],view+'_sensor_timestamp_s':b['sensor_timestamp_s'],
            view+'_age_s':now['time_s']-b['sensor_timestamp_s'],view+'_added_age_s':a['sensor_timestamp_s']-b['sensor_timestamp_s'],
            view+'_camera_shift_m':shift,view+'_camera_rotation_rad':angle})
        for lag in (1,2):
            old=history[-1-lag]['sensors'][view];distance,rotation=pose_error(a,old)
            out[view+f'_reference_lag{lag}_shift_m']=distance
            out[view+f'_reference_lag{lag}_rotation_rad']=rotation
    if spec['delay']:
        require(source['control_step']==step-spec['delay'],'Delay is not in native control steps')
    return out


def episode(wrapper,case,name,monitor,gate_needed):
    import pandas as pd
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    utils=native();r=runtime();c=config();spec=condition(name);receipt=episode_receipt(case,name)
    attempt=receipt.parent/('attempt_'+str(time.time_ns()));attempt.mkdir(parents=True)
    env=None;started=time.monotonic();utils.switch_suite(wrapper,case['suite'])
    try:
        expected_start=native_start_reference(case) if name=='clean' else None
        env,task,initial=create(case);geo=geometry()
        # reset recreates the scene; resolve target references after reset inside the context.
        with warm_history(env,initial,case['rollout_seed']) as (raw,history,samples):
            target=geo['anchor'](env,case['suite'],case['task_id'])
            obs=history[-1]['observation'];arrays=start_arrays(env,obs)
            start=dict(fingerprint=utils.fingerprint(env),rollout_seed=case['rollout_seed'])
            np.savez_compressed(attempt/'START.npz',**arrays);atomic(start,attempt/'START.json')
            paired=ROOT/'paired_starts'/case['start']
            if name=='clean':
                baseline,baseline_meta=expected_start
                require(start==baseline_meta,'Sampler changed native physical/camera/clock fingerprint')
                for k in arrays:require(np.array_equal(arrays[k],baseline[k]),'Sampler changed native reset/RGB: '+k)
                if (paired/'START.json').exists():utils.assert_start(arrays,start,paired)
                else:
                    paired.mkdir(parents=True,exist_ok=True);np.savez_compressed(paired/'START.npz',**arrays);atomic(start,paired/'START.json')
                onsets=[]
            else:
                require(complete(episode_receipt(case,'clean')),'Clean reference missing')
                utils.assert_start(arrays,start,paired)
                clean_meta=read(episode_files(episode_receipt(case,'clean'))['EPISODE.json'])
                onsets=schedule(clean_meta['executed_steps'],c['occlusion_fractions'],c['chunk'])
            gate=utils.interface_gate(wrapper,obs,task.language,case['suite']) if gate_needed else dict(status='PASS')
            first=utils.predict(wrapper,obs,task.language)
            if name!='clean':
                old=np.load(episode_files(episode_receipt(case,'clean'))['FIRST_ACTION.npy'],allow_pickle=False)
                error=float(np.max(np.abs(first-old)));require(error<=c['action_tolerance'],'Paired first clean action differs')
                gate['first_clean_action_error']=error
            np.save(attempt/'FIRST_ACTION.npy',first);gate['passive_sampler_native_reset_identical']=True
            atomic(gate,attempt/'INTERFACE_CHECK.json')
            fps=1/float(env.env.control_timestep);limit=c['max_steps'][case['suite']]
            step=decision=0;success=False;traces=[];steps=[];probes=[];active_decisions=0
            truth=geo['truth'](env,target);first_contact=0 if truth['contact'] else None;first_grasp=0 if truth['grasp'] else None
            deployed,_,_=deploy(history,name,step,onsets,c['occlusion_size'],c['occlusion_fill'],c['occlusion_duration_steps'])
            with utils.CameraWriters(attempt,fps) as writer:
                writer.append(deployed)
                while step<limit and not success:
                    obs=history[-1]['observation']
                    deployed,source,applied=deploy(history,name,step,onsets,c['occlusion_size'],c['occlusion_fill'],c['occlusion_duration_steps'])
                    timing=sample_metrics(history,source,spec,step)
                    clean=first.copy() if step==0 else utils.predict(wrapper,obs,task.language)
                    actions=utils.predict(wrapper,deployed,task.language) if applied else clean.copy()
                    if name=='clean' and step in (16,40,64):
                        # Same-state view/lag comparisons, independent of corrupted-branch drift.
                        before_probe=utils.fingerprint(env)
                        for probe in ('front_delay1','wrist_delay1','front_delay2','wrist_delay2'):
                            image,old,_=deploy(history,probe,step,[]);ps=condition(probe)
                            predicted=utils.predict(wrapper,image,task.language)
                            change=action_difference(predicted,clean,wrapper.vla.norm_stats[wrapper.gen_cfg.unnorm_key]['action'],8)
                            shift,rotation=pose_error(history[-1]['sensors'][ps['view']],old['sensors'][ps['view']])
                            probes.append(dict(step=step,view=ps['view'],delay_steps=ps['delay'],
                                added_age_s=history[-1]['sensors'][ps['view']]['sensor_timestamp_s']-old['sensors'][ps['view']]['sensor_timestamp_s'],
                                camera_shift_m=shift,camera_rotation_rad=rotation,
                                next_xyz_normalized_l2=change['next_xyz_normalized_l2'],all8_xyz_normalized_rmse=change['all8_xyz_normalized_rmse']))
                        require(utils.fingerprint(env)==before_probe,'Offline probes changed native state/sampling')
                    used=0;begin=step;active_decisions+=int(applied)
                    rgb_error={view+'_input_rgb_nrmse':float(np.sqrt(np.mean(((np.asarray(deployed[key],float)-obs[key])/255)**2))) for view,key in KEYS.items()}
                    for slot in range(min(c['chunk'],limit-step)):
                        before=geo['truth'](env,target);command=np.asarray(process_libero_action(actions[slot].copy()),float)
                        require(command.shape==(7,) and np.isfinite(command).all(),'Nonfinite native controller command')
                        raw,_,done,_=env.step(command.tolist());step+=1;used+=1
                        after=geo['truth'](env,target);history.append(snapshot(raw,samples,env,step))
                        if after['contact'] and first_contact is None:first_contact=step
                        if after['grasp'] and first_grasp is None:first_grasp=step
                        steps.append(dict(step=step,time_s=step/fps,decision_block=decision,executed_slot=slot,
                            raw_action=actions[slot].tolist(),same_observation_clean_action=clean[slot].tolist(),controller_command=command.tolist(),
                            eef_xyz=after['eef_xyz'],target_xyz=after['target_xyz'],target_center_distance_m=after['distance_m'],
                            actual_eef_displacement_m=(np.asarray(after['eef_xyz'])-before['eef_xyz']).tolist(),
                            contact=after['contact'],grasp=after['grasp'],success=bool(done)))
                        deployed,_,_=deploy(history,name,step,onsets,c['occlusion_size'],c['occlusion_fill'],c['occlusion_duration_steps']);writer.append(deployed)
                        if done:success=True;break
                    stats=wrapper.vla.norm_stats[wrapper.gen_cfg.unnorm_key]['action']
                    traces.append(dict(decision_block=decision,step_start=begin,step_end=step,executed_slots=used,
                        fault_active=bool(applied),predicted_actions=actions.tolist(),same_observation_clean_actions=clean.tolist(),
                        **timing,**rgb_error,**action_difference(actions,clean,stats,used)))
                    decision+=1;monitor.update(current_start=case['start'],condition=name,executed_steps=step)
            pd.DataFrame(traces).to_parquet(attempt/'TRACE.parquet',index=False)
            pd.DataFrame(steps).to_parquet(attempt/'STEPS.parquet',index=False)
            if name=='clean':pd.DataFrame(probes,columns=['step','view','delay_steps','added_age_s','camera_shift_m','camera_rotation_rad','next_xyz_normalized_l2','all8_xyz_normalized_rmse']).to_parquet(attempt/'PROBES.parquet',index=False)
            atomic(dict(**case,condition=name,checkpoint=c['model_dir'],revision=c['revision'],success=int(success),
                executed_steps=step,max_steps=limit,fps=fps,num_open_loop_steps=8,first_target_contact_step=first_contact,
                first_target_grasp_step=first_grasp,contact_sampling='native control-step endpoint',
                occlusion_onsets=onsets,active_decisions=active_decisions,decisions=decision,model_frozen=True,
                fault=spec,offline_probe_forwards=len(probes),elapsed_s=time.monotonic()-started,**resources()),attempt/'EPISODE.json')
            finish(receipt,[attempt/n for n in ('START.npz','START.json','FIRST_ACTION.npy','INTERFACE_CHECK.json',
                'EPISODE.json','TRACE.parquet','STEPS.parquet','front.mp4','wrist.mp4')]+[paired/n for n in ('START.npz','START.json')]
                +([attempt/'PROBES.parquet'] if name=='clean' else []))
    except BaseException as error:
        atomic(dict(status='FAILED',error=repr(error),case=case,condition=name,elapsed_s=time.monotonic()-started,**resources()),attempt/'FAILURE.json');raise
    finally:
        if env is not None:r.close_environment(env)


def run(stage,index,workers,monitor):
    require(1<=workers<=2,'Screen uses at most two GPUs');m=verify();utils=native()
    from jr_math import weighted_shards
    if stage=='clean':cases=m['cases'];names=['clean']
    else:
        require(stage=='evaluate' and 0<=index<workers,'Invalid shard')
        require(all(complete(episode_receipt(c,'clean')) for c in m['cases']),'All 32 clean references must complete first')
        cases=weighted_shards(m['cases'],workers,config()['max_steps'])[index];names=[v['name'] for v in m['conditions'] if v['name']!='clean']
    receipts=[episode_receipt(c,n) for c in cases for n in names];already=sum(complete(p) for p in receipts)
    monitor.update(completed=already,resumed=already,total=len(receipts));checked=set()
    if already<len(receipts):
        wrapper=utils.load_wrapper(monitor)
        for case in cases:
            # Fixed, seeded condition order avoids making view order a worker-load confound.
            order=list(names)
            if stage!='clean':np.random.default_rng(case['rollout_seed']).shuffle(order)
            for name in order:
                if complete(episode_receipt(case,name)):continue
                episode(wrapper,case,name,monitor,case['suite'] not in checked);checked.add(case['suite'])
                monitor.update(phase='full_task_screen',completed=monitor.data['completed']+1)
    require(all(complete(p) for p in receipts),'Incomplete shard')
    finish(ROOT/f'reports/{stage}_{workers}_{index}_COMPLETE.json',receipts);monitor.update(phase='COMPLETE')

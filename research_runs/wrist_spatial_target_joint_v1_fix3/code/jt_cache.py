"""Native sensor-time render counterfactuals; exact saved-prefix paired execution."""
from functools import wraps
import copy
import gc
import random
import time
import numpy as np
import pandas as pd
import torch
from jt_common import ROOT,source,native,config,verify,selected,source_row,cache_receipt,complete,finish,atomic_torch,atomic,make_engine,require
from jt_math import translations
from sr_geometry import anchor,truth,controller
from sr_math import far_target
from jt_history import replay_previous_resets,sampler_state,model_hashes,numerical_snapshot

def constructor_rng():
    return {'numpy':copy.deepcopy(np.random.get_state()),'python':random.getstate(),'torch':torch.get_rng_state().clone()}

def restore_constructor_rng(state):
    np.random.set_state(state['numpy']);random.setstate(state['python']);torch.set_rng_state(state['torch'])

def fingerprint_difference(expected,actual):
    """Diagnostic only: do not remove fields or weaken the exact equality gate."""
    return {'array_hashes_changed':[k for k in set(expected.get('arrays',{}))|set(actual.get('arrays',{}))
             if expected.get('arrays',{}).get(k)!=actual.get('arrays',{}).get(k)],
            'expected_time':expected.get('time'),'actual_time':actual.get('time'),
            'observable_clocks_changed':{k:{'expected':expected.get('observable_clocks',{}).get(k),
                                           'actual':actual.get('observable_clocks',{}).get(k)}
             for k in set(expected.get('observable_clocks',{}))|set(actual.get('observable_clocks',{}))
             if expected.get('observable_clocks',{}).get(k)!=actual.get('observable_clocks',{}).get(k)}}

def execute_paired_branches(actions,first_start,recreate,execute_one,close,expected):
    """Consume ownership of one fresh environment per condition, even on failure.

    Resetting a simulator that already executed a teacher block is deliberately
    prohibited here. Hidden solver/controller/task state is not assumed reset.
    """
    outcomes=[]
    for index,command in enumerate(actions):
        env,raw,target,current=first_start if index==0 else recreate()
        try:
            require(current==expected,'Paired decision physics/camera/clock differs')
            value=execute_one(env,raw,target,command)
            value['exact_start_fingerprint']=current
            value['environment_isolation']='fresh_environment_for_each_condition'
            outcomes.append(value)
        finally:close(env)
    return outcomes

class SceneTranslation:
    def __init__(self,scene,ids,delta):self.scene=scene;self.ids=set(ids);self.delta=np.asarray(delta);self.calls=0
    def __enter__(self):
        import mujoco
        self.module=mujoco;self.original=mujoco.mjr_render
        def render(*args,**kwargs):
            scene=kwargs.get('scn',args[1] if len(args)>1 else None)
            if scene is not self.scene:return self.original(*args,**kwargs)
            saved=[]
            try:
                for i in range(scene.ngeom):
                    geom=scene.geoms[i]
                    if int(geom.objtype)==5 and int(geom.objid) in self.ids:
                        saved.append((geom,np.array(geom.pos,copy=True)));geom.pos[:]+=self.delta
                require(saved,'Target geometry missing from render scene');self.calls+=1
                return self.original(*args,**kwargs)
            finally:
                for geom,position in saved:geom.pos[:]=position
        mujoco.mjr_render=render;return self
    def __exit__(self,*_):self.module.mjr_render=self.original

class Capture:
    def __init__(self,env,target,render):self.env=env;self.target=target;self.render=render;self.armed=False;self.latest=None
    def arm(self,yes):self.armed=yes;self.latest=None
    def __enter__(self):
        from oracle_camera import fingerprint
        from robosuite import macros
        self.obs=self.env.env._observables['robot0_eye_in_hand_image'];self.original=self.obs._sensor
        @wraps(self.original)
        def sensor(cache):
            pixels=self.original(cache)
            if self.armed:
                before=fingerprint(self.env);geometry=truth(self.env,self.target)
                numeric=numerical_snapshot(self.env)
                delta=translations(geometry['target_xyz'],geometry['eef_xyz'],config()['displacement_m'],config()['direction_epsilon_m'])
                arrays=[];visible=[];masks=[];zero_error=None
                if self.render:
                    context=self.env.sim._render_context_offscreen
                    for shift in [np.zeros(3),*(delta if delta is not None else np.zeros((4,3)))]:
                        with SceneTranslation(context.scn,self.target['render_ids'],shift) as hook:
                            rgb=self.env.sim.render(camera_name='robot0_eye_in_hand',width=256,height=256)
                            seg=self.env.sim.render(camera_name='robot0_eye_in_hand',width=256,height=256,segmentation=True)
                        require(hook.calls==2,'Unexpected render hook path')
                        sign=1 if macros.IMAGE_CONVENTION=='opengl' else -1
                        rgb=np.asarray(rgb)[::sign].copy();seg=np.asarray(seg)[::sign][::-1,::-1]
                        mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],self.target['render_ids'])
                        n=int(np.sqrt(.9)*256);offset=(256-n)//2
                        visible.append(int(mask[offset:offset+n,offset:offset+n].sum()));arrays.append(rgb);masks.append(mask)
                    zero_error=float(np.sqrt(np.mean(((arrays[0].astype(float)-np.asarray(pixels).astype(float))/255.)**2)))
                    require(zero_error<=config()['rgb_nrmse_tolerance'],'Zero render differs from native camera')
                require(fingerprint(self.env)==before,'Render changed physics, camera or clock')
                self.latest={'fingerprint':before,'geometry':geometry,'deltas':delta,'raw_images':arrays,
                             'visible_pixels':visible,'clean_mask':masks[0] if masks else None,
                             'native_pixels':np.array(pixels,copy=True),'zero_rgb_nrmse':zero_error,
                             'numerical_arrays':numeric}
            return pixels
        self.obs._sensor=sensor;return self
    def __exit__(self,*_):self.obs._sensor=self.original

def prefix(env,initial,entry,row,traces,render=False):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action,get_libero_dummy_action
    from oracle_camera import fingerprint
    runtime=native.runtime_module()
    history=replay_previous_resets(env,entry,runtime)
    env.seed(entry['rollout_seed']);runtime.bind_environment_context(env)
    runtime.reset_environment(env);gc.collect();runtime.renderer_metadata(env)
    history['target_samplers']=sampler_state(env);history['model_hashes_before_init']=model_hashes(env)
    raw=env.set_init_state(initial);require(np.array_equal(env.get_sim_state(),initial),'Initial state differs')
    target=anchor(env,entry['suite'],entry['task_id']);controller(env);decision=entry['decision_block']
    with Capture(env,target,render) as camera:
        wait=native.runtime_config()['num_steps_wait']
        for step in range(wait):
            camera.arm(decision==0 and step==wait-1)
            raw,_,done,_=env.step(get_libero_dummy_action());require(not done,'Prefix ended in wait')
        for old in traces.itertuples(index=False):
            if int(old.decision_block)>=decision:break
            actions=native.decode_action_chunk(old.predicted_environment_action_chunk,entry['episode_id'])
            for slot in range(int(old.executed_slots)):
                camera.arm(int(old.decision_block)+1==decision and slot==int(old.executed_slots)-1)
                raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist())
                require(not done,'Saved prefix ended before selected observation')
        captured=camera.latest;require(captured is not None,'Selected native camera sample missing')
        require(np.array_equal(raw['robot0_eye_in_hand_image'],captured['native_pixels']),'Native camera cache stale')
        captured['reset_context']=history
        obs=runtime.observation(raw,native.runtime_config())
        if captured['fingerprint']!=row['sample_fingerprint']:
            stem=ROOT/f'reports/ALIGNMENT_FAILURES/{entry["state_id"]}_{time.time_ns()}'
            stem.parent.mkdir(parents=True,exist_ok=True)
            arrays=dict(captured['numerical_arrays'])
            comparisons={}
            for key in ('full_image','wrist_image','state'):
                expected=np.asarray(row['observation'][key]);actual=np.asarray(obs[key])
                arrays['expected_'+key]=expected;arrays['actual_'+key]=actual
                comparisons[key]={'max_abs_error':float(np.max(np.abs(actual.astype(float)-expected.astype(float)))),
                                  'exact_equal':bool(np.array_equal(actual,expected))}
            np.savez_compressed(stem.with_suffix('.npz'),**arrays)
            atomic({'state_id':entry['state_id'],'expected':row['sample_fingerprint'],'actual':captured['fingerprint'],
                    'difference':fingerprint_difference(row['sample_fingerprint'],captured['fingerprint']),
                    'reset_context':history,'observation_errors':comparisons,
                    'numerical_arrays':str(stem.with_suffix('.npz')),
                    'expected_geometry':row['geometry'],'actual_geometry':captured['geometry'],
                    'tolerance_changed':False},stem.with_suffix('.json'))
            raise RuntimeError('Source native sample fingerprint changed; detailed alignment evidence saved')
        for key in ('full_image','wrist_image','state'):
            require(np.array_equal(np.asarray(obs[key]),np.asarray(row['observation'][key])),'Source observation differs: '+key)
    return raw,target,captured,fingerprint(env)

def execute(env,raw,target,actions,reference):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    from rr_logger import PhysicsLogger
    steps=[]
    with PhysicsLogger(env) as logger:
        for slot,action in enumerate(actions):
            before=truth(env,target);logger.arm()
            raw,_,done,_=env.step(process_libero_action(action.copy()).tolist());after=truth(env,target)
            steps.append({'slot':slot,'before':before,'after':after,'action':action.tolist(),'controller':logger.take(),'done':bool(done)})
            if done:break
    endpoint=truth(env,target)
    return {'distance_to_original_anchor_m':float(np.linalg.norm(np.asarray(endpoint['eef_xyz'])-reference)),
            'endpoint':endpoint,'steps':steps,'executed_slots':len(steps)}

def run(index,monitor):
    native.require_gpu('joint_target_cache')
    require(complete(ROOT/'reports/REPLAY_CHECK_COMPLETE.json'),'Historical replay startup check has not passed')
    value=verify();runtime=native.runtime_module();runtime.prepare_graphics_backend()
    assigned=selected()[index::2];engine=None
    resumed=sum(complete(cache_receipt(r['state_id'])) for r in assigned)
    monitor.update(phase='saved_action_prefix_replay',completed=resumed,resume_completed=resumed,total=len(assigned))
    for n,entry in enumerate(assigned):
        receipt=cache_receipt(entry['state_id'])
        if complete(receipt):continue
        row=source_row(entry);env=None;started=time.monotonic()
        monitor.update(phase='saved_action_prefix_replay',state_id=entry['state_id'])
        try:
            construction_state=constructor_rng()
            env,task,states=runtime.make_env(native.runtime_config(),entry['suite'],entry['task_id'])
            initial=torch.load(value['initial_states'][f"{entry['suite']}/{entry['task_id']}"]['path'],map_location='cpu',weights_only=False)[entry['state_index']]
            require(np.array_equal(initial,states[entry['state_index']]),'Benchmark initial file mismatch')
            traces=pd.read_parquet(value['trace_path'],filters=[('condition','==','clean'),('suite','==',entry['suite']),('task_id','==',entry['task_id']),('state_index','==',entry['state_index'])]).sort_values('decision_block')
            traces=traces[traces.rollout_seed==entry['rollout_seed']]
            require(list(traces.decision_block)==list(range(entry['decision_count'])),'Saved trace incomplete')
            raw,target,captured,start=prefix(env,initial,entry,row,traces,True)
            reasons=[]
            if captured['deltas'] is None:reasons.append('horizontal_direction_degenerate')
            if any(v==0 for v in captured['visible_pixels']):reasons.append('at_least_one_render_target_not_visible')
            if reasons:
                path=receipt.with_suffix('.pt');atomic_torch({'row':row,'valid':False,'invalid_reasons':reasons,'capture':captured},path)
                finish(receipt,[path],valid=False,cohort=entry['cohort'])
                monitor.update(completed=n+1,phase='invalid_target_retained',invalid_reasons=reasons);continue
            reused=value.get('reusable_visual_cache',{}).get(entry['state_id'])
            if reused:
                from jt_common import sha
                require(sha(reused['path'])==reused['sha256'],'Reused visual cache changed')
                parent=torch.load(reused['path'],map_location='cpu',weights_only=False)
                require(parent['valid'] and parent['row']['state_id']==entry['state_id'],'Invalid reusable visual cache')
                require(all(np.array_equal(a,b) for a,b in zip(captured['raw_images'],parent['capture']['raw_images'])),'Reused counterfactual render differs')
                require(np.array_equal(captured['deltas'],parent['capture']['deltas']),'Reused displacement differs')
                clean_action=parent['native_clean_actions'];features=list(parent['goals']);actions=list(parent['teacher_actions'])
                images=list(parent['counterfactual_wrist']);clean_feature=parent['clean_feature'];prepared_teacher=parent['row']['teacher']
                monitor.update(phase='reuse_verified_features_recompute_all_physics',reused_feature_state=entry['state_id'])
            else:
                if engine is None:
                    monitor.update(phase='load_frozen_Goal_model');engine=make_engine()
                monitor.update(phase='DINO_counterfactual_native_forwards')
                engine.verify_patch_path(row)
                obs=row['observation'];clean_action,caps=engine.forward(obs,row['language'])
                normalized,_=engine.action(obs,row['language'])
                require(np.max(abs(engine.physical(normalized)-clean_action))<=1e-5,'Native action path mismatch')
                identity,_=engine.forward(obs,row['language'],{'dino':caps['dino']})
                require(np.max(abs(identity-clean_action))<=1e-5,'Zero DINO replacement not identity')
                features=[];actions=[];images=[]
                for raw_image in captured['raw_images'][1:]:
                    changed_raw=dict(raw);changed_raw['robot0_eye_in_hand_image']=raw_image
                    changed=runtime.observation(changed_raw,native.runtime_config())
                    feature=engine.spatial_observation(changed)
                    a,changed_caps=engine.forward(obs,row['language'],{'dino':feature.numpy()[None]})
                    require(np.array_equal(changed_caps['siglip'],caps['siglip']),'Teacher changed SigLIP')
                    features.append(feature);actions.append(a);images.append(changed['wrist_image'])
                clean_feature=torch.from_numpy(caps['dino'][0]).half()
                command=engine.command(normalized,clipped=True).detach().cpu().numpy()
                relative=np.asarray(truth(env,target)['target_xyz'])-np.asarray(truth(env,target)['eef_xyz'])
                prepared_teacher={'clean':normalized.detach().cpu(),'unit':(relative/max(np.linalg.norm(relative),1e-12)).tolist(),
                    'targets':{'translation_far_boundary':torch.as_tensor(far_target(command,np.full(3,-.05),np.full(3,.05)))},
                    'command_scale':torch.full((3,),.05)}
            reference=np.asarray(truth(env,target)['target_xyz']);outcomes=[]
            monitor.update(phase='fresh_environment_paired_physical_execution')
            def recreate():
                fresh=None
                try:
                    restore_constructor_rng(construction_state)
                    fresh,fresh_task,fresh_states=runtime.make_env(native.runtime_config(),entry['suite'],entry['task_id'])
                    require(fresh_task.name==task.name and np.array_equal(initial,fresh_states[entry['state_index']]),'Fresh environment task/initial differs')
                    fresh_raw,fresh_target,_,current=prefix(fresh,initial,entry,row,traces,False)
                    return fresh,fresh_raw,fresh_target,current
                except BaseException:
                    if fresh is not None:runtime.close_environment(fresh)
                    raise
            # Transfer ownership to the paired executor: no double-close on an exception.
            first_env=env;env=None
            outcomes=execute_paired_branches([clean_action,*actions],(first_env,raw,target,start),recreate,
                lambda e,r,t,a:execute(e,r,t,a,reference),runtime.close_environment,start)
            clean_distance=outcomes[0]['distance_to_original_anchor_m']
            harm=np.clip([(o['distance_to_original_anchor_m']-clean_distance)/config()['displacement_m'] for o in outcomes[1:]],-1,1)
            row=copy.deepcopy(row);row['teacher']=prepared_teacher
            payload={'row':row,'valid':True,'invalid_reasons':[],'clean_feature':clean_feature,
                     'goals':torch.stack(features).half(),'harm':harm,'native_clean_actions':clean_action,
                     'teacher_actions':np.stack(actions),'counterfactual_wrist':np.stack(images),
                     'outcomes':outcomes,'capture':captured,'paired_decision_fingerprint':start,
                     'reference_anchor_xyz':reference.tolist(),'privileged_mechanism_only':True,
                     'reused_visual_cache':reused,'all_physics_recomputed_with_fresh_environments':True,
                     'constructor_rng':construction_state}
            path=receipt.with_suffix('.pt');atomic_torch(payload,path)
            from PIL import Image
            preview=ROOT/'data/previews'/entry['state_id'];preview.mkdir(parents=True,exist_ok=True)
            preview_files=[]
            for name,image in zip(['clean',*config()['rules']],[row['observation']['wrist_image'],*images]):
                p=preview/(name+'.png');Image.fromarray(np.asarray(image,dtype=np.uint8)).save(p);preview_files.append(p)
            audit=receipt.with_name(receipt.stem+'_AUDIT.json')
            atomic({'state_id':entry['state_id'],'cohort':entry['cohort'],'harm':harm.tolist(),
                    'zero_rgb_nrmse':captured['zero_rgb_nrmse'],'deltas_m':captured['deltas'].tolist(),
                    'visible_pixels':captured['visible_pixels'],'sample_geometry':captured['geometry'],
                    'decision_geometry':outcomes[0]['steps'][0]['before'],'exact_paired_prefix':True,
                    'fresh_environment_per_condition':True,'reused_visual_cache':reused,'all_physics_recomputed':True,
                    'constructor_rng_replayed':True,
                    'render_hook_restored':True,'visual_review':'pending; scene-only part displacement may create occlusion or overlap artifacts',
                    'elapsed_s':time.monotonic()-started},audit)
            finish(receipt,[path,audit,*preview_files],valid=True,cohort=entry['cohort'])
        finally:
            if env is not None:runtime.close_environment(env)
        monitor.update(phase='counterfactual_and_one_chunk_control',completed=n+1,total=len(assigned),state_id=entry['state_id'])
    finish(ROOT/f'reports/CACHE_{index}_COMPLETE.json',[cache_receipt(r['state_id']) for r in assigned])

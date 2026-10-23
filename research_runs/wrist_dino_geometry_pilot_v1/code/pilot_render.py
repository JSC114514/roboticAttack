"""Replay prefixes and translate renderer scene geoms at the native wrist sample.

No mj_forward, set_state, step, camera refresh or contact edits inside a capture.
The rendered scene changes; physics labels/contact remain the original values.
"""
from __future__ import annotations
from functools import wraps
import time
import numpy as np
from pilot_common import (ROOT,atlas_path,atomic_json,atomic_npz,commit,complete,config,
                          load_json,progress,require,resources,sha256)


class SceneTranslation:
    """Scoped interception AFTER mjv_updateScene, for exactly one owned scene."""
    def __init__(self,scene,geom_ids,translation,module=None):
        if module is None:
            import mujoco as module
        self.module=module; self.scene=scene; self.ids=set(map(int,geom_ids))
        self.delta=np.asarray(translation,dtype=float); self.calls=0; self.matched=[]

    def __enter__(self):
        self.original=self.module.mjr_render
        def render(*args,**kwargs):
            scene=kwargs.get('scn')
            if scene is None and len(args)>1: scene=args[1]
            if scene is not self.scene: return self.original(*args,**kwargs)
            saved=[]
            try:
                for i in range(scene.ngeom):
                    geom=scene.geoms[i]
                    if int(geom.objtype)==5 and int(geom.objid) in self.ids:
                        saved.append((geom,np.array(geom.pos,copy=True)))
                        geom.pos[:]=geom.pos+self.delta
                require(saved,'Target geom IDs did not reach the owned renderer scene')
                self.calls+=1; self.matched.append(len(saved))
                return self.original(*args,**kwargs)
            finally:
                for geom,position in saved: geom.pos[:]=position
        self.module.mjr_render=render
        return self

    def __exit__(self,*_): self.module.mjr_render=self.original


def clocks(inner):
    # Read scalar observable bookkeeping without invoking sensors/update/reset.
    return {name:{key:value.item() if isinstance(value,np.generic) else value
                  for key,value in vars(obs).items()
                  if key!='_sensor' and isinstance(value,(bool,int,float,str,np.generic))}
            for name,obs in inner._observables.items()}


def fingerprint(env):
    from atlas_labels import array_hash,pose_bundle
    return {'poses':pose_bundle(env),'sim_state':array_hash(np.asarray(env.get_sim_state())),
            'time':float(env.sim.data.time),'clocks':clocks(env.env)}


def render_view(env,geom_ids,translation,body_position):
    from atlas_math import crop_map
    from robosuite import macros
    context=env.sim._render_context_offscreen
    with SceneTranslation(context.scn,geom_ids,translation) as hook:
        rgb,depth=env.sim.render(camera_name='robot0_eye_in_hand',width=224,height=224,depth=True)
        seg=env.sim.render(camera_name='robot0_eye_in_hand',width=224,height=224,segmentation=True)
    require(hook.calls==2,'Unexpected render API path; scene intervention was not applied twice')
    convention=1 if macros.IMAGE_CONVENTION=='opengl' else -1
    rgb=np.asarray(rgb)[::convention][::-1,::-1].copy()
    depth=np.asarray(depth,dtype=np.float64)[::convention][::-1,::-1].copy()
    seg=np.asarray(seg)[::convention][::-1,::-1].copy()
    mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],geom_ids)
    cropped=crop_map(mask.astype(np.uint8),nearest=True).astype(bool)
    extent=float(env.sim.model.stat.extent)
    near=float(env.sim.model.vis.map.znear)*extent; far=float(env.sim.model.vis.map.zfar)*extent
    depth_m=(near/(1-depth*(1-near/far))).astype(np.float32)
    require(np.isfinite(depth_m).all() and (depth_m>0).all(),'Invalid counterfactual metric depth')
    camera=int(env.sim.model.camera_name2id('robot0_eye_in_hand'))
    rotation=env.sim.data.cam_xmat[camera].reshape(3,3)
    position=np.asarray(body_position)+np.asarray(translation)
    p=rotation.T@(position-env.sim.data.cam_xpos[camera])
    tangent=np.tan(np.deg2rad(float(env.sim.model.cam_fovy[camera]))/2)*np.sqrt(.9)
    inside=bool(-p[2]>0 and abs(p[0])<=-p[2]*tangent and abs(p[1])<=-p[2]*tangent)
    y,x=np.where(cropped)
    summary={'subject_visible':bool(len(x)),'subject_area':float(cropped.mean()),
             'subject_center_in_frustum':inside,'subject_occlusion_proxy':inside and not bool(len(x)),
             'subject_u':float(x.mean()/223) if len(x) else None,
             'subject_v':float(y.mean()/223) if len(y) else None,
             'subject_depth_m':float(np.median(crop_map(depth_m)[cropped])) if len(x) else None,
             'rendered_geom_counts':hook.matched,'physical_contacts_unchanged':True}
    return {'rgb':rgb,'target_mask':mask,'depth_m':depth_m},summary


class CounterfactualCamera:
    def __init__(self,env,task):
        from atlas_labels import entity_map
        self.env=env; self.inner=env.env; self.task=task; self.row=None; self.latest=None
        self.entities=entity_map(env)
        self.subject=next(e for e in self.entities if e['name']==task['manipulated_object'])

    def __enter__(self):
        from atlas_labels import physical_labels
        from atlas_sampling import PROPRIO,sampled_proprio
        from atlas_math import rgb_error
        observable=self.inner._observables['robot0_eye_in_hand_image']
        self.observable=observable; self.original=observable._sensor
        @wraps(self.original)
        def sensor(cache):
            pixels=self.original(cache)
            if self.row is not None:
                before=fingerprint(self.env)
                raw={name:np.array(self.inner._observables[name]._sensor(cache),copy=True) for name in PROPRIO}
                truth=physical_labels(self.env,raw,self.entities,self.task)
                for key in ('object_eef_relative_x','object_eef_relative_y','object_eef_relative_z','object_eef_distance'):
                    require(abs(truth[key]-self.row['truth'][key])<1e-9,'Native sample geometry differs from verified atlas')
                original_rgb=np.asarray(pixels)[::-1,::-1].copy()
                body=self.env.sim.data.body_xpos[self.subject['body_id']].copy()
                arrays={}; metadata={}
                clean,summary=render_view(self.env,self.subject['geom_ids'],np.zeros(3),body)
                error=rgb_error(original_rgb,clean['rgb'])
                require(error['normalized_rms']<=1e-4,'Zero translation does not reproduce native cached RGB')
                arrays.update({'clean__'+key:value for key,value in clean.items()})
                metadata['clean']={**summary,'rgb_error':error}
                for condition in self.row['conditions']:
                    value,summary=render_view(self.env,self.subject['geom_ids'],condition['translation_m'],body)
                    name=condition['condition_id']
                    arrays.update({name+'__'+key:a for key,a in value.items()})
                    metadata[name]={**condition,**summary}
                require(fingerprint(self.env)==before,'Counterfactual rendering changed physics/poses/sampling clocks')
                self.latest={'arrays':arrays,'conditions':metadata,'pixels':original_rgb,
                             'proprio':sampled_proprio(raw),'sample_fingerprint':before,
                             'sample_time_s':float(self.env.sim.data.time)}
            return pixels
        observable._sensor=sensor
        return self

    def arm(self,row): self.row=row; self.latest=None
    def __exit__(self,*_): self.observable._sensor=self.original


def render_state(row,episode,task,signature):
    atlas_path()
    from atlas_capture import alignment,make_env,native_observation
    from atlas_runtime import bind_environment_context,close_environment,renderer_metadata,reset_environment
    folder=ROOT/'data/renders'/row['state_id']; receipt=folder/'COMPLETE.json'
    if complete(receipt,signature): return
    folder.mkdir(parents=True,exist_ok=True); started=time.monotonic(); env=None
    for key in ('initial_state_file','trajectory_file'):
        require(sha256(episode[key])==episode[key+'_sha256'],'Replay input changed')
    with np.load(episode['initial_state_file'],allow_pickle=False) as z: initial=z['sim_state'].copy()
    with np.load(episode['trajectory_file'],allow_pickle=False) as z: actions=z['executed_actions'].copy()
    with np.load(row['state_file'],allow_pickle=False) as z: stored={k:z[k].copy() for k in z.files}
    try:
        env=make_env(task['bddl_path']); env.seed(episode['rollout_seed'])
        raw=reset_environment(env); bind_environment_context(env); renderer,vendor=renderer_metadata(env)
        raw=env.set_init_state(initial)
        require(np.array_equal(np.asarray(env.get_sim_state()),initial),'Initial state changed')
        with CounterfactualCamera(env,task) as sampler:
            for warm in range(10):
                sampler.arm(row if warm==9 and row['step_id']==0 else None)
                raw,_,done,_=env.step([0.,0.,0.,0.,0.,0.,-1.])
                require(not done,'Unexpected warmup termination')
            for step in range(row['step_id']):
                sampler.arm(row if step+1==row['step_id'] else None)
                raw,_,done,_=env.step(np.asarray(actions[step],dtype=float).tolist())
                require(not done,'Replay ended before selected frame')
            align=alignment(raw,env,stored)
            require(align['pass'],'Recorded RGB/state/proprio prefix replay mismatch')
            payload=sampler.latest; require(payload is not None,'No native camera capture for selected frame')
            require(np.array_equal(payload['pixels'],native_observation(raw)['wrist_image']),'Camera payload is stale')
            require(np.max(abs(payload['proprio']-stored['proprio']))<=1e-6,'Camera/proprio timing mismatch')
            arrays=folder/'FRAMES.npz'; metadata=folder/'METADATA.json'
            atomic_npz(arrays,**payload['arrays'])
            atomic_json({'state_id':row['state_id'],'alignment':align,'conditions':payload['conditions'],
                         'sample_time_s':payload['sample_time_s'],'checkpoint_time_s':float(env.sim.data.time),
                         'sample_fingerprint':payload['sample_fingerprint'],'renderer':renderer,'vendor':vendor,
                         'physics_modified':False,'scope':'controlled_render_counterfactual_not_physical_rollout'},metadata)
        commit(receipt,[arrays,metadata],signature,state_id=row['state_id'])
    except BaseException as error:
        atomic_json({'state_id':row['state_id'],'error':repr(error),'details':getattr(error,'details',None),
                     'elapsed_s':time.monotonic()-started,**resources()},folder/f'FAILURE_{time.time_ns()}.json')
        raise
    finally:
        if env is not None: close_environment(env)


def render_all(prep):
    atlas_path()
    from atlas_runtime import add_paths,prepare_graphics_backend
    from pilot_common import runtime_config
    add_paths(runtime_config()); prepare_graphics_backend()
    tasks={r['task_id']:r for r in prep['tasks']}; started=time.monotonic()
    for index,row in enumerate(prep['states']):
        render_state(row,prep['episodes'][row['episode_id']],tasks[row['task_id']],prep['signature'])
        elapsed=time.monotonic()-started
        progress('render',completed=index+1,total=40,state_id=row['state_id'],elapsed_s=elapsed,
                 estimated_remaining_s=elapsed/(index+1)*(39-index),vla_loaded=False)
    commit(ROOT/'reports/RENDERS_COMPLETE.json',
           [ROOT/'data/renders'/r['state_id']/'COMPLETE.json' for r in prep['states']],prep['signature'],states=40)

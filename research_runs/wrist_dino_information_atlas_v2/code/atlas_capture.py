"""One Slurm GPU worker: replay stored actions, align labels, capture frozen DINO."""
from __future__ import annotations
import gc
import os
from pathlib import Path
import time
import shutil
import numpy as np
from atlas_common import (ROOT,atomic_json,commit,config,load_json,prepared,progress,receipt_valid,
                          require,resources,runtime_config,sha256_file)
from atlas_labels import array_hash,entity_map,label_specs,physical_labels,pose_bundle,render_labels
from atlas_math import rgb_error
from atlas_sampling import CameraLabels


def native_observation(raw):
    from atlas_runtime import quat2axisangle
    return {'full_image':np.asarray(raw['agentview_image'])[::-1,::-1].copy(),
            'wrist_image':np.asarray(raw['robot0_eye_in_hand_image'])[::-1,::-1].copy(),
            'state':np.concatenate([np.asarray(raw['robot0_eef_pos'],dtype=np.float32),
                                   quat2axisangle(raw['robot0_eef_quat']),
                                   np.asarray(raw['robot0_gripper_qpos'],dtype=np.float32)])}


def alignment(raw,env,stored):
    obs=native_observation(raw); rows={}
    rows['state_exact']=bool(np.array_equal(np.asarray(env.get_sim_state()),stored['sim_state']))
    rows['proprio_max_abs']=float(abs(obs['state']-stored['proprio']).max())
    for view,key in [('front','full_image'),('wrist','wrist_image')]:
        rows[view]=rgb_error(stored[view+'_rgb'],obs[key])
    c=config()['replay']
    rows['pass']=bool(rows['state_exact'] and rows['proprio_max_abs']<=c['proprio_atol'] and
                      all(rows[v]['normalized_rms']<=c['rgb_normalized_rms_limit'] for v in config()['views']))
    return rows


class ArraySink:
    """Task-sized disk arrays; no all-layer/whole-dataset RAM accumulation."""
    def __init__(self,folder,count):
        self.folder=Path(folder); self.count=count; self.maps={}; self.schema={}
        self.folder.mkdir(parents=True,exist_ok=True)

    def put(self,index,values):
        for key,value in values.items():
            value=np.asarray(value)
            if key not in self.maps:
                require(index==0,'Late feature appeared')
                self.maps[key]=np.lib.format.open_memmap(self.folder/(key+'.npy'),mode='w+',dtype=value.dtype,
                                                       shape=(self.count,*value.shape))
                self.schema[key]={'shape':list(value.shape),'dtype':str(value.dtype)}
            require(value.shape==tuple(self.schema[key]['shape']),'Feature layout drift')
            self.maps[key][index]=value
        require(set(values)==set(self.maps),'Incomplete feature capture')

    def close(self):
        for value in self.maps.values(): value.flush()
        self.maps.clear(); gc.collect()


def make_env(bddl):
    from atlas_runtime import prepare_graphics_backend
    prepare_graphics_backend()
    from libero.libero.envs import OffScreenRenderEnv
    # Keep original RGB observables unchanged. Depth/segmentation are direct checkpoint renders.
    return OffScreenRenderEnv(bddl_file_name=bddl,camera_names=['agentview','robot0_eye_in_hand'],
                              camera_heights=224,camera_widths=224)


def replay_episode(episode,states,task,folder,index_offset,old_lookup):
    """Label-only replay. The actual camera callbacks capture each selected frame."""
    from atlas_runtime import bind_environment_context,close_environment,renderer_metadata,reset_environment
    started=time.monotonic(); env=None; records=[]; sid=None; step=None
    state_by_step={r['step_id']:r for r in states}
    for key in ('initial_state_file','trajectory_file'):
        require(sha256_file(episode[key])==episode[key+'_sha256'],'Replay input checksum changed')
    with np.load(episode['initial_state_file'],allow_pickle=False) as z: initial=z['sim_state'].copy()
    with np.load(episode['trajectory_file'],allow_pickle=False) as z: actions=z['executed_actions'].copy()
    try:
        env=make_env(task['bddl_path']); env.seed(episode['rollout_seed'])
        raw=reset_environment(env); bind_environment_context(env); renderer,vendor=renderer_metadata(env)
        raw=env.set_init_state(initial)
        require(np.array_equal(np.asarray(env.get_sim_state()),initial),'Initial simulator state mismatch')
        with CameraLabels(env,task) as sampler:
            for warm in range(config()['replay']['warmup_steps']):
                sampler.arm(state_by_step.get(0,{}).get('state_id') if warm==config()['replay']['warmup_steps']-1 else None)
                raw,_,done,_=env.step([0.,0.,0.,0.,0.,0.,-1.])
                require(not done,'Unexpected warmup termination')
            done=False
            for step,action in enumerate(actions):
                require(not done,'Replay ended before recorded trace')
                if step in state_by_step:
                    row=state_by_step[step]; sid=row['state_id']
                    require(sha256_file(row['state_file'])==row['state_file_sha256'],'Source state checksum changed')
                    with np.load(row['state_file'],allow_pickle=False) as z: stored={k:z[k].copy() for k in z.files}
                    align=alignment(raw,env,stored)
                    if not align['pass']:
                        np.savez_compressed(folder/(sid+'_ALIGNMENT_FAILURE.npz'),
                            **{k:stored[k] for k in ('front_rgb','wrist_rgb','sim_state','proprio')},
                            replay_front=np.asarray(raw['agentview_image'])[::-1,::-1],
                            replay_wrist=np.asarray(raw['robot0_eye_in_hand_image'])[::-1,::-1],
                            replay_sim_state=np.asarray(env.get_sim_state()))
                        atomic_json({'state_id':sid,'alignment':align},folder/(sid+'_ALIGNMENT_FAILURE.json'))
                        raise RuntimeError('RGB/state alignment failed; preserve attempt: '+sid)
                    labels,output_truth,sample_poses,timing=sampler.take(raw)
                    label_arrays={}; visibility={}
                    for view,value in labels.items():
                        for key in ('instance_raw','class_raw','depth_raw','instance_crop','class_tokens','class_confidence',
                                    'depth_tokens','camera_position','camera_rotation'):
                            label_arrays[view+'_'+key]=value[key]
                        label_arrays[view+'_rgb']=stored[view+'_rgb']
                        output_truth.update({view+'_'+k:v for k,v in value['summary'].items()})
                        visibility[view]={k:v for k,v in value['summary'].items() if k.endswith(('visible','frustum','proxy'))}
                        label_arrays[view+'_camera_fovy']=np.asarray(value['camera_fovy'])
                    label_path=folder/'states'/(sid+'.npz'); label_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(label_path,**label_arrays)
                    old=old_lookup.get(sid,{})
                    deltas={new:float(output_truth[new]-old[previous]) for new,previous in
                            [('object_position_x','O1_x'),('object_position_y','O1_y'),('object_position_z','O1_z'),
                             ('object_eef_distance','R2_distance'),('contact','I3_contact'),('grasp','I3_grasp')]
                            if previous in old and output_truth.get(new) is not None}
                    records.append({'state_id':sid,'episode_id':row['episode_id'],'task_id':row['task_id'],
                        'split':row['split'],'step_id':step,'normalized_progress':row['normalized_progress'],
                        'row_index':index_offset+len(records),'truth':output_truth,'proprio':stored['proprio'].astype(float).tolist(),
                        'visibility':visibility,'entities':sampler.entities,'alignment':align,
                        'camera_sample_timing':timing,'label_pose_bundle':sample_poses,
                        'checkpoint_pose_bundle':pose_bundle(env),'old_label_deltas':deltas,
                        'direct_render_rgb_qa':{v:labels[v]['direct_rgb_error'] for v in labels},
                        'source_state_sha256':row['state_file_sha256'],
                        'label_source':'same_model_native_camera_sensor_callback_recorded_action_replay',
                        'original_object_camera_pose_saved':False})
                    progress('replay_labels',task=row['task_id'],episode=row['episode_id'],state=sid,
                             task_completed_states=index_offset+len(records),episode_elapsed_s=time.monotonic()-started)
                sampler.arm(state_by_step.get(step+1,{}).get('state_id'))
                raw,_,done,_=env.step(np.asarray(action,dtype=float).tolist())
        require(len(records)==5 and bool(env.env._check_success()),'Incomplete or unsuccessful clean replay')
        return records,{'episode_id':episode['episode_id'],'states':5,'success':True,
                        'elapsed_s':time.monotonic()-started,'renderer':renderer,'vendor':vendor}
    except BaseException as error:
        if hasattr(error,'evidence'):
            np.savez_compressed(folder/'RENDER_FAILURE.npz',**error.evidence)
        atomic_json({'episode_id':episode['episode_id'],'checkpoint_state_id':sid,'loop_step':step,
                     'armed_camera_target_state_id':getattr(locals().get('sampler'),'tag',None),
                     'error':repr(error),'details':getattr(error,'details',None),'elapsed_s':time.monotonic()-started},
                    folder/'FAILURE_CONTEXT.json')
        raise
    finally:
        if env is not None: close_environment(env)


def publish(attempt,final,signature,**fields):
    """Seal all hashes before exposing an atomic, complete task shard."""
    require(not final.exists(),'Refusing existing shard: '+str(final))
    files=[p for p in sorted(attempt.rglob('*')) if p.is_file()]
    atomic_json({'status':'COMPLETE','signature':signature,**fields,
        'artifacts':{str((final/p.relative_to(attempt)).relative_to(ROOT)):sha256_file(p) for p in files}},attempt/'COMPLETE.json')
    final.parent.mkdir(parents=True,exist_ok=True); os.replace(attempt,final)


def capture_labels(prep):
    import pandas as pd
    signature=prep['signature']; tasks={r['task_id']:r for r in load_json(config()['task_map'])['tasks']}
    old={r['state']:r for path in config()['old_labels'].values() for r in pd.read_parquet(path).to_dict('records')}
    for task_id in range(10):
        final=ROOT/f'data/labels/task_{task_id:02d}'
        if receipt_valid(final/'COMPLETE.json',signature): continue
        started=time.monotonic(); attempt=ROOT/f'data/attempts/labels_{task_id:02d}_{time.time_ns()}'
        attempt.mkdir(parents=True,exist_ok=False); records=[]; episodes=[]
        rows=sorted([r for r in prep['states'] if r['task_id']==task_id],key=lambda r:(r['episode_id'],r['step_id']))
        try:
            for name in sorted({r['episode_id'] for r in rows}):
                result,summary=replay_episode(prep['episodes'][name],[r for r in rows if r['episode_id']==name],
                                             tasks[task_id],attempt,len(records),old)
                records.extend(result); episodes.append(summary)
            require(len(records)==120 and len(episodes)==24,'Incomplete label task grid')
            atomic_json(records,attempt/'RECORDS.json')
            elapsed=time.monotonic()-started
            atomic_json({'episodes':episodes,'elapsed_s':elapsed,'states_per_second':120/elapsed,
                         'remaining_label_estimate_s':elapsed*(9-task_id),'vla_loaded':False,
                         'peak_scope':'capture_process_since_start',**resources()},attempt/'COST.json')
            publish(attempt,final,signature,task_id=task_id,states=120,episodes=24,stage='labels')
            print(f'LABELS task={task_id} states=120 elapsed={elapsed:.1f}s; VLA not loaded',flush=True)
        except BaseException as error:
            atomic_json({'status':'FAILED','error':repr(error),'elapsed_s':time.monotonic()-started,**resources()},attempt/'FAILURE.json')
            raise
    commit(ROOT/'reports/LABELS_COMPLETE.json',[ROOT/f'data/labels/task_{t:02d}/COMPLETE.json' for t in range(10)],
           signature,states=1200,episodes=240,vla_loaded=False)


def capture_features(prep,engine):
    signature=prep['signature']; source={r['state_id']:r for r in prep['states']}
    for task_id in range(10):
        final=ROOT/f'data/task_{task_id:02d}'
        if receipt_valid(final/'COMPLETE.json',signature): continue
        label_folder=ROOT/f'data/labels/task_{task_id:02d}'
        require(receipt_valid(label_folder/'COMPLETE.json',signature),'Unverified label shard')
        attempt=ROOT/f'data/attempts/features_{task_id:02d}_{time.time_ns()}'; attempt.mkdir(parents=True)
        records=load_json(label_folder/'RECORDS.json'); sinks=[ArraySink(attempt/'global',120),ArraySink(attempt/'dense',120)]
        started=time.monotonic()
        try:
            # Immutable label arrays are hard-linked inside this filesystem to avoid duplicate storage.
            shutil.copytree(label_folder/'states',attempt/'states',copy_function=os.link)
            shutil.copyfile(label_folder/'COMPLETE.json',attempt/'LABEL_SOURCE.json')
            for index,row in enumerate(records):
                original=source[row['state_id']]
                require(sha256_file(original['state_file'])==original['state_file_sha256'],'Source state changed')
                with np.load(original['state_file'],allow_pickle=False) as z:
                    obs={'full_image':z['front_rgb'],'wrist_image':z['wrist_rgb'],'state':z['proprio']}
                    features,dense,qa=engine.extract(obs,str(z['language']),validate=index<2)
                require(row['row_index']==index,'Label row index drift')
                sinks[0].put(index,features); sinks[1].put(index,dense); row['feature_qa']=qa
                progress('features',task=task_id,state=row['state_id'],task_completed_states=index+1)
            for sink in sinks: sink.close()
            atomic_json(records,attempt/'RECORDS.json')
            atomic_json({'global':sinks[0].schema,'dense':sinks[1].schema},attempt/'SCHEMA.json')
            atomic_json({'elapsed_s':time.monotonic()-started,'peak_scope':'capture_process_since_start',**resources()},attempt/'COST.json')
            publish(attempt,final,signature,task_id=task_id,states=120,episodes=24,stage='features')
            print(f'FEATURES task={task_id} states=120 elapsed={time.monotonic()-started:.1f}s',flush=True)
        except BaseException as error:
            for sink in sinks: sink.close()
            atomic_json({'status':'FAILED','error':repr(error),'elapsed_s':time.monotonic()-started,**resources()},attempt/'FAILURE.json')
            raise
    commit(ROOT/'reports/CAPTURE_COMPLETE.json',[ROOT/f'data/task_{t:02d}/COMPLETE.json' for t in range(10)],
           signature,states=1200,episodes=240,label_specs=label_specs())


def capture():
    import torch
    from atlas_runtime import add_paths,load_wrapper,prepare_graphics_backend
    from atlas_features import Encoder
    require(os.environ.get('SLURM_JOB_ID'),'Capture requires user-submitted Slurm allocation')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one allocated GPU required')
    require('4090' in torch.cuda.get_device_name(0),'4090 required; no automatic fallback')
    prep=prepared(); signature=prep['signature']; rt=runtime_config(); add_paths(rt); prepare_graphics_backend()
    if all(receipt_valid(ROOT/f'data/task_{t:02d}/COMPLETE.json',signature) for t in range(10)): return
    # Every original frame must acquire aligned labels before any VLA weights are loaded.
    capture_labels(prep)
    require(receipt_valid(ROOT/'reports/LABELS_COMPLETE.json',signature),'Label phase is not complete')
    progress('loading_frozen_vla',labels_complete=1200)
    wrapper=load_wrapper(rt,'libero_spatial')
    capture_features(prep,Encoder(wrapper))

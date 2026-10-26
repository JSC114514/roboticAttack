"""One Slurm GPU worker: replay stored actions, align labels, capture frozen DINO."""
from __future__ import annotations
import gc
import os
from pathlib import Path
import time
import numpy as np
from atlas_common import (ROOT,atomic_json,commit,config,load_json,prepared,progress,receipt_valid,
                          require,resources,runtime_config,sha256_file)
from atlas_labels import array_hash,entity_map,label_specs,physical_labels,pose_bundle,render_labels
from atlas_math import rgb_error


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


def replay_episode(episode,states,task,engine,folder,sinks,index_offset,qa_index,old_lookup):
    from atlas_runtime import bind_environment_context,close_environment,renderer_metadata,reset_environment
    started=time.monotonic(); env=None; records=[]; state_by_step={r['step_id']:r for r in states}
    for key in ('initial_state_file','trajectory_file'):
        require(sha256_file(episode[key])==episode[key+'_sha256'],'Replay input checksum changed')
    with np.load(episode['initial_state_file'],allow_pickle=False) as z: initial=z['sim_state'].copy()
    with np.load(episode['trajectory_file'],allow_pickle=False) as z: actions=z['executed_actions'].copy()
    try:
        env=make_env(task['bddl_path']); env.seed(episode['rollout_seed'])
        # Original collection creates an env, seeds it, then performs one reset.
        raw=reset_environment(env); bind_environment_context(env); renderer,vendor=renderer_metadata(env)
        raw=env.set_init_state(initial)
        require(np.array_equal(np.asarray(env.get_sim_state()),initial),'Initial simulator state mismatch')
        for _ in range(config()['replay']['warmup_steps']):
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
                    raise RuntimeError('RGB/state alignment failed; preserve this attempt: '+sid)
                entities=entity_map(env); poses_before=pose_bundle(env)
                truth=physical_labels(env,raw,entities,task)
                labels=render_labels(env,raw,entities,task)
                require(poses_before==pose_bundle(env),'Label extraction changed model or state')
                index=index_offset+len(records)
                obs={'full_image':stored['front_rgb'],'wrist_image':stored['wrist_rgb'],'state':stored['proprio']}
                features,dense,qa=engine.extract(obs,str(stored['language']),validate=qa_index+len(records)<2)
                sinks[0].put(index,features); sinks[1].put(index,dense)
                label_arrays={}; visibility={}; output_truth={k:(float(v) if np.isfinite(v) else None) for k,v in truth.items()}
                for view,value in labels.items():
                    for key in ('instance_raw','class_raw','depth_raw','instance_crop','class_tokens','class_confidence',
                                'depth_tokens','camera_position','camera_rotation'):
                        label_arrays[view+'_'+key]=value[key]
                    label_arrays[view+'_rgb']=stored[view+'_rgb']
                    output_truth.update({view+'_'+k:v for k,v in value['summary'].items()})
                    visibility[view]={k:v for k,v in value['summary'].items()
                                      if k.endswith(('visible','frustum','proxy'))}
                    label_arrays[view+'_camera_fovy']=np.asarray(value['camera_fovy'])
                label_path=folder/'states'/(sid+'.npz'); label_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(label_path,**label_arrays)
                old=old_lookup.get(sid,{})
                old_deltas={new:float(output_truth[new]-old[previous]) for new,previous in
                            [('object_position_x','O1_x'),('object_position_y','O1_y'),('object_position_z','O1_z'),
                             ('object_eef_distance','R2_distance'),('contact','I3_contact'),('grasp','I3_grasp')]
                            if previous in old and output_truth.get(new) is not None}
                records.append({'state_id':sid,'episode_id':row['episode_id'],'task_id':row['task_id'],
                                'split':row['split'],'step_id':step,'normalized_progress':row['normalized_progress'],
                                'row_index':index,'truth':output_truth,'proprio':stored['proprio'].astype(float).tolist(),
                                'visibility':visibility,'entities':entities,'alignment':align,'feature_qa':qa,
                                'direct_render_rgb_qa':{v:labels[v]['direct_rgb_error'] for v in labels},
                                'pose_bundle':poses_before,'old_label_deltas':old_deltas,
                                'source_state_sha256':row['state_file_sha256'],
                                'label_source':'same_realized_model_recorded_action_replay',
                                'original_object_camera_pose_saved':False})
                progress('capture',task=row['task_id'],episode=row['episode_id'],state=sid,
                         task_completed_states=index+1,episode_elapsed_s=time.monotonic()-started)
            raw,_,done,_=env.step(np.asarray(action,dtype=float).tolist())
        require(len(records)==5 and bool(env.env._check_success()),'Incomplete or unsuccessful clean replay')
        return records,{'episode_id':episode['episode_id'],'states':5,'success':True,
                        'elapsed_s':time.monotonic()-started,'renderer':renderer,'vendor':vendor}
    finally:
        if env is not None: close_environment(env)


def capture():
    import pandas as pd
    import torch
    from atlas_runtime import add_paths,load_wrapper,prepare_graphics_backend
    from atlas_features import Encoder
    require(os.environ.get('SLURM_JOB_ID'),'Capture requires user-submitted Slurm allocation')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one allocated GPU required')
    require('4090' in torch.cuda.get_device_name(0),'4090 required; no automatic fallback')
    prep=prepared(); signature=prep['signature']; rt=runtime_config(); add_paths(rt); prepare_graphics_backend()
    pending=[t for t in range(10) if not receipt_valid(ROOT/f'data/task_{t:02d}/COMPLETE.json',signature)]
    if not pending:
        commit(ROOT/'reports/CAPTURE_COMPLETE.json',[ROOT/f'data/task_{t:02d}/COMPLETE.json' for t in range(10)],
               signature,states=1200,episodes=240,label_specs=label_specs())
        return
    wrapper=load_wrapper(rt,'libero_spatial'); engine=Encoder(wrapper)
    old_lookup={}
    for path in config()['old_labels'].values():
        for row in pd.read_parquet(path).to_dict('records'): old_lookup[row['state']]=row
    tasks={r['task_id']:r for r in load_json(config()['task_map'])['tasks']}
    for task_id in pending:
        started=time.monotonic(); attempt=ROOT/f'data/attempts/task_{task_id:02d}_{time.time_ns()}'
        attempt.mkdir(parents=True,exist_ok=False)
        rows=sorted([r for r in prep['states'] if r['task_id']==task_id],key=lambda r:(r['episode_id'],r['step_id']))
        sinks=[ArraySink(attempt/'global',120),ArraySink(attempt/'dense',120)]
        records=[]; episodes=[]
        try:
            for name in sorted({r['episode_id'] for r in rows}):
                episode_rows=[r for r in rows if r['episode_id']==name]
                result,summary=replay_episode(prep['episodes'][name],episode_rows,tasks[task_id],engine,
                                               attempt,sinks,len(records),len(records),old_lookup)
                records.extend(result); episodes.append(summary)
            require(len(records)==120 and len(episodes)==24,'Incomplete task grid')
            for sink in sinks: sink.close()
            atomic_json(records,attempt/'RECORDS.json')
            atomic_json({'global':sinks[0].schema,'dense':sinks[1].schema},attempt/'SCHEMA.json')
            elapsed=time.monotonic()-started
            atomic_json({'episodes':episodes,'elapsed_s':elapsed,'states_per_second':120/elapsed,
                         'remaining_capture_estimate_s':elapsed*sum(t>task_id for t in pending),
                         'estimate_excludes_queue_and_cpu_analysis':True,
                         'peak_scope':'capture_process_since_start',**resources()},attempt/'COST.json')
            final=ROOT/f'data/task_{task_id:02d}'; require(not final.exists(),'Refusing existing task shard')
            # Hash and seal before the atomic directory rename; never publish an incomplete task.
            files=[p for p in sorted(attempt.rglob('*')) if p.is_file()]
            atomic_json({'status':'COMPLETE','signature':signature,'task_id':task_id,'states':120,'episodes':24,
                         'artifacts':{str((final/p.relative_to(attempt)).relative_to(ROOT)):sha256_file(p) for p in files}},
                        attempt/'COMPLETE.json')
            os.replace(attempt,final)
            print(f'CAPTURE task={task_id} states=120 episodes=24 elapsed={time.monotonic()-started:.1f}s',flush=True)
        except BaseException as error:
            for sink in sinks: sink.close()
            atomic_json({'status':'FAILED','error':repr(error),'elapsed_s':time.monotonic()-started,**resources()},
                        attempt/'FAILURE.json' if attempt.exists() else ROOT/f'reports/FAILED_TASK_{task_id}.json')
            raise
    commit(ROOT/'reports/CAPTURE_COMPLETE.json',[ROOT/f'data/task_{t:02d}/COMPLETE.json' for t in range(10)],
           signature,states=1200,episodes=240,global_sites=len(sinks[0].schema),label_specs=label_specs())

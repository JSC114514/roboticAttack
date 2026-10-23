"""Native PNG closed loops, per-control-step videos and read-only diagnostics."""
import copy
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from sf_common import ROOT, PROJECT, config, source, native, require, read, atomic, complete, finish, verify, external_complete
from audit_replay import SensorCapture, frame, fingerprint
from audit_math import visibility, event_summary
from rr_logger import PhysicsLogger
from sr_geometry import anchor, truth, controller


def conditions():
    v = read(ROOT / 'data/MANIFEST.json')
    if config()['scope'] == 'expansion':
        return {name: v['conditions'][name] for name in v['condition_order']}
    selected = read(ROOT / 'data/SELECTED.json')['candidates']
    return {name: selected[name]['path'] for name in sorted(selected)}


def folder(receipt):
    value = read(receipt)
    paths = [Path(k) for k in value['artifacts'] if k.endswith('/EPISODE.json')]
    require(len(paths) == 1, 'Ambiguous successful attempt')
    return ROOT / paths[0].parent


def paired_start(env, raw, seed, destination, external=None):
    runtime = native.runtime_module(); obs = runtime.observation(raw, native.runtime_config())
    arrays = dict(sim_state=np.array(env.get_sim_state(), copy=True), proprio=obs['state'], front=obs['full_image'], wrist=obs['wrist_image'])
    current = {'fingerprint': fingerprint(env), 'rollout_seed': seed}
    reference = external or destination
    if not (reference / 'START.json').exists():
        require(external is None, 'External clean start unavailable')
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / 'START.npz', **arrays); atomic(current, destination / 'START.json')
    expected = read(reference / 'START.json')
    require(current == expected, 'Paired physical/camera/clock fingerprint changed')
    with np.load(reference / 'START.npz', allow_pickle=False) as old:
        for name in arrays: require(np.array_equal(arrays[name], old[name]), 'Paired reset array/RGB changed: '+name)
    return arrays


def episode(env, task, initial, case, condition, patch, wrapper, attempt, monitor, external=None):
    import imageio.v2 as imageio
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    runtime = native.runtime_module(); cfg = native.runtime_config()
    raw = runtime.reset_wait(env, initial, case['rollout_seed'], cfg['num_steps_wait'])
    paired = ROOT / 'paired_starts' / case['start']
    arrays = paired_start(env, raw, case['rollout_seed'], paired, external)
    np.savez_compressed(attempt / 'START.npz', **arrays)
    target = anchor(env, case['suite'], case['task_id']); controller(env)
    fps = 1 / float(env.env.control_timestep); step = decision = 0; success = False
    events, steps, traces = [], [], []
    def observation():
        obs = runtime.observation(raw, cfg)
        return obs, obs if patch is None else native.paste_numpy(obs, patch)
    def record(captured=None, physics=None):
        geo = truth(env, target)
        vis = visibility(captured['mask'], patch is not None) if captured else None
        events.append({'step': step, 'time_s': step/fps, 'distance_m': geo['distance_m'], 'contact': geo['contact'], 'grasp': geo['grasp'],
                       'visibility_verified': captured is not None, 'sensor_time_s': captured['sample_time_s'] if captured else None,
                       **(vis or {k: None for k in ('raw_visible_pixels','policy_fov_pixels','policy_uncovered_pixels','patch_covered_pixels')}),
                       'tracking_error_peak_m': physics['tracking_error_peak_m'] if physics else None,
                       'peak_robot_contact_force_n': physics['peak_robot_contact_force_n'] if physics else None})
        return geo, vis
    with imageio.get_writer(attempt/'video.mp4', fps=fps, codec='libx264', quality=7, macro_block_size=16,
                            ffmpeg_params=['-threads','2','-pix_fmt','yuv420p']) as writer, SensorCapture(env,target) as camera, PhysicsLogger(env) as logger:
        geo, vis = record(); obs, deployed = observation()
        writer.append_data(frame(deployed, condition, step, fps, geo, vis))
        while step < 300 and not success:
            with torch.no_grad():
                actions = np.asarray(wrapper.predict_action(copy.deepcopy(deployed),task.language),dtype=np.float32).reshape(8,7)
                clean = actions.copy() if patch is None else np.asarray(wrapper.predict_action(copy.deepcopy(obs),task.language),dtype=np.float32).reshape(8,7)
            require(np.isfinite(actions).all() and np.isfinite(clean).all(), 'Nonfinite native actions')
            begin=step; used=0
            for slot in range(min(8,300-step)):
                before=truth(env,target); before_eef=np.asarray(raw['robot0_eef_pos']).copy()
                camera.arm(step+1); logger.arm()
                raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist());step+=1;used+=1
                physics=logger.take();captured=camera.take(raw,step);geo,vis=record(captured,physics)
                obs,deployed=observation();writer.append_data(frame(deployed,condition,step,fps,geo,vis))
                steps.append({'step':step,'decision_block':decision,'executed_slot':slot,'geometry_before':before,'geometry_after':geo,
                              'eef_before':before_eef.tolist(),'eef_after':np.asarray(raw['robot0_eef_pos']).tolist(),
                              'actual_eef_displacement_m':(np.asarray(raw['robot0_eef_pos'])-before_eef).tolist(),
                              'deployed_action':actions[slot].tolist(),'same_observation_clean_chunk_action':clean[slot].tolist(),'controller':physics})
                if done:success=True;break
            traces.append({'decision_block':decision,'step_start':begin,'step_end':step,'executed_slots':used,
                           'predicted_actions':actions.tolist(),'same_observation_clean_actions':clean.tolist(),'success_after_chunk':success})
            decision+=1;monitor.update(current_start=case['start'],condition=condition,executed_steps=step)
    frame_data=pd.DataFrame(events);frame_data.to_parquet(attempt/'EVENTS.parquet',index=False)
    pd.DataFrame(steps).to_parquet(attempt/'STEPS.parquet',index=False)
    pd.DataFrame(traces).to_parquet(attempt/'TRACE.parquet',index=False)
    atomic({**case,'condition':condition,'checkpoint':'libero_goal','png_sha256':None if patch is None else native.sha(conditions()[condition]),
            'success':int(success),'executed_steps':step,'fps':fps,'max_steps':300,'num_open_loop_steps':8,
            'events':event_summary(frame_data),'native_PNG_only':True,'model_frozen':True,
            'target_annotation':{k:v for k,v in target.items() if k!='gripper'},'strict_paired_reset':True},attempt/'EPISODE.json')


def base_root(): return PROJECT / 'research_runs/wrist_spatial_relation_expand_v1'


def base_success(case):
    base=base_root(); receipt=base/'episodes'/case['start']/'clean/COMPLETE.json'
    require(external_complete(base,receipt),'Expansion clean evidence unavailable')
    saved=read(receipt)
    items=[base/p for p in saved['artifacts'] if p.endswith('/EPISODE.json')]
    require(len(items)==1,'Ambiguous expansion clean result')
    result=read(items[0]); require(result['rollout_seed']==case['rollout_seed'] and result['checkpoint']=='libero_goal','Clean reuse identity differs')
    return bool(result['success'])


def run(index, monitor):
    receipt=ROOT/f'reports/EVALUATE_{index}_COMPLETE.json'
    if complete(receipt):return
    manifest=verify();paths=conditions();assigned=manifest['cases'][index::2];receipts=[]
    # Never allocate/load a large model for already completed episodes.
    from sf_train import make_engine
    native.require_gpu('spatial_followup_evaluate');runtime=native.runtime_module();runtime.prepare_graphics_backend()
    engine=make_engine();wrapper=engine.wrapper
    modules=(wrapper.vla,wrapper.action_head,wrapper.proprio_projector,wrapper.noisy_action_projector)
    require(all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in modules if m is not None),'Victim not frozen')
    from PIL import Image
    patches={k:np.asarray(Image.open(p).convert('RGB')).copy() if p else None for k,p in paths.items()}
    for n,case in enumerate(assigned):
        root=ROOT/'episodes'/case['start'];skip=root/'SKIPPED_CLEAN_FAILURE.json'
        if complete(skip):receipts.append(skip);continue
        if all(complete(root/name/'COMPLETE.json') for name in paths):
            receipts.extend(root/name/'COMPLETE.json' for name in paths);continue
        if config()['scope']=='control' and not base_success(case):
            atomic({'reason':'clean failed; not an attack success','start':case['start']},root/'SKIP.json')
            finish(skip,[root/'SKIP.json']);receipts.append(skip);continue
        env,task,states=runtime.make_env(native.runtime_config(),case['suite'],case['task_id'])
        try:
            require(task.name==case['task_name'],'Task mapping changed')
            saved=torch.load(case['initial_file'],map_location='cpu',weights_only=False)
            require(np.array_equal(states[case['state_index']],saved[case['state_index']]),'Initial state file changed')
            for condition in paths:
                if condition!='clean' and config()['scope']=='expansion':
                    clean_folder=folder(root/'clean/COMPLETE.json');old=read(clean_folder/'EPISODE.json')
                    if not old['success']:
                        finish(skip,[root/'clean/COMPLETE.json']);receipts.append(skip);break
                r=root/condition/'COMPLETE.json'
                if complete(r):receipts.append(r);continue
                attempt=r.parent/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True)
                try:
                    episode(env,task,states[case['state_index']],case,condition,patches[condition],wrapper,attempt,monitor,
                            base_root()/'paired_starts'/case['start'] if config()['scope']=='control' else None)
                except BaseException as error:
                    atomic({'status':'ENGINEERING_FAILURE','error':repr(error),'start':case['start'],'condition':condition},attempt/'FAILURE.json');raise
                files=[attempt/name for name in ('START.npz','EPISODE.json','TRACE.parquet','STEPS.parquet','EVENTS.parquet','video.mp4')]
                if config()['scope']=='expansion':files += [ROOT/'paired_starts'/case['start']/name for name in ('START.json','START.npz')]
                finish(r,files);receipts.append(r)
        finally:runtime.close_environment(env)
        monitor.update(completed=n+1,total=len(assigned),phase='native_PNG_closedloop')
    finish(receipt,receipts)

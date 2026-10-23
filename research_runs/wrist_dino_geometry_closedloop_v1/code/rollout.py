"""Native RGB closed loop. Reference predictions never step the simulator."""
from __future__ import annotations
import copy
import hashlib
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from protocol import (ROOT, PROJECT, config, native, require, sha, read_json, atomic_json,
                      atomic_parquet, complete, finish, episode_path, verify_inputs)

def fingerprint(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, array.dtype)).encode() + array.tobytes()).hexdigest()

def patch_array(path):
    with Image.open(path) as im:
        require(im.mode == 'RGB' and im.size == (50, 50), 'Invalid deployment PNG')
        return np.array(im, copy=True)

def ordered_conditions(task, state):
    values = list(config()['conditions'])
    random.Random(config()['order_seed'] + task * 100 + state).shuffle(values)
    return values

def predict(wrapper, obs, language):
    with torch.no_grad():
        action = np.asarray(wrapper.predict_action(copy.deepcopy(obs), language), dtype=np.float32).reshape(8, 7)
    require(np.isfinite(action).all(), 'Nonfinite native action')
    return action

def run_episode(env, wrapper, task, initial, seed, patch, monitor, folder,
                runtime=None, process_action=None, settings=None):
    runtime = native.runtime_module() if runtime is None else runtime
    if process_action is None:
        from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
        process_action = process_libero_action
    settings = config() if settings is None else settings
    raw = runtime.reset_wait(env, initial, seed, settings['num_steps_wait'])
    start_obs = runtime.observation(raw, native.runtime_config())
    start = {'start_sim_sha256': fingerprint(env.get_sim_state()),
             'start_front_sha256': fingerprint(start_obs['full_image']),
             'start_wrist_sha256': fingerprint(start_obs['wrist_image']),
             'start_proprio_sha256': fingerprint(start_obs['state'])}
    step = decision = 0
    success = False
    traces, images = [], []
    started = time.monotonic()
    while step < settings['max_steps'] and not success:
        clean_obs = runtime.observation(raw, native.runtime_config())
        deployed = native.paste_numpy(clean_obs, patch) if patch is not None else clean_obs
        action = predict(wrapper, deployed, task.language)
        # Same current observation, not a frame from a diverged clean trajectory.
        clean_action = predict(wrapper, clean_obs, task.language) if patch is not None else action.copy()
        pre_eef = np.asarray(raw['robot0_eef_pos']).copy()
        before = step
        if decision in settings['keyframe_decisions']:
            path = folder / f'frame_{decision:03d}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.concatenate([deployed['full_image'], deployed['wrist_image']], axis=1)).save(path)
            images.append(path)
        used = 0
        for slot in range(min(8, settings['max_steps'] - step)):
            raw, _reward, done, _info = env.step(process_action(action[slot]).tolist())
            step += 1
            used += 1
            if done:
                success = True
                break
        delta = action - clean_action
        traces.append({'decision_block': decision, 'step_start': before, 'step_end': step,
                       'executed_slots': used, 'predicted_actions': action.tolist(),
                       'same_observation_clean_actions': clean_action.tolist(),
                       'first_translation_delta_l2': float(np.linalg.norm(delta[0, :3])),
                       'executed_translation_delta_l2_mean': float(np.linalg.norm(delta[:used, :3], axis=1).mean()),
                       'executed_rotation_delta_l2_mean': float(np.linalg.norm(delta[:used, 3:6], axis=1).mean()),
                       'executed_gripper_delta_abs_mean': float(np.abs(delta[:used, 6]).mean()),
                       'eef_before': pre_eef.tolist(), 'eef_after': np.asarray(raw['robot0_eef_pos']).tolist(),
                       'gripper_qpos_after': np.asarray(raw['robot0_gripper_qpos']).tolist(),
                       'actual_eef_displacement': (np.asarray(raw['robot0_eef_pos']) - pre_eef).tolist(),
                       'success_after_chunk': success})
        decision += 1
        monitor.update(phase='native_closedloop', executed_steps=step, decision_blocks=decision,
                       current_episode_elapsed_s=time.monotonic() - started)
    path = folder / 'frame_final.png'
    final_obs = runtime.observation(raw, native.runtime_config())
    if patch is not None:
        final_obs = native.paste_numpy(final_obs, patch)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([final_obs['full_image'], final_obs['wrist_image']], axis=1)).save(path)
    images.append(path)
    return {**start, 'success': int(success), 'failure': int(not success), 'executed_steps': step,
            'decision_blocks': decision, 'elapsed_s': time.monotonic() - started}, traces, images

def run_worker(index, monitor):
    native.require_gpu('geometry_closedloop')
    manifest = read_json(ROOT / 'data/MANIFEST.json')
    verify_inputs(manifest)
    c = config()
    tasks = c['workers'][index]
    pending = [(t, s, k) for t in tasks for s in c['states'] for k in c['conditions']
               if not complete(episode_path(t, s, k))]
    if not pending:
        monitor.update(phase='already_complete', completed=20, total=20)
        return
    resume = 20 - len(pending)
    done = resume
    monitor.update(phase='loading_frozen_spatial', completed=done, resume_completed=resume, total=20)
    wrapper = native.load_wrapper('libero_spatial')
    runtime = native.runtime_module()
    patches = {'clean': None, **{k: patch_array(PROJECT / rel) for k, rel in c['patches'].items()}}
    for task_id in tasks:
        if not any(t == task_id for t, _, _ in pending):
            continue
        entry = manifest['initial_states'][str(task_id)]
        require(sha(entry['path']) == manifest['inputs'][entry['path']], 'Initial state file changed')
        env, task, states = runtime.make_env(native.runtime_config(), c['suite'], task_id,
            on_phase=lambda phase, **kw: monitor.update(phase=phase, task_id=task_id, **kw))
        try:
            require(task.name == entry['task_name'], 'Benchmark task mapping changed')
            renderer, vendor = runtime.renderer_metadata(env)
            for state in c['states']:
                for condition in ordered_conditions(task_id, state):
                    receipt = episode_path(task_id, state, condition)
                    if complete(receipt):
                        continue
                    folder = receipt.parent
                    seed = manifest['seeds'][f'{task_id}/{state}']
                    monitor.update(task_id=task_id, state_index=state, condition=condition,
                                   completed=done, total=20)
                    result, traces, images = run_episode(env, wrapper, task, states[state], seed,
                                                        patches[condition], monitor, folder)
                    # Hard pairing gate also works after resume and randomized condition order.
                    keys = ('start_sim_sha256', 'start_front_sha256', 'start_wrist_sha256', 'start_proprio_sha256')
                    for other in c['conditions']:
                        prior = episode_path(task_id, state, other)
                        if complete(prior):
                            paired = read_json(prior.parent / 'EPISODE.json')
                            require(all(result[k] == paired[k] for k in keys),
                                    'Paired reset state/camera mismatch; no scientific success recorded')
                    meta = {'suite': c['suite'], 'checkpoint': 'libero_spatial', 'task_id': task_id,
                            'state_index': state, 'condition': condition, 'rollout_seed': seed,
                            'png_sha256': sha(PROJECT / c['patches'][condition]) if condition != 'clean' else None,
                            'renderer': renderer, 'renderer_vendor': vendor, 'reused_clean': False,
                            'internal_interventions': False, 'source': 'native_RGB', **result}
                    atomic_json(meta, folder / 'EPISODE.json')
                    atomic_parquet(pd.DataFrame(traces), folder / 'TRACE.parquet')
                    finish(receipt, [folder / 'EPISODE.json', folder / 'TRACE.parquet', *images])
                    done += 1
                    monitor.update(completed=done, total=20)
        finally:
            runtime.close_environment(env)
    finish(ROOT / f'evaluation/worker_{index}_COMPLETE.json',
           [episode_path(t, s, k) for t in tasks for s in c['states'] for k in c['conditions']],
           **native.resources())

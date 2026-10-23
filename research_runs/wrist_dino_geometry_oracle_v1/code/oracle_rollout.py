"""Continuous sensor-aligned DINO-only intervention, never a patch result."""
from __future__ import annotations
from contextlib import contextmanager
import gc
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from oracle_common import (ROOT, MATCH, config, native, load_module, require, read_json, atomic_json,
                           atomic_parquet, finish, complete, episode_path, verify_inputs)
from oracle_camera import Camera

@contextmanager
def reset_camera(env, initial, seed, runtime, settings, dummy_action):
    """Native reset_wait sequence, with the hook installed AFTER hard reset.

    Hard reset rewrites observable sensors. Hooking before it silently loses the
    capture callback. Only the final wait step samples the first CF observation.
    """
    env.seed(seed)
    runtime.bind_environment_context(env)
    runtime.reset_environment(env)
    gc.collect()
    runtime.renderer_metadata(env)
    raw = env.set_init_state(initial)
    require(np.array_equal(np.asarray(env.get_sim_state()), np.asarray(initial)), 'Initial state reset mismatch')
    with Camera(env) as camera:
        for step in range(settings['num_steps_wait']):
            camera.arm('decision_0' if step == settings['num_steps_wait'] - 1 else None)
            raw, _, done, _ = env.step(dummy_action())
            require(not done, 'Episode ended during native wait')
        runtime.renderer_metadata(env)
        runtime.validate_camera_frames(raw)
        yield raw, camera

def start_arrays(env, obs):
    return {'sim_state': np.array(env.get_sim_state(), copy=True),
            'front': np.array(obs['full_image'], copy=True), 'wrist': np.array(obs['wrist_image'], copy=True),
            'proprio': np.array(obs['state'], copy=True)}

def compare_start(current, previous):
    rows = {}
    for key, value in current.items():
        before = previous[key]
        shape_match = value.shape == before.shape
        rows[key] = {'equal': bool(np.array_equal(value, before)), 'shape_match': shape_match,
                     'current_shape': list(value.shape), 'reference_shape': list(before.shape),
                     'max_abs_difference': float(np.max(np.abs(value.astype(float) - before.astype(float)))) if shape_match else None,
                     'different_elements': int(np.count_nonzero(value != before)) if shape_match else None}
    return rows

def pairing_gate(env, obs, task_id, state, condition, attempt):
    current = start_arrays(env, obs)
    path = attempt / 'START.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **current)
    for other in config()['conditions']:
        prior = episode_path(task_id, state, other)
        if complete(prior):
            meta = read_json(prior.parent / 'EPISODE.json')
            with np.load(ROOT / meta['start_npz'], allow_pickle=False) as reference:
                details = compare_start(current, reference)
            if not all(v['equal'] for v in details.values()):
                atomic_json({'task_id': task_id, 'state_index': state, 'condition': condition,
                             'reference_condition': other, 'differences': details,
                             'current_start_npz': str(path.relative_to(ROOT)),
                             'reference_start_npz': meta['start_npz']},
                            ROOT / f'reports/PAIRING/{attempt.name}.json')
                raise RuntimeError('Paired reset mismatch BEFORE action execution; see reports/PAIRING')
    return path

def cf_observation(raw, payload, sign, runtime):
    changed = {**raw, 'robot0_eye_in_hand_image': payload['images'][sign][::-1, ::-1].copy()}
    return runtime.observation(changed, native.runtime_config())

def verify_engine(engine, obs, raw, payload, language, runtime):
    """One native GPU gate; it observes effects but sets no efficacy threshold."""
    action, clean = engine.forward(obs, language)
    identity, restored = engine.forward(obs, language, {'dino': clean['dino']})
    require(np.array_equal(action, identity), 'Identity DINO intervention changes native action')
    for sign in ('plus', 'minus'):
        cf = cf_observation(raw, payload, sign, runtime)
        feature = engine.spatial_observation(cf).numpy()
        native_action, features = engine.forward(cf, language)
        require(np.array_equal(feature.reshape(features['dino'].shape), features['dino']),
                'CF standalone DINO differs from native consumed output')
        replaced_action, changed = engine.forward(obs, language,
            {'dino': features['dino'], 'siglip': features['siglip']})
        require(np.array_equal(native_action, replaced_action), 'CF dual-branch replacement differs from native RGB')
        _, dino_only = engine.forward(obs, language, {'dino': feature})
        require(np.array_equal(dino_only['dino'], feature.reshape(dino_only['dino'].shape)), 'DINO target not realized')
        require(np.array_equal(dino_only['siglip'], clean['siglip']), 'SigLIP changed during DINO-only intervention')
    atomic_json({'status': 'PASS', 'identity_action': True, 'cf_native_features': True,
                 'dual_branch_native_action': True, 'dino_only_siglip_clean': True,
                 'no_target_at_gate': payload['no_target'], **native.resources()}, ROOT / 'reports/GPU_GATE.json')

def episode(env, task, initial, seed, condition, task_id, state, engine, runtime, monitor, attempt, gate):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action, get_libero_dummy_action
    c = config()
    traces, images = [], []
    step = decision = 0
    success = False
    started = time.monotonic()
    with reset_camera(env, initial, seed, runtime, c, get_libero_dummy_action) as (raw, camera):
        obs = runtime.observation(raw, native.runtime_config())
        start = pairing_gate(env, obs, task_id, state, condition, attempt)
        while step < c['max_steps'] and not success:
            obs = runtime.observation(raw, native.runtime_config())
            payload = camera.take(raw, f'decision_{decision}')
            if not gate['done']:
                verify_engine(engine, obs, raw, payload, task.language, runtime)
                gate['done'] = True
            clean_action, clean_features = engine.forward(obs, task.language)
            sign = condition.removeprefix('dino_') if condition != 'clean' else None
            cf = None
            feature_l2 = 0.
            if sign is not None and not payload['no_target']:
                cf = cf_observation(raw, payload, sign, runtime)
                feature = engine.spatial_observation(cf).numpy()
                action, features = engine.forward(obs, task.language, {'dino': feature})
                require(np.array_equal(features['dino'], feature.reshape(features['dino'].shape)), 'Imperfect DINO target replacement')
                require(np.array_equal(features['siglip'], clean_features['siglip']), 'DINO intervention contaminated SigLIP')
                feature_l2 = float(np.linalg.norm(feature.reshape(clean_features['dino'].shape) - clean_features['dino']))
            else:
                action = clean_action.copy()
            require(action.shape == (8, 7) and np.isfinite(action).all(), 'Invalid action chunk')
            if decision in c['keyframe_decisions']:
                wrist = obs['wrist_image'] if cf is None else cf['wrist_image']
                path = attempt / f'frame_{decision:03d}.png'
                Image.fromarray(np.concatenate([obs['full_image'], obs['wrist_image'], wrist], axis=1)).save(path)
                images.append(path)
            before_eef = np.array(raw['robot0_eef_pos'], copy=True)
            before = step
            slots = min(8, c['max_steps'] - step)
            used = 0
            for slot in range(slots):
                # Same last-executed-slot callback scheduling as the audited source
                # replay. No render or physics refresh at policy-decision time.
                camera.arm(f'decision_{decision + 1}' if slot == slots - 1 else None)
                raw, _, done, _ = env.step(process_libero_action(action[slot].copy()).tolist())
                step += 1
                used += 1
                if done:
                    success = True
                    break
            delta = action - clean_action
            traces.append({'decision_block': decision, 'step_start': before, 'step_end': step,
                           'executed_slots': used, 'predicted_actions': action.tolist(),
                           'same_observation_clean_actions': clean_action.tolist(),
                           'translation_delta_l2_mean': float(np.linalg.norm(delta[:used, :3], axis=1).mean()),
                           'gripper_disagreement': float(((action[:used, 6] > .5) != (clean_action[:used, 6] > .5)).mean()),
                           'target_available': not payload['no_target'],
                           'intervention_applied': sign is not None and not payload['no_target'],
                           'dino_feature_delta_l2': feature_l2,
                           'target_visible_after_shift': payload['visible_pixels'].get(sign) if sign else None,
                           'sample_time_s': payload['sample_time_s'],
                           'geometry': {k: v for k, v in payload.items() if k not in ('images', 'native_pixels', 'key')},
                           'eef_before': before_eef.tolist(), 'eef_after': np.asarray(raw['robot0_eef_pos']).tolist(),
                           'gripper_qpos_after': np.asarray(raw['robot0_gripper_qpos']).tolist(), 'success_after_chunk': success})
            decision += 1
            monitor.update(phase='oracle_closedloop', executed_steps=step, decision_blocks=decision,
                           capture_calls=camera.captures, capture_elapsed_s=camera.elapsed_s,
                           current_episode_elapsed_s=time.monotonic() - started)
        final = runtime.observation(raw, native.runtime_config())
        path = attempt / 'frame_final.png'
        Image.fromarray(np.concatenate([final['full_image'], final['wrist_image']], axis=1)).save(path)
        images.append(path)
        available = sum(r['target_available'] for r in traces)
        applied_slots = sum(r['executed_slots'] for r in traces if r['intervention_applied'])
        return {'success': int(success), 'failure': int(not success), 'executed_steps': step,
                'decision_blocks': decision, 'available_target_decisions': available,
                'intervened_executed_slots': applied_slots, 'intervention_step_fraction': applied_slots / step,
                'start_npz': str(start.relative_to(ROOT)), 'elapsed_s': time.monotonic() - started,
                'capture_calls': camera.captures, 'capture_elapsed_s': camera.elapsed_s}, traces, [start, *images]

def run(monitor):
    native.require_gpu('geometry_oracle')
    manifest = read_json(ROOT / 'data/MANIFEST.json')
    verify_inputs(manifest)
    c = config()
    all_paths = [episode_path(t, s, k) for t in c['tasks'] for s in c['states'] for k in c['conditions']]
    completed = sum(complete(path) for path in all_paths)
    if completed == 30:
        monitor.update(phase='already_complete', completed=30, total=30)
        return
    monitor.update(phase='loading_frozen_spatial', completed=completed, resume_completed=completed, total=30)
    wrapper = native.load_wrapper('libero_spatial')
    engine = load_module('oracle_native_engine', MATCH / 'code/engine.py').Engine(wrapper)
    runtime = native.runtime_module()
    gate = {'done': False}
    for task_id in c['tasks']:
        if all(complete(episode_path(task_id, s, k)) for s in c['states'] for k in c['conditions']):
            continue
        env, task, initial = runtime.make_env(native.runtime_config(), c['suite'], task_id,
            on_phase=lambda phase, **kw: monitor.update(phase=phase, task_id=task_id, **kw))
        try:
            require(task.name == manifest['initial_states'][str(task_id)]['task_name'], 'Task mapping mismatch')
            for state in c['states']:
                order = list(c['conditions'])
                random.Random(c['order_seed'] + task_id * 100 + state).shuffle(order)
                for condition in order:
                    receipt = episode_path(task_id, state, condition)
                    if complete(receipt):
                        continue
                    attempt = receipt.parent / f'attempt_{time.time_ns()}'
                    attempt.mkdir(parents=True, exist_ok=False)
                    seed = manifest['seeds'][f'{task_id}/{state}']
                    monitor.update(task_id=task_id, state_index=state, condition=condition, completed=completed, total=30)
                    result, traces, artifacts = episode(env, task, initial[state], seed, condition,
                        task_id, state, engine, runtime, monitor, attempt, gate)
                    meta = {'suite': c['suite'], 'checkpoint': 'libero_spatial', 'task_id': task_id,
                            'state_index': state, 'rollout_seed': seed, 'condition': condition,
                            'scope': 'internal_DINO_mechanism_not_RGB_patch', 'reused_clean': False, **result}
                    atomic_json(meta, receipt.parent / 'EPISODE.json')
                    atomic_parquet(pd.DataFrame(traces), receipt.parent / 'TRACE.parquet')
                    finish(receipt, [receipt.parent / 'EPISODE.json', receipt.parent / 'TRACE.parquet', *artifacts])
                    completed += 1
                    monitor.update(completed=completed, total=30)
        finally:
            runtime.close_environment(env)
    finish(ROOT / 'evaluation/COMPLETE.json', all_paths, **native.resources())

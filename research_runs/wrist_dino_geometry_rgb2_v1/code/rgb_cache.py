"""Replay saved native actions; render true 2x RGB targets, never extrapolate features."""
import time
import numpy as np
import pandas as pd
import torch
from rgb_common import (ROOT, RELATION, config, native, Camera, require, signature, read_json,
                        atomic_json, atomic_torch, complete, finish, selections, tasks,
                        cache_receipt, make_engine, verify_inputs)

def alignment(obs, reference):
    errors = {key: float(np.sqrt(np.mean(((np.asarray(obs[key], float) - np.asarray(reference[key], float)) / 255.) ** 2)))
              for key in ('full_image', 'wrist_image')}
    errors['proprio_max_abs'] = float(np.max(np.abs(np.asarray(obs['state']) - np.asarray(reference['state']))))
    return errors

def records(cohort):
    wanted = selections()[cohort]
    for episode in sorted({r['episode_id'] for r in wanted}):
        receipt = cache_receipt(episode)
        require(complete(receipt), 'Target cache incomplete: ' + episode)
        value = torch.load(receipt.with_suffix('.pt'), map_location='cpu', weights_only=False)
        require(value['signature'] == signature(), 'Cache revision changed')
        yield from value['records']

def run(index, monitor):
    native.require_gpu('rgb2_cache')
    manifest = verify_inputs()
    all_rows = [r for values in selections().values() for r in values]
    assigned = tasks()[index::2]
    episodes = {r['episode_id'] for r in all_rows if (r['suite'],r['task_id']) in assigned}
    done = sum(complete(cache_receipt(e)) for e in episodes)
    monitor.update(completed=done, resume_completed=done, total=len(episodes), phase='loading_spatial')
    if done == len(episodes): return
    traces = pd.read_parquet(manifest['trace_path'], filters=[('condition', '==', 'clean')])
    engine = make_engine()
    runtime = native.runtime_module()
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    for suite, task_id in assigned:
        selected = [r for r in all_rows if r['suite'] == suite and r['task_id'] == task_id]
        if all(complete(cache_receipt(r['episode_id'])) for r in selected): continue
        source = torch.load(RELATION / f'data/cache/train/{suite}/task_{task_id:02d}.pt', map_location='cpu', weights_only=False)
        refs = {r['state_id']: r for r in source['records'] if r['state_index'] in (40,41,42,43)}
        env, task, initial = runtime.make_env(native.runtime_config(), suite, task_id,
            on_phase=lambda phase, **kw: monitor.update(phase=phase, suite=suite, task_id=task_id, **kw))
        try:
            require(task.name == manifest['initial_states'][f'{suite}/{task_id}']['task_name'], 'Task mapping changed')
            for episode_id in sorted({r['episode_id'] for r in selected}):
                receipt = cache_receipt(episode_id)
                if complete(receipt): continue
                positions = {r['decision_block']: r for r in selected if r['episode_id'] == episode_id}
                sample = next(iter(positions.values()))
                state, seed = sample['state_index'], sample['rollout_seed']
                current = traces[(traces.suite == suite) & (traces.task_id == task_id) &
                                 (traces.state_index == state) & (traces.rollout_seed == seed)].sort_values('decision_block')
                require(len(current) == sample['decision_count'] and min(positions) > 0, 'Source trace boundary changed')
                final = max(positions)
                required_steps = int(current.loc[current.decision_block < final].executed_slots.sum())
                raw = runtime.reset_wait(env, initial[state], seed, config()['num_steps_wait'])
                output, audits = [], []
                executed = 0
                with Camera(env) as camera:
                    for saved in current.itertuples(index=False):
                        decision = int(saved.decision_block)
                        if decision in positions:
                            row = positions[decision]
                            payload = camera.take(raw, row['state_id'])
                            obs = runtime.observation(raw, native.runtime_config())
                            error = alignment(obs, refs[row['state_id']]['observation'])
                            if max(error['full_image'],error['wrist_image']) > 1e-4 or error['proprio_max_abs'] > 1e-6:
                                atomic_json({'state_id': row['state_id'], 'alignment': error}, ROOT / f'reports/ALIGNMENT/{row["state_id"]}.json')
                                raise RuntimeError('Source cache observation alignment failed: ' + row['state_id'])
                            images, features = {}, {}
                            for sign in ('minus', 'plus'):
                                cf_raw = {**raw, 'robot0_eye_in_hand_image': payload['images'][sign][::-1,::-1].copy()}
                                cf_obs = runtime.observation(cf_raw, native.runtime_config())
                                images['translation_' + sign] = cf_obs['wrist_image']
                                features['translation_' + sign] = engine.spatial_observation(cf_obs).half()
                            record = {**row, 'observation': obs, 'language': task.language,
                                      'counterfactual_wrist': images, 'target_features': features,
                                      'geometry': {k:v for k,v in payload.items() if k not in ('images','native_pixels','key')},
                                      'alignment': error, 'displacement_scale': 2.}
                            engine.verify_patch_path(record)
                            output.append(record)
                            if decision == final: break
                        actions = native.decode_action_chunk(saved.predicted_environment_action_chunk, episode_id)
                        used = int(saved.executed_slots)
                        require(1 <= used <= 8, 'Invalid saved executed slots')
                        for slot in range(used):
                            next_row = positions.get(decision + 1) if slot == used - 1 else None
                            camera.arm(next_row['state_id'] if next_row else None)
                            raw, _, ended, _ = env.step(process_libero_action(actions[slot].copy()).tolist())
                            executed += 1
                            require(not ended or executed >= required_steps, 'Replay ended before selected prefix: ' + episode_id)
                        require(executed == int(saved.step_end), 'Saved step boundary changed')
                require(len(output) == 5 and executed == required_steps, 'Incomplete selected prefix')
                path = receipt.with_suffix('.pt')
                atomic_torch({'signature': signature(), 'records': output, 'episode_id': episode_id}, path)
                finish(receipt, [path], frames=5, cohort=sample['cohort'], replayed_steps=executed)
                done += 1
                monitor.update(phase='aligned_2x_targets', completed=done, total=len(episodes), suite=suite,
                               task_id=task_id, state_index=state, frames=done*5)
        finally:
            runtime.close_environment(env)
        del source, refs


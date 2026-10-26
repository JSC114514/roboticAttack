"""Allocated-GPU worker only. No scheduler submission or training entry points."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import random
import signal
import time
import traceback
from pathlib import Path

from functional77_common import (ROOT, atomic_json, atomic_torch, cases, cfg, completed_episode,
    digest, episode_folder, episode_key, exclusive, load_json, parent_module, prepared, require,
    sha256, signature, source_hashes, specs)
from functional77_progress import Progress


def state_snapshot(env, raw):
    import numpy as np
    result = {'sim_state': np.asarray(env.get_sim_state()).tolist()}
    for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
        result[key] = np.asarray(raw[key]).tolist()
    return result


def array_hash(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return digest({'shape': list(value.shape), 'dtype': str(value.dtype),
                   'bytes': hashlib.sha256(value.tobytes()).hexdigest()})


def episode(env, wrapper, probe, task, initial, rgb, runtime, case, spec, folder, progress,
            model=None, process_action=None):
    """Dependency injection is used only by CPU synthetic integration tests."""
    import numpy as np
    import torch
    from functional77_math import component_replacement, directional_action_stats, group_readouts, norm
    from functional77_probe import seeded
    if model is None:
        model = parent_module('model_runtime')
    if process_action is None:
        from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
        process_action = process_libero_action
    folder = Path(folder)
    attempt = folder / ('attempt_' + str(time.time_ns()) + '_' + str(os.getpid()))
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json({'spec': spec, 'case': case['id'], 'rollout_seed': case['rollout_seed'],
                 'created_unix': time.time(), 'resume': 'restart initial state; preserve unfinished attempt'},
                attempt / 'START.json')
    trace_path, assets = attempt / 'trace.jsonl', []
    started = time.monotonic()
    seed = case['rollout_seed']
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    raw = model.reset_wait(env, initial, seed, 10)
    limit = runtime['max_steps_by_suite'][case['suite']]
    step = block = effective = 0
    success = False
    first_state = first_observation_hashes = None
    active = spec['direction'] != 'native'
    with trace_path.open('x') as journal:
        while step < limit and not success:
            progress.update('policy_decision', episode=episode_key(spec), decision_block=block, simulator_step=step)
            obs, before = model.observation(raw, runtime), state_snapshot(env, raw)
            if block == 0:
                first_state = before
                first_observation_hashes = {key: array_hash(obs[key]) for key in ['full_image', 'wrist_image', 'state']}
            patched_obs = probe.paste(obs, rgb)
            recipient_is_patch = (active and spec['direction'] == 'restore') or (
                not active and spec['component'] == 'png')
            policy_obs = patched_obs if recipient_is_patch else obs
            diagnostic = {'intervention_active': active, 'basis_refreshed': False}
            if active:
                pair = probe.pair(obs, task.language, rgb)
                if block == 0:
                    diagnostic['validation'] = probe.validate(pair)
                rows, checks, gradient_summary, gradients = probe.functional(
                    pair, finite=block == 0, on_phase=progress.update)
                basis = rows['all']
                control_seed = seeded(case['id'], spec['direction'], block, 'random_span')
                replacement, stats = component_replacement(
                    pair['clean_dino'], pair['patched_dino'], basis, spec['component'],
                    spec['direction'], control_seed, cfg()['norm_match_relative_tolerance'])
                endpoint = stats['recipient_endpoint']
                action = probe.intervention(pair, replacement, endpoint)
                actions = probe.environment_actions(action)
                clean_env = probe.environment_actions(pair['clean_action'])
                patched_env = probe.environment_actions(pair['patched_action'])
                clean_commands = np.stack([process_action(a) for a in clean_env])
                patch_commands = np.stack([process_action(a) for a in patched_env])
                commands = np.stack([process_action(a) for a in actions])
                reference_commands = patch_commands if recipient_is_patch else clean_commands
                change = replacement.float() - pair[endpoint + '_dino'].float()
                readouts, _ = group_readouts(pair['clean_action'], pair['patched_action'])
                diagnostic.update(stats, **gradient_summary, basis_refreshed=True, basis_derived_at_block=block,
                    finite_differences=checks, footprint_tokens=int(pair['support'].sum()),
                    clean_normalized_action=pair['clean_action'].cpu().tolist(),
                    patched_normalized_action=pair['patched_action'].cpu().tolist(),
                    intervened_normalized_action=action.cpu().tolist(),
                    clean_environment_action_chunk=clean_env.tolist(),
                    patched_environment_action_chunk=patched_env.tolist(),
                    clean_processed_action_chunk=clean_commands.tolist(),
                    patched_processed_action_chunk=patch_commands.tolist(),
                    recipient_processed_action_chunk=reference_commands.tolist(),
                    intervened_processed_action_chunk=commands.tolist(),
                    processed_recipient_change_l2=float(np.linalg.norm(commands - reference_commands)),
                    predicted_gripper_flips=int(np.count_nonzero(commands[:, -1] != reference_commands[:, -1])),
                    siglip_clean_patch_gap_l2=norm(pair['patched_siglip'].float() - pair['clean_siglip'].float()),
                    **directional_action_stats(pair['clean_action'], pair['patched_action'], action, spec['direction']))
                for name, gradient in gradients.items():
                    diagnostic['linear_prediction_' + name] = float((gradient.to(change.device) * change).sum())
                for name, readout in readouts.items():
                    diagnostic['observed_readout_change_' + name] = float(
                        ((action - pair[endpoint + '_action']) * readout).sum())
                effective += int(stats['actual_change_l2'] > 0)
                if block in cfg()['snapshot_blocks']:
                    path = attempt / ('block_' + str(block).zfill(3) + '.pt')
                    atomic_torch({'spec': spec, 'case': case['id'], 'rollout_seed': seed,
                        'observation': obs, 'task_name': task.language, 'before': before,
                        'clean_dino': pair['clean_dino'].cpu(), 'patched_dino': pair['patched_dino'].cpu(),
                        'replacement': replacement.cpu(), 'basis': basis.cpu(), 'endpoint_vjps': gradients,
                        'support': pair['support'].cpu(), 'diagnostic': diagnostic,
                        'resume_contract': 'diagnostic only; replay verified action prefix from frozen initial state'}, path)
                    assets.append(path)
                del pair, rows, basis, gradients, action, replacement, change
            else:
                with torch.no_grad():
                    actions = np.asarray(wrapper.predict_action(policy_obs, task.language), dtype=np.float32).reshape(8, 7)
            require(actions.shape == (8, 7) and np.isfinite(actions).all(), 'Invalid native action chunk')
            commands = np.stack([process_action(a) for a in actions])
            slots, applied = 0, []
            for slot in range(min(8, limit - step)):
                command = commands[slot].tolist()
                raw, _, done, _ = env.step(command)
                step += 1
                slots += 1
                applied.append(command)
                if done:
                    success = True
                    break
            after = state_snapshot(env, raw)
            hypothetical_rgb_delta = np.asarray(patched_obs['wrist_image'], dtype=np.float32) - np.asarray(obs['wrist_image'], dtype=np.float32)
            rgb_delta = hypothetical_rgb_delta if recipient_is_patch else np.zeros_like(hypothetical_rgb_delta)
            trace = {'decision_block': block, 'executed_slots': slots, 'step_end': step,
                'predicted_environment_action_chunk': actions.tolist(),
                'predicted_processed_action_chunk': commands.tolist(), 'applied_actions': applied,
                'before': before, 'after': after,
                'eef_displacement': (np.asarray(after['robot0_eef_pos']) - np.asarray(before['robot0_eef_pos'])).tolist(),
                'policy_recipient_visual_input': 'actual_png' if recipient_is_patch else 'clean',
                'clean_wrist_sha256': array_hash(obs['wrist_image']),
                'patched_wrist_sha256': array_hash(patched_obs['wrist_image']),
                'policy_wrist_sha256': array_hash(policy_obs['wrist_image']),
                'policy_rgb_changed_pixels': int(np.any(rgb_delta != 0, axis=-1).sum()),
                'policy_rgb_delta_l2_255': float(np.linalg.norm(rgb_delta)),
                'policy_rgb_delta_linf_255': float(np.abs(rgb_delta).max()),
                'hypothetical_png_changed_pixels': int(np.any(hypothetical_rgb_delta != 0, axis=-1).sum()),
                'hypothetical_png_delta_l2_255': float(np.linalg.norm(hypothetical_rgb_delta)),
                'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,
                'cuda_allocated_bytes': int(torch.cuda.memory_allocated()) if torch.cuda.is_initialized() else 0,
                'elapsed_s': time.monotonic() - started, **diagnostic}
            if active:
                reference = np.asarray(trace['recipient_processed_action_chunk'])[:slots]
                executed = np.asarray(applied)
                trace['executed_gripper_flips'] = int(np.count_nonzero(executed[:, -1] != reference[:, -1]))
                trace['executed_recipient_translation_change_l2'] = float(np.linalg.norm(executed[:, :3] - reference[:, :3]))
                trace['executed_recipient_rotation_change_l2'] = float(np.linalg.norm(executed[:, 3:6] - reference[:, 3:6]))
                trace['executed_recipient_processed_change_l2'] = float(np.linalg.norm(executed - reference))
            journal.write(json.dumps(trace, allow_nan=False) + '\n')
            journal.flush()
            block += 1
    row = {**spec, 'suite': case['suite'], 'task_id': case['task_id'], 'state_index': case['state_index'],
           'state_key': case['state_key'], 'patch': 'clean' if spec['component'] == 'clean' else case['patch'],
           'success': int(success), 'failure': int(not success), 'executed_steps': step, 'decision_blocks': block,
           'rollout_seed': seed, 'elapsed_s': time.monotonic() - started, 'intervention_blocks': block if active else 0,
           'effective_intervention_blocks': effective, 'initial_after_wait_state': first_state,
           'initial_observation_hashes': first_observation_hashes, 'trace': str(trace_path.relative_to(folder))}
    return row, [trace_path, *assets]


def run(shard):
    import numpy as np
    import torch
    from PIL import Image
    from functional77_probe import Probe
    receipt, sig = prepared(), signature()
    require(0 <= shard < len(cases()), 'Invalid case shard')
    case = cases()[shard]
    folder = ROOT / 'episodes' / case['id']
    with exclusive(folder / '.worker.lock'):
        grid = specs(case)
        done = {episode_key(s): completed_episode(episode_folder(case, s), sig) for s in grid}
        if all(done.values()):
            path = folder / 'STATUS.json'
            prior = load_json(path) if path.exists() else {}
            require(not prior or prior['signature'] == sig, 'Worker status provenance changed')
            if prior.get('status') != 'COMPLETE':
                atomic_json({**prior, 'status': 'COMPLETE', 'signature': sig, 'case_id': case['id'],
                             'episodes_complete': len(grid), 'episodes_expected': len(grid),
                             'resumed_without_model_load': True}, path)
            return
        progress = Progress(folder / 'STATUS.json', signature=sig, case_id=case['id'], suite=case['suite'],
                            task_id=case['task_id'], episodes_expected=len(grid),
                            episodes_complete=sum(r is not None for r in done.values()))
        env = None
        started = time.monotonic()
        try:
            gpu = parent_module('common').require_gpu('evaluation')
            progress.update('allocated_gpu_verified', gpu_name=gpu, torch_version=torch.__version__)
            model = parent_module('model_runtime')
            runtime = parent_module('common').runtime_config(case['suite'])
            model.prepare_graphics_backend()
            progress.update('load_model')
            wrapper = model.load_wrapper(runtime, case['suite'])
            probe = Probe(wrapper)
            env, task, initial = model.make_env(runtime, case['suite'], case['task_id'], on_phase=progress.update)
            renderer, vendor = model.renderer_metadata(env)
            with Image.open(receipt['patch_paths'][case['patch']]) as image:
                require(image.mode == 'RGB' and image.size == (50, 50), 'Frozen PNG geometry changed')
                rgb = np.asarray(image).copy()
            for spec in grid:
                if done[episode_key(spec)] is not None:
                    continue
                target = episode_folder(case, spec)
                row, assets = episode(env, wrapper, probe, task, initial[case['state_index']], rgb, runtime,
                                     case, spec, target, progress)
                require(all(p.grad is None and not p.requires_grad for module in
                    [wrapper.vla, wrapper.action_head, wrapper.proprio_projector]
                    if module is not None for p in module.parameters()), 'Model gradients/weights unexpectedly enabled')
                require(source_hashes() == receipt['sources'], 'Scientific sources changed during execution')
                row.update(signature=sig, gpu_name=gpu, renderer=renderer, renderer_vendor=vendor,
                           model_parameters_updated=False,
                           source_png_sha256='none' if spec['component'] == 'clean' else sha256(receipt['patch_paths'][case['patch']]))
                complete = {'status': 'COMPLETE', 'signature': sig, 'row': row,
                            'assets': {str(p.relative_to(target)): sha256(p) for p in assets}}
                atomic_json(complete, target / 'COMPLETE.json')
                done[episode_key(spec)] = complete
                progress.update('episode_committed', episodes_complete=sum(r is not None for r in done.values()))
            model.close_environment(env)
            env = None
            progress.finish('COMPLETE', episodes_complete=len(grid), elapsed_s=time.monotonic() - started)
        except BaseException as error:
            progress.finish('FAILED', error=repr(error), traceback=traceback.format_exc(),
                            renderer_details=getattr(error, 'details', {}),
                            episodes_complete=sum(r is not None for r in done.values()))
            raise
        finally:
            if env is not None:
                try:
                    parent_module('model_runtime').close_environment(env)
                except Exception as error:
                    print('CLEANUP_ERROR ' + repr(error), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    def interrupted(number, _frame):
        raise InterruptedError('Signal ' + str(number) + '; preserve completed episodes and unfinished attempts')
    for number in [signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(number, interrupted)
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    try:
        run(args.shard)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()

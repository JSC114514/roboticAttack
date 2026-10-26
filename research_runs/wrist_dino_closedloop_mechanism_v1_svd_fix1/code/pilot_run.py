"""GPU-only, episode-resumable closed-loop pilot. Never submits jobs."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import random
import signal
import threading
import time
import traceback
from pathlib import Path

from pilot_common import (ROOT, atomic_json, atomic_torch, cfg, digest, exclusive, key, load_json,
                          parent_module, parent_root, prepared, require, sha256, shards, signature, specs)


class Progress:
    def __init__(self, path, **values):
        self.path, self.lock = Path(path), threading.Lock()
        self.values = {**values, 'status': 'RUNNING', 'started_unix': time.time(), 'pid': os.getpid(),
                       'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'phase': 'bootstrap'}
        self.stop = threading.Event()
        self.update('bootstrap')
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _write(self):
        self.values['heartbeat_unix'] = time.time()
        atomic_json(dict(self.values), self.path)

    def _heartbeat(self):
        while not self.stop.wait(15):
            with self.lock:
                self._write()

    def update(self, phase, **values):
        with self.lock:
            self.values.update(phase=phase, phase_started_unix=time.time(), **values)
            self._write()
        print('DINO_PILOT_PHASE ' + json.dumps({'phase': phase, **values}, ensure_ascii=False), flush=True)

    def finish(self, status, **values):
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            self.values.update(status=status, finished_unix=time.time(), **values)
            self._write()


def trace_array(values):
    import numpy as np
    return np.stack([np.asarray(item, dtype=np.float32) for item in values])


def verify_reproduction(result, traces, old_row, old_traces):
    import numpy as np
    for field in ['success', 'executed_steps', 'decision_blocks', 'rollout_seed']:
        require(int(result[field]) == int(old_row[field]), 'Old control reproduction differs: ' + field)
    require(len(traces) == len(old_traces), 'Old control decision count differs')
    maximum = 0.
    for new, old in zip(traces, old_traces):
        for field in ['decision_block', 'executed_slots', 'step_end']:
            require(int(new[field]) == int(old[field]), 'Old control trace differs: ' + field)
        a = np.asarray(new['predicted_environment_action_chunk'], dtype=np.float32)
        b = trace_array(old['predicted_environment_action_chunk'])
        error = float(np.max(np.abs(a - b)))
        maximum = max(maximum, error)
    require(maximum <= cfg()['baseline_trace_atol'], 'Old control action trace drift: ' + str(maximum))
    return {'status': 'REPRODUCTION_PASS', 'trace_max_error': maximum,
            'trace_tolerance': cfg()['baseline_trace_atol'], 'outcome_steps_and_seed_match': True}


def completed_episode(folder, sig, selection_hash):
    path = Path(folder) / 'COMPLETE.json'
    if not path.exists():
        return None
    receipt = load_json(path)
    require(receipt['signature'] == sig and receipt['selection_sha256'] == selection_hash, 'Episode resume provenance changed')
    require(all(sha256(Path(folder) / name) == value for name, value in receipt['assets'].items()),
            'Completed episode assets changed')
    return receipt


def state_snapshot(env, raw):
    import numpy as np
    result = {'sim_state': np.asarray(env.get_sim_state()).tolist()}
    for name in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
        result[name] = np.asarray(raw[name]).tolist()
    return result


def episode(env, wrapper, probe, task, initial, seed, rgb, runtime, spec, selection, folder, progress):
    import numpy as np
    import torch
    from pilot_math import action_stats, feature_stats, intervention_window, norm
    from pilot_probe import seeded
    model = parent_module('model_runtime')
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action

    attempt = Path(folder) / ('attempt_' + str(time.time_ns()) + '_' + str(os.getpid()))
    attempt.mkdir(parents=True, exist_ok=False)
    trace_path = attempt / 'trace.jsonl'
    capture_path = attempt / 'CAPTURES.pt'
    started = time.monotonic()
    raw = model.reset_wait(env, initial, seed, 10)
    limit = runtime['max_steps_by_suite'][wrapper.gen_cfg.task_suite_name]
    begin, end = intervention_window(spec['kind'], cfg())
    step, block, success, active_count, effective_count = 0, 0, False, 0, 0
    traces, captures, assets, frozen_basis = [], [], [], None
    with trace_path.open('w') as journal:
        while step < limit and not success:
            progress.update('episode_decision', episode=key(spec), decision_block=block, simulator_step=step)
            obs = model.observation(raw, runtime)
            before = state_snapshot(env, raw)
            active = begin <= block < end
            diagnostic = {'intervention_active': active, 'hook_identity_checked': False}
            original_pixels = np.asarray(obs['wrist_image'])
            patched_obs = obs if rgb is None else probe.paste(obs, rgb)
            native_observation = obs if spec['kind'] == 'whole_transplant' else patched_obs
            basis_vectors = None
            if active:
                require(rgb is not None, 'Feature intervention needs a fixed actual PNG')
                pair = probe.pair(obs, task.language, rgb)
                validation = probe.validate(pair) if block == begin else None
                if validation:
                    diagnostic.update(hook_identity_checked=True, **validation)
                kind = spec['kind']
                if kind == 'whole_restore':
                    replacement = pair['clean_dino']
                    action = probe.intervention(pair, replacement)
                    diagnostic.update(feature_stats(pair['clean_dino'], pair['patched_dino'], replacement))
                elif kind == 'whole_transplant':
                    replacement = pair['patched_dino']
                    action = probe.intervention(pair, replacement, 'clean')
                    diagnostic.update(feature_stats(pair['clean_dino'], pair['patched_dino'], replacement),
                                      recipient_visual_input='clean', dino_donor='same_state_actual_patch')
                    diagnostic['actual_change_l2'] = norm(replacement.float() - pair['clean_dino'].float())
                    diagnostic['change_fraction_of_delta'] = 1. if diagnostic['actual_change_l2'] > 0 else 0.
                    diagnostic['change_cosine_with_clean_direction'] = None
                    diagnostic['dino_change_reference'] = 'recipient_clean'
                else:
                    family = 'spatial' if kind.startswith('spatial_') else 'functional'
                    group = selection[family]['group']
                    if family == 'functional' and frozen_basis is None:
                        progress.update('window_start_vjps', episode=key(spec), decision_block=block, selected_group=group)
                        candidates, _, gradient_summary, basis_vectors = probe.functional(pair, finite=False,
                                                                                          on_phase=progress.update)
                        frozen_basis = candidates[group]
                        diagnostic.update(gradient_summary, basis_derived_at_block=block,
                                          basis_finite_difference_scope='calibration_states_only')
                    strength = .5 if kind == 'functional_half_early' else 1.
                    seed_value = seeded(wrapper.gen_cfg.task_suite_name, spec['patch'], seed, block, family, group)
                    replacement, details = probe.replacement(pair, family, group, strength, seed_value,
                        rows=frozen_basis, random_control='_random_' in kind)
                    require(details.get('norm_match_ok', True), 'BF16 matched random control failed: ' + str(details))
                    diagnostic.update(details, basis_derived_at_block=begin if family == 'functional' else None)
                    action = probe.intervention(pair, replacement)
                    if family == 'functional' and basis_vectors is not None:
                        change = replacement.float() - pair['patched_dino'].float()
                        for name, gradient in basis_vectors.items():
                            diagnostic['linear_prediction_' + name] = float((gradient.to(change.device) * change).sum())
                actions = probe.environment_actions(action)
                clean_commands = np.stack([process_libero_action(a) for a in probe.environment_actions(pair['clean_action'])])
                patch_commands = np.stack([process_libero_action(a) for a in probe.environment_actions(pair['patched_action'])])
                intervention_commands = np.stack([process_libero_action(a) for a in actions])
                diagnostic.update(action_stats(pair['clean_action'], pair['patched_action'], action),
                                  clean_normalized_action=pair['clean_action'].cpu().tolist(),
                                  patched_normalized_action=pair['patched_action'].cpu().tolist(),
                                  intervened_normalized_action=action.cpu().tolist(),
                                  clean_processed_action_chunk=clean_commands.tolist(),
                                  patched_processed_action_chunk=patch_commands.tolist(),
                                  intervened_processed_action_chunk=intervention_commands.tolist(),
                                  processed_action_gap_from_clean_l2=float(np.linalg.norm(intervention_commands - clean_commands)),
                                  processed_action_change_from_patch_l2=float(np.linalg.norm(intervention_commands - patch_commands)),
                                  gripper_command_flips_from_patch=int(np.count_nonzero(intervention_commands[:, -1] != patch_commands[:, -1])),
                                  footprint_tokens=int(pair['support'].sum()),
                                  siglip_clean_patch_gap_l2=norm(pair['patched_siglip'].float() - pair['clean_siglip'].float()))
                active_count += 1
                effective_count += int(diagnostic.get('actual_change_l2', 0.) > 0)
                if block == begin:
                    bundle_path = attempt / ('window_' + str(block).zfill(3) + '.pt')
                    atomic_torch({'signature': signature(), 'spec': spec, 'rollout_seed': seed,
                                  'sim_state': before, 'observation': obs, 'task_name': task.language,
                                  'clean_dino': pair['clean_dino'].cpu(), 'patched_dino': pair['patched_dino'].cpu(),
                                  'replacement': replacement.cpu(), 'support': pair['support'].cpu(),
                                  'basis': None if frozen_basis is None else frozen_basis.cpu(),
                                  'endpoint_vjps': basis_vectors, 'diagnostic': diagnostic,
                                  'resume_contract': 'reset frozen initial state and replay verified action prefix; snapshot is diagnostic'}, bundle_path)
                    assets.append(bundle_path)
                del pair, action, replacement
            else:
                with torch.no_grad():
                    actions = np.asarray(wrapper.predict_action(native_observation, task.language), dtype=np.float32).reshape(8, 7)
            require(actions.shape == (8, 7) and np.isfinite(actions).all(), 'Invalid action chunk')
            if spec['kind'] == 'native' and spec['patch'] != 'clean' and spec['state_index'] == cfg()['screen_state'] and block in cfg()['screen_blocks']:
                captures.append({'observation': {k: np.asarray(v).copy() for k, v in obs.items()},
                                 'task_name': task.language, 'decision_block': block, 'formal_action_chunk': actions.copy()})
            rgb_delta = np.asarray(patched_obs['wrist_image'], dtype=np.float32) - original_pixels.astype(np.float32)
            slots, processed = 0, []
            for slot in range(min(8, limit - step)):
                applied = process_libero_action(actions[slot]).tolist()
                raw, _, done, _ = env.step(applied)
                processed.append(applied)
                step += 1
                slots += 1
                if done:
                    success = True
                    break
            trace = {'decision_block': block, 'executed_slots': slots, 'step_end': step,
                     'predicted_environment_action_chunk': actions.tolist(), 'applied_actions': processed,
                     'before': before, 'after': state_snapshot(env, raw),
                     'rgb_changed_pixels': int(np.any(rgb_delta != 0, axis=-1).sum()),
                     'rgb_delta_l2_255': float(np.linalg.norm(rgb_delta)),
                     'rgb_delta_linf_255': float(np.abs(rgb_delta).max()),
                     'policy_rgb_changed_pixels': 0 if spec['kind'] == 'whole_transplant' else int(np.any(rgb_delta != 0, axis=-1).sum()),
                     'policy_recipient_visual_input': 'clean' if spec['kind'] == 'whole_transplant' or rgb is None else 'actual_png',
                     'clean_wrist_sha256': hashlib.sha256(original_pixels.tobytes()).hexdigest(),
                     'patched_wrist_sha256': hashlib.sha256(np.asarray(patched_obs['wrist_image']).tobytes()).hexdigest(),
                     'policy_wrist_sha256': hashlib.sha256(np.asarray(native_observation['wrist_image']).tobytes()).hexdigest(),
                     'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                     'elapsed_s': time.monotonic() - started, **diagnostic}
            journal.write(json.dumps(trace, allow_nan=False) + '\n')
            journal.flush()
            traces.append(trace)
            block += 1
    if captures:
        atomic_torch(captures, capture_path)
        assets.append(capture_path)
    result = {'success': int(success), 'failure': int(not success), 'executed_steps': step,
              'decision_blocks': block, 'rollout_seed': seed, 'elapsed_s': time.monotonic() - started,
              'intervention_blocks': active_count, 'effective_intervention_blocks': effective_count,
              'window_not_reached': bool(spec['kind'] != 'native' and active_count == 0),
              'trace': str(trace_path.relative_to(folder)),
              'captures': str(capture_path.relative_to(folder)) if captures else None, **spec}
    return result, traces, [trace_path, *assets]


def run(stage, shard):
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from pilot_probe import Probe

    receipt, c, sig = prepared(), cfg(), signature()
    require(0 <= shard < len(shards(stage)), 'Shard out of range')
    suite, task_id = shards(stage)[shard]
    folder = ROOT / 'episodes' / stage / suite / ('task_' + str(task_id).zfill(2))
    with exclusive(folder / '.worker.lock'):
        selection, selection_hash = None, 'none'
        if stage == 'main':
            selection_path = ROOT / 'reports/SELECTION.json'
            selected = load_json(selection_path)
            require(selected['signature'] == sig and selected['status'] == 'FROZEN', 'No validated frozen selection')
            selection, selection_hash = selected['groups'], sha256(selection_path)
            require(all(sha256(ROOT / path) == value for path, value in selected['calibration_assets'].items()),
                    'Calibration assets changed after selection')
        all_specs = specs(stage, task_id)
        done = {key(spec): completed_episode(folder / key(spec), sig, selection_hash) for spec in all_specs}
        screen_status = folder / 'SCREEN_STATUS.json'
        if all(done.values()) and (stage == 'main' or screen_status.exists()):
            if stage == 'calibration':
                verify_screen(folder)
            atomic_json({'status': 'COMPLETE', 'signature': sig, 'episodes_complete': len(done),
                         'selection_sha256': selection_hash, 'resumed_without_model_load': True}, folder / 'STATUS.json')
            return
        progress = Progress(folder / 'STATUS.json', signature=sig, stage=stage, suite=suite,
                            task_id=task_id, episodes_expected=len(all_specs), episodes_complete=sum(bool(v) for v in done.values()))
        env = None
        started = time.monotonic()
        try:
            gpu = parent_module('common').require_gpu('evaluation')
            progress.update('allocated_gpu_verified', gpu_name=gpu, torch_version=torch.__version__,
                            torch_cuda_version=torch.version.cuda, device_capability=list(torch.cuda.get_device_capability(0)))
            model = parent_module('model_runtime')
            runtime = parent_module('common').runtime_config(suite)
            model.prepare_graphics_backend()
            progress.update('load_model')
            wrapper = model.load_wrapper(runtime, suite)
            probe = Probe(wrapper)
            env, task, initial = model.make_env(runtime, suite, task_id, on_phase=progress.update)
            renderer, vendor = model.renderer_metadata(env)
            patches = {}
            for patch, path in receipt['patch_paths'].items():
                with Image.open(path) as image:
                    require(image.mode == 'RGB' and image.size == (50, 50), 'Invalid frozen PNG')
                    patches[patch] = np.asarray(image).copy()
            patches['clean'] = None
            old_rows = pd.read_parquet(parent_root() / 'results/BATCH500_ROLLOUTS.parquet')
            old_traces = pd.read_parquet(parent_root() / 'results/BATCH500_TRACES.parquet')
            old_rows = old_rows[(old_rows.suite == suite) & (old_rows.task_id == task_id)]
            old_traces = old_traces[(old_traces.suite == suite) & (old_traces.task_id == task_id)]
            if stage == 'calibration':
                # Reproduction must pass before the first new closed-loop intervention.
                all_specs.sort(key=lambda s: (s['kind'] != 'native', s['state_index'], s['patch'], s['kind']))
            else:
                random.Random(c['random_seed'] + c['suites'].index(suite) * 100 + task_id).shuffle(all_specs)
            for spec in all_specs:
                episode_folder = folder / key(spec)
                if done[key(spec)]:
                    continue
                reference = old_rows[(old_rows.state_index == spec['state_index']) & (old_rows.condition == spec['patch'])]
                require(len(reference) == 1, 'Missing unique baseline episode')
                baseline = reference.iloc[0].to_dict()
                seed = int(baseline['rollout_seed'])
                frozen_groups = None if selection is None else selection[suite + '|' + spec['patch']]
                row, traces, assets = episode(env, wrapper, probe, task, initial[spec['state_index']], seed,
                                             patches[spec['patch']], runtime, spec, frozen_groups,
                                             episode_folder, progress)
                if spec['kind'] == 'native':
                    previous = old_traces[(old_traces.state_index == spec['state_index']) &
                                          (old_traces.condition == spec['patch'])].sort_values('decision_block').to_dict('records')
                    row['reproduction'] = verify_reproduction(row, traces, baseline, previous)
                row.update(suite=suite, task_id=task_id, signature=sig, selection_sha256=selection_hash,
                           renderer=renderer, renderer_vendor=vendor, gpu_name=gpu,
                           model_parameters_updated=False,
                           source_png_sha256='none' if spec['patch'] == 'clean' else sha256(receipt['patch_paths'][spec['patch']]))
                complete = {'status': 'COMPLETE', 'signature': sig, 'selection_sha256': selection_hash,
                            'row': row, 'assets': {str(path.relative_to(episode_folder)): sha256(path) for path in assets}}
                atomic_json(complete, episode_folder / 'COMPLETE.json')
                done[key(spec)] = complete
                progress.update('episode_committed', episodes_complete=sum(bool(v) for v in done.values()))
            if stage == 'calibration':
                screen(probe, folder, done, patches, suite, task_id, progress)
            require(all(p.grad is None for module in [wrapper.vla, wrapper.action_head, wrapper.proprio_projector]
                        if module is not None for p in module.parameters()), 'Model parameter gradient unexpectedly accumulated')
            progress.update('close_environment')
            model.close_environment(env)
            env = None
            progress.finish('COMPLETE', episodes_complete=len(all_specs), elapsed_s=time.monotonic() - started,
                            selection_sha256=selection_hash)
        except BaseException as error:
            progress.finish('FAILED', error=repr(error), traceback=traceback.format_exc(),
                            episodes_complete=sum(bool(v) for v in done.values()),
                            renderer_details=getattr(error, 'details', {}))
            raise
        finally:
            if env is not None:
                try:
                    parent_module('model_runtime').close_environment(env)
                except Exception as cleanup_error:
                    print('CLEANUP_ERROR ' + repr(cleanup_error), flush=True)


def verify_screen(folder):
    receipt = load_json(Path(folder) / 'SCREEN_STATUS.json')
    require(receipt['status'] == 'COMPLETE' and receipt['signature'] == signature(), 'Screen resume provenance mismatch')
    require(all(sha256(Path(folder) / path) == value for path, value in receipt['assets'].items()), 'Screen artifacts changed')
    return receipt


def screen(probe, folder, completed, patches, suite, task_id, progress):
    import torch
    if (folder / 'SCREEN_STATUS.json').exists():
        return verify_screen(folder)
    assets, contexts = {}, []
    for episode_key, value in sorted(completed.items()):
        row = value['row']
        if row['kind'] != 'native' or row['patch'] == 'clean' or row['state_index'] != cfg()['screen_state']:
            continue
        require(row['captures'] is not None, 'No predeclared calibration captures')
        captures = torch.load(folder / episode_key / row['captures'], map_location='cpu', weights_only=False)
        observed = {int(capture['decision_block']) for capture in captures}
        require(0 in observed and observed <= set(cfg()['screen_blocks']), 'Calibration capture grid changed')
        contexts.append({'patch': row['patch'], 'observed_blocks': sorted(observed),
                         'unreached_blocks': sorted(set(cfg()['screen_blocks']) - observed)})
        for capture in captures:
            context = {'suite': suite, 'task_id': task_id, 'state_index': row['state_index'], 'patch': row['patch'],
                       'decision_block': int(capture['decision_block'])}
            stem = row['patch'].replace('@', '_') + '_b' + str(capture['decision_block']).zfill(3)
            result_path, vector_path = folder / 'screen' / (stem + '.json'), folder / 'screen' / (stem + '.pt')
            if result_path.exists():
                result = load_json(result_path)
                require(result['signature'] == signature() and result['context'] == context and
                        sha256(vector_path) == result['vectors_sha256'], 'Partial screen assets changed')
            else:
                progress.update('same_state_screen', screen_context=context)
                screen_started = time.monotonic()
                records, checks, summary, validation, bundle = probe.screen(capture, patches[row['patch']], context,
                                                                          on_phase=progress.update)
                atomic_torch(bundle, vector_path)
                atomic_json({'signature': signature(), 'context': context, 'records': records,
                             'finite_differences': checks, 'gradient_summary': summary, 'validation': validation,
                             'vectors_sha256': sha256(vector_path), 'elapsed_s': time.monotonic() - screen_started}, result_path)
            for path in [result_path, vector_path]:
                assets[str(path.relative_to(folder))] = sha256(path)
    receipt = {'status': 'COMPLETE', 'signature': signature(), 'contexts': contexts, 'assets': assets}
    atomic_json(receipt, folder / 'SCREEN_STATUS.json')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['calibration', 'main'], required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    def interrupted(number, _frame):
        raise InterruptedError('Slurm/process signal ' + str(number) + '; completed episodes remain reusable')
    for number in [signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(number, interrupted)
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    try:
        run(args.stage, args.shard)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()

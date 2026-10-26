"""Matched finite pulses followed by native clean continuation, within the original horizon."""
import time
import numpy as np
from dp_common import (ROOT, PROJECT, config, read, atomic, require, finish, complete, verify,
                       native, runtime, geometry, resources, episode_receipt, episode_files,
                       reference_files, eligible)
from dp_math import pulse_active, pulse_outcome
from fr_vision import Vision
from sc_rollout import create, warm_history, snapshot, start_arrays
from sc_math import action_difference


def array(value):
    return np.asarray(value.tolist() if isinstance(value, np.ndarray) else value, dtype=np.float32)


def save_state(folder, stem, env, observation, utils, **meta):
    arrays = start_arrays(env, observation)
    np.savez_compressed(folder/(stem+'.npz'), **arrays)
    atomic(dict(fingerprint=utils.fingerprint(env), **meta), folder/(stem+'.json'))
    return arrays


def episode(wrapper, vision, case, condition, patch, monitor, gate_needed):
    import pandas as pd
    from jr_math import paste
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    utils = native(); r = runtime(); c = config(); geo = geometry()
    name = condition['name']; mode = condition['mode']; duration = condition['duration']
    onset = case['onset']; removal = onset + duration; limit = c['max_steps'][case['suite']]
    receipt = episode_receipt(case, name)
    attempt = receipt.parent/('attempt_'+str(time.time_ns())); attempt.mkdir(parents=True)
    env = None; started = time.monotonic(); utils.switch_suite(wrapper, case['suite'])
    try:
        ref = reference_files(case)
        source_steps = pd.read_parquet(ref['STEPS.parquet'])
        source_trace = pd.read_parquet(ref['TRACE.parquet'])
        expected_actions = {int(row.step_start): array(row.predicted_actions) for row in source_trace.itertuples()}
        require(onset in expected_actions, 'Contact onset is not a cached native decision')
        env, task, initial = create(case)
        with warm_history(env, initial, case['rollout_seed']) as (raw, history, samples):
            target = geo['anchor'](env, case['suite'], case['task_id'])
            obs = history[-1]['observation']
            reset_arrays = start_arrays(env, obs)
            reset_meta = dict(fingerprint=utils.fingerprint(env), rollout_seed=case['rollout_seed'])
            # Source asset paths are pinned in the CPU manifest, not copied or rewritten.
            utils.assert_start(reset_arrays, reset_meta, ref['START.json'].parent)
            np.savez_compressed(attempt/'START.npz', **reset_arrays); atomic(reset_meta, attempt/'START.json')
            before = utils.fingerprint(env)
            first, _ = vision.action(obs, task.language)
            reset_error = float(np.max(abs(first-np.load(ref['FIRST_ACTION.npy'], allow_pickle=False))))
            require(reset_error <= c['action_tolerance'], 'Historical reset first action differs')
            require(utils.fingerprint(env) == before, 'Reset forward advanced physics or camera sampling')
            np.save(attempt/'FIRST_ACTION.npy', first)
            fps = 1/float(env.env.control_timestep)
            step = 0; decision = 0; success = False; traces = []; steps = []
            applied_steps = 0; first_contact = None; first_grasp = None
            max_both_error = 0.; max_clean_replay_error = 0.; reached_removal = False
            with utils.CameraWriters(attempt, fps) as writer:
                writer.append(paste(obs, patch if onset == 0 and mode != 'clean' else None, c['patch_xy']))
                # Reproduce the clean prefix with saved controller commands, not a new policy rollout.
                prefix = source_steps[source_steps.step <= onset]
                require(prefix.step.tolist() == list(range(1, onset+1)), 'Clean prefix contains a gap')
                for old in prefix.itertuples():
                    command = np.asarray(old.controller_command, dtype=float)
                    raw, _, done, _ = env.step(command.tolist()); step += 1
                    require(not done, 'Task ended before its clean pre-contact decision')
                    history.append(snapshot(raw, samples, env, step))
                    truth = geo['truth'](env, target)
                    if truth['contact'] and first_contact is None: first_contact = step
                    if truth['grasp'] and first_grasp is None: first_grasp = step
                    record = old._asdict(); record.pop('Index', None)
                    record.update(phase='saved_clean_prefix', pulse_active=False, png_applied=False,
                                  restoration='none', actual_eef_xyz=truth['eef_xyz'])
                    steps.append(record)
                    writer.append(paste(history[-1]['observation'], patch if step == onset and mode != 'clean' else None, c['patch_xy']))
                    monitor.update(current_start=case['start'], condition=name, phase='saved_clean_prefix', executed_steps=step)
                obs = history[-1]['observation']
                before = utils.fingerprint(env)
                onset_action, _ = vision.action(obs, task.language)
                onset_error = float(np.max(abs(onset_action-expected_actions[onset])))
                require(onset_error <= c['action_tolerance'], 'Historical onset clean action differs')
                require(utils.fingerprint(env) == before, 'Onset forward advanced native sampling')
                np.save(attempt/'ONSET_ACTION.npy', onset_action)
                onset_arrays = save_state(attempt, 'ONSET', env, obs, utils, rollout_seed=case['rollout_seed'])
                paired = ROOT/'paired_onsets'/case['start']
                onset_meta = read(attempt/'ONSET.json')
                if (paired/'START.json').exists():
                    utils.assert_start(onset_arrays, onset_meta, paired)
                else:
                    paired.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(paired/'START.npz', **onset_arrays); atomic(onset_meta, paired/'START.json')
                deployed = paste(obs, patch, c['patch_xy'])
                gate = dict(status='PASS', historical_reset_action_error=reset_error,
                            historical_onset_action_error=onset_error, matched_onset_arrays=True)
                if gate_needed:
                    gate.update(utils.interface_gate(wrapper, obs, task.language, case['suite']))
                    gate.update(vision.gate(obs, deployed, task.language, expected_actions[onset]))
                require(utils.fingerprint(env) == before, 'Interface gates altered physics/camera/clocks')
                atomic(gate, attempt/'INTERFACE_CHECK.json')
                while step < limit and not success:
                    obs = history[-1]['observation']; active = pulse_active(step, onset, duration)
                    applied = active and mode != 'clean'
                    before = utils.fingerprint(env)
                    clean, donor = vision.action(obs, task.language)
                    deployed = paste(obs, patch if applied else None, c['patch_xy'])
                    metrics = {k: 0. for k in ('dino_native_delta_rms','dino_used_delta_rms',
                                              'siglip_native_delta_rms','siglip_used_delta_rms')}
                    both_error = None
                    restore = mode if applied and mode in ('dino','siglip') else 'native'
                    if applied:
                        actions, features = vision.action(deployed, task.language, restore, donor)
                        metrics = vision.metrics(features, donor, restore); del features
                        # Both-branch identity is tested at every actually perturbed decision.
                        both, _ = vision.action(deployed, task.language, 'both', donor)
                        both_error = float(np.max(abs(both-clean)))
                        max_both_error = max(max_both_error, both_error)
                        require(both_error <= c['action_tolerance'], 'Same-state both-branch identity failed')
                    else:
                        actions = clean.copy()
                    del donor
                    require(utils.fingerprint(env) == before, 'Feature capture/replacement changed physical state or sensor clocks')
                    if mode == 'clean':
                        require(step in expected_actions, 'Clean continuation exceeds historical trajectory')
                        error = float(np.max(abs(actions-expected_actions[step])))
                        max_clean_replay_error = max(max_clean_replay_error, error)
                        require(error <= c['action_tolerance'], 'Native clean continuation differs from its historical action')
                    begin = step; used = 0; now = history[-1]
                    count = min(c['chunk'], limit-step, removal-step if active else c['chunk'])
                    for slot in range(count):
                        prior = geo['truth'](env, target)
                        command = np.asarray(process_libero_action(actions[slot].copy()), float)
                        require(command.shape == (7,) and np.isfinite(command).all(), 'Invalid controller command')
                        raw, _, done, _ = env.step(command.tolist()); step += 1; used += 1
                        applied_steps += int(applied)
                        history.append(snapshot(raw, samples, env, step)); truth = geo['truth'](env, target)
                        if truth['contact'] and first_contact is None: first_contact = step
                        if truth['grasp'] and first_grasp is None: first_grasp = step
                        steps.append(dict(step=step, time_s=step/fps, decision_block=decision, executed_slot=slot,
                            phase='pulse' if active else 'clean_continuation', pulse_active=active,
                            png_applied=applied, restoration=restore if applied else 'none',
                            raw_action=actions[slot].tolist(), same_state_clean_action=clean[slot].tolist(),
                            controller_command=command.tolist(), eef_xyz=truth['eef_xyz'], target_xyz=truth['target_xyz'],
                            target_center_distance_m=truth['distance_m'],
                            actual_eef_displacement_m=(np.asarray(truth['eef_xyz'])-prior['eef_xyz']).tolist(),
                            contact=truth['contact'], grasp=truth['grasp'], success=bool(done)))
                        next_obs = history[-1]['observation']
                        writer.append(paste(next_obs, patch if mode != 'clean' and pulse_active(step,onset,duration) else None, c['patch_xy']))
                        if step == removal:
                            save_state(attempt, 'REMOVAL', env, next_obs, utils, global_step=step,
                                       remaining_budget=limit-step, task_done=bool(done))
                            reached_removal = True
                        if done:
                            success = True; break
                    stats = wrapper.vla.norm_stats[wrapper.gen_cfg.unnorm_key]['action']
                    traces.append(dict(decision_block=decision, step_start=begin, step_end=step, executed_slots=used,
                        pulse_active=active, png_applied=applied, restoration=restore if applied else 'none',
                        donor_control_step=begin, donor_sim_time_s=before['time'],
                        donor_wrist_frame_id=now['sensors']['wrist']['frame_id'],
                        donor_wrist_timestamp_s=now['sensors']['wrist']['sensor_timestamp_s'],
                        predicted_actions=actions.tolist(), same_state_clean_actions=clean.tolist(),
                        both_restore_max_abs_error=both_error, **metrics, **action_difference(actions, clean, stats, used)))
                    decision += 1
                    monitor.update(phase='pulse' if active else 'clean_continuation', current_start=case['start'],
                                   condition=name, executed_steps=step, remaining_control_steps=limit-step)
                if mode == 'clean':
                    require(success == bool(case['clean_success']) and step == case['clean_executed_steps'],
                            'Unmodified clean continuation changed historical outcome or duration')
                save_state(attempt, 'TERMINAL', env, history[-1]['observation'], utils, global_step=step, success=success)
            pd.DataFrame(traces).to_parquet(attempt/'TRACE.parquet', index=False)
            pd.DataFrame(steps).to_parquet(attempt/'STEPS.parquet', index=False)
            identity = {k: v for k, v in case.items() if k != 'reference_files'}
            outcome = pulse_outcome(success, step, onset, duration, limit)
            atomic(dict(**identity, **outcome, condition=name, mode=mode, pulse_duration=duration,
                patch=c['patch']['name'], png_sha256=c['patch']['sha256'], checkpoint=c['model_dir'], revision=c['revision'],
                executed_steps=step, max_steps=limit, fps=fps, num_open_loop_steps=8, actual_png_steps=applied_steps,
                first_target_contact_step=first_contact, first_target_grasp_step=first_grasp,
                removal_snapshot_saved=reached_removal, model_frozen=True, decisions=decision,
                native_PNG_only=mode in ('clean','png'), oracle_feature_restoration=mode in ('dino','siglip'),
                donor_source='same current branch unpatched observation', both_restore_max_abs_error=max_both_error,
                clean_replay_max_abs_error=max_clean_replay_error, elapsed_s=time.monotonic()-started, **resources()), attempt/'EPISODE.json')
            names = ('START.npz','START.json','FIRST_ACTION.npy','ONSET.npz','ONSET.json','ONSET_ACTION.npy',
                     'INTERFACE_CHECK.json','TERMINAL.npz','TERMINAL.json','EPISODE.json','TRACE.parquet','STEPS.parquet',
                     'front.mp4','wrist.mp4')
            files = [attempt/n for n in names]
            if reached_removal: files += [attempt/'REMOVAL.npz', attempt/'REMOVAL.json']
            finish(receipt, files+[paired/'START.npz', paired/'START.json'])
    except BaseException as error:
        atomic(dict(status='FAILED', error=repr(error), case=case['start'], condition=name,
                    elapsed_s=time.monotonic()-started, **resources()), attempt/'FAILURE.json')
        raise
    finally:
        if env is not None: r.close_environment(env)


def run(index, workers, monitor):
    from jr_math import weighted_shards
    from PIL import Image
    require(1 <= workers <= 2 and 0 <= index < workers, 'Invalid worker shard')
    m = verify(); c = config()
    cases = weighted_shards([case for case in m['cases'] if eligible(case)], workers, c['max_steps'])[index]
    receipts = [episode_receipt(case, row['name']) for case in cases for row in m['conditions']]
    already = sum(complete(p) for p in receipts)
    monitor.update(completed=already, resumed=already, total=len(receipts))
    if already < len(receipts):
        wrapper = native().load_wrapper(monitor); vision = Vision(wrapper)
        patch = np.array(Image.open(PROJECT/c['patch']['path']).convert('RGB'), copy=True)
        checked = set(); timed = 0; elapsed = 0.
        for case in cases:
            begin = time.monotonic(); n = 0
            order = list(m['conditions']); np.random.default_rng(case['rollout_seed']).shuffle(order)
            for row in order:
                if complete(episode_receipt(case, row['name'])): continue
                episode(wrapper, vision, case, row, patch, monitor, case['suite'] not in checked)
                checked.add(case['suite']); n += 1
                monitor.update(completed=monitor.data['completed']+1)
            if n:
                timed += 1; elapsed += time.monotonic()-begin
                monitor.update(starts_timed=timed, mean_complete_start_s=elapsed/timed)
                if timed == 4:
                    atomic(dict(starts=timed, elapsed_s=elapsed, mean_start_s=elapsed/timed, **resources()),
                           ROOT/'reports/THROUGHPUT'/f'worker_{index}_first4.json')
    require(all(complete(p) for p in receipts), 'Incomplete pulse shard')
    finish(ROOT/f'reports/evaluate_{workers}_{index}_COMPLETE.json', receipts)
    monitor.update(phase='COMPLETE')

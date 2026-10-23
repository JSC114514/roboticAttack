"""Native joint-model PNG evaluation. Imported only inside a GPU allocation."""
import copy
import hashlib
from pathlib import Path
import random
import time
import numpy as np
import torch
from jr_common import (ROOT, config, runtime, runtime_config, require, read, atomic,
                       verify, complete, finish, episode_receipt, episode_files, resources)
from jr_math import paste, weighted_shards


def fingerprint(env):
    arrays = {'sim_state': np.asarray(env.get_sim_state())}
    for k in ('body_xpos', 'body_xquat', 'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat', 'cam_xpos', 'cam_xmat'):
        arrays[k] = np.asarray(getattr(env.sim.data, k))
    clocks = {name: {k: v.item() if isinstance(v, np.generic) else v for k, v in vars(obs).items()
                    if k != '_sensor' and isinstance(v, (bool, int, float, str, np.generic))}
              for name, obs in env.env._observables.items()}
    return {'arrays': {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()},
            'time': float(env.sim.data.time), 'observable_clocks': clocks}


def assert_start(arrays, current, reference):
    require(current == read(reference / 'START.json'), 'Paired physical/camera/clock fingerprint differs')
    with np.load(reference / 'START.npz', allow_pickle=False) as expected:
        require(set(arrays) == set(expected.files), 'Paired start fields differ')
        for k, v in arrays.items(): require(np.array_equal(v, expected[k]), 'Paired start differs: ' + k)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def frozen_modules(wrapper):
    return [getattr(wrapper, k) for k in ('vla', 'action_head', 'proprio_projector', 'noisy_action_projector')
            if getattr(wrapper, k, None) is not None]


def load_wrapper(monitor):
    r = runtime(); cfg = runtime_config(); r.prepare_graphics_backend()
    monitor.update(phase='loading_joint_model_once')
    from jr_loading import load_native_wrapper
    wrapper = load_native_wrapper(r, config()['suites'][0])
    devices = {}; offload = []
    for model in frozen_modules(wrapper):
        require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Model not frozen/eval')
        for p in model.parameters(): devices[str(p.device)] = devices.get(str(p.device), 0) + p.numel()
        for name, child in model.named_modules():
            if any(str(d) in ('cpu', 'disk', 'meta') for d in (getattr(child, 'hf_device_map', {}) or {}).values()): offload.append(name)
            if getattr(getattr(child, '_hf_hook', None), 'offload', False): offload.append(name)
    require(len(devices) == 1 and next(iter(devices)).startswith('cuda:') and not offload, 'Model offload/device mismatch')
    for suite in config()['suites']:
        key = cfg['unnorm_keys'][suite]
        require(key == suite + '_no_noops' and key in wrapper.vla.norm_stats, 'Joint normalization key missing: ' + key)
    # The local loader must have selected the joint checkpoint's 300000 components.
    p = Path(config()['model_dir'])
    require([q.name for q in p.glob('action_head--*.pt')] == ['action_head--300000_checkpoint.pt'], 'Ambiguous joint action head')
    require([q.name for q in p.glob('proprio_projector--*.pt')] == ['proprio_projector--300000_checkpoint.pt'], 'Ambiguous joint proprio projector')
    atomic(dict(status='PASS', devices=devices, offload=offload, one_model_load=True, **resources()),
           ROOT / 'reports/GATES' / (monitor.stage + '_residency.json'))
    return wrapper


def switch_suite(wrapper, suite):
    key = runtime_config()['unnorm_keys'][suite]
    wrapper.cfg.task_suite_name = suite; wrapper.gen_cfg.task_suite_name = suite
    wrapper.cfg.unnorm_key = key; wrapper.gen_cfg.unnorm_key = key
    require(key in wrapper.vla.norm_stats, 'Missing runtime normalization')


def predict(wrapper, obs, instruction):
    value = np.asarray(wrapper.predict_action(copy.deepcopy(obs), instruction), dtype=np.float32)
    require(value.shape == (8, 7) and np.isfinite(value).all(), 'Native action chunk invalid')
    return value


def interface_gate(wrapper, obs, instruction, suite):
    """Direct official helper vs wrapper; zero-copy deployment; passive dual-view checks."""
    counts = {'dino': 0, 'siglip': 0}; handles = []
    def hook(key):
        def count(*args): counts[key] += 1
        return count
    try:
        backbone = wrapper.vla.vision_backbone
        for key, model in [('dino', backbone.featurizer), ('siglip', backbone.fused_featurizer)]:
            handles.append(model.register_forward_hook(hook(key)))
        actual = predict(wrapper, obs, instruction)
    finally:
        for handle in handles: handle.remove()
    require(counts == {'dino': 2, 'siglip': 2}, 'Native two-view/two-branch call counts differ')
    ou = wrapper._official_openvla_utils
    with torch.inference_mode():
        # Independent direct call, using the original native observation keys.
        official = np.asarray(ou.get_vla_action(wrapper.gen_cfg, wrapper.vla, wrapper.processor,
            copy.deepcopy(obs), instruction, wrapper.action_head, wrapper.proprio_projector,
            wrapper.noisy_action_projector, use_film=False), dtype=np.float32).reshape(8, 7)
    identity = predict(wrapper, paste(obs, None), instruction)
    error = float(np.max(np.abs(actual - official))); zero = float(np.max(np.abs(actual - identity)))
    require(error <= config()['action_tolerance'] and zero <= config()['action_tolerance'], 'Official/identity action mismatch')
    require(all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in frozen_modules(wrapper)), 'Model changed freeze state')
    return dict(status='PASS', suite=suite, unnorm_key=wrapper.gen_cfg.unnorm_key,
                native_action_max_abs_error=error, zero_action_max_abs_error=zero, passive_calls=counts,
                tolerance=config()['action_tolerance'], head_step=300000, model_frozen=True)


class CameraWriters:
    def __init__(self, folder, fps): self.folder = folder; self.fps = fps; self.writers = {}
    def __enter__(self):
        import imageio.v2 as imageio
        try:
            for view in ('front', 'wrist'):
                self.writers[view] = imageio.get_writer(self.folder / (view + '.mp4'), fps=self.fps,
                    codec='libx264', quality=7, macro_block_size=16,
                    ffmpeg_params=['-threads', '1', '-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-movflags', '+faststart'])
        except BaseException:
            self.__exit__(None, None, None); raise
        return self
    def append(self, obs):
        for view, key in [('front', 'full_image'), ('wrist', 'wrist_image')]:
            frame = np.asarray(obs[key]); require(frame.shape == (224, 224, 3) and frame.dtype == np.uint8, 'Video RGB invalid')
            self.writers[view].append_data(frame)
    def __exit__(self, *args):
        for writer in self.writers.values(): writer.close()


def episode(wrapper, case, condition, patch, monitor, gate_needed):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    import pandas as pd
    r = runtime(); cfg = runtime_config(); receipt = episode_receipt(case, condition['name'])
    attempt = receipt.parent / ('attempt_' + str(time.time_ns())); attempt.mkdir(parents=True)
    env = None; started = time.monotonic(); switch_suite(wrapper, case['suite']); seed_all(case['rollout_seed'])
    try:
        env, task, states = r.make_env(cfg, case['suite'], case['task_id'])
        require(task.name == case['task_name'], 'Task identity differs')
        initial = torch.load(case['initial_file'], map_location='cpu', weights_only=False)[case['state_index']]
        require(np.array_equal(initial, states[case['state_index']]), 'Initial-state source differs')
        raw = r.reset_wait(env, initial, case['rollout_seed'], cfg['num_steps_wait'])
        obs = r.observation(raw, cfg)
        arrays = dict(sim_state=np.array(env.get_sim_state(), copy=True), proprio=obs['state'],
                      front=obs['full_image'], wrist=obs['wrist_image'])
        start = dict(fingerprint=fingerprint(env), rollout_seed=case['rollout_seed'])
        np.savez_compressed(attempt / 'START.npz', **arrays); atomic(start, attempt / 'START.json')
        paired = ROOT / 'paired_starts' / case['start']
        if condition['name'] == 'clean':
            if (paired / 'START.json').exists(): assert_start(arrays, start, paired)
            else:
                paired.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(paired / 'START.npz', **arrays); atomic(start, paired / 'START.json')
        else:
            require(complete(episode_receipt(case, 'clean')), 'Clean reference incomplete')
            assert_start(arrays, start, paired)
        check = interface_gate(wrapper, obs, task.language, case['suite']) if gate_needed else dict(status='PASS', suite=case['suite'])
        first_clean = predict(wrapper, obs, task.language)
        if condition['name'] != 'clean':
            reference = np.load(episode_files(episode_receipt(case, 'clean'))['FIRST_ACTION.npy'], allow_pickle=False)
            check['paired_clean_action_max_abs_error'] = float(np.max(np.abs(first_clean - reference)))
            require(check['paired_clean_action_max_abs_error'] <= config()['action_tolerance'], 'Paired first clean action differs')
        np.save(attempt / 'FIRST_ACTION.npy', first_clean)
        atomic(check, attempt / 'INTERFACE_CHECK.json')
        fps = 1 / float(env.env.control_timestep); limit = config()['max_steps'][case['suite']]
        step = decision = 0; success = False; traces = []; steps = []
        deployed = paste(obs, patch, config()['patch_xy'])
        with CameraWriters(attempt, fps) as writer:
            writer.append(deployed)
            while step < limit and not success:
                actions = first_clean.copy() if step == 0 and patch is None else predict(wrapper, deployed, task.language)
                begin = step; used = 0
                for slot in range(min(8, limit - step)):
                    before = np.array(raw['robot0_eef_pos'], copy=True); command = np.asarray(process_libero_action(actions[slot].copy()), dtype=np.float64)
                    require(command.shape == (7,) and np.isfinite(command).all(), 'Controller command invalid')
                    raw, reward, done, info = env.step(command.tolist()); step += 1; used += 1
                    after = np.array(raw['robot0_eef_pos'], copy=True)
                    steps.append(dict(step=step, time_s=step / fps, decision_block=decision, executed_slot=slot,
                        raw_action=actions[slot].tolist(), controller_command=command.tolist(), eef_before=before.tolist(),
                        eef_after=after.tolist(), actual_eef_displacement_m=(after-before).tolist(),
                        gripper_qpos=np.asarray(raw['robot0_gripper_qpos']).tolist(), task_success=bool(done)))
                    obs = r.observation(raw, cfg); deployed = paste(obs, patch, config()['patch_xy']); writer.append(deployed)
                    if done: success = True; break
                traces.append(dict(decision_block=decision, step_start=begin, step_end=step,
                    executed_slots=used, predicted_actions=actions.tolist(), success_after_chunk=success))
                decision += 1; monitor.update(current_start=case['start'], condition=condition['name'], executed_steps=step)
        pd.DataFrame(steps).to_parquet(attempt / 'STEPS.parquet', index=False)
        pd.DataFrame(traces).to_parquet(attempt / 'TRACE.parquet', index=False)
        atomic(dict(**case, condition=condition['name'], checkpoint=config()['model_dir'], revision=config()['revision'],
                    unnorm_key=wrapper.gen_cfg.unnorm_key, png_sha256=condition['sha256'], success=int(success),
                    executed_steps=step, max_steps=limit, fps=fps, num_open_loop_steps=8,
                    native_PNG_only=True, strict_paired_reset=True, model_frozen=True,
                    elapsed_s=time.monotonic()-started, **resources()), attempt / 'EPISODE.json')
        finish(receipt, [attempt / n for n in ('START.npz', 'START.json', 'FIRST_ACTION.npy', 'EPISODE.json',
            'STEPS.parquet', 'TRACE.parquet', 'INTERFACE_CHECK.json', 'front.mp4', 'wrist.mp4')]
            + [paired / n for n in ('START.npz', 'START.json')])
    except BaseException as error:
        atomic(dict(status='FAILED', error=repr(error), case=case, condition=condition['name'],
                    elapsed_s=time.monotonic()-started, **resources()), attempt / 'FAILURE.json')
        raise
    finally:
        if env is not None: r.close_environment(env)


def assignments(m, stage, index=0, workers=1):
    if stage == 'clean': return m['cases'], [r for r in m['conditions'] if r['name'] == 'clean']
    require(stage == 'evaluate' and 0 <= index < workers <= 4, 'Invalid evaluation shard')
    return weighted_shards(m['cases'], workers, config()['max_steps'])[index], [r for r in m['conditions'] if r['name'] != 'clean']


def run(stage, index, workers, monitor):
    m = verify(); cases, rows = assignments(m, stage, index, workers)
    receipt = ROOT / 'reports' / (f'{stage}_{workers}_{index}_COMPLETE.json')
    targets = [episode_receipt(c, r['name']) for c in cases for r in rows]
    already = sum(complete(p) for p in targets); monitor.update(completed=already, resumed=already, total=len(targets))
    if already < len(targets):
        from PIL import Image
        if stage == 'evaluate': require(all(complete(episode_receipt(c, 'clean')) for c in m['cases']), 'All 40 clean episodes must finish first')
        wrapper = load_wrapper(monitor); checked_suites = set()
        patches = {r['name']: np.array(Image.open(r['path']).convert('RGB'), copy=True) if r['path'] else None for r in rows}
        for case in cases:
            for row in rows:
                if complete(episode_receipt(case, row['name'])): continue
                episode(wrapper, case, row, patches[row['name']], monitor, case['suite'] not in checked_suites)
                checked_suites.add(case['suite'])
                monitor.update(phase='native_closedloop', completed=monitor.data['completed']+1)
    require(all(complete(p) for p in targets), 'Shard incomplete')
    finish(receipt, targets, episodes=len(targets)); monitor.update(phase='COMPLETE', completed=len(targets))

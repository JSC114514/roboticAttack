"""Streaming expert RLDS preparation and native saved-action camera replay."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from wr_common import ROOT, PROJECT, atomic, config, read, require, sha, signature, finish, complete, task_map


def pair_anchors(length, count=8, chunk=8):
    """Midpoints of equal timeline bins; both chunks contain real future labels."""
    maximum = int(length)-2*chunk
    require(maximum >= count-1, 'Trajectory too short for distinct paired anchors')
    start = np.floor((np.arange(count)+.5)*(maximum+1)/count).astype(int)
    require(len(set(start.tolist())) == count, 'Duplicate temporal anchor')
    return np.stack([start, start+chunk], axis=1)


def normalize_bounds(value, stats):
    value = np.asarray(value, dtype=np.float32)
    low, high = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    mask = np.asarray(stats.get('mask', np.ones_like(low)), bool)
    require(value.shape[-1] == low.size and high.shape == low.shape, 'Normalization dimensions differ')
    require(np.all(high[mask] > low[mask]), 'Invalid normalization bounds')
    scaled = np.clip(2*(value-low)/(high-low+1e-8)-1, -1, 1)
    return np.where(mask, scaled, value).astype(np.float32)


def native_expert_actions(raw, action_stats):
    actions = np.asarray(raw, np.float32).copy()
    require(actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all(), 'Invalid expert actions')
    actions[:, 6] = 1-np.clip(actions[:, 6], 0, 1)
    return actions, normalize_bounds(actions, action_stats)


def trajectory_id(path, actions, states):
    h = hashlib.sha256(str(path).encode())
    for x in (actions, states):
        x = np.ascontiguousarray(x, dtype='<f4')
        h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def match_task(path, names):
    name = Path(str(path)).name.lower()
    candidates = [i for i, task in enumerate(names) if name in
                  (task.lower()+'.hdf5', task.lower()+'_demo.hdf5') or
                  name.startswith(task.lower()+'_demo')]
    require(len(candidates) == 1, 'Unknown or ambiguous RLDS task source: '+str(path))
    return candidates[0]


def stage_tasks(stage, c=None):
    c = config() if c is None else c
    require(stage in ('pilot', 'main'), 'Unknown stage')
    return c['pilot_tasks'] if stage == 'pilot' else {s: c['task_splits'][s]['train'] for s in c['suites']}


def select_episodes(metadata, tasks, fit_count, val_count):
    groups = defaultdict(list)
    for item in metadata:
        if item['task_id'] in tasks[item['suite']] and item.get('eligible', False):
            groups[item['suite'], item['task_id']].append(item)
    selected, shortages = [], []
    for suite in tasks:
        for task in tasks[suite]:
            rows = sorted(groups[suite, task], key=lambda x: (x['file_path'], x['episode_id']))
            unique = {x['episode_id']: x for x in rows}
            rows = sorted(unique.values(), key=lambda x: (x['file_path'], x['episode_id']))
            needed = fit_count+val_count
            if len(rows) < needed:
                shortages.append(dict(suite=suite, task_id=task, eligible=len(rows), needed=needed))
            selected.extend(dict(x, split='train' if i < fit_count else 'validation')
                            for i, x in enumerate(rows[:needed]))
    return selected, shortages


def camera_members(stage, c=None, index=None):
    c = config() if c is None else c
    index = read(PROJECT/c['camera_index']) if index is None else index
    tasks, st = stage_tasks(stage, c), c['stages'][stage]
    out = {'train': [], 'validation': []}
    for split, key in (('train', 'camera_fit_states'), ('validation', 'camera_val_states')):
        out[split] = [dict(r, source_split=r['split'], split=split)
                      for r in index if r['split'] == 'train' and r['suite'] in tasks and
                      r['task_id'] in tasks[r['suite']] and r['state_index'] in st[key]]
        groups = defaultdict(list)
        for row in out[split]: groups[row['episode_id']].append(row)
        require(all(len(v) == 5 for v in groups.values()), 'Camera source lost fixed five progress observations')
        expected = sum(map(len, tasks.values()))*len(st[key])*5
        require(len(out[split]) == expected, 'Incomplete full-progress camera source')
    require(not ({r['episode_id'] for r in out['train']} & {r['episode_id'] for r in out['validation']}),
            'Camera trajectory leakage')
    return out


def _dataset(suite, c):
    # Set before importing TensorFlow: CPU preparation must not discover or hold GPU memory.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], 'GPU')
    root = Path(c['expert_root'])/(suite+'_no_noops')/'1.0.0'
    builder = tfds.builder_from_directory(str(root))
    decoder = {'steps': {'observation': {'image': tfds.decode.SkipDecoding(),
                                       'wrist_image': tfds.decode.SkipDecoding()}}}
    ds = builder.as_dataset(split='train', shuffle_files=False, decoders=decoder,
                           read_config=tfds.ReadConfig(try_autocache=False))
    options = tf.data.Options(); options.threading.private_threadpool_size = 2
    return ds.with_options(options)


def _episode_arrays(episode, images=False):
    path = episode['episode_metadata']['file_path'].numpy().decode()
    actions, states, front, wrist, languages = [], [], [], [], []
    for step in episode['steps']:
        actions.append(step['action'].numpy()); states.append(step['observation']['state'].numpy())
        languages.append(step['language_instruction'].numpy().decode())
        if images:
            front.append(step['observation']['image'].numpy())
            wrist.append(step['observation']['wrist_image'].numpy())
    require(len(set(languages)) == 1, 'Instruction changes within expert trajectory')
    return dict(file_path=path, raw_actions=np.asarray(actions, np.float32),
                states=np.asarray(states, np.float32), language=languages[0], front=front, wrist=wrist)


def _save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _decode(jpeg):
    with Image.open(io.BytesIO(jpeg)) as im:
        out = np.asarray(im.convert('RGB'), np.uint8)
    require(out.shape == (256, 256, 3), 'RLDS image size differs from native 256 RGB source')
    # RLDS regenerated images already carry native upright convention. Never rotate twice.
    return out


def prepare(stage, monitor=None):
    receipt = ROOT/f'reports/EXPERT_{stage}_COMPLETE.json'
    manifest_path = ROOT/f'data/{stage}/MANIFEST.json'
    if complete(receipt): return read(manifest_path)
    c, names = config(), task_map(); tasks = stage_tasks(stage); st = c['stages'][stage]
    metadata_path = ROOT/'data/EXPERT_INDEX.json'
    inputs = {}
    for suite in c['suites']:
        directory = Path(c['expert_root'])/(suite+'_no_noops')/'1.0.0'
        for p in sorted(directory.glob('*')):
            if p.is_file(): inputs[str(p)] = [p.stat().st_size, p.stat().st_mtime_ns]
    if metadata_path.exists() and read(metadata_path).get('inputs') == inputs:
        metadata = read(metadata_path)['episodes']
    else:
        metadata = []
        for suite in c['suites']:
            for ordinal, raw in enumerate(_dataset(suite, c)):
                episode = _episode_arrays(raw)
                task = match_task(episode['file_path'], names[suite])
                length = len(episode['raw_actions'])
                metadata.append(dict(suite=suite, task_id=task, file_path=episode['file_path'],
                    episode_id=trajectory_id(episode['file_path'], episode['raw_actions'], episode['states']),
                    length=length, ordinal=ordinal, eligible=length >= 23))
                if monitor: monitor.update(phase='index_expert', suite=suite, completed=len(metadata))
        atomic(dict(inputs=inputs, episodes=metadata), metadata_path)
    selected, shortages = select_episodes(metadata, tasks, st['train_expert_trajectories'], st['val_expert_trajectories'])
    stats_path = Path(c['model_dir'])/'dataset_statistics.json'; stats = read(stats_path)
    manifest = dict(signature=signature(), stage=stage, expert={'train': [], 'validation': []},
                    camera=camera_members(stage), inputs=inputs, stats_sha256=sha(stats_path),
                    shortages=shortages, duplicate_episode_count=len(metadata)-len({x['episode_id'] for x in metadata}),
                    experts_are_policy_predictions=False, camera_has_expert_labels=False)
    if shortages:
        atomic(manifest, manifest_path)
        raise RuntimeError('Expert trajectory coverage insufficient; see '+str(manifest_path))
    selected_by_id = {x['episode_id']: x for x in selected}
    expert_files=[]
    for suite in tasks:
        pending = {k: v for k, v in selected_by_id.items() if v['suite'] == suite}
        for raw in _dataset(suite, c):
            episode = _episode_arrays(raw, images=True)
            identity = trajectory_id(episode['file_path'], episode['raw_actions'], episode['states'])
            if identity not in pending: continue
            row = pending.pop(identity)
            pair_indices = pair_anchors(row['length'], st['pairs_per_trajectory'])
            offsets = pair_indices.reshape(-1)
            target, normalized = native_expert_actions(episode['raw_actions'], stats[suite+'_no_noops']['action'])
            out = ROOT/f'data/experts/{identity}.npz'
            if not out.exists():
                _save_npz(out, front_rgb=np.stack([_decode(episode['front'][i]) for i in offsets]),
                    wrist_rgb=np.stack([_decode(episode['wrist'][i]) for i in offsets]),
                    proprio=episode['states'][offsets],
                    proprio_normalized=normalize_bounds(episode['states'][offsets], stats[suite+'_no_noops']['proprio']),
                    actions=np.stack([normalized[i:i+8] for i in offsets]),
                    native_actions=np.stack([target[i:i+8] for i in offsets]),
                    anchors=pair_indices, language=np.asarray(episode['language']))
            with np.load(out, allow_pickle=False) as saved:
                require(np.array_equal(saved['anchors'], pair_indices), 'Cached expert anchor policy changed')
                require(np.array_equal(saved['actions'],np.stack([normalized[i:i+8] for i in offsets])),
                        'Cached expert normalization changed')
            expert_files.append(out)
            for i in range(len(pair_indices)):
                manifest['expert'][row['split']].append(dict(pair_id=identity+f'_p{i}', suite=suite,
                    task_id=row['task_id'], episode_id=identity, path=str(out), pair_index=i,
                    source_file_path=row['file_path'], anchor_indices=pair_indices[i].tolist()))
            if monitor: monitor.update(phase='extract_expert', completed=len(selected)-len(pending), total=len(selected))
        require(not pending, 'Selected RLDS trajectory disappeared during extraction')
    for split in manifest['expert']:
        manifest['expert'][split].sort(key=lambda r: (r['suite'], r['task_id'], r['episode_id'], r['pair_index']))
    require(not ({r['episode_id'] for r in manifest['expert']['train']} &
                 {r['episode_id'] for r in manifest['expert']['validation']}), 'Expert trajectory leakage')
    atomic(manifest, manifest_path)
    finish(receipt, [manifest_path, metadata_path,*expert_files], train_pairs=len(manifest['expert']['train']),
           validation_pairs=len(manifest['expert']['validation']))
    return manifest


def expert_pair(record):
    """Load just one small trajectory; callers can keep an LRU of these files."""
    with np.load(record['path'], allow_pickle=False) as data:
        i = int(record['pair_index'])*2
        observations = [dict(full_image=data['front_rgb'][j], wrist_image=data['wrist_rgb'][j],
                             state=data['proprio'][j]) for j in (i, i+1)]
        actions = data['actions'][i:i+2].copy()
        language = str(data['language'])
    return observations, actions, language


@lru_cache(None)
def geometry_module():
    # Bind only the geometry reader's tiny dependency contract. Importing the
    # old training package would otherwise rebind unrelated run globals.
    import importlib.util
    from types import SimpleNamespace
    import types
    path=PROJECT/'research_runs/wrist_spatial_approach_pilot_v1/code/sr_geometry.py'
    prior=sys.modules.get('sr_common')
    shim=types.ModuleType('sr_common');shim.require=require;shim.native=SimpleNamespace()
    sys.modules['sr_common']=shim
    try:
        spec=importlib.util.spec_from_file_location('wr_reference_geometry',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        if prior is None:sys.modules.pop('sr_common',None)
        else:sys.modules['sr_common']=prior
    return module

def _fixed_target_ids(env, suite, task):
    target = geometry_module().anchor(env, suite, task)
    return target['render_ids'], {k: v for k, v in target.items() if k != 'gripper'}

@lru_cache(maxsize=1)
def source_rgb(suite,task):
    import torch
    root=(PROJECT/config()['camera_index']).parent
    path=root/f'cache/train/{suite}/task_{task:02d}.pt'
    receipt=read(path.with_suffix('.json'))
    require(receipt.get('status')=='COMPLETE','Historical RGB shard incomplete')
    expected=receipt['artifacts'].get(str(path.relative_to(root.parent)))
    require(expected is not None and sha(path)==expected,'Historical RGB shard hash differs')
    value=torch.load(path,map_location='cpu',weights_only=False)
    return {r['state_id']:dict(observation=r['observation'],sample_fingerprint=r['sample_fingerprint'])
            for r in value['records']}


def _decode_chunk(value):
    if isinstance(value, str): value = json.loads(value)
    if isinstance(value,np.ndarray):value=value.tolist()
    array = np.asarray(value, np.float32)
    if array.size == 56: array = array.reshape(8, 7)
    require(array.shape == (8, 7) and np.isfinite(array).all(), 'Invalid saved native action chunk')
    return array


def replay_alignment(observation, original, sample):
    """Keep exact replay state/timing; audit RGB produced by a different GPU.

    The camera pair uses the current native RGB on BOTH sides. Cached RGB is
    a provenance reference, never the clean half of a newly rendered pair.
    """
    current, reference = sample['fingerprint'], original['sample_fingerprint']
    differences = [key for key in sorted(set(current['arrays']) | set(reference['arrays']))
                   if current['arrays'].get(key) != reference['arrays'].get(key)]
    require(not differences, 'Replay physical/camera arrays differ: '+repr(differences))
    require(current['time'] == reference['time'] and
            sample['sample_time_s'] == reference['time'],
            'Replay camera sampling time differs')
    require(current['observable_clocks'] == reference['observable_clocks'],
            'Replay sensor clocks differ')
    require(np.array_equal(observation['state'], original['observation']['state']),
            'Replay proprioception differs from cache')
    images = {}
    for name in ('full_image', 'wrist_image'):
        a, b = np.asarray(observation[name]), np.asarray(original['observation'][name])
        require(a.shape == b.shape == (224,224,3) and a.dtype == b.dtype == np.uint8,
                'Replay/cache RGB shape or dtype differs: '+name)
        delta = a.astype(np.float64)-b.astype(np.float64)
        images[name] = dict(exact=bool(np.array_equal(a,b)),
            nrmse=float(np.sqrt(np.mean((delta/255.)**2))),
            mean_abs_gray=float(np.mean(np.abs(delta))),
            max_abs_gray=float(np.max(np.abs(delta))),
            changed_pixels=int(np.count_nonzero(np.any(delta != 0,axis=-1))))
    return dict(physical_arrays_exact=True, sample_time_exact=True,
                sensor_clocks_exact=True, proprio_exact=True, cached_rgb=images,
                pair_clean_source='current_native_replay', cached_rgb_used_in_pair=False)


def prepare_camera(stage, index, workers, monitor=None):
    from wr_common import require_gpu, native_runtime, runtime_config
    from wr_camera import CameraSamples
    import pandas as pd
    require_gpu(); c = config(); rt = native_runtime(); rc = runtime_config()
    rt.prepare_graphics_backend(); rt.add_paths(rc)
    require(0<=index<workers,'Invalid camera worker shard')
    manifest_path = ROOT/f'data/{stage}/MANIFEST.json'; manifest = read(manifest_path)
    require(manifest['signature'] == signature(), 'Expert preparation revision changed')
    rows = sum(manifest['camera'].values(), [])
    episodes = sorted({r['episode_id'] for r in rows})[index::workers]
    transforms = [dict(axis=a, shift_m=d) for a in c['camera_axes'] for d in c['camera_train_shifts_m']]
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    completed = 0
    for episode_id in episodes:
        receipt = ROOT/f'data/{stage}/camera/{episode_id}.json'
        if complete(receipt): completed += 1; continue
        selected = {r['decision_block']: r for r in rows if r['episode_id'] == episode_id}
        e = next(iter(selected.values())); suite, task_id = e['suite'], e['task_id']
        trace = pd.read_parquet(PROJECT/c['source_traces'], filters=[('condition', '==', 'clean'),
            ('suite', '==', suite), ('task_id', '==', task_id), ('state_index', '==', e['state_index']),
            ('rollout_seed', '==', e['rollout_seed'])]).sort_values('decision_block')
        require(list(trace.decision_block) == list(range(e['decision_count'])), 'Saved replay action sequence missing')
        env, task, initial = rt.make_env(rc, suite, task_id)
        try:
            raw = rt.reset_wait(env, initial[e['state_index']], e['rollout_seed'], rc['num_steps_wait'])
            try: ids, target = _fixed_target_ids(env, suite, task_id)
            except (RuntimeError, KeyError, ValueError, AttributeError) as error:
                ids, target = None, dict(unavailable=True, reason=repr(error))
            captured, audits = [], []
            with CameraSamples(env, c['rgb_tolerance'], ids) as camera:
                executed = 0
                for saved in trace.itertuples(index=False):
                    decision = int(saved.decision_block)
                    if decision in selected:
                        sample = camera.take(raw, selected[decision]['state_id'])
                        clean = rt.observation(raw, rc)
                        original=source_rgb(suite,task_id)[selected[decision]['state_id']]
                        try:
                            alignment = replay_alignment(clean, original, sample)
                        except RuntimeError as error:
                            atomic(dict(state_id=selected[decision]['state_id'],error=str(error),
                                current=sample['fingerprint'],reference=original['sample_fingerprint']),
                                ROOT/f'reports/REPLAY_ALIGNMENT/{selected[decision]["state_id"]}.json')
                            raise
                        shifted, audit = [], []
                        for transform in transforms:
                            changed, information = camera.shifted_raw(raw, transform['axis'], transform['shift_m'])
                            alternate = rt.observation(changed, rc)
                            require(np.array_equal(alternate['full_image'], clean['full_image']) and
                                    np.array_equal(alternate['state'], clean['state']), 'Camera pair changed front/proprio')
                            shifted.append(alternate['wrist_image']); audit.append(information)
                        captured.append(dict(**selected[decision], observation=clean,
                                             shifted_wrist=np.stack(shifted)))
                        audits.append(dict(state_id=selected[decision]['state_id'],
                                           sample_fingerprint=sample['fingerprint'],
                                           replay_alignment=alignment, transforms=audit))
                        if decision == max(selected): break
                    actions = _decode_chunk(saved.predicted_environment_action_chunk)
                    slots = int(saved.executed_slots)
                    require(1 <= slots <= 8, 'Saved replay slots invalid')
                    for slot in range(slots):
                        nxt = selected.get(decision+1) if slot == slots-1 else None
                        camera.arm(nxt['state_id'] if nxt else None)
                        raw, _, done, _ = env.step(process_libero_action(actions[slot].copy()).tolist())
                        executed += 1
                        require(not done or decision+1 >= max(selected), 'Replay terminated before selected observation')
                    require(executed == int(saved.step_end), 'Replay native executed-step boundary changed')
            require(len(captured) == len(selected), 'Incomplete five-progress camera trajectory')
            out = receipt.with_suffix('.npz')
            _save_npz(out, front_rgb=np.stack([r['observation']['full_image'] for r in captured]),
                wrist_rgb=np.stack([r['observation']['wrist_image'] for r in captured]),
                proprio=np.stack([r['observation']['state'] for r in captured]),
                shifted_wrist=np.stack([r['shifted_wrist'] for r in captured]),
                state_ids=np.asarray([r['state_id'] for r in captured]), language=np.asarray(task.language))
            audit_path = receipt.with_name(receipt.stem+'_AUDIT.json')
            atomic(dict(target=target, samples=audits, transforms=transforms,
                        replayed_saved_actions=True, policy_called=False), audit_path)
            finish(receipt, [out, audit_path], episode_id=episode_id, split=e['split'],
                   samples=len(captured), source_traces_sha256=sha(PROJECT/c['source_traces']))
            completed += 1
            if monitor: monitor.update(phase='camera_replay', completed=completed, total=len(episodes), episode_id=episode_id)
        finally: rt.close_environment(env)
    finish(ROOT/f'reports/CAMERA_{stage}_{workers}_{index}_COMPLETE.json',
           [ROOT/f'data/{stage}/camera/{e}.json' for e in episodes], episodes=len(episodes))


def camera_records(stage, split):
    manifest = read(ROOT/f'data/{stage}/MANIFEST.json')
    out = []
    for row in manifest['camera'][split]:
        path = ROOT/f'data/{stage}/camera/{row["episode_id"]}.npz'
        receipt = path.with_suffix('.json')
        require(complete(receipt), 'Camera cache unfinished: '+str(path))
        with np.load(path, allow_pickle=False) as data:
            ids = data['state_ids'].tolist()
        out.append(dict(row, path=str(path), record_index=ids.index(row['state_id'])))
    return out


if __name__ == '__main__':
    import argparse
    from wr_common import Monitor
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['pilot', 'main'], required=True)
    p.add_argument('--camera-worker', type=int); p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    label = f'prepare_{a.stage}' if a.camera_worker is None else f'camera_{a.stage}_{a.camera_worker}'
    with Monitor(label) as m:
        if a.camera_worker is None: prepare(a.stage, m)
        else: prepare_camera(a.stage, a.camera_worker, a.workers, m)

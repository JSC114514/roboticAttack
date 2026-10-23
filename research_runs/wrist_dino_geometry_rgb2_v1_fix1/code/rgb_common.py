"""Independent RGB feasibility pilot, with pinned read-only source adapters."""
from functools import lru_cache
import ast
import importlib.util
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MATCH = PROJECT / 'research_runs/wrist_dino_geometry_match_v1'
SCALE = PROJECT / 'research_runs/wrist_dino_geometry_oracle_scale2_v1'
CLOSED = PROJECT / 'research_runs/wrist_dino_geometry_closedloop_v1'
RELATION = PROJECT / 'research_runs/wrist_dino_relation_bias_v1'
BROKEN = PROJECT / 'research_runs/wrist_dino_geometry_rgb2_v1'
sys.path.insert(1, str(MATCH / 'code'))
import common as native
sys.path.insert(1, str(SCALE / 'code'))
import oracle_common as scale_source
from oracle_camera import Camera

require, sha, read_json = native.require, native.sha, native.read_json
atomic_json, atomic_torch, atomic_parquet = native.atomic_json, native.atomic_torch, native.atomic_parquet

def config(): return read_json(ROOT / 'config.json')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

@lru_cache(maxsize=1)
def source_rgb():
    return module('rgb_repair_readonly_source', BROKEN / 'code/rgb_common.py')

@lru_cache(maxsize=1)
def signature():
    files = [ROOT / p for p in ('config.json', 'README.md', 'run.sh', 'REPAIR.md')]
    for folder in ('code', 'slurm', 'tests'):
        files += [p for p in (ROOT / folder).glob('*') if p.is_file()]
    return native.digest({'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
                          'match': native.signature(), 'scale2': scale_source.signature(),
                          'repair_source': source_rgb().signature()})

def complete(path):
    path = Path(path)
    if not path.exists(): return False
    value = read_json(path)
    require(value['signature'] == signature(), 'Stale RGB pilot receipt: ' + str(path))
    require(all((ROOT / p).is_file() and sha(ROOT / p) == h for p, h in value['artifacts'].items()),
            'Changed RGB pilot artifact: ' + str(path))
    return value['status'] == 'COMPLETE'

def finish(path, artifacts, **extra):
    atomic_json({'status': 'COMPLETE', 'signature': signature(), 'completed_unix': time.time(),
                 'artifacts': {str(Path(p).relative_to(ROOT)): sha(p) for p in artifacts}, **extra}, path)

def episode_path(task, state, condition):
    return ROOT / f'evaluation/episodes/t{task:02d}_s{state}_{condition}/COMPLETE.json'

def selections():
    rows = read_json(RELATION / 'data/INDEX.json')
    c = config()
    return {cohort: [{**r, 'cohort': cohort} for r in rows if r['split'] == 'train' and
            r['state_index'] in c[key] and r['frame_rank'] in c['frame_ranks']]
            for cohort, key in (('fit', 'fit_states'), ('verify', 'verify_states'))}

def tasks():
    return sorted({(r['suite'], r['task_id']) for r in selections()['fit']})

def cache_receipt(episode_id): return ROOT / f'data/cache/{episode_id}.json'

def cache_complete(index):
    rows = [r for values in selections().values() for r in values]
    assigned = set(tasks()[index::2])
    return all(complete(cache_receipt(e)) for e in {r['episode_id'] for r in rows if (r['suite'], r['task_id']) in assigned})

def verify_inputs():
    manifest = read_json(ROOT / 'data/MANIFEST.json')
    require(manifest['signature'] == signature(), 'Manifest changed')
    for p, h in manifest['inputs'].items(): require(sha(p) == h, 'Source changed: ' + p)
    return manifest

def prepare():
    if complete(ROOT / 'reports/PREPARE_COMPLETE.json'): return verify_inputs()
    import numpy as np
    import torch
    c = config()
    require(native.signature() == c['source_signature'] and scale_source.signature() == c['scale2_signature'], 'Source revision changed')
    require(source_rgb().signature() == c['repair_source_signature'], 'Failed source revision changed')
    require(scale_source.complete(SCALE / 'results/COMPLETE.json'), '2x ideal geometry result incomplete')
    require(c['target_sign'] == 'plus' and c['displacement_scale'] == 2. and not c['eot'] and not c['action_loss'], 'Protocol changed')
    selected = selections()
    require([len(selected[k]) for k in ('fit', 'verify')] == [240, 240] and len(tasks()) == 24, 'Unexpected cohort sizes')
    require(not ({r['episode_id'] for r in selected['fit']} & {r['episode_id'] for r in selected['verify']}), 'Trajectory leakage')
    require(c['visits'] * len(selected['fit']) == c['shared_updates'] == 4800, 'Update budget changed')
    source_manifest = read_json(MATCH / 'data/SOURCE_MANIFEST.json')
    inputs = {str(RELATION / 'data/INDEX.json'): sha(RELATION / 'data/INDEX.json'),
              str(SCALE / 'data/MANIFEST.json'): sha(SCALE / 'data/MANIFEST.json')}
    trace = native.PROJECT / native.cfg()['source_run'] / 'results/LONG4_TRACES.parquet'
    inputs[str(trace)] = sha(trace)
    library = Path(native.runtime_config()['paths']['libero_root']) / 'libero/libero'
    map_path = library / 'benchmark/libero_suite_task_map.py'
    nodes = ast.parse(map_path.read_text()).body
    names = ast.literal_eval(next(n.value for n in nodes if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets)))
    inputs[str(map_path)] = sha(map_path)
    initial = {}
    for suite, task in tasks():
        rel = f'data/cache/train/{suite}/task_{task:02d}.pt'
        path = RELATION / rel
        require(sha(path) == source_manifest['shards'][rel]['sha256'], 'Original cache changed')
        inputs[str(path)] = source_manifest['shards'][rel]['sha256']
        path = library / 'init_files' / suite / (names[suite][task] + '.pruned_init')
        states = torch.load(path, map_location='cpu', weights_only=False)
        require(len(states) > 45, 'Initial states missing')
        initial[f'{suite}/{task}'] = {'path': str(path), 'task_name': names[suite][task]}
        inputs[str(path)] = sha(path)
    for folder in ('data', 'reports/PROGRESS', 'reports/COSTS', 'evaluation', 'results', 'training'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    # Preserve the exact initial PNG shared by both existing loss controls.
    initial_png = MATCH / 'data/initial_s42.png'
    inputs[str(initial_png)] = sha(initial_png)
    shutil.copyfile(initial_png, ROOT / 'data/initial.png')
    for task in c['eval_tasks']:
        for state in c['eval_states']:
            # Clean comes with full START arrays. Other native baselines have
            # independent reset fingerprints, checked against those arrays.
            source = scale_source.episode_path(task, state, 'clean')
            require(scale_source.complete(source), 'Sealed clean unavailable')
            clean_meta = read_json(source.parent / 'EPISODE.json')
            with np.load(SCALE / clean_meta['start_npz'], allow_pickle=False) as arrays:
                for condition in ('clean', 'initial', 'uada'):
                    src_receipt = source if condition == 'clean' else scale_source.source_protocol().episode_path(task, state, condition)
                    old = scale_source if condition == 'clean' else scale_source.source_protocol()
                    require(old.complete(src_receipt), 'Baseline incomplete')
                    inputs[str(src_receipt)] = sha(src_receipt)
                    old_root = SCALE if condition == 'clean' else CLOSED
                    artifacts = []
                    for rel, h in read_json(src_receipt)['artifacts'].items():
                        src, dst = old_root / rel, ROOT / rel
                        inputs[str(src)] = h
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dst)
                        artifacts.append(dst)
                    meta_path = episode_path(task, state, condition).parent / 'EPISODE.json'
                    value = read_json(meta_path)
                    if condition != 'clean':
                        expected_png = initial_png if condition == 'initial' else PROJECT / native.cfg()['uada_png']
                        require(value['png_sha256'] == sha(expected_png), 'Historical baseline PNG changed')
                        import hashlib
                        for key, name in (('start_sim_sha256','sim_state'), ('start_front_sha256','front'),
                                          ('start_wrist_sha256','wrist'), ('start_proprio_sha256','proprio')):
                            a = np.ascontiguousarray(arrays[name])
                            h = hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()
                            require(value[key] == h, 'Historical baseline reset does not match clean')
                    value.update(reused_baseline=True, source_receipt=str(src_receipt), start_npz=clean_meta['start_npz'])
                    atomic_json(value, meta_path)
                    finish(episode_path(task, state, condition), artifacts, reused_baseline=True, source_receipt=str(src_receipt))
    individual = sorted(selected['fit'], key=lambda r: r['state_id'])
    # One fixed observation from every other task, phases cycling over all five
    # ranks; selection never examines action response or validation results.
    ids = [next(r['state_id'] for r in individual if (r['suite'],r['task_id']) == pair and
                r['state_index'] == 40 and r['frame_rank'] == j % 5) for j, pair in enumerate(tasks()[::2])]
    atomic_json(selected, ROOT / 'data/SELECTION.json')
    manifest = {'signature': signature(), 'inputs': inputs, 'initial_states': initial, 'trace_path': str(trace),
                'individual_ids': ids, 'fit_frames': 240, 'verify_frames': 240,
                'trajectories_per_cohort': 48, 'tasks': 24, 'new_closedloop_episodes': 20,
                'reused_closedloop_baselines': 30, 'development_test_not_independent': True}
    reused = []
    expected = {r['state_id']: r for rows in selected.values() for r in rows}
    for receipt in sorted((BROKEN / 'data/cache').glob('*.json')):
        require(source_rgb().complete(receipt), 'Old completed cache changed')
        path = receipt.with_suffix('.pt')
        value = torch.load(path, map_location='cpu', weights_only=False)
        require(value['signature'] == c['repair_source_signature'] and len(value['records']) == 5, 'Invalid reuse shard')
        for row in value['records']:
            require(row['state_id'] in expected and row['cohort'] == expected[row['state_id']]['cohort'], 'Reuse selection changed')
            error = row['alignment']
            require(max(error['full_image'],error['wrist_image']) <= 1e-4 and error['proprio_max_abs'] <= 1e-6, 'Old shard did not pass legacy alignment')
            require(row['displacement_scale'] == 2., 'Reuse target dose changed')
        manifest['inputs'][str(receipt)] = sha(receipt)
        manifest['inputs'][str(path)] = sha(path)
        value.update(signature=signature(), reused_from=str(path), source_signature=c['repair_source_signature'])
        new_receipt = cache_receipt(value['episode_id'])
        new_path = new_receipt.with_suffix('.pt')
        atomic_torch(value,new_path)
        finish(new_receipt,[new_path],reused_cache=True,source_receipt=str(receipt),frames=5,
               cohort=value['records'][0]['cohort'])
        reused.append(value['episode_id'])
    manifest['reused_cache_trajectories'] = reused
    manifest['reused_cache_frames'] = len(reused)*5
    atomic_json(manifest, ROOT / 'data/MANIFEST.json')
    finish(ROOT / 'reports/PREPARE_COMPLETE.json', [ROOT / 'data/MANIFEST.json', ROOT / 'data/SELECTION.json', ROOT / 'data/initial.png'])
    return manifest

def make_engine():
    mod = module('rgb_private_engine', MATCH / 'code/engine.py')
    mod.ROOT, mod.signature = ROOT, signature
    return mod.Engine(native.load_wrapper('libero_spatial'))

@lru_cache(maxsize=1)
def monitoring():
    value = module('rgb_private_monitoring', CLOSED / 'code/protocol.py')
    value.ROOT, value.signature = ROOT, signature
    return value

def Monitor(key): return monitoring().Monitor(key)
def cost(key): return monitoring().cost(key)

"""Pinned read-only adapters; all new artifacts live below ROOT."""
from functools import lru_cache
from pathlib import Path
import importlib.util
import sys
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
SOURCE = PROJECT/'research_runs/wrist_dino_geometry_rgb2_v1_fix1'
MATCH = PROJECT/'research_runs/wrist_dino_geometry_match_v1'
PILOT = PROJECT/'research_runs/wrist_dino_geometry_pilot_v1'
sys.path.insert(1, str(SOURCE/'code'))
import rgb_common as source
native = source.native
require, read_json, sha = native.require, native.read_json, native.sha
atomic_json, atomic_torch, atomic_parquet = native.atomic_json, native.atomic_torch, native.atomic_parquet


def config(): return read_json(ROOT/'config.json')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


@lru_cache(None)
def signature():
    files = [ROOT/n for n in ('README.md', 'config.json', 'run.sh')]
    for directory in ('code', 'slurm', 'tests'):
        files += [p for p in (ROOT/directory).glob('*') if p.is_file()]
    return native.digest({'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
                          'source': source.signature(),
                          'readout': sha(PILOT/'readouts/b22.npz'),
                          'readout_metadata': sha(PILOT/'readouts/b22.json')})


def complete(path):
    path = Path(path)
    if not path.exists(): return False
    v = read_json(path)
    require(v['signature'] == signature(), 'Stale local receipt: '+str(path))
    require(all((ROOT/p).is_file() and sha(ROOT/p) == h for p,h in v.get('artifacts', {}).items()),
            'Local artifact changed: '+str(path))
    return v['status'] == 'COMPLETE'


def finish(path, artifacts=(), **fields):
    atomic_json({'status': 'COMPLETE', 'signature': signature(), 'completed_unix': time.time(),
                 'artifacts': {str(Path(p).relative_to(ROOT)): sha(p) for p in artifacts}, **fields}, path)


@lru_cache(None)
def monitoring():
    m = module('rr_private_monitor', source.CLOSED/'code/protocol.py')
    m.ROOT, m.signature = ROOT, signature
    return m


def Monitor(key): return monitoring().Monitor(key)
def cost(key): return monitoring().cost(key)


def manifest():
    value = read_json(ROOT/'data/MANIFEST.json')
    require(value['signature'] == signature(), 'Manifest revision changed')
    return value


def verify_inputs():
    v = manifest()
    for p,h in v['inputs'].items(): require(sha(p) == h, 'Pinned input changed: '+p)
    return v


def selections():
    return {k: [r for r in rows if r['suite'] == 'libero_spatial']
            for k,rows in source.selections().items()}


class Pool:
    def __init__(self, cohort):
        self.items = read_json(ROOT/'data/SELECTION.json')[cohort]
    def __len__(self): return len(self.items)
    @lru_cache(maxsize=2)
    def episode(self, episode_id):
        import torch
        v = torch.load(SOURCE/f'data/cache/{episode_id}.pt', map_location='cpu', weights_only=False)
        require(v['signature'] == config()['source_signature'], 'Cached tensor revision changed')
        return {r['state_id']: r for r in v['records']}
    def __getitem__(self, index):
        item = self.items[index]
        return self.episode(item['episode_id'])[item['state_id']]
    def __iter__(self):
        for i in range(len(self)): yield self[i]


def target(row):
    import numpy as np
    geometry = row['geometry']
    if geometry['no_target']: return None
    delta = np.asarray(geometry['deltas']['plus'], dtype=np.float64)
    require(delta.shape == (3,) and np.isfinite(delta).all(), 'Invalid actual geometry displacement')
    return delta if np.linalg.norm(delta) > 1e-12 else None


@lru_cache(None)
def engine_module():
    m = module('rr_private_engine', MATCH/'code/engine.py')
    m.ROOT, m.signature = ROOT, signature
    # Imported functions still use the pinned historical cfg; assert compatibility in prepare.
    return m


def make_engine(): return engine_module().Engine(native.load_wrapper('libero_spatial'))


def episode_path(task, state, condition):
    return ROOT/f'evaluation/episodes/t{task:02d}_s{state}_{condition}/COMPLETE.json'


def prepare():
    import torch
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'): return verify_inputs()
    c = config()
    require(source.signature() == c['source_signature'], 'Source changed')
    require(native.cfg()['patch_xy'] == c['patch_xy'] and native.cfg()['patch_shape'] == c['patch_shape'],
            'Read-only patch adapter geometry differs')
    require(c['layer'] == 22 and c['displacement_scale'] == 2 and not c['eot'], 'Protocol changed')
    selected = selections()
    require([len(selected[k]) for k in ('fit','verify')] == [60,60], 'Expected 60/60 observations')
    require(all(sorted({r['task_id'] for r in rows}) == c['tasks'] for rows in selected.values()),
            'Spatial task IDs changed')
    require(all(len({r['episode_id'] for r in selected[k]}) == 12 for k in selected), 'Expected 12 trajectories')
    require(not ({r['episode_id'] for r in selected['fit']} & {r['episode_id'] for r in selected['verify']}),
            'Fit/verify trajectories overlap')
    decoder = read_json(PILOT/'readouts/b22.json')
    require(decoder['labels'][:3] == ['object_eef_relative_'+x for x in 'xyz'] and
            decoder['training_split'] == 'discovery', 'Readout provenance changed')
    require(not ({r['episode_id'] for r in selected['verify']} & set(decoder['training_episode_ids'])),
            'Decoder/verify trajectory overlap')
    inputs = {str(PILOT/'readouts/b22.npz'): sha(PILOT/'readouts/b22.npz'),
              str(PILOT/'readouts/b22.json'): sha(PILOT/'readouts/b22.json')}
    atlas_preflight=PILOT.parent/'wrist_dino_information_atlas_v2/reports/PREFLIGHT.json'
    atlas=read_json(atlas_preflight)
    inputs[str(atlas_preflight)]=sha(atlas_preflight)
    fit_episode_ids=set(decoder['training_episode_ids'])
    decoder_episodes=[r for key,r in atlas['episodes'].items() if key in fit_episode_ids]
    require(len(decoder_episodes)==120,'Decoder training episode provenance incomplete')
    decoder_seeds={(r['task_id'],r['rollout_seed']) for r in decoder_episodes}
    require(not ({(r['task_id'],r['rollout_seed']) for r in selected['verify']} & decoder_seeds),
            'Decoder/verify replay seed overlap')
    invalid = []
    for cohort, rows in selected.items():
        for episode in sorted({r['episode_id'] for r in rows}):
            receipt = SOURCE/f'data/cache/{episode}.json'
            require(source.complete(receipt), 'Source cache incomplete: '+episode)
            for path in (receipt, receipt.with_suffix('.pt')): inputs[str(path)] = sha(path)
            payload = torch.load(receipt.with_suffix('.pt'), map_location='cpu', weights_only=False)
            require(payload['signature'] == c['source_signature'], 'Cache signature changed')
            expected = {r['state_id'] for r in rows if r['episode_id'] == episode}
            require({r['state_id'] for r in payload['records']} == expected, 'Cache selection mismatch')
            for r in payload['records']:
                require(r['cohort'] == cohort and r['displacement_scale'] == 2., 'Cohort/dose mismatch')
                if target(r) is None: invalid.append({'state_id': r['state_id'], 'cohort': cohort, 'reason': r['geometry'].get('reason')})
    old = read_json(SOURCE/'data/MANIFEST.json')
    # Episode names differ between the two banks. Also compare actual initial
    # arrays, rather than infer separation from naming conventions alone.
    import numpy as np
    verify_initials={}
    for task in c['tasks']:
        path=old['initial_states'][f'libero_spatial/{task}']['path']
        states=torch.load(path,map_location='cpu',weights_only=False)
        verify_initials[task]=[np.asarray(states[i]) for i in c['verify_states']]
    for row in decoder_episodes:
        path=row['initial_state_file']
        require(sha(path)==row['initial_state_file_sha256'],'Decoder initial state changed')
        inputs[path]=row['initial_state_file_sha256']
        with np.load(path,allow_pickle=False) as before:
            require(not any(np.array_equal(before['sim_state'],x) for x in verify_initials.get(row['task_id'],[])),
                    'Decoder fitted a verify initial state under a different episode name')
    for p,h in old['inputs'].items():
        # Pin actual model code and initial state files without importing the policy on login.
        if p.endswith('.pruned_init') or Path(p).name in ('modeling_prismatic.py','configuration_prismatic.py'):
            require(sha(p) == h, 'Model/init source changed: '+p); inputs[p] = h
    for name in ('config.json','modeling_prismatic.py','configuration_prismatic.py'):
        p = Path(native.runtime_config()['checkpoints']['libero_spatial'])/name
        inputs[str(p)] = sha(p)
    for folder in ('logs','data','reports/locks','reports/PROGRESS','reports/COSTS','results','training','evaluation'):
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    origin = MATCH/'data/initial_s42.png'
    inputs[str(origin)] = sha(origin)
    shutil.copyfile(origin, ROOT/'data/initial.png')
    # Baselines are references, never copied or rewritten. Audit metadata/artifacts per start.
    baselines = {}
    scale_manifest = read_json(source.SCALE/'data/MANIFEST.json')
    for task in c['eval_tasks']:
        for state in c['eval_states']:
            start = {}; reason = None
            for condition in ('clean','initial','uada'):
                receipt = SOURCE/f'evaluation/episodes/t{task:02d}_s{state}_{condition}/COMPLETE.json'
                if not source.complete(receipt): reason = 'missing_baseline'; break
                meta_path = receipt.parent/'EPISODE.json'; meta = read_json(meta_path)
                if meta['checkpoint'] != 'libero_spatial': reason = 'checkpoint_mismatch'; break
                expected_png = None if condition == 'clean' else origin if condition == 'initial' else PROJECT/native.cfg()['uada_png']
                if expected_png is not None and meta.get('png_sha256') != sha(expected_png):
                    reason = 'png_mismatch'; break
                if meta['rollout_seed'] != scale_manifest['seeds'][f'{task}/{state}']:
                    reason = 'seed_mismatch'; break
                paths = [receipt,meta_path,SOURCE/meta['start_npz']]
                if condition=='clean' and meta['start_npz'] not in read_json(receipt)['artifacts']:
                    reason='clean_start_not_sealed_by_receipt';break
                for path in paths: inputs[str(path)] = sha(path)
                start[condition] = {'meta':str(meta_path),'receipt':str(receipt),'start':str(SOURCE/meta['start_npz'])}
            baselines[f'{task}/{state}'] = {'eligible':reason is None,'reason':reason,'conditions':start,
                'seed':scale_manifest['seeds'][f'{task}/{state}']}
    selection_path = ROOT/'data/SELECTION.json'; atomic_json(selected, selection_path)
    value = {'signature':signature(),'inputs':inputs,'baselines':baselines,'invalid_targets':invalid,
             'initial_states':{str(t):old['initial_states'][f'libero_spatial/{t}'] for t in c['eval_tasks']},
             'fit_observations':60,'verify_observations':60,'trajectories_each':12,
             'readout_fit':'600 Discovery observations, 120 trajectories; frozen',
             'split_checks':['episode_id','task_and_rollout_seed','exact_initial_sim_state_array'],
             'readout_is_causal_control':'unverified','test_scope':'development_not_independent'}
    atomic_json(value,ROOT/'data/MANIFEST.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',selection_path,ROOT/'data/initial.png'])
    return value

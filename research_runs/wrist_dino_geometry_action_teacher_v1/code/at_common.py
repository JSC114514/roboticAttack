"""Independent, sealed experiment; historical modules and data remain read-only."""
from pathlib import Path
from functools import lru_cache
from datetime import datetime
import ast
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
SOURCE = PROJECT/'research_runs/wrist_dino_geometry_rgb2_v1_fix1'
sys.path.insert(1, str(SOURCE/'code'))
import rgb_common as source

native = source.native
require, sha, read_json = native.require, native.sha, native.read_json
atomic_json, atomic_torch, atomic_parquet = native.atomic_json, native.atomic_torch, native.atomic_parquet

def config(): return read_json(ROOT/'config.json')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

@lru_cache(None)
def signature():
    paths = [ROOT/'config.json', ROOT/'README.md', ROOT/'run.sh']
    for d in ('code', 'slurm', 'tests'):
        paths += [p for p in (ROOT/d).glob('*') if p.is_file()]
    return native.digest({'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
                          'source': source.signature()})

def complete(path):
    path = Path(path)
    if not path.exists(): return False
    v = read_json(path)
    require(v['signature'] == signature(), 'Receipt revision changed: '+str(path))
    require(v['status'] == 'COMPLETE', 'Invalid completion receipt')
    require(all((ROOT/p).is_file() and sha(ROOT/p) == h for p,h in v['artifacts'].items()),
            'Artifact changed: '+str(path))
    return True

def finish(path, artifacts, **fields):
    import time
    atomic_json({'status':'COMPLETE', 'signature':signature(), 'completed_unix':time.time(),
        'artifacts':{str(Path(p).relative_to(ROOT)):sha(p) for p in artifacts}, **fields}, path)

def cutoff(): return datetime.fromisoformat(config()['train_until']).timestamp()

def candidates():
    return [(f'{m}_s{s}', m, s) for s in config()['seeds'] for m in config()['methods']]

@lru_cache(None)
def reference_engine_module():
    m = module('at_reference_engine', source.MATCH/'code/engine.py')
    m.ROOT, m.signature = ROOT, signature
    return m

def make_engine(checkpointing=False):
    from at_engine import Engine
    return Engine(native.load_wrapper('libero_spatial'), checkpointing=checkpointing)

@lru_cache(None)
def monitoring():
    # Use the independent I/O lock in MATCH rather than concurrent writes to one temp file.
    m = module('at_private_monitor', source.MATCH/'code/common.py')
    m.ROOT, m.signature = ROOT, signature
    return m

def Monitor(key): return monitoring().Monitor(key)
def cost(key): return monitoring().cost_attempt(key, path=ROOT/f'reports/COSTS/{key}_{__import__("time").time_ns()}.json')

def teacher_receipt(episode_id): return ROOT/f'data/teachers/{episode_id}.json'

def source_shards(): return read_json(ROOT/'data/MANIFEST.json')['source_shards']

def records(cohort, with_teacher=False):
    import torch
    for entry in source_shards():
        if entry['cohort'] != cohort: continue
        payload = torch.load(entry['path'], map_location='cpu', weights_only=False)
        require(payload['signature'] == config()['source_signature'], 'Source payload revision changed')
        targets = None
        if with_teacher:
            receipt = teacher_receipt(payload['episode_id'])
            require(complete(receipt), 'Teacher unavailable: '+payload['episode_id'])
            saved = torch.load(receipt.with_suffix('.pt'), map_location='cpu', weights_only=False)
            require(saved['signature'] == signature(), 'Teacher revision changed')
            targets = {r['state_id']:r for r in saved['records']}
        for r in payload['records']:
            require(r['cohort'] == cohort, 'Source cohort changed')
            yield {**r, 'teacher':targets[r['state_id']]} if targets is not None else r

def cache_complete(index):
    return all(complete(teacher_receipt(e['episode_id'])) for e in source_shards()[index::2])

def verify_inputs():
    v = read_json(ROOT/'data/MANIFEST.json')
    require(v['signature'] == signature(), 'Preparation revision changed')
    require(source.signature() == config()['source_signature'], 'Historical code changed')
    for p,h in v['inputs'].items(): require(sha(p) == h, 'Historical input changed: '+p)
    return v

def prepare():
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'): return verify_inputs()
    import numpy as np
    import torch
    from PIL import Image
    c = config()
    require(source.signature() == c['source_signature'], 'Unexpected RGB2 revision')
    require(source.complete(SOURCE/'results/COMPLETE.json'), 'Source experiment incomplete')
    source.verify_inputs()
    for d in ('data/teachers','reports/locks','reports/COSTS','reports/PROGRESS','reports/GATES',
              'training','evaluation/episodes','evaluation/offline','results','logs'):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    inputs = {}; shards = []; cohorts = {'fit':[], 'verify':[]}
    for receipt in sorted((SOURCE/'data/cache').glob('*.json')):
        require(source.complete(receipt), 'Source shard incomplete')
        p = receipt.with_suffix('.pt')
        v = torch.load(p,map_location='cpu',weights_only=False)
        require(len(v['records']) == 5, 'Source shard size changed')
        r = v['records'][0]; cohort = r['cohort']
        require(cohort in cohorts and all(x['displacement_scale']==2. for x in v['records']), 'Target protocol changed')
        cohorts[cohort] += [{k:x[k] for k in ('state_id','episode_id','suite','task_id','state_index','frame_rank')} for x in v['records']]
        inputs[str(p)] = sha(p); inputs[str(receipt)] = sha(receipt)
        shards.append({'path':str(p),'episode_id':v['episode_id'],'cohort':cohort})
        del v
    require([len(cohorts[k]) for k in ('fit','verify')] == [240,240], 'Expected 240/240 observations')
    require(not ({x['episode_id'] for x in cohorts['fit']} & {x['episode_id'] for x in cohorts['verify']}), 'Trajectory leakage')
    require(all(len({(x['suite'],x['task_id']) for x in cohorts[k]})==24 for k in cohorts), 'Task coverage changed')
    for seed in c['seeds']:
        # Retain the already deployed initialization for seed42, deterministic seed43.
        p=ROOT/f'data/initial_s{seed}.png'
        if seed==42:
            src=SOURCE/'data/initial.png';inputs[str(src)]=sha(src);p.write_bytes(src.read_bytes())
        else:
            x=torch.rand((3,50,50),generator=torch.Generator().manual_seed(seed))
            Image.fromarray(x.permute(1,2,0).mul(255).round().byte().numpy()).save(p)
    runtime = native.runtime_config()
    library = Path(runtime['paths']['libero_root'])/'libero/libero'
    task_map=library/'benchmark/libero_suite_task_map.py'
    nodes=ast.parse(task_map.read_text()).body
    names=ast.literal_eval(next(n.value for n in nodes if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets)))
    inputs[str(task_map)]=sha(task_map); init={}
    for suite in c['suites']:
        for task in range(10):
            p=library/'init_files'/suite/(names[suite][task]+'.pruned_init')
            states=torch.load(p,map_location='cpu',weights_only=False)
            require(len(states)>max(c['eval_states']), 'Evaluation initial states missing')
            inputs[str(p)]=sha(p);init[f'{suite}/{task}']={'path':str(p),'task_name':names[suite][task]}
    uada=PROJECT/native.cfg()['uada_png']; inputs[str(uada)]=sha(uada)
    # Model/config hashes and native dependency pinning without loading model weights.
    native.runtime_dependency_preflight()
    ckpt=Path(runtime['checkpoints']['libero_spatial'])
    for name in ('config.json','modeling_prismatic.py','configuration_prismatic.py'):
        p=ckpt/name;inputs[str(p)]=sha(p)
    v={'signature':signature(),'inputs':inputs,'source_shards':shards,'cohorts':cohorts,
       'initial_states':init,'uada_png':str(uada),'checkpoint':str(ckpt),'fit_frames':240,
       'verify_frames':240,'max_new_closedloop_episodes':640,'independent_confirmation':False}
    atomic_json(v,ROOT/'data/MANIFEST.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',*[ROOT/f'data/initial_s{s}.png' for s in c['seeds']]])
    return v

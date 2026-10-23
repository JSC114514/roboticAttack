"""Pinned, isolated ideal-DINO experiment; no training or source writes."""
from __future__ import annotations
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MATCH = PROJECT / 'research_runs/wrist_dino_geometry_match_v1'
RELATION = PROJECT / 'research_runs/wrist_dino_relation_bias_v1'
CLOSED = PROJECT / 'research_runs/wrist_dino_geometry_closedloop_v1'
sys.path.insert(1, str(MATCH / 'code'))
import common as native

require, read_json, sha = native.require, native.read_json, native.sha
atomic_json, atomic_parquet = native.atomic_json, native.atomic_parquet

def config():
    return read_json(ROOT / 'config.json')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@lru_cache(maxsize=1)
def source_protocol():
    return load_module('oracle_readonly_closed_protocol', CLOSED / 'code/protocol.py')

@lru_cache(maxsize=1)
def geometry():
    module = load_module('oracle_readonly_geometry', RELATION / 'code/geometry.py')
    # That file imports an unqualified common.cfg. Bind its original configuration
    # in this private module only; never alter historical modules or files.
    module.cfg = lambda: read_json(RELATION / 'config.json')
    return module

@lru_cache(maxsize=1)
def signature():
    files = [ROOT / p for p in ('config.json', 'README.md', 'run.sh')]
    for folder in ('code', 'slurm', 'tests'):
        files += [p for p in (ROOT / folder).glob('*') if p.is_file()]
    return native.digest({'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
                          'source': native.signature(), 'closedloop': source_protocol().signature(),
                          'geometry': sha(RELATION / 'code/geometry.py'),
                          'relation_config': sha(RELATION / 'config.json')})

def complete(path):
    path = Path(path)
    if not path.exists():
        return False
    value = read_json(path)
    require(value['signature'] == signature(), 'Stale oracle receipt: ' + str(path))
    require(all((ROOT / p).is_file() and sha(ROOT / p) == digest for p, digest in value['artifacts'].items()),
            'Oracle artifact changed: ' + str(path))
    return value['status'] == 'COMPLETE'

def finish(path, artifacts, **extra):
    atomic_json({'status': 'COMPLETE', 'signature': signature(), 'completed_unix': time.time(),
                 'artifacts': {str(Path(p).relative_to(ROOT)): sha(p) for p in artifacts}, **extra}, path)

def episode_path(task, state, condition):
    return ROOT / f'evaluation/episodes/t{task:02d}_s{state}_{condition}/COMPLETE.json'

def rollout_complete():
    return all(complete(episode_path(t, s, c)) for t in config()['tasks']
               for s in config()['states'] for c in config()['conditions'])

def verify_inputs(manifest):
    require(manifest['signature'] == signature(), 'Manifest revision changed')
    for path, digest in manifest['inputs'].items():
        require(sha(path) == digest, 'Read-only input changed: ' + path)

def prepare():
    if complete(ROOT / 'reports/PREPARE_COMPLETE.json'):
        manifest = read_json(ROOT / 'data/MANIFEST.json')
        verify_inputs(manifest)
        return manifest
    c = config()
    require(c['tasks'] == [0, 1, 2, 6, 7] and c['states'] == [44, 45] and c['suite'] == 'libero_spatial', 'Scope changed')
    require(native.signature() == c['source_signature'], 'Frozen model/runtime source changed')
    old = source_protocol()
    require(old.signature() == c['closedloop_source_signature'], 'Closed-loop source changed')
    require(old.complete(CLOSED / 'reports/PREPARE_COMPLETE.json'), 'Previous data preflight missing')
    previous = read_json(CLOSED / 'data/MANIFEST.json')
    inputs = {str(CLOSED / 'data/MANIFEST.json'): sha(CLOSED / 'data/MANIFEST.json')}
    initial = {}
    for task in c['tasks']:
        entry = previous['initial_states'][str(task)]
        initial[str(task)] = entry
        inputs[entry['path']] = previous['inputs'][entry['path']]
        for state in c['states']:
            for condition in ('clean', 'initial', 'projection', 'full', 'uada'):
                path = old.episode_path(task, state, condition)
                require(old.complete(path), 'Previously paired start not complete: ' + str(path))
                inputs[str(path)] = sha(path)
    require(geometry().cfg()['max_displacement_m'] == .03 and geometry().cfg()['bearing_degrees'] == 30.,
            'Original geometry rules changed')
    for name in ('modeling_prismatic.py', 'configuration_prismatic.py', 'config.json'):
        checkpoint = Path(native.runtime_config()['checkpoints']['libero_spatial']) / name
        inputs[str(checkpoint)] = sha(checkpoint)
    for folder in ('logs', 'data', 'reports/PROGRESS', 'reports/COSTS', 'reports/locks', 'results', 'evaluation'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    manifest = {'signature': signature(), 'inputs': inputs, 'initial_states': initial,
                'seeds': {f'{t}/{s}': previous['seeds'][f'{t}/{s}'] for t in c['tasks'] for s in c['states']},
                'episodes': 30, 'checkpoint': native.runtime_config()['checkpoints']['libero_spatial'],
                'scope': 'internal_DINO_mechanism_not_RGB_patch', 'clean_rerun': True}
    verify_inputs(manifest)
    atomic_json(manifest, ROOT / 'data/MANIFEST.json')
    finish(ROOT / 'reports/PREPARE_COMPLETE.json', [ROOT / 'data/MANIFEST.json'], slurm_submitted=False)
    return manifest

# Reuse the tested monitor/cost implementation in a PRIVATE module whose ROOT and
# signature are rebound to this directory. Historical progress/results stay read-only.
@lru_cache(maxsize=1)
def monitoring():
    module = load_module('oracle_private_monitoring', CLOSED / 'code/protocol.py')
    module.ROOT, module.signature = ROOT, signature
    return module

def Monitor(key):
    return monitoring().Monitor(key)

def cost(key):
    return monitoring().cost(key)

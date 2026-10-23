"""Isolated, pinned protocol. Historical inputs are always read-only."""
from __future__ import annotations
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / 'research_runs/wrist_dino_geometry_match_v1'
sys.path.insert(1, str(SOURCE / 'code'))
import common as native

require = native.require
read_json = native.read_json
sha = native.sha
atomic_json = native.atomic_json
atomic_parquet = native.atomic_parquet

def config():
    return read_json(ROOT / 'config.json')

@lru_cache(maxsize=1)
def signature():
    files = [ROOT / name for name in ('config.json', 'README.md', 'run.sh')]
    for folder in ('code', 'slurm', 'tests'):
        files.extend(p for p in (ROOT / folder).glob('*') if p.is_file())
    return native.digest({'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
                          'source_signature': native.signature()})

def complete(path):
    path = Path(path)
    if not path.exists():
        return False
    value = read_json(path)
    require(value.get('signature') == signature(), 'Stale local receipt: ' + str(path))
    if value.get('status') != 'COMPLETE':
        return False
    for rel, digest in value['artifacts'].items():
        require((ROOT / rel).is_file() and sha(ROOT / rel) == digest, 'Changed artifact: ' + rel)
    return True

def finish(path, artifacts, **extra):
    atomic_json({'status': 'COMPLETE', 'signature': signature(), 'completed_unix': time.time(),
                 'artifacts': {str(Path(p).relative_to(ROOT)): sha(p) for p in artifacts}, **extra}, path)

def episode_path(task, state, condition):
    return ROOT / f'evaluation/episodes/t{task:02d}_s{state}_{condition}/COMPLETE.json'

def worker_complete(index):
    return all(complete(episode_path(t, s, c)) for t in config()['workers'][index]
               for s in config()['states'] for c in config()['conditions'])

def prepare():
    """Small CPU preflight: no big feature bank, no rendering and no model load."""
    if complete(ROOT / 'reports/PREPARE_COMPLETE.json'):
        manifest = read_json(ROOT / 'data/MANIFEST.json')
        verify_inputs(manifest)
        return manifest
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    c = config()
    require(native.signature() == c['source_signature'], 'Completed source revision changed')
    require(native.complete(SOURCE / 'results/pilot/COMPLETE.json'), 'Source pilot is not sealed/complete')
    require(native.complete(SOURCE / 'evaluation/pilot/CHECK_COMPLETE.json'), 'Source native PNG checks missing')
    require(c['tasks'] == sorted(t for group in c['workers'] for t in group), 'Worker tasks missing/duplicated')
    require(len(c['workers']) == 3 and c['states'] == [44, 45], 'Unexpected scope')
    require(c['patch_xy'] == native.cfg()['patch_xy'] and c['max_steps'] == 220, 'Deployment protocol mismatch')
    require(c['checkpoint_suite'] == c['suite'] == 'libero_spatial', 'Checkpoint confound')
    require(set(c['states']).isdisjoint(native.cfg()['pilot_fit_states'] + native.cfg()['pilot_verify_states']),
            'Initial states overlap this pilot fit/verification')
    inputs = {}
    for name, rel in c['patches'].items():
        path = PROJECT / rel
        with Image.open(path) as im:
            require(im.mode == 'RGB' and im.size == (50, 50), 'Invalid PNG: ' + name)
        inputs[str(path)] = sha(path)
    runtime_cfg = native.runtime_config()
    library = Path(runtime_cfg['paths']['libero_root']) / 'libero/libero'
    map_path = library / 'benchmark/libero_suite_task_map.py'
    nodes = ast.parse(map_path.read_text()).body
    task_map = ast.literal_eval(next(n.value for n in nodes if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets)))
    inputs[str(map_path)] = sha(map_path)
    initial = {}
    for task in c['tasks']:
        name = task_map[c['suite']][task]
        path = library / 'init_files' / c['suite'] / (name + '.pruned_init')
        states = torch.load(path, map_location='cpu', weights_only=False)
        require(len(states) > max(c['states']), 'Unavailable initial state')
        require(all(np.isfinite(np.asarray(states[s])).all() for s in c['states']), 'Invalid initial state')
        initial[str(task)] = {'path': str(path), 'task_name': name, 'count': len(states)}
        inputs[str(path)] = sha(path)
    reference_path = PROJECT / c['seed_reference']
    frame = pd.read_parquet(reference_path)
    clean = frame[(frame.suite == c['suite']) & (frame.condition == 'clean') &
                  frame.task_id.isin(c['tasks']) & frame.state_index.isin(c['states'])]
    require(len(clean) == 12 and not clean.duplicated(['task_id', 'state_index']).any(), 'Historical seed pairing unavailable')
    inputs[str(reference_path)] = sha(reference_path)
    seeds = {f'{int(r.task_id)}/{int(r.state_index)}': int(r.rollout_seed) for r in clean.itertuples()}
    manifest = {'signature': signature(), 'source_signature': native.signature(), 'inputs': inputs,
                'initial_states': initial, 'seeds': seeds, 'historical_clean_successes': int(clean.success.sum()),
                'episodes': 60, 'conditions': c['conditions'], 'runtime_checkpoints': runtime_cfg['checkpoints'][c['suite']],
                'independent_tasks': False, 'new_to_pilot_initial_states': True,
                'initial_states_in_old_long_training_pool': True}
    for folder in ('logs', 'data', 'reports/PROGRESS', 'reports/COSTS', 'evaluation', 'results'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, ROOT / 'data/MANIFEST.json')
    finish(ROOT / 'reports/PREPARE_COMPLETE.json', [ROOT / 'data/MANIFEST.json'], slurm_submitted=False)
    return manifest

def verify_inputs(manifest):
    require(manifest['signature'] == signature(), 'Manifest revision changed')
    for path, digest in manifest['inputs'].items():
        require(sha(path) == digest, 'Read-only input changed: ' + path)

class Monitor:
    def __init__(self, key):
        self.key = key
        self.started = time.monotonic()
        self.fields = {'status': 'RUNNING', 'phase': 'starting'}
        self.lock = threading.RLock()
        self.event = threading.Event()
    def update(self, **fields):
        with self.lock:
            self.fields.update(fields)
            elapsed = time.monotonic() - self.started
            done = self.fields.get('completed', 0) - self.fields.get('resume_completed', 0)
            remaining = self.fields.get('total', 0) - self.fields.get('completed', 0)
            atomic_json({'signature': signature(), 'worker': self.key,
                         'job_id': os.environ.get('SLURM_JOB_ID'), 'heartbeat_unix': time.time(),
                         'elapsed_s': elapsed, 'estimated_remaining_s': elapsed / done * remaining if done > 0 else None,
                         **native.resources(), **self.fields}, ROOT / f'reports/PROGRESS/{self.key}.json')
    def loop(self):
        while not self.event.wait(30):
            self.update()
    def __enter__(self):
        self.update()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, kind, error, tb):
        self.event.set()
        self.thread.join()
        self.update(status='FAILED' if kind else 'COMPLETE', error=repr(error) if error else None)

@contextlib.contextmanager
def cost(key):
    started = time.monotonic()
    path = ROOT / f'reports/COSTS/{key}_{time.time_ns()}.json'
    fields = {'signature': signature(), 'job_id': os.environ.get('SLURM_JOB_ID'), 'worker': key,
              'started_unix': time.time(), 'measurement_scope': 'this_process_RSS_HWM_and_CUDA_peaks'}
    atomic_json({**fields, 'status': 'STARTED'}, path)
    try:
        yield
    except BaseException as error:
        atomic_json({**fields, 'status': 'FAILED', 'elapsed_s': time.monotonic() - started,
                     'error': repr(error), 'traceback': traceback.format_exc(), **native.resources()}, path)
        raise
    else:
        atomic_json({**fields, 'status': 'COMPLETE', 'elapsed_s': time.monotonic() - started,
                     **native.resources()}, path)

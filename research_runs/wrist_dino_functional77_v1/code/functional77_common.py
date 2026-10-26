"""Lightweight pilot contracts. Importing this module does not load a model."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PYTHON = Path('/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


@lru_cache(maxsize=1)
def cfg():
    result = load_json(ROOT / 'config.json')
    require(result['new_episode_count'] == 77 and result['max_gpu_concurrent'] == 4,
            'Frozen episode/resource budget changed')
    require(result['components'] == ['functional', 'random', 'orthogonal'] and
            result['directions'] == ['restore', 'inject'], 'Frozen factorial changed')
    return result


@lru_cache(maxsize=1)
def cases():
    rows = load_json(ROOT / 'CASES.json')['cases']
    require(len(rows) == 10 and [r['shard'] for r in rows] == list(range(10)), 'Invalid case grid')
    require(len({r['id'] for r in rows}) == 10, 'Duplicate case identifier')
    seen = set()
    for row in rows:
        state = (row['suite'], row['task_id'], row['state_index'])
        require(row['clean_control_owner'] == (state not in seen), 'Shared clean control ownership changed')
        seen.add(state)
    require(len(seen) == 7 and {r['suite'] for r in rows} == set(cfg()['suites']), 'Incomplete four-suite coverage')
    return rows


def specs(case):
    """Controls first; independently randomized six-condition order per case."""
    import random
    rows = [{'case_id': case['id'], 'direction': 'native', 'component': 'png'}]
    if case['clean_control_owner']:
        rows.insert(0, {'case_id': case['id'], 'direction': 'native', 'component': 'clean'})
    arms = [{'case_id': case['id'], 'direction': d, 'component': c}
            for d in cfg()['directions'] for c in cfg()['components']]
    random.Random(cfg()['random_seed'] + case['shard']).shuffle(arms)
    return rows + arms


def episode_key(spec):
    return spec['direction'] + '__' + spec['component']


def episode_folder(case, spec):
    return ROOT / 'episodes' / case['id'] / episode_key(spec)


def budget():
    n = sum(len(specs(case)) for case in cases())
    require(n == 77, 'Condition grid exceeds frozen budget')
    return {'total_new': n, 'intervention': 60, 'fresh_clean': 7, 'fresh_actual_png': 10,
            'case_workers': 10, 'max_gpu_concurrent': 4, 'new_training': 0}


def parent_root():
    return ROOT.parent / cfg()['parent']


def parent_module(name):
    directory = str(parent_root() / 'code')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve().parent == (parent_root() / 'code').resolve(),
            'Unexpected native module: ' + name)
    return module


def source_hashes():
    files = [ROOT / n for n in ['config.json', 'PLAN.md', 'CASES.json', 'SOURCE_PROVENANCE.json']]
    files += [*ROOT.glob('code/*.py'), *ROOT.glob('tests/*.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def operational_hashes():
    return {str(p.relative_to(ROOT)): sha256(p)
            for p in sorted([ROOT / 'run.sh', *ROOT.glob('slurm/*')]) if p.is_file()}


def runtime_sources():
    paths = load_json(parent_root() / 'config.json')['evaluation']['paths']
    files = set((Path(paths['robotic_attack_root']) / 'VLAAttacker/multiview').glob('*.py'))
    for folder in [Path(paths['openvla_oft_root']) / 'prismatic',
                   Path(paths['openvla_oft_root']) / 'experiments/robot',
                   Path(paths['libero_root']) / 'libero/libero']:
        require(folder.is_dir(), 'Missing native runtime: ' + str(folder))
        files.update(folder.rglob('*.py'))
    return {str(p.resolve()): sha256(p) for p in sorted(files) if p.is_file()}


def library_versions():
    result = {}
    for name in ['torch', 'torchvision', 'transformers', 'timm', 'numpy', 'pandas', 'pyarrow',
                 'mujoco', 'robosuite', 'tensorflow', 'peft', 'flash-attn']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'NOT_INSTALLED_AS_DISTRIBUTION'
    return result


@lru_cache(maxsize=1)
def signature():
    return digest({'sources': source_hashes(), 'native_signature': cfg()['parent_signature'],
                   'runtime': runtime_sources(), 'libraries': library_versions()})


@contextmanager
def exclusive(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process holds lock: ' + str(path)) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def preflight():
    """Checksums/stat only: no model import, renderer, CUDA or scheduler."""
    require(PYTHON.is_file(), 'Research Python missing')
    native = parent_module('common')
    require(native.run_signature() == cfg()['parent_signature'], 'Frozen native sources changed')
    receipt = native.prepared()
    require(load_json(parent_root() / 'results/STATUS.json')['status'] == 'COMPLETE', 'Native batch incomplete')
    provenance = load_json(ROOT / 'SOURCE_PROVENANCE.json')
    inputs = {**provenance['frozen_coarse_sources'], **load_json(ROOT / 'CASES.json')['historical_inputs']}
    validation = load_json(parent_root() / 'reports/AUTO_VALIDATION.json')
    patches = {}
    for patch in cfg()['patches']:
        method, epoch = patch.split('@')
        path = parent_root() / 'training' / method / ('outer_' + str(int(epoch)).zfill(4) + '.png')
        inputs[str(path.resolve())] = validation['patches'][patch]['png']
        patches[patch] = str(path.resolve())
    require(all(sha256(p) == h for p, h in inputs.items()), 'Historical input/source/PNG checksum mismatch')
    for case in cases():
        records = {name: load_json(ref['receipt']) for name, ref in case['history'].items()}
        for name, expected in [('clean', 1), ('png', 0), ('whole_restore', 1)]:
            row = records[name]['row']
            require(records[name]['status'] == 'COMPLETE' and row['success'] == expected and
                    records[name]['signature'] == provenance['frozen_coarse_signature'], 'Case selection evidence changed')
            require(row['rollout_seed'] == case['rollout_seed'] and row['state_index'] == case['state_index']
                    and row['suite'] == case['suite'] and row['task_id'] == case['task_id'], 'Case pairing changed')
    return {'status': 'CPU_PREFLIGHT_PASS', 'signature': signature(), 'budget': budget(),
            'sources': source_hashes(), 'operational_hashes': operational_hashes(),
            'runtime_sources': runtime_sources(), 'library_versions': library_versions(),
            'frozen_inputs': inputs, 'patch_paths': patches,
            'native_source_files_verified': len(receipt['source_code_hashes']),
            'models_loaded': False, 'cuda_initialized': False, 'slurm_submitted': False,
            'runtime_validation': 'PENDING_USER_SUBMISSION', 'created_unix': time.time()}


def prepared():
    path = ROOT / 'reports/PREPARATION.json'
    current = load_json(ROOT / 'reports/CPU_PREFLIGHT.json')
    require(current['status'] == 'CPU_PREFLIGHT_PASS', 'Latest CPU preflight failed')
    receipt = load_json(path)
    failed = ROOT / 'reports/PREPARE_FAILED.json'
    require(not failed.exists() or failed.stat().st_mtime_ns <= path.stat().st_mtime_ns,
            'Unresolved preparation failure')
    require(receipt['signature'] == signature(), 'Frozen source/config changed; preserve submitted run')
    require(receipt['library_versions'] == library_versions(), 'Runtime libraries changed')
    require(all(sha256(p) == h for p, h in receipt['frozen_inputs'].items()), 'Frozen evidence changed')
    parent_module('common').prepared()
    return receipt


def completed_episode(folder, sig):
    path = Path(folder) / 'COMPLETE.json'
    if not path.exists():
        return None
    receipt = load_json(path)
    require(receipt['status'] == 'COMPLETE' and receipt['signature'] == sig, 'Episode provenance changed')
    require(bool(receipt['assets']) and
            all(sha256(Path(folder) / p) == h for p, h in receipt['assets'].items()), 'Episode assets changed')
    return receipt



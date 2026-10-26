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
    require(result['new_episode_count'] == 592, 'Episode budget changed')
    require(result['max_gpu_concurrent'] <= 4, 'Four GPU concurrency cap exceeded')
    return result


def parent_root():
    return ROOT.parent / cfg()['parent']


def parent_module(name):
    directory = str(parent_root() / 'code')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module = importlib.import_module(name)
    # These names must resolve to the frozen, internally consistent package.
    require(Path(module.__file__).resolve().parent == (parent_root() / 'code').resolve(),
            'Unexpected imported parent module: ' + name)
    return module


def source_hashes():
    files = [ROOT / 'config.json', ROOT / 'PLAN.md', ROOT / 'RECOVERY_PROVENANCE.json',
             *ROOT.glob('code/*.py'), *ROOT.glob('tests/*.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files) if p.is_file()}


def operational_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted([ROOT / 'run.sh', *ROOT.glob('slurm/*')]) if p.is_file()}


def runtime_sources():
    paths = load_json(parent_root() / 'config.json')['evaluation']['paths']
    native = Path(paths['robotic_attack_root']) / 'VLAAttacker/multiview'
    files = set(native.glob('*.py'))
    for folder in [Path(paths['openvla_oft_root']) / 'prismatic',
                   Path(paths['openvla_oft_root']) / 'experiments/robot',
                   Path(paths['libero_root']) / 'libero/libero']:
        require(folder.is_dir(), 'Missing native runtime source folder: ' + str(folder))
        files.update(folder.rglob('*.py'))
    return {str(path.resolve()): sha256(path) for path in sorted(files) if path.is_file()}


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
    return digest({'sources': source_hashes(), 'parent': cfg()['parent_signature'],
                   'runtime_sources': runtime_sources(), 'libraries': library_versions()})


def old_condition(patch):
    return 'clean' if patch == 'clean' else patch


def specs(stage, task_id):
    c = cfg()
    if stage == 'calibration':
        require(task_id in c['calibration_tasks'], 'Not a calibration task')
        kinds = ['native', 'whole_restore', 'whole_transplant']
        return [{'state_index': state, 'patch': patch, 'kind': kind, 'cohort': 'calibration'}
                for state in c['calibration_states']
                for patch in ['clean', *c['patches']]
                for kind in (['native'] if patch == 'clean' else kinds)]
    require(stage == 'main' and task_id in c['main_tasks'] + c['confirmation_tasks'], 'Not a main task')
    confirm = task_id in c['confirmation_tasks']
    kinds = ['functional_early', 'functional_random_early'] if confirm else [
        'spatial_early', 'spatial_random_early', 'functional_early', 'functional_random_early',
        'functional_half_early', 'functional_late']
    return [{'state_index': c['main_state'], 'patch': patch, 'kind': kind,
             'cohort': 'confirmation' if confirm else 'main'} for patch in c['patches'] for kind in kinds]


def shards(stage):
    tasks = cfg()['calibration_tasks'] if stage == 'calibration' else cfg()['main_tasks'] + cfg()['confirmation_tasks']
    return [(suite, task) for task in tasks for suite in cfg()['suites']]


def key(spec):
    return str(spec['state_index']) + '__' + spec['patch'].replace('@', '_') + '__' + spec['kind']


def budget():
    totals = {stage: sum(len(specs(stage, task)) for _, task in shards(stage)) for stage in ['calibration', 'main']}
    require(sum(totals.values()) == cfg()['new_episode_count'], 'Condition table does not match budget')
    return {**totals, 'new_native_reproduction': 64, 'whole_path': 96,
            'fine_grained': 360, 'confirmation': 72, 'total_new': sum(totals.values()),
            'new_training': 0, 'reused_control_episodes': 320}


@contextmanager
def exclusive(path):
    """Nonblocking process lock; a crash releases the kernel lock."""
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
    """CPU metadata/hash checks only. No data-loader, model, CUDA or scheduler."""
    import pandas as pd
    c, parent = cfg(), parent_root()
    require(PYTHON.is_file(), 'Missing research Python environment')
    frozen = parent_module('common')
    require(frozen.run_signature() == c['parent_signature'], 'Frozen parent source/signature changed')
    prepared_parent = frozen.prepared()
    require(prepared_parent['signature'] == c['parent_signature'], 'Invalid parent preparation')
    old = load_json(parent / 'results/STATUS.json')
    require(old['status'] == 'COMPLETE' and old['signature'] == c['parent_signature'], 'Old run is incomplete')
    files = {parent / 'results/BATCH500_ROLLOUTS.parquet': old['raw_sha256'],
             parent / 'results/BATCH500_TRACES.parquet': old['traces_sha256']}
    validation = load_json(parent / 'reports/AUTO_VALIDATION.json')
    require(validation['status'] == 'COMPLETE' and validation['signature'] == c['parent_signature'], 'Invalid patch assets')
    patches = {}
    for patch in c['patches']:
        method, epoch = patch.split('@')
        path = parent / 'training' / method / ('outer_' + str(int(epoch)).zfill(4) + '.png')
        files[path] = validation['patches'][patch]['png']
        patches[patch] = str(path)
    require(all(sha256(p) == h for p, h in files.items()), 'Frozen input checksum mismatch')
    rows = pd.read_parquet(parent / 'results/BATCH500_ROLLOUTS.parquet')
    subset = rows[rows.condition.isin(['clean', *c['patches']])]
    require(len(subset) == 320 and not subset.duplicated(['suite', 'task_id', 'state_index', 'condition']).any(),
            'Incomplete old control grid')
    require(set(subset.signature) == {c['parent_signature']}, 'Wrong old control signature')
    for suite in c['suites']:
        for task in range(10):
            for state in [40, 41]:
                pair = subset[(subset.suite == suite) & (subset.task_id == task) & (subset.state_index == state)]
                require(len(pair) == 4 and pair.rollout_seed.nunique() == 1, 'Control pairing/seed mismatch')
    return {'status': 'CPU_PREFLIGHT_PASS', 'signature': signature(), 'budget': budget(),
            'source_hashes': source_hashes(), 'operational_hashes': operational_hashes(),
            'runtime_sources': runtime_sources(), 'library_versions': library_versions(),
            'frozen_inputs': {str(p): h for p, h in files.items()}, 'patch_paths': patches,
            'parent_signature': c['parent_signature'], 'parent_source_files_verified': len(prepared_parent['source_code_hashes']),
            'models_loaded': False, 'cuda_initialized': False, 'slurm_submitted': False,
            'runtime_validation': 'PENDING_USER_SUBMISSION', 'created_unix': time.time()}


def prepared():
    receipt = load_json(ROOT / 'reports/PREPARATION.json')
    current = load_json(ROOT / 'reports/CPU_PREFLIGHT.json')
    require(current['status'] == 'CPU_PREFLIGHT_PASS', 'Latest CPU preflight failed; GPU stages are blocked')
    failure = ROOT / 'reports/PREPARE_FAILED.json'
    require(not failure.exists() or failure.stat().st_mtime_ns <= (ROOT / 'reports/PREPARATION.json').stat().st_mtime_ns,
            'Unresolved preparation failure; inspect reports/PREPARE_FAILED.json')
    require(receipt['signature'] == signature(), 'Pilot source/config changed: rerun CPU preflight')
    require(receipt['library_versions'] == library_versions(), 'Runtime library version drift')
    require(all(sha256(p) == h for p, h in receipt['frozen_inputs'].items()), 'Frozen inputs changed')
    old = parent_module('common')
    require(old.run_signature() == cfg()['parent_signature'], 'Frozen parent changed')
    old.prepared()
    return receipt

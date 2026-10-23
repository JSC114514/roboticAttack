"""CPU-safe contracts, provenance and monitoring for an isolated evaluation."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import resource
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PYTHON = '/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'


def require(ok, message):
    if not bool(ok): raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())
def config(): return read(ROOT / 'config.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(value, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name('.' + p.name + '.' + str(os.getpid()) + '.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    q.replace(p)


def dependencies():
    r = PROJECT / config()['runtime_run'] / 'code'
    paths = [r / n for n in ('model_runtime.py', 'readonly_model.py', 'common.py')]
    paths += [PROJECT / 'VLAAttacker/multiview' / n for n in ('openvla_oft_wrapper.py', 'openvla_oft_experiments.py')]
    paths += [PROJECT / 'research_tools/export_singleview_videos.py']
    official = Path(read(PROJECT / config()['runtime_run'] / 'config.json')['evaluation']['paths']['openvla_oft_root'])
    paths += [official / n for n in ('experiments/robot/openvla_utils.py', 'prismatic/extern/hf/modeling_prismatic.py',
                                    'prismatic/extern/hf/configuration_prismatic.py')]
    # Pin reset/sampling code without importing the simulator on login.
    b = Path('/home/caijiasheng25/Desktop/LIBERO/libero/libero/envs')
    paths += [b / n for n in ('bddl_base_domain.py', 'env_wrapper.py', 'regions/object_property_sampler.py')]
    return paths


@lru_cache(None)
def signature():
    paths = [ROOT / n for n in ('config.json', 'README.md', 'run.sh')]
    paths += [p for d in ('code', 'slurm', 'tests') for p in (ROOT / d).glob('*') if p.is_file()]
    paths += dependencies()
    return digest({str(p): sha(p) for p in sorted(paths)})


def complete(path, artifacts=True):
    p = Path(path)
    if not p.exists(): return False
    v = read(p)
    require(v.get('signature') == signature() and v.get('status') == 'COMPLETE', 'Invalid/stale receipt: ' + str(p))
    for name, expected in (v['artifacts'].items() if artifacts else []):
        q = (ROOT / name).resolve()
        require(q.is_relative_to(ROOT) and q.is_file() and sha(q) == expected, 'Changed artifact: ' + str(q))
    return True


def finish(path, files, **extra):
    atomic(dict(status='COMPLETE', signature=signature(), completed_unix=time.time(),
        artifacts={str(Path(p).relative_to(ROOT)): sha(p) for p in files}, **extra), path)


def manifest(): return read(ROOT / 'data/MANIFEST.json')


def verify():
    m = manifest(); require(m['signature'] == signature(), 'Prepared revision differs')
    for p, h in m['inputs'].items(): require(Path(p).is_file() and sha(p) == h, 'Changed input: ' + p)
    for p, st in m['model_file_stats'].items():
        s = Path(p).stat(); require([s.st_size, s.st_mtime_ns] == st, 'Changed model component: ' + p)
    return m


def episode_receipt(case, condition): return ROOT / 'episodes' / case['start'] / condition / 'COMPLETE.json'


def episode_files(receipt): return {Path(p).name: ROOT / p for p in read(receipt)['artifacts']}


@contextmanager
def lock(name):
    p = ROOT / 'reports/locks' / (name + '.lock'); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def resources():
    out = dict(process_rss_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               child_rss_hwm_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
               cuda_peak_allocated_bytes=0, cuda_peak_reserved_bytes=0)
    t = sys.modules.get('torch')
    if t is not None and t.cuda.is_initialized():
        out.update(cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),
                   cuda_peak_reserved_bytes=t.cuda.max_memory_reserved(), device=t.cuda.get_device_name(0))
    return out


class Monitor:
    def __init__(self, stage):
        self.stage = stage; self.data = {}; self.guard = threading.Lock()
        self.stop = threading.Event(); self.started = time.monotonic()
    def update(self, **values):
        with self.guard: self.data.update(values)
    def write(self):
        with self.guard: v = dict(self.data)
        elapsed = time.monotonic() - self.started
        done = v.get('completed', 0) - v.get('resumed', 0)
        if done > 0 and v.get('total'):
            v.update(throughput_per_s=done / elapsed, estimated_remaining_s=max(0, v['total'] - v['completed']) * elapsed / done)
        atomic({**v, 'stage': self.stage, 'heartbeat_unix': time.time(), 'elapsed_s': elapsed,
                'job_id': os.environ.get('SLURM_JOB_ID'), **resources()}, ROOT / 'reports/PROGRESS' / (self.stage + '.json'))
    def __enter__(self):
        def pulse():
            while not self.stop.wait(30): self.write()
        self.thread = threading.Thread(target=pulse, daemon=True); self.thread.start(); self.write(); return self
    def __exit__(self, kind, error, tb):
        self.stop.set(); self.thread.join(5)
        v = dict(status='FAILED' if kind else 'COMPLETE', error=repr(error) if kind else None,
                 traceback=''.join(traceback.format_exception(kind, error, tb)) if kind else None)
        self.update(**v); self.write()
        atomic({**v, 'stage': self.stage, 'signature': signature(), 'progress': dict(self.data),
                'job_id': os.environ.get('SLURM_JOB_ID'), 'elapsed_s': time.monotonic() - self.started,
                **resources()}, ROOT / 'reports/COSTS' / (self.stage + '_' + str(time.time_ns()) + '.json'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def runtime_config():
    c = config(); old = read(PROJECT / c['runtime_run'] / 'config.json')['evaluation']
    return {**old, 'checkpoints': {s: c['model_dir'] for s in c['suites']},
            'env_image_resolution': 256, 'policy_image_resolution': 224,
            'num_steps_wait': c['num_steps_wait'], 'num_open_loop_steps': c['chunk'],
            'max_steps_by_suite': c['max_steps']}


@lru_cache(None)
def runtime():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'),
            'Model/simulator work requires a user-submitted GPU allocation')
    p = PROJECT / config()['runtime_run'] / 'code'
    sys.path.insert(0, str(p))
    return module('joint_native_runtime', p / 'model_runtime.py')

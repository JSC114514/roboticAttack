"""Independent output roots and sealed, read-only historical adapters."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import fcntl
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
SOURCE = PROJECT / 'research_runs/wrist_spatial_approach_pilot_v1'
BASE = PROJECT / 'research_runs/wrist_spatial_relation_expand_v1'
for directory in (SOURCE, PROJECT/'research_runs/wrist_spatial_failure_audit_v1',
                  PROJECT/'research_runs/wrist_spatial_relation_rescue_v1'):
    sys.path.insert(1, str(directory/'code'))
import sr_common as source
native = source.native
require, read, sha = native.require, native.read_json, native.sha
atomic, atomic_torch = native.atomic_json, native.atomic_torch

def config(): return read(ROOT/'config.json')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

@lru_cache(None)
def signature():
    require(source.signature() == config()['source_signature'], 'Historical source revision changed')
    paths = [ROOT/'config.json', ROOT/'README.md', ROOT/'run.sh']
    paths += [p for d in ('code','slurm','tests') for p in (ROOT/d).glob('*') if p.is_file()]
    # Include directly reused renderer, camera, video, physics and evaluation code.
    paths += [PROJECT/p for p in (
        'research_runs/wrist_spatial_approach_pilot_v1/code/sr_geometry.py',
        'research_runs/wrist_spatial_failure_audit_v1/code/audit_replay.py',
        'research_runs/wrist_spatial_failure_audit_v1/code/audit_math.py',
        'research_runs/wrist_spatial_relation_rescue_v1/code/rr_logger.py',
        'research_runs/wrist_spatial_relation_shuffle_v1_fix1/code/rfix_evaluate.py',
        'research_tools/export_singleview_videos.py')]
    reset_sources=[Path('/home/caijiasheng25/Desktop/LIBERO/libero/libero/envs')/p for p in (
        'bddl_base_domain.py','env_wrapper.py','regions/object_property_sampler.py')]
    return native.digest({'files': {str(p.relative_to(PROJECT)): sha(p) for p in sorted(paths)},
                          'installed_reset_sources':{str(p):sha(p) for p in reset_sources},
                          'source': source.signature()})

def complete(path):
    path = Path(path)
    if not path.exists(): return False
    value = read(path)
    require(value['signature'] == signature() and value['status'] == 'COMPLETE', 'Stale receipt: '+str(path))
    require(all((ROOT/p).is_file() and sha(ROOT/p) == h for p,h in value['artifacts'].items()), 'Changed artifact: '+str(path))
    return True

def finish(path, files, **fields):
    atomic({'status':'COMPLETE', 'signature':signature(), 'completed_unix':time.time(),
            'artifacts':{str(Path(p).relative_to(ROOT)):sha(p) for p in files}, **fields}, path)

def verify_external(root, receipt):
    value = read(receipt)
    expected = read(root/'data/MANIFEST.json')['signature']
    require(value['status']=='COMPLETE' and value['signature']==expected, 'Invalid external receipt')
    require(all((root/p).is_file() and sha(root/p)==h for p,h in value['artifacts'].items()), 'Changed external artifacts')
    return value

@contextmanager
def lock(name):
    path = ROOT/'reports/locks'/(name+'.lock'); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def resources():
    value = {'process_rss_hwm_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             'child_rss_hwm_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
             'cuda_peak_allocated_bytes':0, 'cuda_peak_reserved_bytes':0}
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t=sys.modules['torch']; value.update(cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=t.cuda.max_memory_reserved(), device=t.cuda.get_device_name(0))
    return value

class Monitor:
    def __init__(self, stage):
        self.stage=stage; self.data={}; self.guard=threading.Lock(); self.stop=threading.Event(); self.started=time.monotonic()
    def update(self, **values):
        with self.guard: self.data.update(values)
    def write(self):
        with self.guard: value=dict(self.data)
        elapsed=time.monotonic()-self.started
        done=value.get('completed',0)-value.get('resume_completed',0)
        if done>0 and value.get('total'):
            value.update(throughput_per_s=done/elapsed, estimated_remaining_s=(value['total']-value['completed'])*elapsed/done)
        atomic({**value,'stage':self.stage,'heartbeat_unix':time.time(),'elapsed_s':elapsed,
                'job_id':os.environ.get('SLURM_JOB_ID'),**resources()},ROOT/'reports/PROGRESS'/(self.stage+'.json'))
    def __enter__(self):
        def pulse():
            while not self.stop.wait(30): self.write()
        self.thread=threading.Thread(target=pulse,daemon=True);self.thread.start();self.write();return self
    def __exit__(self, typ, error, tb):
        self.stop.set();self.thread.join(5)
        scientific_stop=isinstance(error,SystemExit) and error.code==3
        result={'status':'STOPPED' if scientific_stop else 'FAILED' if typ else 'COMPLETE','error':repr(error) if typ else None,
                'traceback':''.join(traceback.format_exception(typ,error,tb)) if typ else None}
        self.update(**result);self.write()
        atomic({**result,'stage':self.stage,'signature':signature(),'progress':dict(self.data),'elapsed_s':time.monotonic()-self.started,**resources()},
               ROOT/'reports/COSTS'/f'{self.stage}_{time.time_ns()}.json')

def manifest(): return read(ROOT/'data/MANIFEST.json')
def verify():
    value=manifest();require(value['signature']==signature(),'Preparation revision changed')
    for p,h in value['inputs'].items(): require(sha(p)==h,'Changed input: '+p)
    for p,expected in value['model_file_stats'].items():
        st=Path(p).stat();require([st.st_size,st.st_mtime_ns]==expected,'Model changed: '+p)
    return value

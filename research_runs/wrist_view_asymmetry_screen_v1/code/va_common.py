"""Isolated, signed outputs. Importing this module never imports a simulator."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import fcntl
import hashlib
import json
import os
import resource
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
FAIR = PROJECT / 'research_runs/wrist_spatial_relation_fair_v1'
SOURCE = PROJECT / 'research_runs/wrist_spatial_approach_pilot_v1'
PYTHON = '/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'

def require(ok, message):
    if not bool(ok): raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text())
def config(): return read(ROOT/'config.json')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def atomic(value, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)

def save_torch(value, path):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    torch.save(value, tmp); tmp.replace(p)

@lru_cache(None)
def signature():
    files = [ROOT/'config.json', ROOT/'README.md', ROOT/'run.sh']
    files += [p for d in ('code','slurm','tests') for p in (ROOT/d).glob('*') if p.is_file()]
    files += [PROJECT/p for p in (
        'research_runs/wrist_spatial_approach_pilot_v1/code/sr_geometry.py',
        'research_runs/wrist_spatial_failure_audit_v1/code/audit_replay.py',
        'research_runs/wrist_spatial_failure_audit_v1/code/audit_math.py',
        'research_runs/wrist_spatial_relation_fair_v1/code/fc_common.py',
        'research_tools/export_singleview_videos.py')]
    value = {str(p.relative_to(PROJECT)):sha(p) for p in sorted(files)}
    value['fair_signature'] = config()['source_signature']
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def complete(path):
    p = Path(path)
    if not p.exists(): return False
    v = read(p)
    require(v['signature']==signature() and v['status']=='COMPLETE', 'Stale completion: '+str(p))
    require(all((ROOT/k).is_file() and sha(ROOT/k)==h for k,h in v['artifacts'].items()), 'Changed artifact: '+str(p))
    return True

def finish(path, files, **extra):
    atomic(dict(status='COMPLETE',signature=signature(),completed_unix=time.time(),
                artifacts={str(Path(p).relative_to(ROOT)):sha(p) for p in files},**extra),path)

def manifest(): return read(ROOT/'data/MANIFEST.json')

def verify():
    v = manifest(); require(v['signature']==signature(), 'Prepared revision differs')
    for p,h in v['inputs'].items(): require(sha(p)==h, 'Input changed: '+p)
    for p,expected in v['model_file_stats'].items():
        s=Path(p).stat(); require([s.st_size,s.st_mtime_ns]==expected, 'Model changed: '+p)
    return v

@lru_cache(None)
def adapters():
    # Historical modules are imported read-only; no historical stage is invoked.
    sys.path.insert(1,str(FAIR/'code'))
    import fc_common
    require(fc_common.signature()==config()['source_signature'], 'Historical adapter signature differs')
    return fc_common.source, fc_common.native

def cases(): return manifest()['cases']
def branch_receipt(case, condition): return ROOT/'episodes'/case['episode_id']/condition/'COMPLETE.json'
def reference_path(case): return ROOT/'data/references'/case['episode_id']/'reference.pt'

def branch_files(receipt):
    return {Path(p).name:ROOT/p for p in read(receipt)['artifacts']}

@contextmanager
def lock(name):
    p=ROOT/'reports/locks'/(name+'.lock'); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); yield

def resources():
    out={'rss_hwm_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         'cuda_peak_allocated_bytes':0,'cuda_peak_reserved_bytes':0}
    t=sys.modules.get('torch')
    if t is not None and t.cuda.is_initialized():
        out.update(cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),
                   cuda_peak_reserved_bytes=t.cuda.max_memory_reserved(),device=t.cuda.get_device_name(0))
    return out

class Monitor:
    def __init__(self,stage):
        self.stage=stage;self.data={};self.start=time.monotonic();self.guard=threading.Lock();self.stop=threading.Event()
        self.cost_path=ROOT/f'reports/COSTS/{stage}_{time.time_ns()}.json'
    def update(self,**v):
        with self.guard: self.data.update(v)
    def write(self):
        with self.guard: v=dict(self.data)
        elapsed=time.monotonic()-self.start; done=v.get('completed',0)-v.get('resume_completed',0)
        if done>0 and v.get('total'):
            v.update(throughput_per_s=done/elapsed,eta_s=max(0,(v['total']-v['completed'])*elapsed/done))
        atomic(dict(v,stage=self.stage,heartbeat_unix=time.time(),elapsed_s=elapsed,**resources()),ROOT/f'reports/PROGRESS/{self.stage}.json')
    def __enter__(self):
        atomic(dict(status='STARTED',stage=self.stage,signature=signature(),job_id=os.environ.get('SLURM_JOB_ID'),
                    started_unix=time.time(),elapsed_s=0.,**resources()),self.cost_path)
        def pulse():
            while not self.stop.wait(30): self.write()
        self.thread=threading.Thread(target=pulse,daemon=True); self.thread.start(); self.write(); return self
    def __exit__(self,kind,error,tb):
        self.stop.set(); self.thread.join(5)
        result=dict(status='FAILED' if kind else 'COMPLETE',error=repr(error) if kind else None,
                    traceback=''.join(traceback.format_exception(kind,error,tb)) if kind else None)
        self.update(**result); self.write()
        atomic(dict(result,stage=self.stage,signature=signature(),job_id=os.environ.get('SLURM_JOB_ID'),
                    elapsed_s=time.monotonic()-self.start,progress=self.data,**resources()),self.cost_path)

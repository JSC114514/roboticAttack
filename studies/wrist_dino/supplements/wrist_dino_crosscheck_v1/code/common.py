# 发布阅读说明：CPU安全I/O、哈希、冻结配置与运行资源记录。原始来源路径保留在配置快照；发布入口用显式run-root选择工作区。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""CPU-safe state, atomic results, provenance and 30-second worker heartbeat.

The Slurm controller is invoked by the active Codex session. This heartbeat
does not schedule work or call a model API. Historical experiment files are
read-only dependencies; every new receipt/result/video belongs to ROOT.
"""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
SOURCE=PROJECT/'supplements/wrist_dino_sensitivity_v1'
FACTORIAL=PROJECT/'supplements/wrist_dino_factorial_v1'
PYTHON='/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'
PI_PYTHON='/home/caijiasheng25/openpi/.venv/bin/python'
JAX_DEVICE=None  # Set by the main thread only after native policy loading.

def require(value,message):
    if not bool(value):raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text())

def atomic(value,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name('.'+p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def frozen():return read(ROOT/'configs/FROZEN.json')

def resources():
    a=dict(rss_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    torch=sys.modules.get('torch')
    cuda=getattr(torch,'cuda',None)
    if cuda is not None and callable(getattr(cuda,'is_initialized',None)) and cuda.is_initialized():
        a.update(gpu=torch.cuda.get_device_name(0),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if JAX_DEVICE is not None:
        try:
            device=JAX_DEVICE;stats=device.memory_stats() or {}
            a.update(gpu=device.device_kind,jax_memory_stats=stats)
        except Exception as e:a['jax_memory_error']=repr(e)
    return a

def done(path,files=(),**kwargs):
    atomic(dict(status='DONE',signature=sha(ROOT/'configs/FROZEN.json'),completed_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),artifacts={str(Path(p).relative_to(ROOT)):sha(p) for p in files},**kwargs),path)

def complete(path):
    p=Path(path)
    if not p.is_file():return False
    a=read(p);require(a['status']=='DONE' and a['signature']==sha(ROOT/'configs/FROZEN.json'),'Changed completion identity: '+str(p))
    for name,value in a.get('artifacts',{}).items():require((ROOT/name).is_file() and sha(ROOT/name)==value,'Completed artifact changed: '+name)
    return True

class Monitor:
    """Write progress and measured costs without polling Slurm in a worker."""
    def __init__(self,name):self.name=name;self.values={};self.stop=threading.Event();self.start=time.monotonic()
    def update(self,**values):self.values.update(values)
    def write(self):
        elapsed=time.monotonic()-self.start;v=dict(self.values)
        if v.get('completed',0)>v.get('resumed',0) and v.get('total'):
            rate=(v['completed']-v.get('resumed',0))/elapsed;v.update(rate_per_s=rate,remaining_s=max(0,v['total']-v['completed'])/rate)
        atomic(dict(name=self.name,time_unix=time.time(),elapsed_s=elapsed,job_id=os.environ.get('SLURM_JOB_ID'),**v,**resources()),ROOT/'reports/progress'/f'{self.name}.json')
    def __enter__(self):
        def loop():
            while not self.stop.wait(30):self.write()
        atomic({str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'run.sh',*sorted((ROOT/'code').glob('*.py'))]},ROOT/'reports/source_snapshots'/f'{self.name}-{os.environ.get("SLURM_JOB_ID","cpu")}.json')
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();self.write();return self
    def __exit__(self,kind,error,tb):
        self.stop.set();self.thread.join();self.update(status='FAILED' if kind else 'DONE',error=repr(error) if kind else None);self.write()
        atomic(dict(name=self.name,status='FAILED' if kind else 'DONE',error=repr(error) if kind else None,traceback=''.join(traceback.format_exception(kind,error,tb)) if kind else None,elapsed_s=time.monotonic()-self.start,job_id=os.environ.get('SLURM_JOB_ID'),**resources()),ROOT/'reports/costs'/f'{self.name}-{time.time_ns()}.json')

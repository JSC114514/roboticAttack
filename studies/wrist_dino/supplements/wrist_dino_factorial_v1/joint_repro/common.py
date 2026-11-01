# 发布阅读说明：CPU安全I/O、哈希、冻结配置与运行资源记录。原始来源路径保留在配置快照；发布入口用显式run-root选择工作区。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Small shared I/O, provenance, deterministic seeds and resource reporting.

This module is CPU-safe. It never imports TensorFlow, Torch, a model or MuJoCo
on import. Workers write only inside this experiment directory.
"""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import json
import os
import resource
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
LAUNCH_SOURCES=None
PYTHON='/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'

def require(ok,message):
    if not bool(ok):raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text())
def config():return read(ROOT/'configs/PROTOCOL.json')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic(value,path):
    """Atomic JSON replacement; reject NaN rather than writing fake valid metrics."""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def save_npz(path,**arrays):
    import numpy as np
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(p)

def save_torch(value,path):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp')
    torch.save(value,tmp);tmp.replace(p)

def source_hashes():
    c=config();base=Path(c['project']);paths=list((ROOT/'joint_repro').glob('*.py'))+[ROOT/'configs/PROTOCOL.json']
    paths += [ROOT/'configs/ABLATION.json']
    paths += [base/p for p in ['VLAAttacker/white_patch/appply_random_transform.py','VLAAttacker/white_patch/UADA_OFT_SOURCE_COPY.py','VLAAttacker/multiview/openvla_oft_wrapper.py','VLAAttacker/multiview/openvla_oft_experiments.py']]
    return {str(p):sha(p) for p in sorted(paths)}

def finish(path,files,**extra):
    atomic(dict(status='DONE',protocol_sha256=sha(ROOT/'configs/PROTOCOL.json'),supplement_sha256=sha(ROOT/'configs/ABLATION.json'),source_hashes=LAUNCH_SOURCES if LAUNCH_SOURCES is not None else source_hashes(),
        job_id=os.environ.get('SLURM_JOB_ID'),completed_unix=time.time(),
        artifacts={str(Path(p).relative_to(ROOT)):sha(p) for p in files},**extra),path)

def complete(path):
    p=Path(path)
    if not p.is_file():return False
    r=read(p);require(r.get('status')=='DONE' and r['protocol_sha256']==sha(ROOT/'configs/PROTOCOL.json'),'Stale protocol receipt: '+str(p))
    if 'supplement_sha256' in r:require(r['supplement_sha256']==sha(ROOT/'configs/ABLATION.json'),'Changed frozen supplement: '+str(p))
    for name,h in r['artifacts'].items():require((ROOT/name).is_file() and sha(ROOT/name)==h,'Changed completed artifact: '+name)
    return True

def resources():
    r=dict(rss_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    t=sys.modules.get('torch')
    if t is not None and t.cuda.is_initialized():r.update(cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=t.cuda.max_memory_reserved(),gpu=t.cuda.get_device_name(0))
    return r

class Monitor:
    """30-second file heartbeat; the agent's full audit remains every30minutes."""
    def __init__(self,name):self.name=name;self.values={};self.start=time.monotonic();self.stop=threading.Event()
    def update(self,**values):self.values.update(values)
    def write(self):
        elapsed=time.monotonic()-self.start;v=dict(self.values)
        if v.get('completed',0)>v.get('resumed',0) and v.get('total'):
            speed=(v['completed']-v.get('resumed',0))/elapsed;v.update(rate_per_s=speed,remaining_s=max(0,v['total']-v['completed'])/speed)
        atomic(dict(**v,name=self.name,time_unix=time.time(),elapsed_s=elapsed,job_id=os.environ.get('SLURM_JOB_ID'),**resources()),ROOT/'reports/progress'/f'{self.name}.json')
    def __enter__(self):
        def loop():
            while not self.stop.wait(30):self.write()
        global LAUNCH_SOURCES
        LAUNCH_SOURCES=source_hashes()
        atomic(LAUNCH_SOURCES,ROOT/'reports/source_snapshots'/f'{self.name}-{os.environ.get("SLURM_JOB_ID","local")}.json')
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();self.write();return self
    def __exit__(self,kind,error,tb):
        self.stop.set();self.thread.join();self.update(status='FAILED' if kind else 'DONE',error=str(error) if kind else None);self.write()
        atomic(dict(name=self.name,status='FAILED' if kind else 'DONE',error=repr(error) if kind else None,
            traceback=''.join(traceback.format_exception(kind,error,tb)) if kind else None,
            elapsed_s=time.monotonic()-self.start,job_id=os.environ.get('SLURM_JOB_ID'),**resources()),ROOT/'reports/costs'/f'{self.name}_{time.time_ns()}.json')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def task_map():
    import ast
    p=Path(config()['libero_root'])/'libero/libero/benchmark/libero_suite_task_map.py'
    tree=ast.parse(p.read_text());return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='libero_task_map' for t in n.targets)))

def setup_paths():
    c=config();p=Path(c['project']);sys.path[:0]=[str(p),str(p/'VLAAttacker'),str(p/'scripts')]
    return p

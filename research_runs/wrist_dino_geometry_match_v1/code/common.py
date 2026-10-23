"""Isolated experiment protocol, native runtime and provenance."""
from __future__ import annotations
import hashlib, importlib, importlib.util, json, os, resource, sys, time, traceback, threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
PYTHON="/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python"

def cfg():
    return read_json(ROOT/"config.json")

def origin_run():
    return PROJECT/cfg()["origin_run"]

def runtime_run():
    return PROJECT/cfg()["runtime_run"]

def require(value, message):
    if not bool(value):
        raise RuntimeError(message)

def read_json(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)

def atomic_torch(value, path):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary); os.replace(temporary, path)

def atomic_parquet(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)

def resources():
    value = {"process_rss_hwm_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
             "process_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime)}
    try:
        import torch
        if torch.cuda.is_available():
            value.update(cuda_device_name=torch.cuda.get_device_name(0),
                         cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                         cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    except Exception:
        pass
    return value

@contextmanager
def cost_attempt(stage, *, suite=None, path=None):
    started = time.monotonic()
    path = Path(path) if path is not None else ROOT/f"reports/COSTS/{stage}_{time.time_ns()}.json"
    identity = {"stage":stage, "suite":suite, "job_id":os.environ.get("SLURM_JOB_ID"),
                "pid":os.getpid(), "measurement_scope":"this_process", "signature":signature(),
                "submitted_signature":os.environ.get('GEOM_EXPECTED_SIGNATURE')}
    atomic_json({**identity, "status":"STARTED", "started_unix":time.time()}, path)
    try:
        yield path
    except BaseException as error:
        atomic_json({**identity, "status":"FAILED", "error":repr(error),
                     "traceback":traceback.format_exc(), "elapsed_s":time.monotonic()-started,
                     **resources()}, path)
        raise
    else:
        atomic_json({**identity, "status":"COMPLETE", "elapsed_s":time.monotonic()-started,
                     **resources()}, path)

def finish(path, artifacts, **extra):
    artifacts = [Path(p) for p in artifacts]
    atomic_json({"status": "COMPLETE", "signature": signature(), "completed_unix": time.time(),
                 "artifacts": {str(p.relative_to(ROOT)): sha(p) for p in artifacts}, **extra}, path)

@lru_cache(maxsize=1)
def runtime_config():
    value = read_json(runtime_run()/"config.json")["evaluation"]
    return {**value, "env_image_resolution": 256, "policy_image_resolution": 224}

@lru_cache(maxsize=1)
def runtime_module():
    code = runtime_run()/"code"
    if str(code) not in sys.path: sys.path.insert(1, str(code))
    spec = importlib.util.spec_from_file_location("relation_native_runtime", code/"model_runtime.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def load_wrapper(suite):
    wrapper = runtime_module().load_wrapper(runtime_config(), suite)
    wrapper.freeze_model()
    for module in (wrapper.vla, wrapper.action_head, wrapper.proprio_projector, wrapper.noisy_action_projector):
        if module is not None:
            require(not module.training and all(not p.requires_grad for p in module.parameters()), "Victim model is not frozen/eval")
    return wrapper

def runtime_dependency_preflight():
    runtime_module()
    loader = importlib.import_module("readonly_model")
    require(Path(loader.__file__).resolve() == (runtime_run()/"code/readonly_model.py").resolve(),
            "Unexpected readonly checkpoint loader")
    config = runtime_config()
    checked = {}
    for suite in cfg()["suites"]:
        checked[suite] = loader.verify_checkpoint(config["checkpoints"][suite], config["paths"]["openvla_oft_root"])
    return {"readonly_loader_import":"PASS", "checkpoint_logic":checked}

def save_png(patch, path):
    import numpy as np
    from PIL import Image
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    value = patch.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    Image.fromarray(np.asarray(value)).save(temporary, format="PNG"); os.replace(temporary, path)

def load_png(path, device="cpu"):
    import numpy as np, torch
    from PIL import Image
    with Image.open(path) as image:
        require(image.mode == "RGB" and image.size == (50, 50), "Expected one RGB 50x50 PNG")
        value = np.array(image, copy=True)
    return torch.as_tensor(value, device=device).permute(2, 0, 1).float().div(255)

def paste_numpy(observation, patch):
    import numpy as np
    x, y = cfg()["patch_xy"]
    output = {k: np.array(v, copy=True) for k, v in observation.items()}
    wrist = output["wrist_image"]
    require(wrist.shape == (224, 224, 3), "Unexpected wrist image shape")
    wrist[y:y+50, x:x+50] = np.asarray(patch, dtype=np.uint8)
    return output

def decode_action_chunk(value, context="saved trace"):
    """Decode both numeric arrays and Parquet's object array of slot arrays."""
    import numpy as np
    prefix="Invalid saved action chunk ("+context+"): "
    try:
        with np.errstate(over="ignore",invalid="ignore"):
            slots=[np.asarray(slot,dtype=np.float32) for slot in value]
    except (TypeError,ValueError,OverflowError) as error:
        raise RuntimeError(prefix+"cannot decode numeric slots") from error
    require(len(slots)==8 and all(slot.shape==(7,) for slot in slots),
            prefix+"expected exactly 8 slots of 7 values")
    actions=np.stack(slots,axis=0)
    require(np.isfinite(actions).all(),prefix+"non-finite float32 action")
    return actions

load_json=read_json
sha256_file=sha

@lru_cache(maxsize=1)
def signature():
    files=[ROOT/'config.json',ROOT/'README.md',ROOT/'PLAN.md',ROOT/'run.sh']
    for folder in ('code','slurm','tests'):
        files.extend(p for p in (ROOT/folder).glob('*') if p.is_file())
    origin=origin_run()
    files += [origin/'config.json',origin/'data/INDEX.json',origin/'reports/PREFLIGHT.json',
              origin/'reports/CACHE_REUSE.json',origin/'training/translation/COMPLETE.json',
              origin/'training/translation/epoch_00.png',origin/'training/translation/epoch_50.png',
              PROJECT/cfg()['uada_png'],runtime_run()/'config.json']
    files += list((runtime_run()/'code').glob('*.py'))
    files += [PROJECT/'VLAAttacker/multiview/openvla_oft_wrapper.py',
              PROJECT/'VLAAttacker/multiview/openvla_oft_experiments.py']
    return digest({str(p.resolve()):sha(p) for p in sorted(files)})

def complete(path):
    path=Path(path)
    if not path.exists():return False
    value=read_json(path)
    require(value.get('signature')==signature(),'Stale experiment receipt: '+str(path))
    if value.get('status')!='COMPLETE':return False
    for relative,fingerprint in value.get('artifacts',{}).items():
        require((ROOT/relative).is_file() and sha(ROOT/relative)==fingerprint,'Artifact changed: '+relative)
    return True

def initial_patch(device='cpu',seed=None):
    import torch
    generator=torch.Generator(device='cpu').manual_seed(cfg()['seed'] if seed is None else seed)
    return torch.rand(tuple(cfg()['patch_shape']),generator=generator,dtype=torch.float32).to(device)

def require_gpu(stage):
    import torch
    require(os.environ.get('SLURM_JOB_ID'),stage+' requires a user-submitted Slurm allocation')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one allocated GPU required')
    name=torch.cuda.get_device_name(0)
    require('4090' in name,'This experiment requires one 4090, got '+name)
    return name

class Monitor:
    """Independent per-worker heartbeat; completed work never shares a progress file."""
    def __init__(self,key):
        self.key=key;self.started=time.monotonic();self.fields={};self.lock=threading.Lock();self.io_lock=threading.Lock()
        self.event=threading.Event();self.path=ROOT/f'reports/PROGRESS/{key}.json'
    def update(self,**fields):
        with self.lock:self.fields.update(fields)
        self.flush()
    def flush(self):
        with self.io_lock:self._flush()
    def _flush(self):
        with self.lock:fields=dict(self.fields)
        elapsed=time.monotonic()-self.started
        done,total=fields.get('completed',0),fields.get('total',0)
        new_done=done-fields.get('resume_completed',0)
        estimate=elapsed/new_done*(total-done) if new_done>0 and total>=done else None
        atomic_json({'signature':signature(),'worker':self.key,'job_id':os.environ.get('SLURM_JOB_ID'),
                     'heartbeat_unix':time.time(),'elapsed_s':elapsed,'estimated_remaining_s':estimate,
                     **resources(),**fields},self.path)
    def _loop(self):
        while not self.event.wait(30):self.flush()
    def __enter__(self):
        self.update(status='RUNNING',phase='starting')
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start();return self
    def __exit__(self,kind,error,tb):
        self.event.set();self.thread.join();self.update(status='FAILED' if kind else 'COMPLETE',error=repr(error) if error else None)

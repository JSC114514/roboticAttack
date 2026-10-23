"""Immutable protocol, source validation, atomic outputs, and runtime loading."""
from __future__ import annotations
import hashlib, importlib, importlib.util, json, os, resource, sys, time, traceback
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PYTHON = "/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python"

def require(value, message):
    if not bool(value):
        raise RuntimeError(message)

def read_json(path):
    return json.loads(Path(path).read_text())

def cfg():
    value = read_json(ROOT / "config.json")
    require(value["methods"] == ["translation", "bearing"], "Method grid changed")
    require(value["patch_shape"] == [3, 50, 50] and value["epochs"] == 50, "Patch budget changed")
    require(value["updates_per_epoch"] == 960, "Training cohort/update budget changed")
    return value

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# The reused readonly checkpoint loader imports these historical API names.
load_json = read_json
sha256_file = sha

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def source_run():
    return PROJECT / cfg()["source_run"]

def runtime_run():
    return PROJECT / cfg()["runtime_run"]

def source_files():
    files = [ROOT / "config.json", ROOT / "PLAN.md", ROOT / "README.md", ROOT / "run.sh"]
    for folder in ("code", "slurm", "tests"):
        files.extend(p for p in (ROOT / folder).glob("*") if p.is_file())
    external = [source_run()/"results/LONG4_ROLLOUTS.parquet",
                source_run()/"results/LONG4_TRACES.parquet",
                PROJECT/cfg()["uada_png"], runtime_run()/"config.json",
                runtime_run()/"code/model_runtime.py", runtime_run()/"code/preprocessing.py",
                runtime_run()/"code/readonly_model.py",
                PROJECT/"VLAAttacker/multiview/openvla_oft_wrapper.py",
                PROJECT/"VLAAttacker/multiview/openvla_oft_experiments.py"]
    runtime = read_json(runtime_run()/"config.json")["evaluation"]
    bddl = Path(runtime["paths"]["libero_root"])/"libero/libero/bddl_files"
    external.extend(p for suite in cfg()["suites"] for p in sorted((bddl/suite).glob("*.bddl")))
    return {str(p.resolve()): sha(p) for p in sorted(files + external) if p.is_file()}

@lru_cache(maxsize=1)
def signature():
    return digest({"protocol": source_files()})

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
                "pid":os.getpid(), "measurement_scope":"this_process", "signature":signature()}
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

def complete(path):
    path = Path(path)
    if not path.exists(): return False
    value = read_json(path)
    require(value.get("signature") == signature() or reusable_cache_receipt(path,value),
            "Existing artifact belongs to another protocol: " + str(path))
    if value.get("status") != "COMPLETE": return False
    for relative, expected in value.get("artifacts", {}).items():
        target = ROOT / relative
        require(target.is_file() and sha(target) == expected, "Completed artifact changed: " + relative)
    return True

def reusable_cache_receipt(path,value):
    """Only explicitly audited, byte-identical cache receipts may cross revisions."""
    try: relative=Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:return False
    if not relative.startswith("data/cache/") or not Path(relative).name.startswith("task_"):return False
    manifest=ROOT/"reports/CACHE_REUSE.json"
    if not manifest.is_file():return False
    audit=read_json(manifest)
    if audit.get("target_signature")!=signature():return False
    entry=audit.get("receipts",{}).get(relative,{})
    return (entry.get("origin_signature")==value.get("signature") and entry.get("sha256")==sha(path))

def cache_shard_origin_matches(shard,receipt):
    require(complete(receipt),"Cache receipt is incomplete")
    return shard.get("signature")==read_json(receipt)["signature"]

def progress(stage, **fields):
    atomic_json({"stage": stage, "signature": signature(), "updated_unix": time.time(), **fields}, ROOT/"reports/PROGRESS.json")

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

def require_gpu(stage):
    import torch
    require(os.environ.get("SLURM_JOB_ID"), stage + " requires a user-submitted Slurm allocation")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one allocated GPU required")
    name = torch.cuda.get_device_name(0)
    stage_key="validate" if stage=="validation" else stage
    resource=cfg()["resources"].get(stage_key,{})
    require(resource.get("gpus")==1,"No single-GPU allocation configured for "+stage)
    expected=resource["gpu_type"]
    require(expected.lower() in name.lower(), "This protocol requests one GPU of type "+expected+", got "+name)
    return name

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

def initial_patch(device="cpu"):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(cfg()["seed"])
    return torch.rand(tuple(cfg()["patch_shape"]), generator=generator, dtype=torch.float32).to(device)

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

def validate_trace_actions(traces):
    """CPU-only validation of every source chunk before a GPU model is loaded."""
    checked=0
    for row in traces.itertuples(index=False):
        context=", ".join(f"{key}={getattr(row,key,'?')}" for key in
                          ("suite","task_id","state_index","decision_block"))
        decode_action_chunk(row.predicted_environment_action_chunk,context)
        try: executed=int(row.executed_slots)
        except (TypeError,ValueError,OverflowError) as error:
            raise RuntimeError("Invalid executed_slots ("+context+")") from error
        require(executed==row.executed_slots and 1<=executed<=8,
                "Invalid executed_slots ("+context+"): expected integer in [1,8]")
        checked+=1
    return {"status":"PASS","chunks_checked":checked,"shape":[8,7],"dtype":"float32",
            "finite":True,"slot_order_preserved":True,"executed_slots_range":[1,8]}

def preflight():
    started=time.monotonic()
    import pandas as pd
    from PIL import Image
    c = cfg(); suites = c["suites"]
    dependencies = runtime_dependency_preflight()
    require(set(c["task_splits"]) == set(suites), "Suite split map incomplete")
    for suite in suites:
        split = c["task_splits"][suite]
        flat = split["train"] + split["validation"] + split["heldout"]
        require(len(split["train"]) == 6 and len(split["validation"]) == len(split["heldout"]) == 2,
                "Expected 6/2/2 tasks per suite")
        require(sorted(flat) == list(range(10)) and len(set(flat)) == 10, "Task splits overlap or omit tasks")
    uada = PROJECT/c["uada_png"]
    with Image.open(uada) as image:
        require(image.mode == "RGB" and image.size == (50, 50), "Frozen UADA reference is not RGB 50x50")
    rollouts = pd.read_parquet(source_run()/"results/LONG4_ROLLOUTS.parquet",filters=[("condition","==","clean")])
    traces = pd.read_parquet(source_run()/"results/LONG4_TRACES.parquet",filters=[("condition","==","clean")])
    keys = ["suite", "task_id", "state_index", "rollout_seed"]
    require(len(rollouts) == 400 and not rollouts.duplicated(keys).any(), "Clean rollout grid changed")
    require(len(traces) == 8062 and set(rollouts.suite) == set(suites), "Clean trace source changed")
    action_validation=validate_trace_actions(traces)
    rows = []
    for suite in suites:
        for split, tasks in c["task_splits"][suite].items():
            states = c["trajectory_states"][split]
            for task in tasks:
                for state in states:
                    episode = rollouts.loc[(rollouts.suite == suite) & (rollouts.task_id == task) & (rollouts.state_index == state)]
                    require(len(episode) == 1, "Missing source episode")
                    seed = int(episode.iloc[0].rollout_seed)
                    current = traces.loc[(traces.suite == suite) & (traces.task_id == task) &
                                         (traces.state_index == state) & (traces.rollout_seed == seed)].sort_values("decision_block")
                    require(len(current) >= 9 and current.decision_block.tolist() == list(range(len(current))), "Invalid source decision sequence")
                    indices = [round(f * (len(current)-1)) for f in c["frame_fractions"]]
                    require(len(set(indices)) == 5, "Progress sampling produced duplicate frames")
                    require(min(indices)>0, "Selected frames require a preceding native camera sampling step")
                    episode_id = f"{suite}_t{task:02d}_s{state:02d}"
                    for rank, decision in enumerate(indices):
                        rows.append({"state_id": f"{episode_id}_d{decision:03d}", "episode_id": episode_id,
                                     "suite": suite, "task_id": int(task), "state_index": int(state),
                                     "rollout_seed": seed, "split": split, "frame_rank": rank,
                                     "decision_block": int(decision), "decision_count": int(len(current)),
                                     "normalized_progress": float(decision/(len(current)-1))})
    require(len(rows) == 1280 and len({r["episode_id"] for r in rows}) == 256, "Cohort size changed")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "heldout")}
    require(counts == {"train": 960, "validation": 160, "heldout": 160}, "Split frame counts changed")
    # Preserve superseded preparation/submission evidence before writing new receipts.
    previous = ROOT/"reports/PREFLIGHT.json"
    if previous.exists() and read_json(previous).get("signature") != signature():
        old_signature = read_json(previous)["signature"]
        archive = ROOT/"reports/REVISIONS"/old_signature
        for name in ("data/INDEX.json", "reports/PREFLIGHT.json", "reports/PREPARE_COMPLETE.json", "reports/SUBMISSION.json",
                     "reports/LOCAL_VALIDATION.json","reports/NATIVE_PATH_CHECK.json","reports/checks/initial.png"):
            original = ROOT/name; target = archive/name
            if original.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
    path = ROOT/"data/INDEX.json"; atomic_json(rows, path)
    report = ROOT/"reports/PREFLIGHT.json"
    atomic_json({"status": "CPU_PREFLIGHT_PASS", "signature": signature(), "frames": counts,
                 "trajectories": {"train": 192, "validation": 32, "heldout": 32},
                 "source_clean_decisions": len(traces), "source_clean_trajectories": len(rollouts),
                 "runtime_dependencies": dependencies,
                 "source_action_validation": action_validation,
                 "elapsed_s":time.monotonic()-started, "preflight_process_resources":resources(),
                 "source_hashes": source_files(), "scheduler_submitted": False}, report)
    finish(ROOT/"reports/PREPARE_COMPLETE.json", [path, report], frames=counts)
    return read_json(report)

def prepared():
    require(complete(ROOT/"reports/PREPARE_COMPLETE.json"), "CPU preparation is incomplete")
    return read_json(ROOT/"reports/PREFLIGHT.json")

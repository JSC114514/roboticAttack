"""One new first-order DINO method; native optimizer, targets and EOT preserved."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHOD, config, run_signature, load_bank, load_json, require,
                    require_gpu, runtime_config, atomic_json, atomic_torch, atomic_parquet)
from UADA_SOURCE_COPY import build_uada_adamw
from meta import update
from monitor import Monitor, rng_state, restore_rng, preserve_rng
from objectives import rgb_metrics
from resources import snapshot

def save_patch(folder, patch, initial, outer, cfg, signature):
    path = folder / f"outer_{outer:04d}.pt"
    value = {"signature": signature, "method": METHOD, "patch": patch.detach().cpu(),
             "initial_patch": initial.cpu(), "config": cfg, "outer_iterations": outer,
             "optimizer_updates": outer*cfg["inner_loop"]}
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=False)
        require(old["signature"] == signature and torch.equal(old["patch"], value["patch"]), "Refusing conflicting snapshot overwrite")
    else:
        atomic_torch(value, path)
    rgb = value["patch"].permute(1, 2, 0).mul(255).round().byte().numpy()
    png = path.with_suffix(".png")
    if png.exists():
        with Image.open(png) as im:
            require(np.array_equal(np.asarray(im), rgb), "Existing PNG conflicts with RGB snapshot")
    else:
        temp = png.with_suffix(".png.tmp"); Image.fromarray(rgb).save(temp, format="PNG"); temp.replace(png)

def validate_resume(saved, cfg, signature):
    outer = saved["outer"]
    require(saved["signature"] == signature and saved["config"] == cfg and saved["method"] == METHOD, "Resume provenance drift")
    require(0 <= outer <= cfg["outer_iterations"] and len(saved["rows"]) == outer
            and saved["scheduler"]["last_epoch"] == outer, "Resume outer/scheduler/history clock changed")
    expected = outer*cfg["inner_loop"]
    states = saved["optimizer"]["state"]
    require((expected == 0 and not states) or (len(states) == 1 and all(int(s["step"]) == expected for s in states.values())), "Virtual steps advanced real Adam or moments were lost")
    for s in states.values():
        require(all(k in s and s[k].shape == saved["patch"].shape and torch.isfinite(s[k]).all()
                    for k in ("exp_avg", "exp_avg_sq")), "Invalid Adam moments")
    require(saved["patch"].shape == tuple(cfg["patch_shape"]) and torch.isfinite(saved["patch"]).all()
            and ((saved["patch"] >= 0) & (saved["patch"] <= 1)).all(), "Invalid resumed RGB")
    require(set(saved["rng"]) == {"python", "numpy", "torch", "cuda"}, "Incomplete RNG checkpoint")

def epochs(engine, bank, patch, optimizer, scheduler, cfg, start=0, rows=None, on_step=None,
           on_outer=None, diagnostic=None, elapsed_before=0., stop_after=None):
    """Production loop with CPU analytic seams; exactly50 real updates per row."""
    rows = [] if rows is None else rows
    began = time.monotonic()
    stop = cfg["outer_iterations"] if stop_after is None else min(cfg["outer_iterations"], stop_after)
    for outer in range(start+1, stop+1):
        pair = bank["pairs"][outer-1]
        record = bank["records"][pair["uid"]]
        successor = None if pair["query_uid"] is None else bank["records"][pair["query_uid"]]
        before_outer = patch.detach().clone(); learning_rate = float(optimizer.param_groups[0]["lr"])
        step_rows = []
        for inner in range(cfg["inner_loop"]):
            seed = cfg["data_seed"]*1000000 + outer*10000 + inner
            metrics, tensors = update(engine, record, successor, patch, optimizer, seed)
            row = {"outer": outer, "inner": inner, "uid": pair["uid"], "query_uid": pair["query_uid"],
                   "eot_seed": seed, "learning_rate": learning_rate, **metrics}
            if on_step:
                on_step(row)
            if diagnostic and outer in cfg["diagnostic_outers"] and inner in cfg["diagnostic_inners"]:
                with preserve_rng():
                    diagnostic(engine, record, successor, seed, tensors, outer, inner)
            step_rows.append(metrics)
            del tensors
        scheduler.step()
        row = {"outer_iteration": outer, "optimizer_updates": outer*cfg["inner_loop"], "uid": pair["uid"],
               "query_uid": pair["query_uid"], "learning_rate": learning_rate,
               "elapsed_s": elapsed_before+time.monotonic()-began,
               "paired_updates": sum(s["used_query"] for s in step_rows),
               "native_fallback_updates": sum(not s["used_query"] for s in step_rows),
               "current_uada_loss": float(np.mean([s["current_uada_loss"] for s in step_rows])),
               "rgb_outer_mae": float((patch.detach()-before_outer).abs().mean()),
               "effective_rgb_updates": sum(s["rgb_update_l2"] > 0 for s in step_rows),
               **rgb_metrics(patch, bank["initial"])}
        rows.append(row)
        if on_outer:
            on_outer(outer, rows)
        engine.clear_bases()
    return rows

def train():
    cfg, signature = config(), run_signature(); gpu = require_gpu("training")
    folder = ROOT / "training" / METHOD; folder.mkdir(parents=True, exist_ok=True)
    status_path, resume = folder / "status.json", folder / "resume.pt"
    if status_path.exists():
        old = load_json(status_path); require(old["signature"] == signature, "Training identity changed")
        if old["status"] == "COMPLETE":
            require((folder / "outer_0500.png").exists() and resume.exists(), "Completed training artifacts missing")
            return
    bank = load_bank(); initial = torch.rand(cfg["patch_shape"], generator=torch.Generator().manual_seed(cfg["seed"]))
    patch = initial.to("cuda").detach().requires_grad_(True); bank["initial"] = initial
    optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"], cfg["warmup_outer_iterations"], cfg["schedule_outer_iterations"], weight_decay=0.)
    start, rows, elapsed_before, saved = 0, [], 0., None
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False); validate_resume(saved, cfg, signature)
        require(torch.equal(saved["initial"], initial), "Fresh seed42 initialization changed")
        with torch.no_grad(): patch.copy_(saved["patch"].to(patch.device))
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        for state in optimizer.state.values():
            for k, value in state.items():
                if isinstance(value, torch.Tensor): state[k] = value.to(patch.device)
        start, rows, elapsed_before = saved["outer"], saved["rows"], saved["elapsed_s"]
    manifest = {"signature": signature, "method": METHOD, "config": cfg, "first_order": True,
                "gradient": "0.5*(native_current(P)+native_successor(Pprime)); virtual direction is wrist-DINO",
                "model_parameters_updated": False, "virtual_optimizer_steps": 0, "gap": 8,
                "initialization": "fresh seed42, same as parent UADA; no warm start"}
    manifest_path = folder / "RUN_MANIFEST.json"
    if manifest_path.exists(): require(load_json(manifest_path) == manifest, "Existing manifest changed")
    else: atomic_json(manifest, manifest_path)
    if saved is None: save_patch(folder, patch, initial, 0, cfg, signature)
    phase, completed, monitor = "model_load", start, None
    began = time.monotonic()
    try:
        atomic_json({"status": "RUNNING", "signature": signature, "outer": start, "phase": phase, "gpu_name": gpu}, status_path)
        torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
        from engine import MetaEngine
        from model_runtime import load_wrapper
        engine = MetaEngine(load_wrapper(runtime_config(), "libero_10"), cfg)
        phase = "native_interface"
        atomic_json(engine.validate(bank["records"][bank["pairs"][0]["uid"]], patch), folder / f"GPU_INTERFACE_{time.time_ns()}.json")
        if saved:
            require(len(saved["rng"]["cuda"]) == 1, "Single-GPU resume RNG missing")
            restore_rng(saved["rng"])
        monitor = Monitor(folder, signature, cfg); phase = "rgb_updates"
        def checkpoint(outer, history):
            nonlocal completed
            completed = outer
            elapsed = elapsed_before + time.monotonic()-began
            history[-1]["elapsed_s"] = elapsed
            if outer in cfg["checkpoint_outer_iterations"]:
                save_patch(folder, patch, initial, outer, cfg, signature)
            if outer % cfg["resume_every_outer"] == 0 or outer == cfg["outer_iterations"]:
                atomic_parquet(pd.DataFrame(history), folder / "training.parquet")
                atomic_torch({"signature": signature, "method": METHOD, "config": cfg, "outer": outer,
                    "patch": patch.detach().cpu(), "initial": initial, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rows": history, "elapsed_s": elapsed, "rng": rng_state()}, resume)
                atomic_json({"status": "COMPLETE" if outer == cfg["outer_iterations"] else "RUNNING", "signature": signature,
                    "outer": outer, "optimizer_updates": outer*cfg["inner_loop"], "elapsed_s": elapsed,
                    "phase": "checkpoint", **snapshot()}, status_path)
            if outer % 25 == 0:
                recent = history[-min(25, len(history)):]
                speed = (recent[-1]["elapsed_s"]-recent[0]["elapsed_s"])/max(1, len(recent)-1)
                print(f"DINO_CROSS_OBS {outer}/500 loss={history[-1]['current_uada_loss']:.6f} "
                      f"paired={history[-1]['paired_updates']}/50 elapsed_h={elapsed/3600:.2f} ETA_h={(500-outer)*speed/3600:.2f}", flush=True)
        epochs(engine, bank, patch, optimizer, scheduler, cfg, start, rows, monitor.write, checkpoint,
               monitor.capture, elapsed_before)
    except Exception as error:
        atomic_json({"status": "FAILED", "signature": signature, "outer": completed, "phase": phase,
                     "error": repr(error), "resume_every_outer": cfg["resume_every_outer"], **snapshot()}, status_path)
        raise
    finally:
        if monitor: monitor.close()

if __name__ == "__main__": train()

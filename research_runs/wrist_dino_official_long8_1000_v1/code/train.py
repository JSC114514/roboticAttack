"""Exploring copied RGB loop; official expert labels, eight independent methods."""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, atomic_json, atomic_parquet, atomic_torch, config, flags,
                    load_bank, load_json, prepared, require, require_gpu, run_signature)
from UADA_SOURCE_COPY import build_uada_adamw
from engine import OFTEngine
from model_runtime import load_wrapper
from common import runtime_config
from objectives import (action_loss, attention_loss, combine_gradients, geometry_loss,
                        precondition_step, rgb_metrics, sampled_index, target_action)

def patch_device():
    """Production is GPU-only; a separate seam lets CPU tests use a mock VLA."""
    return torch.device("cuda")

def save_patch(folder, patch, initial, method, outer, cfg, signature):
    path = folder / f"outer_{outer:04d}.pt"
    atomic_torch({"patch": patch.detach().cpu(), "initial_patch": initial.cpu(), "method": method,
                  "outer_iterations": outer, "optimizer_updates": outer * cfg["inner_loop"],
                  "parameter_count": patch.numel(), "config": cfg, "signature": signature}, path)
    rgb = patch.detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(rgb).save(temporary, format="PNG")
    temporary.replace(path.with_suffix(".png"))

def validate_resume(saved, method, cfg, signature):
    require(saved["signature"] == signature and saved["method"] == method and saved["config"] == cfg,
            "Resume source/config/method changed")
    outer = saved["outer"]
    require(0 <= outer <= cfg["outer_iterations"] and len(saved["rows"]) == outer, "Resume clock/history changed")
    require(saved["scheduler"]["last_epoch"] == outer, "Resume scheduler clock changed")
    steps = outer * cfg["inner_loop"]
    require((steps == 0 and not saved["optimizer"]["state"]) or
            (len(saved["optimizer"]["state"]) == 1 and all(int(s["step"]) == steps for s in saved["optimizer"]["state"].values())),
            "Patch Adam moments or update counter lost")
    for state in saved["optimizer"]["state"].values():
        require(all(k in state and state[k].shape == (3, 50, 50) and torch.isfinite(state[k]).all()
                    for k in ("exp_avg", "exp_avg_sq")), "Patch Adam moments missing/invalid")
    require(all(k in saved for k in ("patch", "initial", "python_rng", "numpy_rng", "torch_rng", "cuda_rng",
                                     "frequency_scale", "elapsed_s")), "Incomplete training checkpoint")
    require(saved["patch"].shape == (3, 50, 50) and torch.isfinite(saved["patch"]).all()
            and ((saved["patch"] >= 0) & (saved["patch"] <= 1)).all(), "Invalid resumed RGB")
    require(saved["torch_rng"].dtype == torch.uint8 and len(saved["cuda_rng"]) == 1
            and saved["cuda_rng"][0].dtype == torch.uint8, "Single-GPU training RNG state missing/invalid")

def select_record(engine, bank, patch, outer, spec, cfg):
    uids = bank["candidates"][outer - 1]
    if not spec["sampling"]:
        return bank["records"][uids[0]], {"selected_candidate": 0, "candidate_scores": [0.] * 4, "candidate_weights": [1.] * 4}
    scores = []
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        with torch.no_grad():
            rgb = patch.detach().mul(255).round().div(255)
            for uid in uids:
                record = bank["records"][uid]
                inputs, proprio = engine.base(record)
                _clean_action, clean, _ = engine.forward(inputs, proprio)
                target = target_action(record["expert_actions"].to(patch.device), spec["family"], cfg)
                pixels, _ = engine.pixels(record, rgb, cfg["data_seed"] + outer, diverse=False)
                attacked, caps, _ = engine.forward(inputs, proprio, pixels)
                restored, restored_caps, _ = engine.forward(inputs, proprio, pixels, replacement=clean["dino"])
                require(torch.equal(caps["siglip"], restored_caps["siglip"]), "DINO sampling restore changed SigLIP")
                scores.append(max(0., float(action_loss(restored, target, spec["family"], cfg)
                                           - action_loss(attacked, target, spec["family"], cfg))))
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
    index, weights = sampled_index(scores, cfg["data_seed"] * 100000 + outer)
    return bank["records"][uids[index]], {"selected_candidate": index, "candidate_scores": scores, "candidate_weights": weights}

def train(method):
    cfg, spec, signature = config(), flags(method), run_signature()
    prepared()
    folder = ROOT / "training" / method
    folder.mkdir(parents=True, exist_ok=True)
    status_path, resume_path = folder / "status.json", folder / "resume.pt"
    if status_path.exists():
        old = load_json(status_path)
        require(old["signature"] == signature, "Existing method identity changed")
        if old["status"] == "COMPLETE":
            require(old["outer"] == cfg["outer_iterations"] and
                    (folder / f"outer_{cfg['outer_iterations']:04d}.png").exists(), "Completed method missing final snapshot")
            return
    require_gpu("training")
    bank = load_bank()
    initial = torch.rand(cfg["patch_shape"], generator=torch.Generator().manual_seed(cfg["seed"]))
    patch = initial.to(patch_device()).detach().clone().requires_grad_(True)
    optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"], cfg["warmup_outer_iterations"], cfg["outer_iterations"],
                                          weight_decay=cfg["weight_decay"])
    require(len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]["params"]) == 1
            and optimizer.param_groups[0]["params"][0] is patch, "Only external RGB may be optimized")
    outer, rows, elapsed_before, frequency_scale, saved = 0, [], 0., None, None
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(saved, method, cfg, signature)
        with torch.no_grad():
            patch.copy_(saved["patch"].to(patch.device))
        initial = saved["initial"]
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(patch.device)
        outer, rows, elapsed_before, frequency_scale = saved["outer"], saved["rows"], saved["elapsed_s"], saved["frequency_scale"]
    manifest = {"signature": signature, "method": method, "flags": spec, "config": cfg,
                "dataset": "official Long train demonstrations", "action_target_reference": "expert labels or fixed XYZ",
                "initialization": "fresh seed42 random RGB, no old patch/teacher/prefix", "model_parameters_updated": False}
    if (folder / "RUN_MANIFEST.json").exists():
        require(load_json(folder / "RUN_MANIFEST.json") == manifest, "Training manifest changed")
    else:
        atomic_json(manifest, folder / "RUN_MANIFEST.json")
    if saved is None:
        save_patch(folder, patch, initial, method, 0, cfg, signature)
    started = time.monotonic()
    phase = "model_load"
    try:
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": outer, "phase": phase}, status_path)
        torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
        wrapper = load_wrapper(runtime_config(), "libero_10")
        engine = OFTEngine(wrapper)
        phase = "native_interface"
        atomic_json(engine.validate(bank["records"][bank["candidates"][0][0]], patch), folder / "GPU_INTERFACE.json")
        if saved:
            random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        for current in range(outer + 1, cfg["outer_iterations"] + 1):
            phase = "sample_selection"
            record, selection = select_record(engine, bank, patch, current, spec, cfg)
            inputs, proprio = engine.base(record)
            target = target_action(record["expert_actions"].to(patch.device), spec["family"], cfg)
            mechanism = None
            if spec["geometry"] or spec["attention"]:
                mechanism = engine.mechanism_reference(record, inputs, proprio, target, cfg,
                                                       geometry=spec["geometry"], attention=spec["attention"])
            before_outer = patch.detach().clone()
            lr, moved, statistics = float(optimizer.param_groups[0]["lr"]), 0, []
            phase = "rgb_updates"
            for inner in range(cfg["inner_loop"]):
                optimizer.zero_grad(set_to_none=True)
                seed = cfg["data_seed"] * 1000000 + current * 10000 + inner
                pixels, footprint = engine.pixels(record, patch, seed, diverse=inner >= cfg["eot"]["fixed_inner_count"])
                action, captures, qkv = engine.forward(inputs, proprio, pixels, attention=spec["attention"])
                main = action_loss(action, target, spec["family"], cfg)
                auxiliary, aux_metrics = None, {"aux_active": 0.}
                if spec["geometry"]:
                    auxiliary, aux_metrics = geometry_loss(captures["dino"], mechanism["clean"], mechanism["direction"],
                                                           mechanism["weights"], footprint)
                elif spec["attention"]:
                    auxiliary, aux_metrics = attention_loss(qkv, engine.attentions, mechanism["weights"], footprint,
                                                            mechanism["attention_scales"])
                require(torch.isfinite(main), "Nonfinite continuous action objective")
                main_grad = torch.autograd.grad(main, patch, retain_graph=auxiliary is not None)[0]
                auxiliary_grad = None if auxiliary is None else torch.autograd.grad(auxiliary, patch)[0]
                patch.grad, grad_metrics = combine_gradients(main_grad, auxiliary_grad, cfg["auxiliary_gradient_ratio"])
                before = patch.detach().clone()
                optimizer.step()
                with torch.no_grad():
                    if spec["frequency"]:
                        value, frequency_scale = precondition_step(before, patch.detach(), frequency_scale, cfg)
                        patch.copy_(value)
                    else:
                        patch.clamp_(0, 1)
                changed = not torch.equal(before, patch.detach())
                moved += int(changed)
                statistics.append({"main_loss": float(main.detach()), "auxiliary_loss": 0. if auxiliary is None else float(auxiliary.detach()),
                                   "gradient_l2": float(patch.grad.norm()), "rgb_inner_l2": float((patch.detach() - before).norm()),
                                   **aux_metrics, **grad_metrics})
                optimizer.zero_grad(set_to_none=True)
                del pixels, action, captures, qkv, main, auxiliary, main_grad, auxiliary_grad
            scheduler.step()
            outer = current
            elapsed = elapsed_before + time.monotonic() - started
            means = {key: float(np.mean([s[key] for s in statistics])) for key in statistics[0]}
            rows.append({"outer_iteration": outer, "optimizer_updates": outer * cfg["inner_loop"], "uid": record["uid"],
                         "episode_id": record["episode_id"], "step_index": record["step_index"], "task_id": record["task_id"],
                         "learning_rate": lr, "effective_rgb_updates": moved, "elapsed_s": elapsed,
                         "rgb_outer_mae": float((patch.detach() - before_outer).abs().mean()), "frequency_scale": frequency_scale,
                         "geometry_reference": -1 if mechanism is None else mechanism["geometry_selected"],
                         "geometry_reference_scores": [0.] * 4 if mechanism is None else mechanism["geometry_scores"],
                         **selection, **means, **rgb_metrics(patch, initial)})
            if outer in cfg["checkpoint_outer_iterations"]:
                save_patch(folder, patch, initial, method, outer, cfg, signature)
            if outer % cfg["resume_every_outer"] == 0 or outer == cfg["outer_iterations"]:
                phase = "checkpoint"
                atomic_parquet(pd.DataFrame(rows), folder / "training.parquet")
                atomic_torch({"signature": signature, "method": method, "config": cfg, "outer": outer,
                              "patch": patch.detach().cpu(), "initial": initial.cpu(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "rows": rows, "elapsed_s": elapsed,
                              "frequency_scale": frequency_scale, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                              "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, resume_path)
                atomic_json({"status": "COMPLETE" if outer == cfg["outer_iterations"] else "RUNNING", "signature": signature,
                             "method": method, "outer": outer, "optimizer_updates": outer * cfg["inner_loop"], "elapsed_s": elapsed,
                             "phase": phase, "frequency_scale": frequency_scale}, status_path)
            if outer % 25 == 0 or outer == cfg["outer_iterations"]:
                recent = rows[-min(25, len(rows)):]
                seconds_per_outer = (recent[-1]["elapsed_s"] - recent[0]["elapsed_s"]) / max(1, len(recent)-1)
                print(f"OFFICIAL_LONG8 {method} {outer}/{cfg['outer_iterations']} loss={means['main_loss']:.5f} "
                      f"RGB_MAE={rows[-1]['rgb_outer_mae']:.6g} updates={moved}/{cfg['inner_loop']} lr={lr:.5g} "
                      f"elapsed_h={elapsed/3600:.2f} ETA_h={(cfg['outer_iterations']-outer)*seconds_per_outer/3600:.2f}", flush=True)
    except Exception as error:
        atomic_json({"status": "FAILED", "signature": signature, "method": method, "outer": outer,
                     "phase": phase, "error": repr(error), "resume_every_outer": cfg["resume_every_outer"]}, status_path)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    train(parser.parse_args().method)

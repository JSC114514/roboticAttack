"""Exploring-derived RGB loop; three DINO-only objectives, no policy updates."""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, BASE_SIGNATURE, atomic_json, atomic_parquet, atomic_torch,
                    config, load_bank, load_json, prepared, require, require_gpu, run_signature, runtime_config)
from UADA_SOURCE_COPY import build_uada_adamw
from engine import OFTEngine
from model_runtime import load_wrapper
from objectives import rgb_metrics
from new_objectives import features_forward, multilayer_reference, objective


def patch_device():
    return torch.device("cuda")


def initialize_attack(cfg, device):
    initial = torch.rand(cfg["patch_shape"], generator=torch.Generator().manual_seed(cfg["seed"]))
    patch = initial.to(device).clone().detach().requires_grad_(True)
    optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"], cfg["warmup_outer_iterations"],
                                          cfg["schedule_outer_iterations"], weight_decay=cfg["weight_decay"])
    return patch, initial, optimizer, scheduler


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
    require(0 < outer <= cfg["outer_iterations"] and len(saved["rows"]) == outer, "Resume clock/history changed")
    states, steps = saved["optimizer"]["state"], outer * cfg["inner_loop"]
    require(len(states) == 1 and saved["parameter_shapes"] == [[3, 50, 50]], "RGB optimizer layout changed")
    state = next(iter(states.values()))
    require(int(state["step"]) == steps and all(k in state and tuple(state[k].shape) == (3, 50, 50)
            and torch.isfinite(state[k]).all() for k in ("exp_avg", "exp_avg_sq")), "Adam clock/moments invalid")
    require(saved["patch"].shape == saved["initial"].shape == (3, 50, 50)
            and torch.isfinite(saved["patch"]).all() and ((saved["patch"] >= 0) & (saved["patch"] <= 1)).all(), "Invalid resumed RGB")
    require(saved["scheduler"]["last_epoch"] == outer, "Outer LR clock changed")
    require(saved["torch_rng"].dtype == torch.uint8 and len(saved["cuda_rng"]) == 1, "Single-GPU RNG missing")


def verify_final_assets(folder, saved, method, cfg, signature):
    require(saved["outer"] == cfg["outer_iterations"], "Final resume clock incomplete")
    path = folder / f"outer_{cfg['outer_iterations']:04d}.pt"
    asset = torch.load(path, map_location="cpu", weights_only=False)
    require(asset["signature"] == signature and asset["method"] == method and asset["config"] == cfg
            and torch.equal(asset["patch"], saved["patch"]) and torch.equal(asset["initial_patch"], saved["initial"]),
            "Final snapshot differs from committed RGB state")
    with Image.open(path.with_suffix(".png")) as im:
        expected = saved["patch"].permute(1, 2, 0).mul(255).round().byte().numpy()
        require(im.mode == "RGB" and np.array_equal(np.asarray(im), expected), "Final PNG differs from committed RGB")


def train(method):
    cfg, signature = config(), run_signature()
    require(method in METHODS, "Unknown objective")
    prepared()
    folder = ROOT / "training" / method
    folder.mkdir(parents=True, exist_ok=True)
    status_path, resume_path = folder / "status.json", folder / "resume.pt"
    if status_path.exists():
        old = load_json(status_path)
        require(old["signature"] == signature, "Existing experiment identity changed")
        if old["status"] == "COMPLETE":
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_resume(saved, method, cfg, signature)
            verify_final_assets(folder, saved, method, cfg, signature)
            return
    # A power loss may occur AFTER final resume commit but BEFORE status commit.
    # Recover the receipt without another model load or optimizer update.
    if resume_path.exists():
        committed = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(committed, method, cfg, signature)
        if committed["outer"] == cfg["outer_iterations"]:
            verify_final_assets(folder, committed, method, cfg, signature)
            atomic_parquet(pd.DataFrame(committed["rows"]), folder / "training.parquet")
            atomic_json({"status": "COMPLETE", "signature": signature, "method": method,
                         "outer": committed["outer"], "optimizer_updates": committed["outer"] * cfg["inner_loop"],
                         "elapsed_s": committed["elapsed_s"], "phase": "recovered_final_receipt"}, status_path)
            return
    outer, rows, elapsed_before, saved = 0, [], 0., None
    phase, started = "initialization", time.monotonic()
    try:
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": 0, "phase": phase}, status_path)
        require_gpu("training")
        # Before any convolution/model initialization, not after first forward.
        torch.backends.cudnn.enabled = False
        bank = load_bank()
        patch, initial, optimizer, scheduler = initialize_attack(cfg, patch_device())
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_resume(saved, method, cfg, signature)
            initial = saved["initial"]
            with torch.no_grad():
                patch.copy_(saved["patch"].to(patch.device))
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            outer, rows, elapsed_before = saved["outer"], saved["rows"], saved["elapsed_s"]
        manifest = {"signature": signature, "base_signature": BASE_SIGNATURE, "method": method,
                    "config": cfg, "initialization_seed": cfg["seed"], "old_patch_loaded": False,
                    "model_parameters_updated": False, "generator_parameterization": False,
                    "trainable_parameter_count": 7500, "data_selection": "same frozen ordinary Long expert-window order",
                    "target": "DINO objective only; no auxiliary action/teacher/sampling"}
        manifest_path = folder / "RUN_MANIFEST.json"
        if manifest_path.exists():
            require(load_json(manifest_path) == manifest, "Training manifest changed")
        else:
            atomic_json(manifest, manifest_path)
        if saved is None:
            save_patch(folder, initial, initial, method, 0, cfg, signature)
        phase = "model_load"
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": outer, "phase": phase}, status_path)
        torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
        engine = OFTEngine(load_wrapper(runtime_config(), "libero_10"))
        require(not engine.dino.training, "DINO must be frozen in eval mode")
        policy_ids = {id(p) for module in (engine.wrapper.vla, engine.wrapper.action_head,
                                           engine.wrapper.proprio_projector) if module is not None for p in module.parameters()}
        require(id(patch) not in policy_ids and all(len(group["params"]) == 1 and group["params"][0] is patch for group in optimizer.param_groups),
                "Policy parameter entered external optimizer")
        phase = "native_interface"
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": outer, "phase": phase}, status_path)
        first = bank["records"][bank["candidates"][0][0]]
        atomic_json(engine.validate(first, initial.to(patch_device()).clone().requires_grad_(True)), folder / "GPU_INTERFACE.json")
        if saved:
            random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        for current in range(outer + 1, cfg["outer_iterations"] + 1):
            record = bank["records"][bank["candidates"][current - 1][0]]
            inputs, _ = engine.base(record)
            reference = None
            if method != "dino_attention":
                with torch.no_grad():
                    clean, captures = features_forward(engine, inputs["pixel_values"], method, cfg)
                    reference = multilayer_reference(captures, cfg["objective_epsilon"]) if method == "dino_multilayer" else clean.detach()
                    del clean, captures
            statistics, moved = [], 0
            before_outer = patch.detach().clone()
            lr, phase = float(optimizer.param_groups[0]["lr"]), "RGB_updates"
            for inner in range(cfg["inner_loop"]):
                optimizer.zero_grad(set_to_none=True)
                before = patch.detach().clone()
                seed = cfg["data_seed"] * 1000000 + current * 10000 + inner
                pixels, footprint = engine.pixels(record, patch, seed, diverse=True)
                dense, captures = features_forward(engine, pixels, method, cfg)
                loss, detail = objective(method, dense, captures, reference, engine, footprint, cfg)
                require(torch.isfinite(loss), "Nonfinite actual DINO objective")
                loss.backward()
                require(patch.grad is not None and torch.isfinite(patch.grad).all(), "RGB backward disconnected/nonfinite")
                gradient = float(patch.grad.detach().float().norm())
                optimizer.step()
                with torch.no_grad():
                    patch.clamp_(0, 1)
                require(torch.isfinite(patch).all(), "Nonfinite learned RGB")
                moved += int(not torch.equal(before, patch))
                statistics.append({"main_loss": float(loss.detach()), "gradient_l2": gradient,
                                   "rgb_inner_l2": float((patch.detach() - before).norm()),
                                   "visible_patch_tokens": float((footprint > 1e-6).sum()), **detail})
                optimizer.zero_grad(set_to_none=True)
                del pixels, footprint, dense, captures, loss
            scheduler.step()
            outer, elapsed = current, elapsed_before + time.monotonic() - started
            means = {key: float(np.mean([s[key] for s in statistics])) for key in statistics[0]}
            rows.append({"outer_iteration": outer, "optimizer_updates": outer * cfg["inner_loop"], "uid": record["uid"],
                         "episode_id": record["episode_id"], "step_index": record["step_index"], "task_id": record["task_id"],
                         "learning_rate": lr, "effective_rgb_updates": moved, "elapsed_s": elapsed,
                         "rgb_outer_mae": float((patch.detach() - before_outer).abs().mean()), **means, **rgb_metrics(patch, initial)})
            if outer in cfg["checkpoint_outer_iterations"]:
                save_patch(folder, patch, initial, method, outer, cfg, signature)
            if outer % cfg["resume_every_outer"] == 0 or outer == cfg["outer_iterations"]:
                phase = "checkpoint"
                atomic_parquet(pd.DataFrame(rows), folder / "training.parquet")
                atomic_torch({"signature": signature, "method": method, "config": cfg, "outer": outer,
                              "patch": patch.detach().cpu(), "initial": initial.cpu(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "rows": rows, "elapsed_s": elapsed,
                              "parameter_shapes": [[3, 50, 50]], "python_rng": random.getstate(),
                              "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state_all()}, resume_path)
                atomic_json({"status": "COMPLETE" if outer == cfg["outer_iterations"] else "RUNNING",
                             "signature": signature, "method": method, "outer": outer,
                             "optimizer_updates": outer * cfg["inner_loop"], "elapsed_s": elapsed, "phase": phase}, status_path)
            if outer % 25 == 0 or outer == cfg["outer_iterations"]:
                recent = rows[-min(25, len(rows)):]
                rate = (recent[-1]["elapsed_s"] - recent[0]["elapsed_s"]) / max(1, len(recent) - 1)
                print(f"DINO_OBJECTIVES {method} {outer}/{cfg['outer_iterations']} loss={means['main_loss']:.6f} "
                      f"RGB_MAE={rows[-1]['rgb_outer_mae']:.6g} updates={moved}/{cfg['inner_loop']} lr={lr:.5g} "
                      f"elapsed_h={elapsed/3600:.2f} ETA_h={(cfg['outer_iterations']-outer)*rate/3600:.2f}", flush=True)
    except Exception as error:
        atomic_json({"status": "FAILED", "signature": signature, "method": method, "outer": outer,
                     "phase": phase, "error": repr(error), "resume_every_outer": cfg["resume_every_outer"]}, status_path)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    train(parser.parse_args().method)

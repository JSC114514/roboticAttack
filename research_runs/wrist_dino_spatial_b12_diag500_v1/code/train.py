"""Exploring-derived fresh 500x50 loops; exact B averaging, external weights only."""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, BASE_SIGNATURE, METHODS, atomic_json, atomic_parquet, atomic_torch,
                    arm_info, config, eot_seeds, load_bank, load_json, prepared, require,
                    require_gpu, run_signature, runtime_config, batch_uids)
from UADA_SOURCE_COPY import build_uada_adamw
from engine import OFTEngine
from model_runtime import load_wrapper
from objectives import rgb_metrics
from feature_objectives import new_generator, generated_rgb
from batch_runtime import prepare_references, pixel_batch_backward, generator_batch_backward
from monitor import selected, scalar_row, before_probe, after_probe


def patch_device():
    return torch.device("cuda")


def initialize_attack(method, cfg, device):
    family = arm_info(method)["objective"]
    if family.startswith("advencoder_"):
        generator, noise = new_generator(cfg["generator"], cfg["seed"], device)
        with torch.no_grad():
            initial = generated_rgb(generator, noise).detach().cpu().clone()
        optimizer = torch.optim.Adam(generator.parameters(), lr=cfg["generator"]["learning_rate"],
                                     betas=tuple(cfg["generator"]["betas"]))
        return None, generator, noise, initial, optimizer, None
    initial = torch.rand(cfg["patch_shape"], generator=torch.Generator().manual_seed(cfg["seed"]))
    patch = initial.to(device).clone().detach().requires_grad_(True)
    optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"], cfg["warmup_outer_iterations"],
                                          cfg["schedule_outer_iterations"], weight_decay=cfg["weight_decay"])
    return patch, None, None, initial, optimizer, scheduler


def save_patch(folder, patch, initial, method, outer, cfg, signature):
    path = folder / f"outer_{outer:04d}.pt"
    atomic_torch({"patch": patch.detach().cpu(), "initial_patch": initial.cpu(), "method": method,
                  "outer_iterations": outer, "optimizer_updates": outer * cfg["inner_loop"],
                  "sample_updates": outer * cfg["inner_loop"] * arm_info(method)["batch"],
                  "parameter_count": patch.numel(), "config": cfg, "signature": signature}, path)
    rgb = patch.detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(rgb).save(temporary, format="PNG")
    temporary.replace(path.with_suffix(".png"))


def validate_resume(saved, method, cfg, signature):
    require(saved["signature"] == signature and saved["method"] == method and saved["config"] == cfg,
            "Resume source/config/method changed")
    info, outer, shapes = arm_info(method), saved["outer"], saved["parameter_shapes"]
    require(0 < outer <= cfg["outer_iterations"] and len(saved["rows"]) == outer, "Resume clock/history changed")
    states, steps = saved["optimizer"]["state"], outer * cfg["inner_loop"]
    require(len(states) == len(shapes) and all(int(s["step"]) == steps for s in states.values()), "Adam clock lost")
    for state, shape in zip(states.values(), shapes):
        require(all(k in state and tuple(state[k].shape) == tuple(shape) and torch.isfinite(state[k]).all()
                    for k in ("exp_avg", "exp_avg_sq")), "Adam moments invalid")
    require(saved["patch"].shape == saved["initial"].shape == (3, 50, 50) and torch.isfinite(saved["patch"]).all()
            and ((saved["patch"] >= 0) & (saved["patch"] <= 1)).all(), "Invalid resumed RGB")
    require(saved["torch_rng"].dtype == torch.uint8 and len(saved["cuda_rng"]) == 1, "Single-GPU RNG missing")
    if info["objective"].startswith("advencoder_"):
        require(saved["scheduler"] is None and saved["generator"] is not None and saved["noise"].shape == (1, 100, 1, 1),
                "Generator state/layout lost")
        require(all(torch.isfinite(v).all() for v in saved["generator"].values()), "Generator state invalid")
    else:
        require(saved["scheduler"]["last_epoch"] == outer and shapes == [[3, 50, 50]], "Pixel scheduler/layout changed")


def verify_final_assets(folder, saved, method, cfg, signature):
    require(saved["outer"] == cfg["outer_iterations"], "Final resume clock incomplete")
    path = folder / f"outer_{cfg['outer_iterations']:04d}.pt"
    asset = torch.load(path, map_location="cpu", weights_only=False)
    require(asset["signature"] == signature and asset["method"] == method and asset["config"] == cfg
            and torch.equal(asset["patch"], saved["patch"]) and torch.equal(asset["initial_patch"], saved["initial"]),
            "Final snapshot differs from committed RGB")
    with Image.open(path.with_suffix(".png")) as im:
        expected = saved["patch"].permute(1, 2, 0).mul(255).round().byte().numpy()
        require(im.mode == "RGB" and np.array_equal(np.asarray(im), expected), "Final PNG differs from committed RGB")


def fixed_center(bank, device, engine):
    import fcntl
    path = ROOT / "data/SPATIAL_DINO_MEAN.pt"
    with (ROOT / "data/CENTER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
        else:
            total = None
            with torch.no_grad():
                for uid in bank["center_reference_uids"]:
                    inputs, _ = engine.base(bank["records"][uid])
                    value = engine.direct_dino(inputs["pixel_values"]).float()
                    total = value.clone() if total is None else total + value
            saved = {"signature": run_signature(), "uids": bank["center_reference_uids"],
                     "mean": (total / len(bank["center_reference_uids"])).cpu(),
                     "source": "Spatial train only, Spatial checkpoint"}
            atomic_torch(saved, path)
    require(saved["signature"] == run_signature() and saved["uids"] == bank["center_reference_uids"]
            and saved["mean"].shape[:2] == (1, 256) and torch.isfinite(saved["mean"]).all(), "Train-only mean changed")
    return saved["mean"].to(device).detach()


def train(method):
    cfg, signature, info = config(), run_signature(), arm_info(method)
    prepared()
    folder = ROOT / "training" / method
    folder.mkdir(parents=True, exist_ok=True)
    status_path, resume_path = folder / "status.json", folder / "resume.pt"
    saved = None
    if status_path.exists():
        require(load_json(status_path)["signature"] == signature, "Existing experiment identity changed")
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(saved, method, cfg, signature)
        if saved["outer"] == cfg["outer_iterations"]:
            verify_final_assets(folder, saved, method, cfg, signature)
            atomic_parquet(pd.DataFrame(saved["rows"]), folder / "training.parquet")
            atomic_json({"status": "COMPLETE", "signature": signature, "method": method, "outer": saved["outer"],
                         "optimizer_updates": saved["outer"] * cfg["inner_loop"], "elapsed_s": saved["elapsed_s"],
                         "phase": "verified_committed_final", "training_image_batch_size": info["batch"]}, status_path)
            print(f"BATCH500_SKIP_COMPLETE {method}", flush=True)
            return
    elif status_path.exists():
        require(load_json(status_path)["status"] != "COMPLETE", "Complete receipt without optimizer state")
    outer, rows, elapsed_before = (0, [], 0.) if saved is None else (saved["outer"], saved["rows"], saved["elapsed_s"])
    step_rows = [] if saved is None else saved["step_rows"]
    diagnostic_rows = [] if saved is None else saved["diagnostic_rows"]
    phase, started = "initialization", time.monotonic()
    try:
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": outer, "phase": phase}, status_path)
        require_gpu("training")
        # Proven 4090 fix must precede FIRST generator convolution, not model load.
        torch.backends.cudnn.enabled = False
        bank = load_bank()
        patch, generator, noise, initial, optimizer, scheduler = initialize_attack(method, cfg, patch_device())
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        shapes = [list(p.shape) for p in parameters]
        if saved is not None:
            require(saved["parameter_shapes"] == shapes, "External parameter layout changed")
            initial = saved["initial"]
            if generator is None:
                with torch.no_grad():
                    patch.copy_(saved["patch"].to(patch.device))
                scheduler.load_state_dict(saved["scheduler"])
            else:
                generator.load_state_dict(saved["generator"])
                noise = saved["noise"].to(patch_device()).detach()
            optimizer.load_state_dict(saved["optimizer"])
        manifest = {"signature": signature, "base_signature": BASE_SIGNATURE, "method": method, "config": cfg,
                    "candidate_id": info["id"], "objective": info["objective"], "training_image_batch_size": info["batch"],
                    "initialization_seed": cfg["seed"], "old_patch_loaded": False, "model_parameters_updated": False,
                    "generator_parameterization": generator is not None, "generator_latent_batch_size": 1 if generator is not None else None,
                    "trainable_parameter_count": sum(p.numel() for p in parameters), "export_rgb_parameter_count": 7500,
                    "data_selection": "Spatial train250 window pool, cyclic batches; no DINO sampling",
                    "nce_degenerate_control": info["id"] == 9 and info["batch"] == 1}
        manifest_path = folder / "RUN_MANIFEST.json"
        if manifest_path.exists():
            require(load_json(manifest_path) == manifest, "Training manifest changed")
        else:
            atomic_json(manifest, manifest_path)
        if saved is None:
            save_patch(folder, initial, initial, method, 0, cfg, signature)
        phase = "model_load"
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": outer, "phase": phase}, status_path)
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        engine = OFTEngine(load_wrapper(runtime_config(), cfg["training_suite"]))
        require(not engine.dino.training, "Frozen DINO must use eval mode")
        policy_ids = {id(p) for module in (engine.wrapper.vla, engine.wrapper.action_head, engine.wrapper.proprio_projector)
                      if module is not None for p in module.parameters()}
        require(not policy_ids.intersection(id(p) for p in parameters), "Policy parameter entered external optimizer")
        phase = "native_interface"
        atomic_json(engine.validate(bank["records"][bank["candidates"][0][0]], initial.to(patch_device()).clone().requires_grad_(True)),
                    folder / "GPU_INTERFACE.json")
        center = fixed_center(bank, patch_device(), engine) if info["objective"] == "dino_center" else None
        if saved is not None:
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        for current in range(outer + 1, cfg["outer_iterations"] + 1):
            uids = batch_uids(bank["candidates"], current, method)
            records = [bank["records"][uid] for uid in uids]
            references = prepare_references(engine, records, info["objective"], cfg)
            with torch.no_grad():
                before_outer = (patch if generator is None else generated_rgb(generator, noise)).detach().clone()
            statistics, moved = [], 0
            lr, phase = float(optimizer.param_groups[0]["lr"]), "external_updates"
            for inner in range(cfg["inner_loop"]):
                optimizer.zero_grad(set_to_none=True)
                current_rgb = patch if generator is None else generated_rgb(generator, noise)
                current_rgb.retain_grad()
                before = current_rgb.detach().clone()
                seeds = eot_seeds(current, inner, len(records))
                tracing = selected(current, inner)
                engine._training_sample_gradients = [] if tracing else None
                if generator is None:
                    loss_value, detail = pixel_batch_backward(engine, records, references, current_rgb, seeds, info["objective"], cfg, center)
                else:
                    loss_value, detail = generator_batch_backward(engine, records, references, current_rgb, seeds, info["objective"], cfg)
                require(np.isfinite(loss_value) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),
                        "External gradient disconnected or nonfinite")
                gradient = float(sum(p.grad.detach().float().square().sum() for p in parameters).sqrt())
                require(current_rgb.grad is not None, "RGB gradient monitor disconnected")
                actual_gradient = current_rgb.grad.detach().clone()
                probe = None
                if tracing:
                    components = engine._training_sample_gradients or [actual_gradient]
                    probe = before_probe(engine, records, references, before, seeds, info["objective"],
                                         cfg, center, actual_gradient, components)
                    probe["actual_loss_before"] = loss_value
                engine._training_sample_gradients = None
                optimizer.step()
                with torch.no_grad():
                    if generator is None:
                        proposal = patch.detach().clone()
                        patch.clamp_(0, 1)
                        final_rgb = patch.detach()
                    else:
                        proposal = None
                        final_rgb = generated_rgb(generator, noise).detach()
                require(torch.isfinite(final_rgb).all(), "Nonfinite learned RGB")
                step_rows.append({"method": method, "training_image_batch_size": len(records),
                                  "optimizer_parameter_gradient_l2": gradient,
                                  "learning_rate": lr,
                                  **scalar_row(current, inner, loss_value, actual_gradient,
                                               before, proposal, final_rgb, detail)})
                if tracing:
                    diagnostic_rows.extend(after_probe(engine, records, references, before, final_rgb, seeds,
                                                       cfg, probe, method, current, inner, actual_gradient))
                    atomic_parquet(pd.DataFrame(diagnostic_rows), folder / "diagnostics.parquet")
                    print(f"INTERNAL_PROBE {method} outer={current} inner={inner} rows={len(diagnostic_rows)}", flush=True)
                moved += int(not torch.equal(before, final_rgb))
                statistics.append({"main_loss": loss_value, "gradient_l2": gradient,
                                   "rgb_inner_l2": float((final_rgb - before).norm()), **detail})
                optimizer.zero_grad(set_to_none=True)
                del current_rgb
            if scheduler is not None:
                scheduler.step()
            outer, elapsed = current, elapsed_before + time.monotonic() - started
            means = {key: float(np.mean([s[key] for s in statistics])) for key in statistics[0]}
            rows.append({"outer_iteration": outer, "optimizer_updates": outer * cfg["inner_loop"], "uid": uids[0], "uids": uids,
                         "episode_id": records[0]["episode_id"], "episode_ids": [r["episode_id"] for r in records],
                         "task_id": records[0]["task_id"], "learning_rate": lr, "effective_rgb_updates": moved, "elapsed_s": elapsed,
                         "sample_updates": outer * cfg["inner_loop"] * len(uids), "training_image_batch_size": len(uids),
                         "rgb_outer_mae": float((final_rgb - before_outer).abs().mean()), **means, **rgb_metrics(final_rgb, initial)})
            if outer in cfg["checkpoint_outer_iterations"]:
                save_patch(folder, final_rgb, initial, method, outer, cfg, signature)
            if outer % cfg["resume_every_outer"] == 0 or outer == cfg["outer_iterations"]:
                phase = "checkpoint"
                atomic_parquet(pd.DataFrame(rows), folder / "training.parquet")
                atomic_parquet(pd.DataFrame(step_rows), folder / "inner_steps.parquet")
                atomic_torch({"signature": signature, "method": method, "config": cfg, "outer": outer,
                              "patch": final_rgb.cpu(), "initial": initial.cpu(), "optimizer": optimizer.state_dict(),
                              "scheduler": None if scheduler is None else scheduler.state_dict(), "rows": rows, "elapsed_s": elapsed,
                              "parameter_shapes": shapes, "generator": None if generator is None else
                              {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()},
                              "noise": None if noise is None else noise.cpu(), "step_rows": step_rows,
                              "diagnostic_rows": diagnostic_rows, "python_rng": random.getstate(),
                              "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, resume_path)
                atomic_json({"status": "COMPLETE" if outer == cfg["outer_iterations"] else "RUNNING", "signature": signature,
                             "method": method, "outer": outer, "optimizer_updates": outer * cfg["inner_loop"], "elapsed_s": elapsed,
                             "phase": phase, "training_image_batch_size": len(uids)}, status_path)
            if outer % 25 == 0 or outer == cfg["outer_iterations"]:
                recent = rows[-min(25, len(rows)):]
                rate = (recent[-1]["elapsed_s"] - recent[0]["elapsed_s"]) / max(1, len(recent) - 1)
                print(f"BATCH500 {method} {outer}/500 loss={means['main_loss']:.6f} RGB_MAE={rows[-1]['rgb_outer_mae']:.6g} "
                      f"moves={moved}/50 lr={lr:.5g} elapsed_h={elapsed/3600:.2f} ETA_h={(500-outer)*rate/3600:.2f}", flush=True)
            del references
    except Exception as error:
        if step_rows:
            atomic_parquet(pd.DataFrame(step_rows), folder / "inner_steps_failure_partial.parquet")
        if diagnostic_rows:
            atomic_parquet(pd.DataFrame(diagnostic_rows), folder / "diagnostics_failure_partial.parquet")
        atomic_json({"status": "FAILED", "signature": signature, "method": method, "outer": outer,
                     "phase": phase, "error": repr(error), "resume_every_outer": cfg["resume_every_outer"]}, status_path)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    train(parser.parse_args().method)

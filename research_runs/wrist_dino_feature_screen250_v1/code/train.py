"""Four fresh, short, Exploring-derived loops; only external attack is learned."""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, atomic_json, atomic_parquet, atomic_torch, config,
                    load_bank, load_json, prepared, require, require_gpu, run_signature, runtime_config)
from UADA_SOURCE_COPY import build_uada_adamw
from engine import OFTEngine
from model_runtime import load_wrapper
from objectives import action_loss, target_action, rgb_metrics
from feature_objectives import (centered_loss, weighted_key_loss, key_forward,
                                flattened_cosine, generator_loss, new_generator, generated_rgb)


def patch_device():
    return torch.device("cuda")


def initialize_attack(method, cfg, device):
    if method == "advencoder_cos":
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
    steps = outer * cfg["inner_loop"]
    states = saved["optimizer"]["state"]
    shapes = saved["parameter_shapes"]
    require(len(states) == len(shapes) and all(int(s["step"]) == steps for s in states.values()), "Adam update counter lost")
    for state, shape in zip(states.values(), shapes):
        require(all(k in state and tuple(state[k].shape) == tuple(shape) and torch.isfinite(state[k]).all()
                    for k in ("exp_avg", "exp_avg_sq")), "Adam moments missing/invalid")
    require(saved["patch"].shape == saved["initial"].shape == (3, 50, 50)
            and torch.isfinite(saved["patch"]).all() and ((saved["patch"] >= 0) & (saved["patch"] <= 1)).all(), "Invalid resumed RGB")
    require(saved["torch_rng"].dtype == torch.uint8 and len(saved["cuda_rng"]) == 1, "Single-GPU RNG state missing")
    if method == "advencoder_cos":
        require(saved["scheduler"] is None and saved["generator"] is not None
                and saved["noise"].shape == (1, cfg["generator"]["latent_dimension"], 1, 1), "Generator state missing")
        require(all(torch.isfinite(v).all() for v in saved["generator"].values()), "Invalid generator state")
    else:
        require(saved["scheduler"]["last_epoch"] == outer and shapes == [[3, 50, 50]], "Pixel scheduler/shape changed")
    if method == "dino_center":
        require(saved["center_mean"] is not None and saved["center_mean"].shape[:2] == (1, 256)
                and torch.isfinite(saved["center_mean"]).all(), "Fixed train-only mean missing")


def compute_center_mean(engine, bank, folder, signature):
    started, total = time.monotonic(), None
    with torch.no_grad():
        for uid in bank["center_reference_uids"]:
            inputs, _ = engine.base(bank["records"][uid])
            dense = engine.direct_dino(inputs["pixel_values"]).float()
            total = dense.clone() if total is None else total + dense
    mean = total / len(bank["center_reference_uids"])
    require(torch.isfinite(mean).all(), "Train-only feature mean invalid")
    elapsed = time.monotonic() - started
    atomic_torch({"signature": signature, "mean": mean.cpu(), "uids": bank["center_reference_uids"],
                  "source": "official Long train only", "elapsed_s": elapsed}, folder / "CENTER_MEAN.pt")
    return mean.detach(), elapsed


def train(method):
    cfg, signature = config(), run_signature()
    prepared()
    folder = ROOT / "training" / method
    folder.mkdir(parents=True, exist_ok=True)
    status_path, resume_path = folder / "status.json", folder / "resume.pt"
    if status_path.exists():
        old = load_json(status_path)
        require(old["signature"] == signature, "Existing experiment changed")
        if old["status"] == "COMPLETE":
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_resume(saved, method, cfg, signature)
            require(saved["outer"] == cfg["outer_iterations"]
                    and (folder / f"outer_{cfg['outer_iterations']:04d}.png").exists(), "Incomplete final training")
            return
    require_gpu("training")
    bank = load_bank()
    patch, generator, noise, initial, optimizer, scheduler = initialize_attack(method, cfg, patch_device())
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    shapes = [list(p.shape) for p in parameters]
    outer, rows, elapsed_before, center, mean_elapsed, saved = 0, [], 0., None, 0., None
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(saved, method, cfg, signature)
        require(saved["parameter_shapes"] == shapes, "External optimizer layout changed")
        initial = saved["initial"]
        if generator is None:
            with torch.no_grad():
                patch.copy_(saved["patch"].to(patch.device))
            scheduler.load_state_dict(saved["scheduler"])
        else:
            generator.load_state_dict(saved["generator"])
            noise = saved["noise"].to(patch_device()).detach()
        optimizer.load_state_dict(saved["optimizer"])
        outer, rows, elapsed_before = saved["outer"], saved["rows"], saved["elapsed_s"]
        center = None if saved["center_mean"] is None else saved["center_mean"].to(patch_device())
        mean_elapsed = saved["center_mean_elapsed_s"]
    manifest = {"signature": signature, "method": method, "config": cfg, "initialization_seed": cfg["seed"],
                "model_parameters_updated": False, "data_selection": "one shared uniform expert-record schedule",
                "target": "expert X opposite boundary" if method == "uada_original" else "DINO feature objective only",
                "generator_parameterization": generator is not None, "old_patch_loaded": False,
                "trainable_parameter_count": sum(p.numel() for p in parameters),
                "export_rgb_parameter_count": 7500}
    if (folder / "RUN_MANIFEST.json").exists():
        require(load_json(folder / "RUN_MANIFEST.json") == manifest, "Training manifest changed")
    else:
        atomic_json(manifest, folder / "RUN_MANIFEST.json")
    if saved is None:
        save_patch(folder, initial, initial, method, 0, cfg, signature)
    started, phase = time.monotonic(), "model_load"
    try:
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": outer, "phase": phase}, status_path)
        torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
        engine = OFTEngine(load_wrapper(runtime_config(), "libero_10"))
        require(not engine.dino.training, "Frozen DINO must be in evaluation mode")
        policy_ids = {id(p) for module in (engine.wrapper.vla, engine.wrapper.action_head,
                                           engine.wrapper.proprio_projector) if module is not None
                      for p in module.parameters()}
        require(not (policy_ids & {id(p) for p in parameters}), "Policy parameter entered external optimizer")
        phase = "native_interface"
        atomic_json(engine.validate(bank["records"][bank["candidates"][0][0]], initial.to(patch_device()).clone().requires_grad_(True)),
                    folder / "GPU_INTERFACE.json")
        if method == "dino_center" and center is None:
            phase = "train_only_center_mean"
            atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": outer, "phase": phase}, status_path)
            center, mean_elapsed = compute_center_mean(engine, bank, folder, signature)
        if saved:
            random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        for current in range(outer + 1, cfg["outer_iterations"] + 1):
            record = bank["records"][bank["candidates"][current - 1][0]]
            inputs, proprio = engine.base(record)
            target = clean = clean_key = saliency = clean_rgb = None
            layer = -1
            if method == "uada_original":
                target = target_action(record["expert_actions"].to(patch_device()), "uada", cfg)
            else:
                with torch.no_grad():
                    if method == "dino_key":
                        clean, clean_key, saliency, layer = key_forward(engine, inputs["pixel_values"])
                    else:
                        clean = engine.direct_dino(inputs["pixel_values"]).detach()
                    if generator is not None:
                        clean_rgb = torch.as_tensor(record["observation"]["wrist_image"], device=patch_device()).permute(2, 0, 1)[None].float() / 255
            statistics, moved = [], 0
            if generator is None:
                current_rgb = patch
            else:
                with torch.no_grad():
                    current_rgb = generated_rgb(generator, noise).detach()
            before_outer = current_rgb.detach().clone()
            lr = float(optimizer.param_groups[0]["lr"])
            phase = "external_updates"
            for inner in range(cfg["inner_loop"]):
                optimizer.zero_grad(set_to_none=True)
                current_rgb = patch if generator is None else generated_rgb(generator, noise)
                before = current_rgb.detach().clone()
                seed = cfg["data_seed"] * 1000000 + current * 10000 + inner
                rendered = engine.pixels(record, current_rgb, seed, diverse=True, return_rgb=generator is not None)
                pixels = rendered[0]
                detail = {}
                if method == "uada_original":
                    action, captures, qkv = engine.forward(inputs, proprio, pixels)
                    loss = action_loss(action, target, "uada", cfg)
                    del action, captures, qkv
                elif method == "dino_key":
                    dense, key, _, _ = key_forward(engine, pixels)
                    loss = weighted_key_loss(clean_key, key, saliency)
                    detail = {"feature_cosine": float(loss.detach()), "key_layer": layer,
                              "saliency_entropy": float(-(saliency * saliency.clamp_min(1e-12).log()).sum())}
                    del dense, key
                else:
                    dense = engine.direct_dino(pixels)
                    if method == "dino_center":
                        loss = centered_loss(clean, dense, center)
                        detail = {"feature_cosine": float(loss.detach()), "uncentered_cosine": float(flattened_cosine(clean, dense).detach())}
                    else:
                        loss, detail = generator_loss(clean, dense, clean_rgb, rendered[2], cfg["generator"])
                    del dense
                require(torch.isfinite(loss), "Nonfinite actual training objective")
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), "External backward disconnected/nonfinite")
                gradient = float(sum(p.grad.detach().float().square().sum() for p in parameters).sqrt())
                optimizer.step()
                with torch.no_grad():
                    if generator is None:
                        patch.clamp_(0, 1)
                        current_rgb = patch.detach()
                    else:
                        current_rgb = generated_rgb(generator, noise).detach()
                require(torch.isfinite(current_rgb).all(), "Nonfinite learned RGB")
                changed = not torch.equal(before, current_rgb)
                moved += int(changed)
                statistics.append({"main_loss": float(loss.detach()), "gradient_l2": gradient,
                                   "rgb_inner_l2": float((current_rgb - before).norm()), **detail})
                optimizer.zero_grad(set_to_none=True)
                del pixels, rendered, loss
            if scheduler is not None:
                scheduler.step()
            outer, elapsed = current, elapsed_before + time.monotonic() - started
            means = {key: float(np.mean([s[key] for s in statistics])) for key in statistics[0]}
            rows.append({"outer_iteration": outer, "optimizer_updates": outer * cfg["inner_loop"], "uid": record["uid"],
                         "episode_id": record["episode_id"], "step_index": record["step_index"], "task_id": record["task_id"],
                         "learning_rate": lr, "effective_rgb_updates": moved, "elapsed_s": elapsed,
                         "center_mean_elapsed_s": mean_elapsed, "rgb_outer_mae": float((current_rgb - before_outer).abs().mean()),
                         **means, **rgb_metrics(current_rgb, initial)})
            if outer in cfg["checkpoint_outer_iterations"]:
                save_patch(folder, current_rgb, initial, method, outer, cfg, signature)
            if outer % cfg["resume_every_outer"] == 0 or outer == cfg["outer_iterations"]:
                phase = "checkpoint"
                atomic_parquet(pd.DataFrame(rows), folder / "training.parquet")
                atomic_torch({"signature": signature, "method": method, "config": cfg, "outer": outer,
                              "patch": current_rgb.cpu(), "initial": initial.cpu(), "optimizer": optimizer.state_dict(),
                              "scheduler": None if scheduler is None else scheduler.state_dict(), "rows": rows,
                              "elapsed_s": elapsed, "parameter_shapes": shapes,
                              "generator": None if generator is None else {k: v.cpu() for k, v in generator.state_dict().items()},
                              "noise": None if noise is None else noise.cpu(),
                              "center_mean": None if center is None else center.cpu(), "center_mean_elapsed_s": mean_elapsed,
                              "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                              "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, resume_path)
                atomic_json({"status": "COMPLETE" if outer == cfg["outer_iterations"] else "RUNNING", "signature": signature,
                             "method": method, "outer": outer, "optimizer_updates": outer * cfg["inner_loop"],
                             "elapsed_s": elapsed, "phase": phase, "center_mean_elapsed_s": mean_elapsed}, status_path)
            if outer % 25 == 0 or outer == cfg["outer_iterations"]:
                recent = rows[-min(25, len(rows)):]
                rate = (recent[-1]["elapsed_s"] - recent[0]["elapsed_s"]) / max(1, len(recent) - 1)
                print(f"DINO_FEATURE_SCREEN {method} {outer}/{cfg['outer_iterations']} loss={means['main_loss']:.6f} "
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

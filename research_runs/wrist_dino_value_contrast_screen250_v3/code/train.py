"""Three fresh Exploring-derived loops; only external RGB/generator is learned."""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, REFERENCE_SIGNATURE, atomic_json, atomic_parquet, atomic_torch,
                    config, load_bank, load_json, prepared, require, require_gpu, run_signature,
                    runtime_config, batch_uids)
from UADA_SOURCE_COPY import build_uada_adamw
from engine import OFTEngine
from model_runtime import load_wrapper
from objectives import rgb_metrics
from feature_objectives import new_generator, generated_rgb
from remaining_objectives import value_forward, value_loss, cached_generator_backward


def patch_device():
    return torch.device("cuda")


def initialize_attack(method, cfg, device):
    if method.startswith("advencoder_"):
        generator, noise = new_generator(cfg["generator"], cfg["seed"], device)
        with torch.no_grad():
            initial = generated_rgb(generator, noise).detach().cpu().clone()
        optimizer = torch.optim.Adam(generator.parameters(), lr=cfg["generator"]["learning_rate"],
                                     betas=tuple(cfg["generator"]["betas"]))
        return None, generator, noise, initial, optimizer, None
    require(method == "dino_value", "Unknown attack initializer")
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
    outer, shapes = saved["outer"], saved["parameter_shapes"]
    require(0 < outer <= cfg["outer_iterations"] and len(saved["rows"]) == outer, "Resume clock/history changed")
    states, steps = saved["optimizer"]["state"], outer * cfg["inner_loop"]
    require(len(states) == len(shapes) and all(int(s["step"]) == steps for s in states.values()), "Adam clock lost")
    for state, shape in zip(states.values(), shapes):
        require(all(k in state and tuple(state[k].shape) == tuple(shape) and torch.isfinite(state[k]).all()
                    for k in ("exp_avg", "exp_avg_sq")), "Adam moments invalid")
    require(saved["patch"].shape == saved["initial"].shape == (3, 50, 50) and torch.isfinite(saved["patch"]).all()
            and ((saved["patch"] >= 0) & (saved["patch"] <= 1)).all(), "Invalid resumed RGB")
    require(saved["torch_rng"].dtype == torch.uint8 and len(saved["cuda_rng"]) == 1, "Single-GPU RNG missing")
    if method.startswith("advencoder_"):
        require(saved["scheduler"] is None and saved["generator"] is not None and saved["noise"].shape == (1, 100, 1, 1),
                "Generator state/layout lost")
        require(all(torch.isfinite(v).all() for v in saved["generator"].values()), "Generator buffers/weights invalid")
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


def train(method):
    cfg, signature = config(), run_signature()
    require(method in METHODS, "Unknown final-screen method")
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
    if resume_path.exists():
        committed = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(committed, method, cfg, signature)
        if committed["outer"] == cfg["outer_iterations"]:
            verify_final_assets(folder, committed, method, cfg, signature)
            atomic_parquet(pd.DataFrame(committed["rows"]), folder / "training.parquet")
            atomic_json({"status": "COMPLETE", "signature": signature, "method": method, "outer": committed["outer"],
                         "optimizer_updates": committed["outer"] * cfg["inner_loop"], "elapsed_s": committed["elapsed_s"],
                         "phase": "recovered_final_receipt"}, status_path)
            return
    outer, rows, elapsed_before, saved = 0, [], 0., None
    phase, started = "initialization", time.monotonic()
    try:
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": 0, "phase": phase}, status_path)
        require_gpu("training")
        # Keep the real 4090 repair BEFORE the very first generator convolution.
        torch.backends.cudnn.enabled = False
        bank = load_bank()
        patch, generator, noise, initial, optimizer, scheduler = initialize_attack(method, cfg, patch_device())
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        shapes = [list(p.shape) for p in parameters]
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_resume(saved, method, cfg, signature)
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
            outer, rows, elapsed_before = saved["outer"], saved["rows"], saved["elapsed_s"]
        manifest = {"signature": signature, "reference_signature": REFERENCE_SIGNATURE, "method": method, "config": cfg,
                    "initialization_seed": cfg["seed"], "old_patch_loaded": False, "model_parameters_updated": False,
                    "generator_parameterization": generator is not None, "trainable_parameter_count": sum(p.numel() for p in parameters),
                    "generator_latent_batch_size": 1 if generator is not None else None,
                    "training_image_batch_size": cfg["batch_sizes"][method], "export_rgb_parameter_count": 7500,
                    "data_selection": "same read-only official Long pool; paired cyclic B8 for generators",
                    "target": "native Value CosDist" if generator is None else "cross-image " + ("negative InfoNCE" if "nce" in method else "cosine")}
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
        require(not engine.dino.training, "Frozen DINO must use eval mode")
        policy_ids = {id(p) for module in (engine.wrapper.vla, engine.wrapper.action_head, engine.wrapper.proprio_projector)
                      if module is not None for p in module.parameters()}
        require(not policy_ids.intersection(id(p) for p in parameters), "Policy parameter entered external optimizer")
        phase = "native_interface"
        atomic_json({"status": "RUNNING", "signature": signature, "method": method, "outer": outer, "phase": phase}, status_path)
        atomic_json(engine.validate(bank["records"][bank["candidates"][0][0]], initial.to(patch_device()).clone().requires_grad_(True)),
                    folder / "GPU_INTERFACE.json")
        if saved:
            random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        for current in range(outer + 1, cfg["outer_iterations"] + 1):
            uids = batch_uids(bank["candidates"], current, method)
            records = [bank["records"][uid] for uid in uids]
            with torch.no_grad():
                if generator is None:
                    inputs, _proprio = engine.base(records[0])
                    reference = {i: value.detach() for i, value in value_forward(engine, inputs["pixel_values"], cfg["value_layers"]).items()}
                else:
                    clean = []
                    for record in records:
                        inputs, _proprio = engine.base(record)
                        clean.append(engine.direct_dino(inputs["pixel_values"]).detach())
                    reference = torch.cat(clean)
                    clean_rgb = [torch.as_tensor(np.asarray(record["observation"]["wrist_image"]), device=patch_device())
                                 .permute(2, 0, 1)[None].float() / 255 for record in records]
                before_outer = (patch if generator is None else generated_rgb(generator, noise)).detach().clone()
            statistics, moved = [], 0
            lr, phase = float(optimizer.param_groups[0]["lr"]), "external_updates"
            for inner in range(cfg["inner_loop"]):
                optimizer.zero_grad(set_to_none=True)
                current_rgb = patch if generator is None else generated_rgb(generator, noise)
                before = current_rgb.detach().clone()
                seeds = [cfg["data_seed"] * 1000000 + current * 10000 + inner * 100 + j for j in range(len(records))]
                if generator is None:
                    # Exactly the same A-vs-previous B1 EOT seed, no new transform.
                    seed = cfg["data_seed"] * 1000000 + current * 10000 + inner
                    pixels, _footprint = engine.pixels(records[0], current_rgb, seed, diverse=True)
                    values = value_forward(engine, pixels, cfg["value_layers"])
                    loss, detail = value_loss(reference, values)
                    require(torch.isfinite(loss), "Nonfinite actual Value loss")
                    loss.backward()
                    loss_value = float(loss.detach())
                    detail.update(training_image_batch_size=1, attacked_dino_forwards_per_update=1)
                    del pixels, values, loss
                else:
                    criterion = "nce" if method == "advencoder_nce_b8" else "cos"
                    loss_value, detail = cached_generator_backward(engine, records, reference, clean_rgb, current_rgb, seeds, criterion, cfg)
                require(np.isfinite(loss_value) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),
                        "External gradient disconnected or nonfinite")
                gradient = float(sum(p.grad.detach().float().square().sum() for p in parameters).sqrt())
                optimizer.step()
                with torch.no_grad():
                    if generator is None:
                        patch.clamp_(0, 1)
                        final_rgb = patch.detach()
                    else:
                        final_rgb = generated_rgb(generator, noise).detach()
                require(torch.isfinite(final_rgb).all(), "Nonfinite learned RGB")
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
                         "episode_id": records[0]["episode_id"], "episode_ids": [record["episode_id"] for record in records],
                         "step_index": records[0]["step_index"], "task_id": records[0]["task_id"], "learning_rate": lr,
                         "effective_rgb_updates": moved, "elapsed_s": elapsed, "sample_updates": outer * cfg["inner_loop"] * len(uids),
                         "rgb_outer_mae": float((final_rgb - before_outer).abs().mean()), **means, **rgb_metrics(final_rgb, initial)})
            if outer in cfg["checkpoint_outer_iterations"]:
                save_patch(folder, final_rgb, initial, method, outer, cfg, signature)
            if outer % cfg["resume_every_outer"] == 0 or outer == cfg["outer_iterations"]:
                phase = "checkpoint"
                atomic_parquet(pd.DataFrame(rows), folder / "training.parquet")
                atomic_torch({"signature": signature, "method": method, "config": cfg, "outer": outer,
                              "patch": final_rgb.cpu(), "initial": initial.cpu(), "optimizer": optimizer.state_dict(),
                              "scheduler": None if scheduler is None else scheduler.state_dict(), "rows": rows, "elapsed_s": elapsed,
                              "parameter_shapes": shapes, "generator": None if generator is None else
                              {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()},
                              "noise": None if noise is None else noise.cpu(), "python_rng": random.getstate(),
                              "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, resume_path)
                atomic_json({"status": "COMPLETE" if outer == cfg["outer_iterations"] else "RUNNING", "signature": signature,
                             "method": method, "outer": outer, "optimizer_updates": outer * cfg["inner_loop"], "elapsed_s": elapsed,
                             "phase": phase, "training_image_batch_size": len(uids)}, status_path)
            if outer % 25 == 0 or outer == cfg["outer_iterations"]:
                recent = rows[-min(25, len(rows)):]
                rate = (recent[-1]["elapsed_s"] - recent[0]["elapsed_s"]) / max(1, len(recent) - 1)
                print(f"DINO_FINAL_SCREEN {method} {outer}/{cfg['outer_iterations']} loss={means['main_loss']:.6f} "
                      f"RGB_MAE={rows[-1]['rgb_outer_mae']:.6g} updates={moved}/{cfg['inner_loop']} image_B={len(uids)} "
                      f"lr={lr:.5g} elapsed_h={elapsed/3600:.2f} ETA_h={(cfg['outer_iterations']-outer)*rate/3600:.2f}", flush=True)
    except Exception as error:
        atomic_json({"status": "FAILED", "signature": signature, "method": method, "outer": outer,
                     "phase": phase, "error": repr(error), "resume_every_outer": cfg["resume_every_outer"]}, status_path)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    train(parser.parse_args().method)

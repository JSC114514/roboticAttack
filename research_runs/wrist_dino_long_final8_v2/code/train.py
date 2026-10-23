"""Exploring-derived fresh RGB loop: AdamW, 50 real inner updates, outer cosine."""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, atomic_json, atomic_parquet, atomic_torch, clean_records, config,
                    family, load_bank, load_json, method_flags, observation_key, prepared, require,
                    require_gpu, run_signature, runtime_config, sha256_file)
from UADA_SOURCE_COPY import build_uada_adamw
from engine import OFTEngine
from model_runtime import load_wrapper
from objectives import (action_loss, appearance_loss, distillation_loss, draw_record, precondition_step,
                        rgb_metrics, scored_weights, target_action)


def save_patch(folder, patch, initial, method, outer, cfg, signature):
    path = folder / f"outer_{outer:04d}.pt"
    atomic_torch({"patch": patch.detach().cpu(), "initial_patch": initial.cpu(), "method": method,
                  "outer_iterations": outer, "optimizer_updates": outer * cfg["inner_loop"],
                  "parameter_count": 7500, "config": cfg, "signature": signature}, path)
    rgb = patch.detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(rgb).save(temporary, format="PNG")
    temporary.replace(path.with_suffix(".png"))


def validate_resume(saved, method, cfg, signature):
    require(saved["signature"] == signature and saved["method"] == method and saved["config"] == cfg,
            "Resume source/config/method changed")
    outer = saved["outer"]
    require(len(saved["rows"]) == outer and 0 <= outer <= 2000, "Resume history mismatch")
    require(saved["scheduler"]["last_epoch"] == outer, "Resume scheduler clock changed")
    steps = outer * cfg["inner_loop"]
    require((steps == 0 and not saved["optimizer"]["state"]) or
            (len(saved["optimizer"]["state"]) == 1 and all(int(s["step"]) == steps
             for s in saved["optimizer"]["state"].values())), "Adam moments/step counter lost")
    require(all(key in saved for key in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng", "weights",
                                         "distill_coefficient", "frequency_scale")), "Incomplete training resume")


def restore_scores(engine, bank, patch, family_name, prep, cfg):
    scores = []
    rgb = patch.detach().mul(255).round().div(255)
    # Diagnostic passes cannot perturb the future EOT sequence or dropout RNG.
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        with torch.no_grad():
            for clip in bank["clips"]:
                record = bank["observations"][clip["uids"][1]]
                inputs, proprio = engine.base(record)
                clean_action, clean = engine.forward(inputs, proprio)
                target = target_action(clean_action, family_name, prep["bounds"], cfg)
                pixels = engine.pixels(record, rgb, cfg["data_seed"], fixed=True)
                attacked, caps = engine.forward(inputs, proprio, pixels)
                restored, restored_caps = engine.forward(inputs, proprio, pixels, replacement=clean["dino"])
                require(torch.equal(caps["siglip"], restored_caps["siglip"]), "DINO score changed Wrist-SigLIP")
                scores.append(max(0., float(action_loss(restored, target, family_name, cfg)
                                            - action_loss(attacked, target, family_name, cfg))))
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
    return scored_weights(bank["clips"], scores), scores


def fixed_panel(engine, bank, patch, method, prep, cfg, outer):
    rows = []
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        with torch.no_grad():
            for task in range(10):
                uid = next(k for k in bank["clean"] if bank["observations"][k]["task_id"] == task)
                record = bank["observations"][uid]
                inputs, proprio = engine.base(record)
                clean_action, _ = engine.forward(inputs, proprio)
                target = target_action(clean_action, family(method), prep["bounds"], cfg)
                action, _ = engine.forward(inputs, proprio, engine.pixels(record, patch.detach().mul(255).round().div(255),
                                                                         cfg["data_seed"], fixed=True))
                rows.append({"outer": outer, "task_id": task, "uid": uid,
                             "target_mse": float(action_loss(action, target, family(method), cfg)),
                             "action_change_rms": float((action - clean_action).square().mean().sqrt())})
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
    return rows


def train(method, stop=2000):
    cfg, prep, flags = config(), prepared(), method_flags(method)
    require(stop == 2000 or (stop == 250 and method == "uada_original"), "Unapproved training budget")
    signature = run_signature()
    folder = ROOT / "training" / method
    folder.mkdir(parents=True, exist_ok=True)
    status_path, resume_path = folder / "status.json", folder / "resume.pt"
    if status_path.exists():
        old = load_json(status_path)
        require(old["signature"] == signature, "Existing training identity changed")
        if old.get("outer", 0) >= stop:
            require((folder / f"outer_{stop:04d}.pt").is_file(), "Completed snapshot missing")
            return
    require_gpu("training")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["data_seed"])
    random.seed(cfg["data_seed"])
    if stop == 250:
        observations, keys = {}, []
        for r in clean_records():
            uid = "clean:" + observation_key(r["task_id"], r["state_index"], r["decision_index"])
            observations[uid] = r
            keys.append(uid)
        bank = {"clean": keys, "observations": observations, "clips": []}
    else:
        bank = load_bank()
    teacher_records = {}
    if flags["teacher"]:
        path = ROOT / "teachers" / f"{flags['family']}_BANK.pt"
        teacher_bank = torch.load(path, map_location="cpu", weights_only=False)
        require(load_json(path.with_suffix(".json"))["sha256"] == sha256_file(path), "Teacher bank bytes changed")
        require(teacher_bank["signature"] == signature and teacher_bank["family"] == flags["family"], "Wrong teacher family")
        teacher_records = teacher_bank["records"]
        require(set(teacher_records) == set(bank["observations"]), "Teacher/student states not aligned")
    initial = torch.rand(cfg["patch_shape"], generator=torch.Generator().manual_seed(cfg["seed"]))
    patch = initial.to("cuda").requires_grad_(True)
    optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"], cfg["warmup_outer_iterations"], 2000,
                                          weight_decay=cfg["weight_decay"])
    outer, rows, panels, elapsed_before, frequency_scale, coefficient = 0, [], [], 0., None, None
    weights = [1.] * len(bank["clips"])
    saved = None
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(saved, method, cfg, signature)
        patch.data.copy_(saved["patch"].to(patch.device))
        initial = saved["initial"]
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(patch.device)
        outer, rows, panels, elapsed_before = saved["outer"], saved["rows"], saved["panels"], saved["elapsed_s"]
        frequency_scale, coefficient = saved["frequency_scale"], saved["distill_coefficient"]
        # uada_original has no replay weight dependency during its counted prefix.
        if flags["replay"]:
            require(len(saved["weights"]) == len(weights), "Replay pool changed on resume")
            weights = saved["weights"]
    manifest = {"signature": signature, "method": method, "config": cfg, "from_random": True,
                "parameter_count": 7500, "flags": flags, "source_bank": cfg["training_bank_sha256"],
                "algorithm": "Exploring source copies; OFT continuous action adaptation",
                "initialization": "seed42 random RGB, no teacher or old learned patch initialization"}
    if (folder / "RUN_MANIFEST.json").exists():
        require(load_json(folder / "RUN_MANIFEST.json") == manifest, "Manifest changed")
    else:
        atomic_json(manifest, folder / "RUN_MANIFEST.json")
    if saved is None:
        save_patch(folder, patch, initial, method, 0, cfg, signature)
    started = time.monotonic()
    wrapper = load_wrapper(runtime_config(), "libero_10")
    engine = OFTEngine(wrapper)
    first_record = bank["observations"][bank["clean"][0]]
    atomic_json(engine.validate(first_record, patch), folder / "GPU_INTERFACE.json")
    if saved:
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
    try:
        for current in range(outer + 1, stop + 1):
            after = current - 1
            if flags["weighted"] and (after == 250 or after in cfg["refresh_after_outer"]):
                weights, scores = restore_scores(engine, bank, patch, flags["family"], prep, cfg)
                atomic_json({"signature": signature, "outer": after, "weights": weights, "scores": scores},
                            folder / f"scores_after_{after:04d}.json")
            selection = draw_record(bank, method, current, weights, cfg)
            anchors = {}
            for uid in dict.fromkeys(selection["uids"]):
                record = bank["observations"][uid]
                inputs, proprio = engine.base(record)
                with torch.no_grad():
                    clean_action, clean = engine.forward(inputs, proprio)
                    target = target_action(clean_action, flags["family"], prep["bounds"], cfg)
                anchors[uid] = record, inputs, proprio, clean, target
            before_outer = patch.detach().clone()
            statistics, moved, lr = [], 0, float(optimizer.param_groups[0]["lr"])
            for inner in range(cfg["inner_loop"]):
                uid = selection["uids"][inner % len(selection["uids"])]
                record, inputs, proprio, clean, target = anchors[uid]
                seed = cfg["data_seed"] * 1000000 + current * 10000 + inner
                geometry = True if not flags["diverse"] else inner >= cfg["inner_loop"] // 2
                diverse = flags["diverse"] and geometry
                optimizer.zero_grad(set_to_none=True)
                pixels = engine.pixels(record, patch, seed, geometry, diverse)
                action, captures = engine.forward(inputs, proprio, pixels)
                main = action_loss(action, target, flags["family"], cfg)
                auxiliary = main.new_zeros(())
                used_teacher = False
                if flags["teacher"] and teacher_records[uid]["valid"]:
                    with torch.no_grad():
                        teacher_pixels = engine.pixels(record, teacher_records[uid]["patch"].to(patch.device),
                                                       seed, geometry, diverse)
                        teacher_dino = engine.direct_dino(teacher_pixels).detach()
                    auxiliary = distillation_loss(captures["dino"], teacher_dino, clean["dino"], cfg["distill_rms_floor"])
                    if coefficient is None:
                        grad_main = torch.autograd.grad(main, patch, retain_graph=True)[0]
                        grad_aux = torch.autograd.grad(auxiliary, patch, retain_graph=True)[0]
                        if float(grad_main.norm()) > 1e-12 and float(grad_aux.norm()) > 1e-12:
                            coefficient = cfg["distill_gradient_ratio"] * float(grad_main.norm() / grad_aux.norm())
                            atomic_json({"uid": uid, "outer": current, "inner": inner, "coefficient": coefficient,
                                         "gradient_ratio": cfg["distill_gradient_ratio"], "scope": "Long source only"},
                                        folder / "DISTILL_CALIBRATION.json")
                        del grad_main, grad_aux
                    used_teacher = coefficient is not None
                tv = appearance_loss(patch, cfg) if flags["tv"] else main.new_zeros(())
                total = main + tv + (0. if coefficient is None else coefficient) * auxiliary
                require(torch.isfinite(total), "Nonfinite objective")
                total.backward()
                require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Missing/nonfinite RGB gradient")
                before = patch.detach().clone()
                optimizer.step()
                with torch.no_grad():
                    if flags["frequency"]:
                        value, frequency_scale = precondition_step(before, patch.detach(), frequency_scale, cfg)
                        patch.copy_(value)
                    else:
                        patch.clamp_(0, 1)
                moved += int(not torch.equal(before, patch.detach()))
                statistics.append({"main_loss": float(main.detach()), "auxiliary_loss": float(auxiliary.detach()),
                                   "tv_loss": float(tv.detach()), "teacher_used": int(used_teacher),
                                   "gradient_l2": float(patch.grad.norm())})
                optimizer.zero_grad(set_to_none=True)
                del pixels, action, captures, main, total, auxiliary, tv
            scheduler.step()
            outer = current
            elapsed = elapsed_before + time.monotonic() - started
            means = {k: float(np.mean([r[k] for r in statistics])) for k in statistics[0]}
            rows.append({"outer_iteration": outer, "optimizer_updates": outer * 50, "origin": selection["origin"],
                         "task_id": record["task_id"], "learning_rate": lr, "effective_rgb_updates": moved,
                         "rgb_outer_mae": float((patch.detach() - before_outer).abs().mean()), "elapsed_s": elapsed,
                         "frequency_scale": frequency_scale, "distill_coefficient": coefficient,
                         **means, **rgb_metrics(patch, initial)})
            if outer in cfg["checkpoint_outer_iterations"]:
                save_patch(folder, patch, initial, method, outer, cfg, signature)
                if outer in cfg["evaluation_checkpoints"]:
                    panels.extend(fixed_panel(engine, bank, patch, method, prep, cfg, outer))
                    atomic_parquet(pd.DataFrame(panels), folder / "FIXED_SOURCE_PANEL.parquet")
            if outer % cfg["resume_every_outer"] == 0 or outer == stop:
                atomic_parquet(pd.DataFrame(rows), folder / "training.parquet")
                atomic_torch({"signature": signature, "method": method, "config": cfg, "outer": outer,
                              "patch": patch.detach().cpu(), "initial": initial.cpu(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "rows": rows, "panels": panels, "elapsed_s": elapsed,
                              "weights": weights, "frequency_scale": frequency_scale, "distill_coefficient": coefficient,
                              "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                              "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, resume_path)
                atomic_json({"status": "COMPLETE" if outer == 2000 else "PREFIX_COMPLETE" if outer == stop else "RUNNING",
                             "signature": signature, "method": method, "outer": outer, "optimizer_updates": outer * 50,
                             "elapsed_s": elapsed, "frequency_scale": frequency_scale, "distill_coefficient": coefficient}, status_path)
            if outer % 25 == 0 or outer == stop:
                print(f"FINAL8_TRAIN {method} {outer}/2000 loss={means['main_loss']:.5f} lr={lr:.5g} "
                      f"rgb_mae={rows[-1]['rgb_outer_mae']:.5g} elapsed_h={elapsed/3600:.2f}", flush=True)
    except Exception as error:
        atomic_json({"status": "FAILED", "signature": signature, "method": method, "outer": outer,
                     "error": repr(error), "resume_outer": (outer // cfg["resume_every_outer"]) * cfg["resume_every_outer"]}, status_path)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stop", type=int, choices=(250, 2000), default=2000)
    args = parser.parse_args()
    train(args.method, args.stop)

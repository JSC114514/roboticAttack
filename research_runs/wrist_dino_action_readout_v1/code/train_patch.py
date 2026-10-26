"""Two matched fresh RGB optimizations, with native-policy gradient/step tracing."""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image
from readout import ActionReadout, energy_logits, patch_objective
from run_core import (ARMS, ROOT, BASE, BASE_METHOD, atomic_write, bank, cfg, finish, locked, native,
                      prepared, read_json, require, require_gpu, save_frame, save_torch, sha, signature,
                      stage_done, write_json)


def cosine(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def component_gradients(action, helper, clean, wrong, arm, outer, patch):
    """Exact loss-path components; detach only in these diagnostic graphs."""
    native_loss, _ = patch_objective(action, None if helper is None else helper.detach(), clean, wrong, arm, outer)
    gn = torch.autograd.grad(native_loss, patch, retain_graph=True)[0].detach()
    if helper is None:
        return gn, torch.zeros_like(gn)
    helper_loss, _ = patch_objective(action.detach(), helper, clean, wrong, arm, outer)
    gh = torch.autograd.grad(helper_loss, patch, retain_graph=True)[0].detach()
    return gn, gh


def snapshot(folder, patch, initial, outer):
    path = folder / f"outer_{outer:04d}.pt"
    save_torch({"signature": signature(), "patch": patch.detach().cpu(), "initial": initial.cpu(), "outer": outer}, path)
    rgb = patch.detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
    atomic_write(path.with_suffix(".png"), lambda p: Image.fromarray(rgb).save(p, format="PNG"))


def validate_resume(saved, arm, initial, head_hash):
    require(saved["signature"] == signature() and saved["arm"] == arm and saved["head_sha256"] == head_hash, "Patch resume identity changed")
    require(torch.equal(saved["initial"].cpu(), initial.cpu()), "Patch initialization changed")
    outer = saved["outer"]
    require(0 < outer <= 500 and len(saved["steps"]) == outer * 50 and len(saved["history"]) == outer, "Patch resume clock incomplete")
    states = saved["optimizer"]["state"]
    require(len(states) == 1 and all(int(s["step"]) == outer * 50 for s in states.values()), "Adam clock changed")
    require(saved["patch"].shape == (3, 50, 50) and torch.isfinite(saved["patch"]).all(), "Invalid resumed RGB")


def train(arm):
    require(arm in ARMS, "Unknown patch arm")
    prepared()
    folder = ROOT / "training" / arm
    with locked(folder / "worker.lock"):
        status_path = folder / "STATUS.json"
        if stage_done(status_path, [folder / "outer_0500.png"]):
            print(arm, "ALREADY_COMPLETE", flush=True)
            return
        require_gpu("training")
        from native_bridge import load_engine, wrong_target
        from UADA_SOURCE_COPY import build_uada_adamw
        source = bank()
        initial = torch.load(BASE / "training" / BASE_METHOD / "outer_0000.pt", map_location="cpu", weights_only=False)["patch"]
        patch = initial.to("cuda").clone().detach().requires_grad_(True)
        optimizer, scheduler = build_uada_adamw(patch, .002, 20, 2000, weight_decay=0.)
        require(all(p is patch for group in optimizer.param_groups for p in group["params"]), "Only RGB may enter optimizer")
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        engine = load_engine()
        helper, head_hash = None, None
        if arm == ARMS[1]:
            require(stage_done(ROOT / "readout/STATUS.json"), "Readout not complete")
            head_path = ROOT / "readout/best.pt"
            head_hash = sha(head_path)
            weights = torch.load(head_path, map_location="cuda", weights_only=False)
            require(weights["signature"] == signature(), "Wrong readout checkpoint")
            require(weights["cache_sha256"] == sha(ROOT / "teacher/INDEX.json"), "Readout teacher cache identity changed")
            helper = ActionReadout(**weights["dimensions"]).cuda().eval()
            helper.load_state_dict(weights["state_dict"])
            helper.requires_grad_(False)
        require(all(not p.requires_grad for mod in (engine.wrapper.vla, engine.wrapper.action_head, engine.wrapper.proprio_projector)
                    if mod is not None for p in mod.parameters()), "Native policy must remain frozen")
        manifest = {"signature": signature(), "arm": arm, "head_sha256": head_hash,
                    "native_signature": native().run_signature(), "initial_pt_sha256": sha(BASE / "training" / BASE_METHOD / "outer_0000.pt"),
                    "trainable_rgb_parameters": patch.numel(), "patch_seed": 42, "training_batch_size": 1,
                    "patch_pool_windows": 250, "head_extra_source_windows": 1259 if helper else 0,
                    "native_policy_updated": False, "native_branches_detached": False, "old_trained_patch_loaded": False}
        mp = folder / "RUN_MANIFEST.json"
        if mp.exists():
            require(read_json(mp) == manifest, "Existing training manifest changed")
        write_json(manifest, mp)
        history, steps, diagnostics, first, elapsed = [], [], [], 1, 0.
        resume = folder / "resume.pt"
        if resume.exists():
            saved = torch.load(resume, map_location="cpu", weights_only=False)
            validate_resume(saved, arm, initial, head_hash)
            with torch.no_grad(): patch.copy_(saved["patch"].to(patch.device))
            optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
            history, steps, diagnostics = saved["history"], saved["steps"], saved["diagnostics"]
            first, elapsed = saved["outer"] + 1, saved["elapsed_s"]
            random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["cpu_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
        else:
            snapshot(folder, patch, initial, 0)
        references, started = {}, time.monotonic()
        def reference(uid):
            if uid not in references:
                record = source["records"][uid]
                context = engine.context(record)
                with torch.no_grad(): a0, d0 = engine.run(record, context)
                references[uid] = (context, a0.detach(), wrong_target(record, patch.device), d0.detach().cpu())
            return references[uid]
        # Clean action equivalence is checked on the actual loaded native model.
        uid0 = source["candidates"][0][0]
        record0 = source["records"][uid0]
        context0, action0, _, _ = reference(uid0)
        with torch.no_grad():
            official = np.asarray(engine.wrapper.predict_action(record0["observation"], record0["task_name"])).reshape(8, 7)
            normalized, _, _ = engine.forward(context0["inputs"], context0["proprio"])
            ours = engine.wrapper.vla._unnormalize_actions(normalized.cpu().numpy()[0], engine.wrapper.gen_cfg.unnorm_key)
            error = float(np.abs(official - ours).max())
            require(error <= .02, "Native action interface mismatch")
        write_json({"signature": signature(), "status": "PASS", "native_action_max_error": error,
                    "feature_replacement_used": False}, folder / "GPU_INTERFACE.json")
        for outer in range(first, 501):
            uid = source["candidates"][(outer - 1) % 250][0]
            record = source["records"][uid]
            context, clean, wrong, clean_features = reference(uid)
            lr, moved = float(optimizer.param_groups[0]["lr"]), 0
            outer_losses = []
            for inner in range(50):
                seed = native().eot_seeds(outer, inner, 1)[0]
                tracing = outer in cfg()["diagnostic_outers"] and inner in [0, 49]
                optimizer.zero_grad(set_to_none=True)
                before = patch.detach().clone()
                action, features = engine.run(record, context, patch, seed)
                helper_action = None if helper is None or outer > 400 else helper(features, context["language"], context["state"])
                loss, detail = patch_objective(action, helper_action, clean, wrong, arm, outer)
                require(torch.isfinite(loss), "Nonfinite patch objective")
                native_gradient, helper_gradient, native_component = None, None, None
                if tracing:
                    native_gradient = torch.autograd.grad(F.mse_loss(action, wrong), patch, retain_graph=True)[0].detach()
                    native_component, helper_gradient = component_gradients(action, helper_action, clean, wrong, arm, outer, patch)
                loss.backward()
                gradient = patch.grad.detach().clone()
                require(torch.isfinite(gradient).all(), "Disconnected or nonfinite RGB gradient")
                action_before = action.detach()
                loss_value = float(loss.detach())
                feature_summary = None
                if tracing:
                    delta_features = features.detach().float() - clean_features.to(features.device).float()
                    feature_summary = {"dino_front_delta_l2": float(delta_features[:, 0].norm()),
                                       "dino_wrist_delta_l2": float(delta_features[:, 1].norm())}
                optimizer.step()
                with torch.no_grad():
                    proposal = patch.detach().clone(); patch.clamp_(0, 1)
                    delta = patch - before
                    step = {"outer": outer, "inner": inner, "uid": uid, "eot_seed": seed,
                            "loss": loss_value, "learning_rate": lr, "gradient_l2": float(gradient.norm()),
                            "rgb_update_rms": float(delta.square().mean().sqrt()), "rgb_update_max_abs": float(delta.abs().max()),
                            "gradient_dot_actual_step": float((gradient * delta).sum()),
                            "descent_step_cosine": cosine(-gradient, delta),
                            "clipped_fraction": float((proposal != patch).float().mean()),
                            "saturated_fraction": float(((patch <= 0) | (patch >= 1)).float().mean()),
                            "png_changed_fraction": float((patch.mul(255).round() != before.mul(255).round()).float().mean()), **detail}
                steps.append(step); outer_losses.append(loss_value)
                moved += int(not torch.equal(before, patch.detach()))
                if tracing:
                    with torch.no_grad():
                        after_action, _ = engine.run(record, context, patch, seed)
                        quantized = patch.detach().mul(255).round().div(255)
                        png_action, _ = engine.run(record, context, quantized, seed)
                    diag = {"outer": outer, "inner": inner, "uid": uid, "eot_seed": seed,
                            "beta": detail["beta"],
                            "native_uada_before": float(F.mse_loss(action_before, wrong)),
                            "native_uada_after": float(F.mse_loss(after_action, wrong)),
                            "native_action_step_l2": float((after_action - action_before).norm()),
                            "png_action_gap_l2": float((png_action - after_action).norm()),
                            "native_rgb_gradient_l2": float(native_gradient.norm()),
                            "native_gradient_vs_update_cosine": cosine(-native_gradient, patch.detach() - before),
                            "helper_rgb_gradient_l2": float(helper_gradient.norm()),
                            "native_component_rgb_gradient_l2": float(native_component.norm()),
                            "native_helper_gradient_cosine": cosine(native_component, helper_gradient),
                            "uada_helper_gradient_cosine": cosine(native_gradient, helper_gradient),
                            "helper_native_norm_ratio": float(helper_gradient.norm() / native_component.norm().clamp_min(1e-20)),
                            "component_gradient_sum_relative_error": float((gradient - native_component - helper_gradient).norm() / gradient.norm().clamp_min(1e-20)),
                            **feature_summary}
                    if helper_action is not None:
                        diag["helper_native_action_delta_error"] = float(((helper_action.detach() - clean) - (action_before - clean)).square().mean())
                    diagnostics.append(diag)
                    save_torch({"signature": signature(), "rgb_before": before.cpu(), "rgb_after": patch.detach().cpu(),
                                "actual_gradient": gradient.cpu(), "native_gradient": native_gradient.cpu(),
                                "native_component_gradient": native_component.cpu(), "helper_gradient": helper_gradient.cpu(),
                                "native_action_before": action_before.cpu(), "native_action_after": after_action.cpu(),
                                "png_action": png_action.cpu(), "clean_action": clean.cpu(), "wrong_target": wrong.cpu(),
                                "helper_action": None if helper_action is None else helper_action.detach().cpu(),
                                "dino_clean": clean_features, "dino_patched": features.detach().cpu(), **diag},
                               folder / f"probe_{outer:04d}_{inner:02d}.pt")
                    save_frame(pd.DataFrame(diagnostics), folder / "diagnostics.parquet")
                del action, features, helper_action, loss, gradient
            scheduler.step()
            elapsed_s = elapsed + time.monotonic() - started
            history.append({"outer": outer, "uid": uid, "episode_id": record["episode_id"], "mean_loss": float(np.mean(outer_losses)),
                            "learning_rate": lr, "rgb_updates": moved, "elapsed_s": elapsed_s})
            if outer in [125, 250, 375, 500]: snapshot(folder, patch, initial, outer)
            if outer % 10 == 0 or outer == 500:
                save_frame(pd.DataFrame(steps), folder / "inner_steps.parquet")
                save_frame(pd.DataFrame(history), folder / "history.parquet")
                save_torch({"signature": signature(), "arm": arm, "head_sha256": head_hash, "outer": outer,
                            "patch": patch.detach().cpu(), "initial": initial, "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "steps": steps, "history": history, "diagnostics": diagnostics,
                            "elapsed_s": elapsed_s, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, resume)
                write_json({"status": "RUNNING", "signature": signature(), "outer": outer, "updates": outer * 50,
                            "elapsed_s": elapsed_s, "estimated_remaining_s": elapsed_s / outer * (500 - outer)}, status_path)
            if outer % 25 == 0:
                print(f"PATCH {arm} outer={outer}/500 loss={np.mean(outer_losses):.6g} moves={moved}/50 elapsed_h={elapsed_s/3600:.2f} ETA_h={elapsed_s/outer*(500-outer)/3600:.2f}", flush=True)
        require(len(steps) == 25000 and len(diagnostics) == len(cfg()["diagnostic_outers"]) * 2, "Training trace incomplete")
        finish(status_path, [folder / p for p in ["outer_0500.pt", "outer_0500.png", "resume.pt", "inner_steps.parquet", "history.parquet", "diagnostics.parquet", "RUN_MANIFEST.json"]],
               outer=500, update_attempts=25000, actual_rgb_updates=sum(x["rgb_updates"] for x in history), elapsed_s=elapsed_s)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("arm", choices=ARMS)
    train(p.parse_args().arm)

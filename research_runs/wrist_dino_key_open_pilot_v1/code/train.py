"""One GPU/model load: clean intersection -> Key targets -> single fits -> two universal pilots."""
from __future__ import annotations
import json
import math
import random
import time
from collections import Counter
import numpy as np
import torch
from PIL import Image
from common import (ROOT, METHODS, config, run_signature, require, require_gpu, prepared, load_json,
                    atomic_json, atomic_torch, sha256_file, runtime_config)
from data import split_records, balanced
from key_math import close_slots, feature_loss, open_loss, transactional_step, action_metrics
from key_runtime import KeyEngine
from model_runtime import load_wrapper
from resources import snapshot

def initial_patch(): return torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(42))

def save_png(patch, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(patch.detach().cpu().permute(1, 2, 0).mul(255).round().byte().numpy(), "RGB").save(tmp)
    tmp.replace(path)

def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def restore_rng(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if s["cuda"]: torch.cuda.set_rng_state_all(s["cuda"])

def target_device(t, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}

def evaluate_features(engine, records, targets, patch, label):
    """Bounded PNG readback probes; never select a checkpoint or advance training RNG."""
    state = rng_state(); rows = []
    try:
        rgb = patch.detach().mul(255).round().div(255)
        with torch.no_grad():
            for r in records:
                t = target_device(targets[r["uid"]], patch.device)
                for seed, fixed in [(42, True), (42, False), (43, False), (44, False), (45, False)]:
                    a, k = engine.native(r, rgb, seed, fixed)
                    row = {"uid": r["uid"], "task_id": r["task_id"], "split": r["pilot_split"],
                           "label": label, "seed": seed, "fixed": fixed, "target_outcome": t["outcome"],
                           "open_loss": float(open_loss(a, t["mask"])), **action_metrics(a, t["mask"])}
                    if t["nonzero"]:
                        row.update({m+"_loss": float(feature_loss(k, t["clean_key"], t["target_key"], t["scale"], m)) for m in METHODS})
                    rows.append(row)
    finally: restore_rng(state)
    return rows

def fit(engine, records, targets, method, folder, universal, probe_records):
    cfg, signature = config(), run_signature(); folder.mkdir(parents=True, exist_ok=True)
    done = folder / "status.json"; resume = folder / "resume.pt"
    if done.exists() and load_json(done)["status"] == "COMPLETE":
        s = load_json(done); require(s["signature"] == signature and sha256_file(folder / "final.png") == s["png_sha256"], "Completed patch changed")
        return
    patch = initial_patch().to(engine.wrapper.device).requires_grad_(True)
    optimizer = torch.optim.AdamW([patch], lr=cfg["learning_rate"], weight_decay=0, eps=1e-6)
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
    if saved:
        require(saved["signature"] == signature, "Resume identity changed")
        with torch.no_grad(): patch.copy_(saved["patch"].to(patch))
        optimizer.load_state_dict(saved["optimizer"]); restore_rng(saved["rng"])
    completed = saved["completed"] if saved else 0
    counts = Counter(saved["counts"] if saved else {}); elapsed_before = saved["elapsed_s"] if saved else 0.
    total = cfg["universal_outer"]*cfg["inner_loop"] if universal else cfg["single_candidates"]
    attempt = str(time.time_ns()); attempt_dir = folder / "attempts" / attempt; attempt_dir.mkdir(parents=True)
    started = time.monotonic()
    if not saved:
        save_png(patch, folder / "initial.png")
        atomic_json(evaluate_features(engine, probe_records, targets, patch, "initial"), folder / "probe_initial.json")
    with (attempt_dir / "INNER_STEPS.jsonl").open("w") as stream:
        for index in range(completed, total):
            outer, inner = divmod(index, cfg["inner_loop"])
            r = records[outer % len(records)] if universal else records[0]
            t = target_device(targets[r["uid"]], patch.device)
            seed = cfg["data_seed"]*1000000 + (outer+1)*10000 + inner
            fixed = not universal
            if universal:
                factor = min(1., outer/cfg["warmup_outer"])
                if outer >= cfg["warmup_outer"]:
                    factor = .5*(1+math.cos(math.pi*(outer-cfg["warmup_outer"])/(cfg["cosine_outer"]-cfg["warmup_outer"])))
                optimizer.param_groups[0]["lr"] = cfg["learning_rate"]*factor
            row = {"index": index, "outer": outer+1, "inner": inner, "uid": r["uid"], "seed": seed,
                   "learning_rate": optimizer.param_groups[0]["lr"], "method": method, "attempt": attempt}
            before = patch.detach().clone(); optimizer.zero_grad(set_to_none=True)
            if not t["nonzero"]:
                counts["no_target"] += 1; row.update(outcome="NO_TARGET", gradient_l2=0.)
            else:
                # Gradient uses only the DINO Key path. Full native actions below are no-grad acceptance checks.
                key = engine.rgb_key(r, patch, seed, fixed)
                loss = feature_loss(key, t["clean_key"], t["target_key"], t["scale"], method)
                loss.backward(); gradient = float(patch.grad.norm())
                require(torch.isfinite(patch.grad).all(), "Nonfinite RGB Key gradient")
                del loss, key
                def check(p):
                    with torch.no_grad():
                        a, k = engine.native(r, p, seed, fixed)
                        return float(feature_loss(k, t["clean_key"], t["target_key"], t["scale"], method)), float(open_loss(a, t["mask"]))
                key_before, open_before = check(patch)
                result = transactional_step(patch, optimizer, check, key_before, open_before, cfg["backtracking"])
                category = "accepted" if result["accepted"] else "rejected"; counts[category] += 1
                row.update(outcome=category.upper(), gradient_l2=gradient, key_before=key_before,
                           open_before=open_before, **result)
            row.update(rgb_step_l2=float((patch.detach()-before).norm()),
                       quantized_pixels_changed=int((patch.detach().mul(255).round()!=before.mul(255).round()).sum()),
                       rgb_min=float(patch.detach().min()), rgb_max=float(patch.detach().max()))
            if patch.grad is not None:
                delta = patch.detach()-before
                row["gradient_dot_rgb_step"] = float((patch.grad*delta).sum())
                row["negative_gradient_step_cosine"] = float(-(patch.grad*delta).sum() /
                    (patch.grad.norm()*delta.norm()).clamp_min(1e-20))
            stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush()
            if (index+1) % cfg["inner_loop"] == 0 or index+1 == total:
                elapsed = elapsed_before + time.monotonic()-started
                atomic_torch({"signature": signature, "patch": patch.detach().cpu(), "optimizer": optimizer.state_dict(),
                              "completed": index+1, "counts": dict(counts), "rng": rng_state(), "elapsed_s": elapsed}, resume)
                atomic_json({"status": "RUNNING", "signature": signature, "completed_candidates": index+1,
                             "planned_candidates": total, "counts": dict(counts), "elapsed_s": elapsed, **snapshot()}, done)
                print(f"KEY_PILOT {folder.name} {method} {index+1}/{total} {dict(counts)}", flush=True)
    save_png(patch, folder / "final.png")
    probes = evaluate_features(engine, probe_records, targets, patch, "final")
    atomic_json(probes, folder / "probe_final.json")
    atomic_json({"status": "COMPLETE", "signature": signature, "completed_candidates": total,
                 "planned_candidates": total, "counts": dict(counts), "elapsed_s": elapsed_before+time.monotonic()-started,
                 "png_sha256": sha256_file(folder / "final.png"), "initial_png_sha256": sha256_file(folder / "initial.png"),
                 "only_key_loss_rgb_gradient": True, **snapshot()}, done)

def train():
    cfg, signature = config(), run_signature(); prepared(); require_gpu("training")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    final = ROOT / "training/STATUS.json"
    if final.exists() and load_json(final)["status"] in ("COMPLETE", "NO_TARGET", "NO_PIXEL_CHANGE"):
        require(load_json(final)["signature"] == signature, "Completed run identity drift")
        print("TRAINING_ALREADY_COMPLETE"); return
    bank = torch.load(ROOT / "data/EXPERT_CLOSE_BANK.pt", map_location="cpu", weights_only=False)
    engine = KeyEngine(load_wrapper(runtime_config(), "libero_spatial"))
    subset_file = ROOT / "data/DUAL_CLOSE.pt"
    if subset_file.exists():
        subset = torch.load(subset_file, map_location="cpu", weights_only=False)
        require(subset["signature"] == signature, "Subset drift")
        selected = subset["records"]
    else:
        eligible = []; native_rows = []
        for r in bank["records"]:
            with torch.no_grad(): action, _ = engine.native(r)
            mask = close_slots(r["expert_actions"], action.cpu())
            native_rows.append({"uid": r["uid"], "task_id": r["task_id"], "dual_close_slots": int(mask.sum()),
                                "clean_action": action.cpu().tolist()})
            if mask.any(): eligible.append({**r, "close_slots": mask, "clean_action": action.cpu()})
        train_set, validation = split_records(eligible, cfg["train_observations"], cfg["validation_observations"])
        selected = [{**r, "pilot_split": "train"} for r in train_set] + [{**r, "pilot_split": "validation"} for r in validation]
        atomic_torch({"signature": signature, "records": selected}, subset_file)
        atomic_json({"signature": signature, "expert_close_observations": 150, "dual_close_observations": len(eligible),
                     "train_uids": [r["uid"] for r in train_set], "validation_uids": [r["uid"] for r in validation],
                     "native_rows": native_rows}, ROOT / "reports/DUAL_CLOSE.json")
    del bank
    if not selected:
        atomic_json({"status": "NO_DUAL_CLOSE", "signature": signature,
                     "reason": "No expert/clean dual-close observation", **snapshot()}, final)
        return
    train_set = [r for r in selected if r["pilot_split"] == "train"]
    qa_path = ROOT / "reports/GPU_INTERFACE.json"
    if not qa_path.exists():
        qa = engine.validate(selected[0], initial_patch().to(engine.wrapper.device).requires_grad_(True))
        with torch.no_grad():
            a, k = engine.native(selected[0]); identity, _ = engine.native(selected[0], replacement=k)
        require(torch.allclose(a, identity, atol=1e-4, rtol=1e-4), "Zero Key intervention altered native action")
        with torch.no_grad():
            p = initial_patch().to(engine.wrapper.device)
            native, consumed = engine.native(selected[0], p, 42)
            direct = engine.rgb_key(selected[0], p, 42, False)
            require(torch.allclose(consumed, direct, atol=1e-5, rtol=1e-5), "Key gradient readout differs from consumed native Key")
            fixed, _ = engine.native(selected[0], p, 42, True)
            quantized, _ = engine.native(selected[0], p.mul(255).round().div(255), 42, True)
            require(torch.allclose(fixed, quantized, atol=1e-4, rtol=1e-4), "Fixed-placement PNG roundtrip altered native action")
        qa.update(signature=signature, post_k_norm_identity=True, direct_native_key_parity=True, png_fixed_parity=True,
                  replacements="Wrist spatial Key only; current-layer Q/V and prefix unchanged")
        atomic_json(qa, qa_path)
    target_path = ROOT / "data/KEY_TARGETS.pt"
    cached_targets = torch.load(target_path, map_location="cpu", weights_only=False) if target_path.exists() else None
    if cached_targets: require(cached_targets["signature"] == signature, "Cached Key targets changed identity")
    targets = cached_targets["targets"] if cached_targets else {}
    for r in selected:
        if r["uid"] in targets: continue
        targets[r["uid"]] = engine.make_target(r, r["clean_action"], cfg)
        atomic_torch({"signature": signature, "targets": targets}, target_path)
        print("KEY_TARGET", r["uid"], targets[r["uid"]]["outcome"], flush=True)
    atomic_json({"signature": signature, "all": dict(Counter(t["outcome"] for t in targets.values())),
                 "train": dict(Counter(targets[r["uid"]]["outcome"] for r in train_set)),
                 "train_count": len(train_set), "validation_count": len(selected)-len(train_set)}, ROOT / "reports/TARGET_SUMMARY.json")
    if not any(targets[r["uid"]]["nonzero"] for r in train_set):
        atomic_json({"status": "NO_TARGET", "signature": signature, "reason": "No nonzero action-improving Key target in training subset", **snapshot()}, final)
        return
    singles = balanced(train_set, cfg["single_observations"])
    for i, r in enumerate(singles):
        for method in METHODS:
            fit(engine, [r], targets, method, ROOT / "training/single" / f"sample_{i:02d}" / method, False, [r])
    for method in METHODS:
        fit(engine, train_set, targets, method, ROOT / "training/universal" / method, True, selected)
    statuses = [load_json(ROOT / "training/universal" / m / "status.json") for m in METHODS]
    unchanged = all(s["png_sha256"] == s["initial_png_sha256"] for s in statuses)
    atomic_json({"status": "NO_PIXEL_CHANGE" if unchanged else "COMPLETE", "signature": signature,
                 "universal_candidates_per_arm": 2000, "single_candidates_per_fit": cfg["single_candidates"],
                 "train_count": len(train_set), "validation_count": len(selected)-len(train_set), **snapshot()}, final)

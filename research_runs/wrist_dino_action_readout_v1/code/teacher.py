"""Finite RGB-response supervision from the complete frozen native policy."""
from __future__ import annotations
import time
import torch
from torch.nn import functional as F
from run_core import (ROOT, BASE, BASE_METHOD, bank, cfg, finish, locked, prepared, read_json,
                      require, require_gpu, save_torch, sha, signature, stage_done, write_json)


def build_cache():
    prepared()
    status_path = ROOT / "teacher/STATUS.json"
    with locked(ROOT / "teacher/worker.lock"):
        if stage_done(status_path, [ROOT / "teacher/INDEX.json"]):
            print("TEACHER_ALREADY_COMPLETE", flush=True)
            return
        require_gpu("training")
        from native_bridge import load_engine, wrong_target
        engine, source = load_engine(), bank()
        eta = cfg()["response_eta"]
        paths = [BASE / "training" / BASE_METHOD / "outer_0500.pt",
                 BASE / "training/m02_dino_key_b1/outer_0500.pt"]
        library = [torch.load(p, map_location="cpu", weights_only=False)["patch"] for p in paths]
        started, entries = time.monotonic(), []
        split = read_json(ROOT / "data/SPLIT.json")
        validation = set(split["validation_uids"])
        for index, uid in enumerate(sorted(source["records"])):
            path = ROOT / "teacher/records" / (str(index).zfill(5) + ".pt")
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                require(saved["signature"] == signature() and saved["uid"] == uid, "Teacher resume identity changed")
            else:
                record = source["records"][uid]
                context = engine.context(record)
                variant = index % 3
                value = library[variant] if variant < 2 else torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(cfg()["teacher_seed"] + index))
                p = value.to(engine.wrapper.device).detach().clone().requires_grad_(True)
                seed = cfg()["teacher_seed"] * 1000000 + index
                wrong = wrong_target(record, p.device)
                with torch.no_grad():
                    clean_action, clean_features = engine.run(record, context)
                action, features = engine.run(record, context, p, seed)
                clean_action, clean_features = clean_action.cpu(), clean_features.cpu()
                action_cpu, features_cpu = action.detach().cpu(), features.detach().cpu()
                gradient = torch.autograd.grad(F.mse_loss(action, wrong), p)[0]
                require(torch.isfinite(gradient).all(), "Nonfinite teacher RGB direction")
                gradient_l2 = float(gradient.norm())
                direction = -gradient.detach().sign()
                plus, minus = (p.detach() + eta * direction).clamp(0, 1), (p.detach() - eta * direction).clamp(0, 1)
                del action, features, gradient
                with torch.no_grad():
                    ap, fp = engine.run(record, context, plus, seed)
                    am, fm = engine.run(record, context, minus, seed)
                saved = {"signature": signature(), "uid": uid, "episode_id": record["episode_id"],
                         "task_id": record["task_id"], "split": "validation" if uid in validation else "train",
                         "features": torch.cat([clean_features, features_cpu, fp.cpu(), fm.cpu()]),
                         "actions": torch.cat([clean_action, action_cpu, ap.cpu(), am.cpu()]),
                         "language": context["language"].cpu(), "proprio": context["state"].cpu(),
                         "variant_order": ["clean", "p", "p_plus", "p_minus"], "patch_source": ["uada", "dino_key", "random"][variant],
                         "teacher_gradient_l2": gradient_l2,
                         "eot_seed": seed, "eta": eta, "actual_plus_l2": float((plus - p.detach()).norm()),
                         "actual_minus_l2": float((minus - p.detach()).norm()),
                         "plus_minus_quantized_changed_fraction": float((plus.mul(255).round() != minus.mul(255).round()).float().mean())}
                require(saved["features"].shape == (4, 2, 256, 1024) and saved["actions"].shape == (4, 8), "Teacher tensor layout mismatch")
                save_torch(saved, path)
                del context, p, plus, minus, direction, ap, am, fp, fm
            entries.append({"uid": uid, "path": str(path.relative_to(ROOT)), "sha256": sha(path),
                            "split": saved["split"], "episode_id": saved["episode_id"], "task_id": saved["task_id"]})
            if (index + 1) % 20 == 0 or index == 0:
                write_json({"status": "RUNNING", "signature": signature(), "complete_windows": index + 1,
                            "total_windows": len(source["records"]), "elapsed_s": time.monotonic() - started}, status_path)
                print(f"TEACHER {index+1}/{len(source['records'])} elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True)
        write_json({"signature": signature(), "entries": entries, "teacher": "complete native frozen OFT",
                    "variants": 4, "eta": eta}, ROOT / "teacher/INDEX.json")
        finish(status_path, [ROOT / "teacher/INDEX.json"], windows=len(entries), elapsed_s=time.monotonic() - started,
               teacher_forwards=4 * len(entries), teacher_backwards=len(entries))


if __name__ == "__main__":
    build_cache()

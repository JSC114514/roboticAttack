"""Train only the small readout; cache all BF16 teacher features on the GPU."""
from __future__ import annotations
import time
import pandas as pd
import torch
from readout import ActionReadout, response_loss
from run_core import (ROOT, cfg, finish, locked, prepared, read_json, require, require_gpu,
                      save_frame, save_torch, sha, signature, stage_done, write_json)


def predict(model, features, language, proprio):
    n = features.shape[0]
    return model(features.flatten(0, 1), language.repeat_interleave(4, 0), proprio.repeat_interleave(4, 0)).reshape(n, 4, 8)


def fit():
    prepared()
    status = ROOT / "readout/STATUS.json"
    with locked(ROOT / "readout/worker.lock"):
        if stage_done(status, [ROOT / "readout/best.pt"]):
            print("HEAD_ALREADY_COMPLETE", flush=True)
            return
        require_gpu("training")
        require(stage_done(ROOT / "teacher/STATUS.json"), "Teacher cache incomplete")
        index = read_json(ROOT / "teacher/INDEX.json")
        rows = []
        for entry in index["entries"]:
            path = ROOT / entry["path"]
            require(sha(path) == entry["sha256"], "Teacher feature changed")
            rows.append(torch.load(path, map_location="cpu", weights_only=False))
        device = torch.device("cuda")
        features = torch.stack([x["features"] for x in rows]).to(device)
        actions = torch.stack([x["actions"] for x in rows]).to(device).float()
        language = torch.cat([x["language"] for x in rows]).to(device).float()
        proprio = torch.cat([x["proprio"] for x in rows]).to(device).float()
        train = torch.tensor([i for i, x in enumerate(rows) if x["split"] == "train"], device=device)
        validation = torch.tensor([i for i, x in enumerate(rows) if x["split"] == "validation"], device=device)
        task_ids = [x["task_id"] for x in rows]
        del rows
        eta = cfg()["response_eta"]
        ta = actions[train]
        scales = {"action": ta.flatten(0, 1).var(0, unbiased=False).clamp_min(1e-6),
                  "delta": (ta[:, 1:] - ta[:, :1]).flatten(0, 1).var(0, unbiased=False).clamp_min(1e-6),
                  "response": ((ta[:, 2] - ta[:, 3]) / (2 * eta)).var(0, unbiased=False).clamp_min(1e-6)}
        dimensions = {"feature_dim": features.shape[-1], "language_dim": language.shape[-1], "width": 128, "tokens": 256}
        torch.manual_seed(cfg()["head_seed"])
        model = ActionReadout(**dimensions).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        history, best, first, elapsed = [], float("inf"), 1, 0.
        resume = ROOT / "readout/resume.pt"
        if resume.exists():
            saved = torch.load(resume, map_location=device, weights_only=False)
            require(saved["signature"] == signature() and saved["cache_sha256"] == sha(ROOT / "teacher/INDEX.json"), "Head resume changed")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            history, best, first, elapsed = saved["history"], saved["best"], saved["epoch"] + 1, saved["elapsed_s"]
            torch.set_rng_state(saved["cpu_rng"].cpu()); torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        started = time.monotonic()
        def evaluate(ids, feature_ids=None):
            totals, count = {"action": 0., "delta": 0., "response": 0.}, 0
            with torch.no_grad():
                for start in range(0, len(ids), 64):
                    batch = ids[start:start+64]
                    fidx = batch if feature_ids is None else feature_ids[start:start+64]
                    output = predict(model, features[fidx], language[batch], proprio[batch])
                    _, terms = response_loss(output, actions[batch], scales, eta)
                    for key in totals:
                        totals[key] += float(terms[key]) * len(batch)
                    count += len(batch)
            return {k: v / count for k, v in totals.items()}
        for epoch in range(first, 31):
            model.train()
            order = train[torch.randperm(len(train), device=device)]
            total = 0.
            for start in range(0, len(order), 64):
                batch = order[start:start+64]
                optimizer.zero_grad(set_to_none=True)
                output = predict(model, features[batch], language[batch], proprio[batch])
                loss, _ = response_loss(output, actions[batch], scales, eta)
                require(torch.isfinite(loss), "Nonfinite head objective")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step(); total += float(loss.detach()) * len(batch)
            model.eval()
            metrics = evaluate(validation)
            score = sum(metrics.values())
            require(all(torch.isfinite(p).all() for p in model.parameters()), "Nonfinite head weights")
            history.append({"epoch": epoch, "training_loss": total / len(train), "validation_loss": score,
                            **{"validation_" + k: v for k, v in metrics.items()}})
            if score < best:
                best = score
                save_torch({"signature": signature(), "state_dict": model.state_dict(), "dimensions": dimensions,
                            "scales": scales, "epoch": epoch, "validation_loss": best,
                            "cache_sha256": sha(ROOT / "teacher/INDEX.json")}, ROOT / "readout/best.pt")
            save_torch({"signature": signature(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "history": history, "best": best, "elapsed_s": elapsed + time.monotonic() - started,
                        "cache_sha256": sha(ROOT / "teacher/INDEX.json"), "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state()}, resume)
            save_frame(pd.DataFrame(history), ROOT / "readout/history.parquet")
            write_json({"status": "RUNNING", "signature": signature(), "epoch": epoch,
                        "validation_loss": score, "best_validation_loss": best}, status)
            print(f"HEAD epoch={epoch}/30 val={score:.6g} best={best:.6g}", flush=True)
        saved = torch.load(ROOT / "readout/best.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"]); model.eval()
        plain = evaluate(validation)
        shuffled = validation.clone()
        for task in set(task_ids):
            positions = [i for i, source_idx in enumerate(validation.tolist()) if task_ids[source_idx] == task]
            if len(positions) > 1:
                pos = torch.tensor(positions, device=device)
                shuffled[pos] = validation[pos.roll(1)]
        shuffle_metrics = evaluate(validation, shuffled)
        write_json({"signature": signature(), "validation": plain, "same_task_feature_shuffle": shuffle_metrics,
                    "shuffle_error_increase": sum(shuffle_metrics.values()) - sum(plain.values()),
                    "parameter_count": sum(p.numel() for p in model.parameters()), "selected_epoch": saved["epoch"],
                    "train_windows": len(train), "validation_windows": len(validation)}, ROOT / "readout/DIAGNOSTICS.json")
        finish(status, [ROOT / "readout/best.pt", ROOT / "readout/history.parquet", ROOT / "readout/DIAGNOSTICS.json"],
               elapsed_s=elapsed + time.monotonic() - started, best_validation_loss=best)


if __name__ == "__main__":
    fit()

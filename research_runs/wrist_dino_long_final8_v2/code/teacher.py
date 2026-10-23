"""Sample-specific, actually RGB-realizable teachers; frozen VLA, 100 steps/state."""
from __future__ import annotations

import argparse
import time
import torch
from common import (ROOT, atomic_json, atomic_torch, config, digest, load_bank, prepared,
                    require, require_gpu, run_signature, runtime_config)
from engine import OFTEngine
from model_runtime import load_wrapper
from objectives import target_action, action_loss


def train_teachers(family, shard):
    cfg, prep, bank = config(), prepared(), load_bank()
    require_gpu("training")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    signature = run_signature()
    path = ROOT / "teachers" / f"{family}_{shard}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
    if saved:
        require(saved["signature"] == signature and saved["family"] == family and saved["shard"] == shard,
                "Teacher resume identity changed")
        if saved["complete"]:
            return
    records = {} if saved is None else saved["records"]
    keys = sorted(bank["observations"])[shard::cfg["teacher_shards_per_family"]]
    wrapper = load_wrapper(runtime_config(), "libero_10")
    engine = OFTEngine(wrapper)
    started = time.monotonic()
    for number, uid in enumerate(keys):
        if uid in records:
            continue
        record = bank["observations"][uid]
        seed = cfg["data_seed"] + int(digest(uid)[:8], 16)
        initial = torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(seed))
        patch = initial.to(wrapper.device).requires_grad_(True)
        inputs, proprio = engine.base(record)
        with torch.no_grad():
            clean_action, clean = engine.forward(inputs, proprio)
            target = target_action(clean_action, family, prep["bounds"], cfg)
            clean_loss = float(action_loss(clean_action, target, family, cfg))
            first, _ = engine.forward(inputs, proprio, engine.pixels(record, patch, seed, True, True))
            first_loss = float(action_loss(first, target, family, cfg))
        optimizer = torch.optim.AdamW([patch], lr=cfg["teacher_lr"], weight_decay=0)
        for step in range(cfg["teacher_updates"]):
            optimizer.zero_grad(set_to_none=True)
            pixels = engine.pixels(record, patch, seed + step, True, True)
            action, _ = engine.forward(inputs, proprio, pixels)
            loss = action_loss(action, target, family, cfg)
            loss.backward()
            require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Invalid teacher RGB gradient")
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
            del loss, action, pixels, _
        with torch.no_grad():
            # Evaluate the same deployment-quantized RGB permission and same EOT as initial.
            displayed = patch.detach().mul(255).round().div(255)
            pixels = engine.pixels(record, displayed, seed, True, True)
            action, caps = engine.forward(inputs, proprio, pixels)
            restored, restored_caps = engine.forward(inputs, proprio, pixels, replacement=clean["dino"])
            require(torch.equal(caps["siglip"], restored_caps["siglip"]), "Restore changed attacked SigLIP")
            final_loss = float(action_loss(action, target, family, cfg))
            restored_loss = float(action_loss(restored, target, family, cfg))
            valid = final_loss < min(first_loss, clean_loss) and restored_loss > final_loss
        records[uid] = {"patch": displayed.cpu(), "valid": bool(valid), "initial_target_mse": first_loss,
                        "clean_target_mse": clean_loss,
                        "final_target_mse": final_loss, "dino_restore_target_mse": restored_loss,
                        "dino": caps["dino"].detach().cpu(), "eot_seed": seed, "updates": 100}
        del optimizer, pixels, action, restored, caps, restored_caps, patch
        if len(records) % 10 == 0 or len(records) == len(keys):
            atomic_torch({"signature": signature, "family": family, "shard": shard,
                          "records": records, "complete": len(records) == len(keys)}, path)
        if (number + 1) % 10 == 0 or len(records) == len(keys):
            print(f"TEACHER {family} shard={shard} {len(records)}/{len(keys)} valid={sum(r['valid'] for r in records.values())}", flush=True)
    atomic_json({"status": "COMPLETE", "signature": signature, "family": family,
                 "states": len(records), "elapsed_s": time.monotonic() - started}, path.with_suffix(".json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("uada", "tma"), required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    train_teachers(args.family, args.shard)

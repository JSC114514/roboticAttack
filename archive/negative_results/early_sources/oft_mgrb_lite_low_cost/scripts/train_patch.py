#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import time

import pandas as pd
import torch

from common import CONFIG_PATH, ROOT, atomic_json, atomic_parquet, atomic_torch, load_json, load_manifest, require, require_gpu, sha256_file
from model_runtime import differentiable_action_and_features, load_wrapper
from patch_ops import render_latent


def load_samples(expected_manifest_sha256: str) -> list[dict]:
    samples = []
    for task in range(10):
        path = ROOT / "outputs" / "anchors" / f"task_{task:02d}.pt"
        require(path.is_file(), f"Missing anchor shard: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(len(payload["records"]) == 2, f"Invalid anchor shard: {path}")
        require(payload["manifest_sha256"] == expected_manifest_sha256, f"Stale anchor shard: {path}")
        samples.extend(payload["records"])
    require(len(samples) == 20, "Expected 20 attack-training samples")
    return sorted(samples, key=lambda row: (row["task_id"], row["state_index"]))


def save_checkpoint(method: str, step: int, latent: torch.Tensor, cfg: dict, metadata: dict) -> str:
    path = ROOT / "artifacts" / "patches" / method / f"step_{step:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch({
        "experiment_id": cfg["experiment_id"], "method": method, "step": step,
        "latent": latent.detach().float().cpu(), "rendered_shape": cfg["rendered_patch_shape"],
        **metadata,
    }, path)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mgrb", "generic"], required=True)
    args = parser.parse_args()
    print(json.dumps(require_gpu()), flush=True)
    cfg = load_json(CONFIG_PATH)
    train_manifest = load_manifest("MGRB_ATTACK_TRAIN_STATES.json")
    samples = load_samples(train_manifest["manifest_sha256"])
    suite = cfg["primary_suite"]
    wrapper = load_wrapper(cfg, suite)
    init_path = ROOT / "artifacts" / "PATCH_INIT.pt"
    init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
    latent = init_payload["latent"].to(device=wrapper.device, dtype=torch.float32).clone().requires_grad_(True)
    require(latent.numel() == 192, "Trainable parameter count drift")
    optimizer = torch.optim.AdamW(
        [latent], lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    checkpoints = set(int(x) for x in cfg["optimization_steps"])
    final_step = max(checkpoints)
    final_output = ROOT / "outputs" / f"training_{args.method}.parquet"
    status_output = ROOT / "reports" / f"TRAIN_{args.method.upper()}_STATUS.json"
    if final_output.is_file() and status_output.is_file():
        complete = pd.read_parquet(final_output)
        require(len(complete) == final_step and int(complete.step.max()) == final_step, "Corrupt completed training output")
        print(f"ALREADY_COMPLETE method={args.method}")
        return
    metadata = {
        "seed": int(cfg["seed"]), "patch_init_sha256": sha256_file(init_path),
        "config_sha256": sha256_file(CONFIG_PATH), "trainable_parameters": 192,
        "training_states": 20, "optimizer": cfg["optimizer"],
        "learning_rate": float(cfg["learning_rate"]), "lambda_align": float(cfg["lambda_align"]),
        "dino_hook": "vla.vision_backbone.featurizer output call_index=1 (frozen VF1 site)",
    }
    checkpoint_hashes = {0: save_checkpoint(args.method, 0, latent, cfg, metadata)}
    rows = []
    start_step = 1
    elapsed_before = 0.0
    resume_path = ROOT / "outputs" / f"training_{args.method}.resume.pt"
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=wrapper.device, weights_only=False)
        require(resume["method"] == args.method and resume["patch_init_sha256"] == sha256_file(init_path), "Resume provenance drift")
        require(resume["config_sha256"] == sha256_file(CONFIG_PATH), "Resume config drift")
        latent.data.copy_(resume["latent"].to(wrapper.device))
        optimizer.load_state_dict(resume["optimizer"])
        rows = resume["rows"]
        start_step = int(resume["step"]) + 1
        elapsed_before = float(resume["elapsed_s"])
        checkpoint_hashes.update({int(k): v for k, v in resume.get("checkpoint_hashes", {}).items()})
        print(f"RESUME method={args.method} from_step={start_step}", flush=True)
    started = time.time()
    torch.manual_seed(int(cfg["seed"]))
    try:
        for step in range(start_step, final_step + 1):
            sample = samples[(step - 1) % len(samples)]
            patch = render_latent(latent)
            optimizer.zero_grad(set_to_none=True)
            wrapper.vla.zero_grad(set_to_none=True)
            adv_action, adv_features, _pixels = differentiable_action_and_features(
                wrapper, sample["observation"], sample["task_name"], patch,
                f"upa_rfas_{suite}", ("dino",)
            )
            clean_action = sample["clean_action_normalized"].to(wrapper.device).float()
            action_l2 = torch.linalg.vector_norm(adv_action.float() - clean_action)
            action_loss = -action_l2
            clean_dino = sample["clean_dino"].to(wrapper.device).float()
            patch_delta = (adv_features["dino"].float() - clean_dino).reshape(-1)
            patch_norm = torch.linalg.vector_norm(patch_delta).clamp_min(1e-8)
            anchor = sample["geometry_anchor"].to(wrapper.device).float()
            alignment_cosine = torch.dot(patch_delta / patch_norm, anchor)
            align_loss = 1.0 - alignment_cosine
            loss = action_loss if args.method == "generic" else action_loss + float(cfg["lambda_align"]) * align_loss
            require(loss.requires_grad and torch.isfinite(loss).item(), "Training loss is non-differentiable or non-finite")
            component_grad_action = float("nan")
            component_grad_align = float("nan")
            if step == 1 and args.method == "mgrb":
                grad_action = torch.autograd.grad(action_loss, latent, retain_graph=True, allow_unused=False)[0]
                grad_align = torch.autograd.grad(align_loss, latent, retain_graph=True, allow_unused=False)[0]
                component_grad_action = float(torch.linalg.vector_norm(grad_action.float()).item())
                component_grad_align = float(torch.linalg.vector_norm(grad_align.float()).item())
                require(component_grad_action > 0 and component_grad_align > 0, "A registered MGRB loss component has zero initial gradient")
            loss.backward()
            require(latent.grad is not None and torch.isfinite(latent.grad).all(), "Invalid latent gradient")
            grad_norm = float(torch.linalg.vector_norm(latent.grad.detach()).item())
            torch.nn.utils.clip_grad_norm_([latent], max_norm=float(cfg["gradient_clip_norm"]))
            optimizer.step()
            with torch.no_grad():
                latent.clamp_(0.0, 1.0)
            row = {
                "method": args.method, "step": step, "task_id": int(sample["task_id"]),
                "state_index": int(sample["state_index"]), "total_loss": float(loss.detach()),
                "action_loss": float(action_loss.detach()), "action_l2": float(action_l2.detach()),
                "align_loss": float(align_loss.detach()), "alignment_cosine": float(alignment_cosine.detach()),
                "patch_delta_norm": float(patch_norm.detach()), "latent_grad_l2": grad_norm,
                "component_grad_action_l2": component_grad_action,
                "component_grad_align_l2": component_grad_align,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "elapsed_s": elapsed_before + time.time() - started,
            }
            rows.append(row)
            if step in checkpoints:
                checkpoint_hashes[step] = save_checkpoint(args.method, step, latent, cfg, {
                    **metadata, "walltime_s": elapsed_before + time.time() - started,
                    "forward_passes": step, "backward_passes": step,
                    "training_sample_visits": step,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                })
            if step % 10 == 0 or step in checkpoints:
                atomic_parquet(pd.DataFrame(rows), ROOT / "outputs" / f"training_{args.method}.partial.parquet")
                atomic_torch({
                    "method": args.method, "step": step, "latent": latent.detach().float().cpu(),
                    "optimizer": optimizer.state_dict(), "rows": rows,
                    "elapsed_s": elapsed_before + time.time() - started,
                    "patch_init_sha256": sha256_file(init_path), "config_sha256": sha256_file(CONFIG_PATH),
                    "checkpoint_hashes": checkpoint_hashes,
                }, resume_path)
                print(f"TRAIN method={args.method} step={step}/{final_step} action_l2={float(action_l2):.6f} align={float(alignment_cosine):.6f}", flush=True)
        frame = pd.DataFrame(rows)
        atomic_parquet(frame, final_output)
        partial = ROOT / "outputs" / f"training_{args.method}.partial.parquet"
        partial.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
        atomic_json({
            "status": "COMPLETE", "method": args.method, "steps": final_step,
            "checkpoints": checkpoint_hashes, "walltime_s": elapsed_before + time.time() - started,
            "gpu_hours": (elapsed_before + time.time() - started) / 3600.0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }, status_output)
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

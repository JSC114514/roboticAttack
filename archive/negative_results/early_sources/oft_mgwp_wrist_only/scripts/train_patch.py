#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time

import pandas as pd
import torch
from PIL import Image

from common import CONFIG_PATH, ROOT, action_scale, atomic_json, atomic_parquet, atomic_torch, load_json, require, require_gpu, sha256_file
from runtime import differentiable, load_wrapper


def metrics(action: torch.Tensor, clean: torch.Tensor, scale: torch.Tensor, beta: float):
    delta = (action.float() - clean.float()) / scale
    d_action = torch.linalg.vector_norm(delta, dim=-1).mean()
    d_persist = torch.linalg.vector_norm(delta[:, :, :6].mean(dim=1), dim=-1).mean()
    return d_action, d_persist, d_action + beta * d_persist


def save_patch(method: str, step: int, patch: torch.Tensor, metadata: dict) -> str:
    directory = ROOT / "artifacts" / "patches" / method
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step:04d}.pt"
    atomic_torch({"method": method, "step": step, "patch": patch.detach().float().cpu(), **metadata}, path)
    rgb = patch.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    Image.fromarray(rgb).save(directory / f"step_{step:04d}.png")
    return sha256_file(path)


def training_index(step: int, size: int, seed: int) -> int:
    epoch, offset = divmod(step - 1, size)
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["persistent", "dino_norm", "mgwp"], required=True)
    args = parser.parse_args()
    print(json.dumps(require_gpu()), flush=True)
    cfg = load_json(CONFIG_PATH)
    bank_path = ROOT / "artifacts" / "GEOMETRY_TRAIN_BANK.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    records = bank["records"]
    require(len(records) == 300, "Training bank incomplete")
    xy = tuple(bank["selected_xy"])
    wrapper = load_wrapper(cfg)
    torch.manual_seed(int(cfg["seed"]))
    patch = torch.rand(tuple(cfg["patch_shape"]), device=wrapper.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([patch], lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scale = torch.as_tensor(action_scale(cfg), device=wrapper.device)
    beta = float(cfg["beta_persistent"])
    checkpoints = set(cfg["checkpoint_steps"])
    final_step = int(cfg["optimization_steps"])
    resume_path = ROOT / "outputs" / f"train_{args.method}.resume.pt"
    rows, start_step, lambda_aux = [], 1, None
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=wrapper.device, weights_only=False)
        patch.data.copy_(resume["patch"])
        optimizer.load_state_dict(resume["optimizer"])
        rows, start_step, lambda_aux = resume["rows"], int(resume["step"]) + 1, resume["lambda_aux"]
    metadata = {"experiment_id": cfg["experiment_id"], "selected_xy": list(xy), "training_states": 300,
                "beta_persistent": beta, "mechanism_gradient_ratio": cfg["mechanism_gradient_ratio"],
                "bank_sha256": sha256_file(bank_path), "config_sha256": sha256_file(CONFIG_PATH)}
    if start_step == 1:
        save_patch(args.method, 0, patch, metadata)
    started = time.time()
    try:
        for step in range(start_step, final_step + 1):
            record = records[training_index(step, len(records), int(cfg["seed"]))]
            optimizer.zero_grad(set_to_none=True)
            wrapper.vla.zero_grad(set_to_none=True)
            action, dino = differentiable(wrapper, record["observation"], record["task_name"], patch, xy)
            clean_action = record["clean_action"].to(wrapper.device).float()
            d_action, d_persist, main_score = metrics(action, clean_action, scale, beta)
            delta = (dino.float() - record["clean_dino"].to(wrapper.device).float()).reshape(-1)
            dino_norm = torch.linalg.vector_norm(delta)
            qr = torch.dot(delta, record["range_direction"].to(wrapper.device).float())
            ql = torch.dot(delta, record["lateral_direction"].to(wrapper.device).float())
            tau = float(cfg["smoothmax_tau"])
            mechanism = tau * torch.logsumexp(torch.stack((qr, ql)) / tau, dim=0)
            main_loss = -main_score
            aux_score = dino_norm if args.method == "dino_norm" else mechanism
            aux_loss = -aux_score
            if args.method == "persistent":
                main_loss.backward()
                grad_main_norm = float(torch.linalg.vector_norm(patch.grad))
                grad_aux_norm = 0.0
                conflict = False
            else:
                g_main = torch.autograd.grad(main_loss, patch, retain_graph=True)[0]
                g_aux = torch.autograd.grad(aux_loss, patch)[0]
                grad_main_norm = float(torch.linalg.vector_norm(g_main))
                grad_aux_norm = float(torch.linalg.vector_norm(g_aux))
                require(grad_main_norm > 0 and grad_aux_norm > 0, "Zero component gradient")
                if lambda_aux is None:
                    lambda_aux = float(cfg["mechanism_gradient_ratio"]) * grad_main_norm / grad_aux_norm
                dot = torch.sum(g_main * g_aux)
                conflict = bool(dot < 0)
                if args.method == "mgwp" and conflict:
                    g_aux = g_aux - dot / torch.sum(g_main * g_main).clamp_min(1e-12) * g_main
                patch.grad = g_main + float(lambda_aux) * g_aux
            require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Invalid patch gradient")
            total_grad = float(torch.linalg.vector_norm(patch.grad))
            torch.nn.utils.clip_grad_norm_([patch], float(cfg["gradient_clip_norm"]))
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
            rows.append({"method": args.method, "step": step, "task_id": record["task_id"],
                         "state_index": record["state_index"], "phase_proxy": record["phase_proxy"],
                         "d_action": float(d_action.detach()), "d_persistent": float(d_persist.detach()),
                         "dino_norm": float(dino_norm.detach()), "q_range": float(qr.detach()), "q_lateral": float(ql.detach()),
                         "mechanism_score": float(mechanism.detach()), "lambda_aux": float(lambda_aux or 0),
                         "gradient_conflict": int(conflict), "grad_main_l2": grad_main_norm,
                         "grad_aux_l2": grad_aux_norm, "grad_total_l2": total_grad,
                         "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "elapsed_s": time.time() - started})
            if step % 10 == 0 or step in checkpoints:
                atomic_parquet(pd.DataFrame(rows), ROOT / "outputs" / f"train_{args.method}.partial.parquet")
                atomic_torch({"method": args.method, "step": step, "patch": patch.detach(), "optimizer": optimizer.state_dict(),
                              "rows": rows, "lambda_aux": lambda_aux}, resume_path)
                print(f"TRAIN method={args.method} step={step}/{final_step} action={float(d_action):.4f} persistent={float(d_persist):.4f}", flush=True)
            if step in checkpoints:
                save_patch(args.method, step, patch, {**metadata, "lambda_aux": float(lambda_aux or 0)})
        atomic_parquet(pd.DataFrame(rows), ROOT / "outputs" / f"train_{args.method}.parquet")
        atomic_json({"status": "COMPLETE", "method": args.method, "steps": final_step,
                     "lambda_aux": float(lambda_aux or 0), "selected_xy": list(xy),
                     "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())},
                    ROOT / "reports" / f"TRAIN_{args.method.upper()}.json")
        resume_path.unlink(missing_ok=True)
        (ROOT / "outputs" / f"train_{args.method}.partial.parquet").unlink(missing_ok=True)
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

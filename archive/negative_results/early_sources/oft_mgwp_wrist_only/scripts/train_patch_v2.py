#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from common import ROOT, V2_CONFIG_PATH, atomic_json, atomic_parquet, atomic_torch, load_json, require, require_gpu, sha256_file
from runtime import differentiable, load_wrapper
from v2_utils import action_terms, combine_gradients, load_direction_artifact, patch_path, record_key, scale_tensor, signed_scores, v2_auxiliary_loss


METHODS = ("persistent", "dino_norm", "mgwp_v1", "mgwp_v2")


def training_index(step: int, size: int, seed: int) -> int:
    epoch, offset = divmod(step - 1, size)
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def save_patch(method: str, step: int, patch: torch.Tensor, metadata: dict) -> None:
    path = patch_path(method, step)
    atomic_torch({"method": method, "step": step, "patch": patch.detach().float().cpu(), **metadata}, path)
    rgb = patch.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    Image.fromarray(rgb).save(path.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    print(json.dumps(require_gpu()), flush=True)
    cfg = load_json(V2_CONFIG_PATH)
    require(tuple(cfg["train_methods"]) == METHODS, "Training method contract drift")
    bank_path = ROOT / "artifacts" / "GEOMETRY_TRAIN_BANK.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    records = bank["records"]
    require(len(records) == cfg["training_state_count"], "Training bank incomplete")
    require(list(bank["selected_xy"]) == cfg["patch_xy"], "Patch location drift")
    artifact = load_direction_artifact("cpu")
    directions = {record_key(row): row for row in artifact["records"]}
    require(len(directions) == len(records), "Direction-key coverage drift")

    output_dir = ROOT / "outputs" / "mgwp_v2"
    final_output = output_dir / f"train_{args.method}.parquet"
    final_report = ROOT / "reports" / f"MGWP_V2_TRAIN_{args.method.upper()}.json"
    final_patch = patch_path(args.method, int(cfg["optimization_steps"]))
    if final_output.is_file() and final_report.is_file() and final_patch.is_file():
        frame = pd.read_parquet(final_output)
        require(len(frame) == cfg["optimization_steps"] and int(frame.step.max()) == cfg["optimization_steps"], "Completed training output is invalid")
        print(f"ALREADY_COMPLETE method={args.method}")
        return

    wrapper = load_wrapper(cfg)
    torch.manual_seed(int(cfg["seed"]))
    patch = torch.rand(tuple(cfg["patch_shape"]), device=wrapper.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([patch], lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scale = scale_tensor(cfg, wrapper.device)
    beta = float(cfg["beta_persistent"])
    rho = float(cfg["rho_primary"])
    xy = tuple(map(int, cfg["patch_xy"]))
    checkpoints = set(map(int, cfg["checkpoint_steps"]))
    final_step = int(cfg["optimization_steps"])
    resume_path = output_dir / f"train_{args.method}.resume.pt"
    partial_path = output_dir / f"train_{args.method}.partial.parquet"
    rows: list[dict] = []
    start_step = 1
    fixed_lambda: float | None = None
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=wrapper.device, weights_only=False)
        require(resume["method"] == args.method, "Resume method mismatch")
        patch.data.copy_(resume["patch"])
        optimizer.load_state_dict(resume["optimizer"])
        rows = resume["rows"]
        start_step = int(resume["step"]) + 1
        fixed_lambda = resume["fixed_lambda"]
    metadata = {
        "experiment_id": cfg["experiment_id"], "selected_xy": list(xy),
        "training_states": len(records), "rho_primary": rho,
        "direction_sha256": sha256_file(ROOT / "artifacts" / "mgwp_v2" / "ACTION_SENSITIVE_DIRECTION.pt"),
        "bank_sha256": sha256_file(bank_path), "config_sha256": sha256_file(V2_CONFIG_PATH),
    }
    if start_step == 1:
        save_patch(args.method, 0, patch, metadata)
    started = time.time()
    try:
        for step in range(start_step, final_step + 1):
            index = training_index(step, len(records), int(cfg["seed"]))
            record = records[index]
            direction = directions[record_key(record)]
            optimizer.zero_grad(set_to_none=True)
            wrapper.vla.zero_grad(set_to_none=True)
            action, dino = differentiable(wrapper, record["observation"], record["task_name"], patch, xy)
            clean_action = record["clean_action"].to(wrapper.device).float()
            clean_dino = record["clean_dino"].to(wrapper.device).float()
            delta_action, d_action, d_persistent, main_score = action_terms(action, clean_action, scale, beta)
            delta_dino = dino.float() - clean_dino
            dino_norm = torch.linalg.vector_norm(delta_dino)
            u_range = record["range_direction"].to(wrapper.device).float().reshape(-1)
            u_lateral = record["lateral_direction"].to(wrapper.device).float().reshape(-1)
            q_range = torch.dot(delta_dino.reshape(-1), u_range)
            q_lateral = torch.dot(delta_dino.reshape(-1), u_lateral)
            tau = float(cfg["smoothmax_tau"])
            v1_mechanism = tau * torch.logsumexp(torch.stack((q_range, q_lateral)) / tau, dim=0)
            signed_dino, signed_action = signed_scores(delta_action, delta_dino, direction, artifact)
            v2_aux, mechanism_margin_loss, action_margin_loss = v2_auxiliary_loss(signed_dino, signed_action, cfg)
            main_loss = -main_score

            grad_aux_norm = 0.0
            effective_ratio = 0.0
            conflict = False
            lambda_value = 0.0
            if args.method == "persistent":
                main_loss.backward()
                grad_main_norm = float(torch.linalg.vector_norm(patch.grad))
            else:
                if args.method == "dino_norm":
                    aux_loss = -dino_norm
                elif args.method == "mgwp_v1":
                    aux_loss = -v1_mechanism
                else:
                    aux_loss = v2_aux
                g_main = torch.autograd.grad(main_loss, patch, retain_graph=True)[0]
                g_aux = torch.autograd.grad(aux_loss, patch)[0]
                grad_main_norm = float(torch.linalg.vector_norm(g_main))
                grad_aux_norm = float(torch.linalg.vector_norm(g_aux))
                if args.method == "mgwp_v2":
                    total, lambda_value, conflict, grad_aux_norm, effective_ratio = combine_gradients(
                        g_main, g_aux, rho=rho, project_conflict=True)
                else:
                    dot = torch.sum(g_main * g_aux)
                    conflict = bool(dot < 0)
                    adjusted = g_aux
                    if args.method == "mgwp_v1" and conflict:
                        adjusted = adjusted - dot / torch.sum(g_main * g_main).clamp_min(1e-12) * g_main
                    if fixed_lambda is None:
                        fixed_lambda = rho * grad_main_norm / float(torch.linalg.vector_norm(adjusted).clamp_min(1e-12))
                    lambda_value = float(fixed_lambda)
                    total = g_main + lambda_value * adjusted
                    effective_ratio = float(torch.linalg.vector_norm(lambda_value * adjusted) / torch.linalg.vector_norm(g_main).clamp_min(1e-12))
                patch.grad = total
            require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Invalid patch gradient")
            total_grad_norm = float(torch.linalg.vector_norm(patch.grad))
            torch.nn.utils.clip_grad_norm_([patch], float(cfg["gradient_clip_norm"]))
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
                saturation = float(((patch <= 0.01) | (patch >= 0.99)).float().mean())
            rows.append({
                "method": args.method, "step": step, "record_index": index,
                "task_id": int(record["task_id"]), "state_index": int(record["state_index"]),
                "phase_proxy": record["phase_proxy"], "d_action": float(d_action.detach()),
                "d_persistent": float(d_persistent.detach()), "dino_norm": float(dino_norm.detach()),
                "q_range": float(q_range.detach()), "q_lateral": float(q_lateral.detach()),
                "v1_mechanism": float(v1_mechanism.detach()), "signed_dino": float(signed_dino.detach()),
                "signed_action": float(signed_action.detach()), "mechanism_margin_loss": float(mechanism_margin_loss.detach()),
                "action_margin_loss": float(action_margin_loss.detach()), "lambda_aux": lambda_value,
                "effective_aux_grad_ratio": effective_ratio, "gradient_conflict": int(conflict),
                "grad_main_l2": grad_main_norm, "grad_aux_l2": grad_aux_norm,
                "grad_total_l2": total_grad_norm, "patch_saturation_fraction": saturation,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "elapsed_s": time.time() - started,
            })
            if step % 10 == 0 or step in checkpoints:
                atomic_parquet(pd.DataFrame(rows), partial_path)
                atomic_torch({"method": args.method, "step": step, "patch": patch.detach(),
                              "optimizer": optimizer.state_dict(), "rows": rows,
                              "fixed_lambda": fixed_lambda}, resume_path)
            if step % 50 == 0 or step in checkpoints:
                print(f"TRAIN method={args.method} step={step}/{final_step} action={float(d_action):.4f} "
                      f"persistent={float(d_persistent):.4f} signed_dino={float(signed_dino):.4f} "
                      f"ratio={effective_ratio:.3f}", flush=True)
            if step in checkpoints:
                save_patch(args.method, step, patch, {**metadata, "fixed_lambda": fixed_lambda})
        frame = pd.DataFrame(rows)
        require(len(frame) == final_step and not frame.step.duplicated().any(), "Training row contract failed")
        require(torch.isfinite(torch.as_tensor(frame.select_dtypes("number").to_numpy())).all(), "Non-finite training metric")
        atomic_parquet(frame, final_output)
        report = {
            "status": "COMPLETE", "method": args.method, "steps": final_step,
            "primary_checkpoint": final_step, "fixed_lambda": fixed_lambda,
            "rho_primary": rho if args.method == "mgwp_v2" else None,
            "selected_xy": list(xy), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "final_patch_sha256": sha256_file(final_patch), "config_sha256": sha256_file(V2_CONFIG_PATH),
        }
        atomic_json(report, final_report)
        resume_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

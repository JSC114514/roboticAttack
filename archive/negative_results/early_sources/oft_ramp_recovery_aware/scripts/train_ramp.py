#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch

from common import (CONFIG_PATH, ROOT, action_scale, atomic_json, atomic_parquet,
                    atomic_torch, load_json, patch_path, require, require_gpu, sha256_file)
from ramp_utils import (METHODS, action_quantities, group_trajectories,
                        normalized_phase_weights, project_and_scale, training_index)
from runtime import differentiable, load_wrapper


def load_patch_payload(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    patch = payload["patch"] if isinstance(payload, dict) else payload
    require(tuple(patch.shape) == (3, 50, 50), f"Invalid initial patch shape: {tuple(patch.shape)}")
    require(torch.isfinite(patch).all(), "Initial patch contains non-finite values")
    return patch.float()


def save_patch(method: str, update: int, patch: torch.Tensor, metadata: dict) -> None:
    atomic_torch({"method": method, "trajectory_update": update,
                  "patch": patch.detach().float().cpu(), **metadata}, patch_path(method, update))


def trajectory_update(wrapper, patch, trajectory, method, cfg, scale, weights):
    xy = tuple(map(int, cfg["patch_xy"]))
    beta = float(cfg["beta_persistent"])
    # Pass 1 fixes the local direction of the trajectory norm without retaining six graphs.
    means = []
    with torch.no_grad():
        for row in trajectory:
            action, _dino = differentiable(wrapper, row["observation"], row["task_name"], patch, xy)
            clean_action = row["clean_action"].to(wrapper.device).float()
            _delta, _da, _dp, _score, mean_six = action_quantities(action, clean_action, scale, beta)
            means.append(mean_six)
    cumulative = sum(weights[index] * value for index, value in enumerate(means))
    direction = cumulative / torch.linalg.vector_norm(cumulative).clamp_min(1e-12)

    base_sum = torch.zeros_like(patch)
    trajectory_sum = torch.zeros_like(patch)
    geometry_sum = torch.zeros_like(patch)
    metrics = {"action": 0.0, "persistent": 0.0, "dino": 0.0, "geometry": 0.0}
    for index, row in enumerate(trajectory):
        wrapper.vla.zero_grad(set_to_none=True)
        action, dino = differentiable(wrapper, row["observation"], row["task_name"], patch, xy)
        clean_action = row["clean_action"].to(wrapper.device).float()
        clean_dino = row["clean_dino"].to(wrapper.device).float()
        _delta, d_action, persistent, state_score, mean_six = action_quantities(action, clean_action, scale, beta)
        delta_dino = dino.float() - clean_dino
        dino_norm = torch.linalg.vector_norm(delta_dino)
        range_direction = row["range_direction"].to(wrapper.device).float().reshape(-1)
        lateral_direction = row["lateral_direction"].to(wrapper.device).float().reshape(-1)
        q_range = torch.dot(delta_dino.reshape(-1), range_direction)
        q_lateral = torch.dot(delta_dino.reshape(-1), lateral_direction)
        geometry = torch.sqrt(q_range.square() + q_lateral.square() + float(cfg["geometry_epsilon"]))

        base_loss = -(state_score + float(cfg["lambda_dino"]) * dino_norm) / 6.0
        trajectory_loss = -weights[index] * torch.dot(mean_six, direction)
        geometry_loss = -geometry / 6.0
        need_trajectory = method in {"ramp_t", "ramp_tg"}
        need_geometry = method == "ramp_tg"
        base_gradient = torch.autograd.grad(base_loss, patch, retain_graph=need_trajectory or need_geometry)[0]
        base_sum.add_(base_gradient.detach())
        if need_trajectory:
            trajectory_gradient = torch.autograd.grad(
                trajectory_loss, patch, retain_graph=need_geometry)[0]
            trajectory_sum.add_(trajectory_gradient.detach())
        if need_geometry:
            geometry_gradient = torch.autograd.grad(geometry_loss, patch)[0]
            geometry_sum.add_(geometry_gradient.detach())
        metrics["action"] += float(d_action.detach()) / 6.0
        metrics["persistent"] += float(persistent.detach()) / 6.0
        metrics["dino"] += float(dino_norm.detach()) / 6.0
        metrics["geometry"] += float(geometry.detach()) / 6.0
        del action, dino, delta_dino, base_gradient

    total = base_sum.clone()
    traj_scale = geom_scale = traj_effective = geom_effective = 0.0
    traj_conflict = geom_conflict = False
    if method in {"ramp_t", "ramp_tg"}:
        scaled, traj_scale, traj_conflict, traj_effective = project_and_scale(
            base_sum, trajectory_sum, float(cfg["trajectory_gradient_ratio"]))
        total.add_(scaled)
    if method == "ramp_tg":
        scaled, geom_scale, geom_conflict, geom_effective = project_and_scale(
            base_sum, geometry_sum, float(cfg["geometry_gradient_ratio"]))
        total.add_(scaled)
    require(torch.isfinite(total).all(), "Non-finite RAMP gradient")
    patch.grad = total
    return {
        **metrics, "trajectory_cumulative": float(torch.linalg.vector_norm(cumulative)),
        "base_grad_l2": float(torch.linalg.vector_norm(base_sum)),
        "trajectory_grad_l2": float(torch.linalg.vector_norm(trajectory_sum)),
        "geometry_grad_l2": float(torch.linalg.vector_norm(geometry_sum)),
        "trajectory_scale": traj_scale, "geometry_scale": geom_scale,
        "trajectory_conflict": int(traj_conflict), "geometry_conflict": int(geom_conflict),
        "trajectory_effective_ratio": traj_effective, "geometry_effective_ratio": geom_effective,
        "total_grad_l2": float(torch.linalg.vector_norm(total)),
    }


def load_training(cfg):
    bank_path = Path(cfg["paths"]["training_bank"])
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    require(list(bank["selected_xy"]) == cfg["patch_xy"], "Patch location differs from frozen bank")
    trajectories = group_trajectories(bank["records"], int(cfg["training_trajectory_count"]))
    return bank_path, trajectories


def run(method: str) -> None:
    gpu = require_gpu()
    print(json.dumps(gpu), flush=True)
    cfg = load_json(CONFIG_PATH)
    require(tuple(cfg["train_methods"]) == METHODS, "Method contract drift")
    bank_path, trajectories = load_training(cfg)
    wrapper = load_wrapper(cfg)
    initial = load_patch_payload(cfg["paths"]["initial_patch"])
    patch = initial.to(wrapper.device).requires_grad_(True)
    optimizer = torch.optim.AdamW([patch], lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scale = torch.as_tensor(action_scale(cfg), device=wrapper.device, dtype=torch.float32)
    weights = normalized_phase_weights(cfg["phase_weights"], wrapper.device)
    output_dir = ROOT / "outputs" / "train"
    final_output = output_dir / f"{method}.parquet"
    final_report = ROOT / "reports" / f"TRAIN_{method.upper()}.json"
    final_update = int(cfg["trajectory_updates"])
    if final_output.is_file() and final_report.is_file() and patch_path(method, final_update).is_file():
        frame = pd.read_parquet(final_output)
        require(len(frame) == final_update, "Completed training output is incomplete")
        print(f"ALREADY_COMPLETE method={method}")
        return
    resume_path = output_dir / f"{method}.resume.pt"
    partial_path = output_dir / f"{method}.partial.parquet"
    rows, start = [], 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=wrapper.device, weights_only=False)
        require(resume["method"] == method, "Resume method mismatch")
        patch.data.copy_(resume["patch"])
        optimizer.load_state_dict(resume["optimizer"])
        rows, start = resume["rows"], int(resume["update"]) + 1
    metadata = {
        "experiment_id": cfg["experiment_id"], "patch_xy": cfg["patch_xy"],
        "initial_patch_sha256": sha256_file(cfg["paths"]["initial_patch"]),
        "training_bank_sha256": sha256_file(bank_path), "config_sha256": sha256_file(CONFIG_PATH),
    }
    if start == 1:
        save_patch(method, 0, patch, metadata)
    started = time.time()
    try:
        for update in range(start, final_update + 1):
            index = training_index(update, len(trajectories), int(cfg["seed"]))
            optimizer.zero_grad(set_to_none=True)
            values = trajectory_update(wrapper, patch, trajectories[index], method, cfg, scale, weights)
            torch.nn.utils.clip_grad_norm_([patch], float(cfg["gradient_clip_norm"]))
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
                saturation = float(((patch <= 0.01) | (patch >= 0.99)).float().mean())
            first = trajectories[index][0]
            rows.append({"method": method, "trajectory_update": update,
                         "trajectory_index": index, "task_id": int(first["task_id"]),
                         "trajectory_rank": int(first["trajectory_rank"]),
                         **values, "patch_saturation_fraction": saturation,
                         "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                         "elapsed_s": time.time() - started})
            if update % 10 == 0:
                atomic_parquet(pd.DataFrame(rows), partial_path)
                atomic_torch({"method": method, "update": update, "patch": patch.detach(),
                              "optimizer": optimizer.state_dict(), "rows": rows}, resume_path)
            if update % 25 == 0:
                print(f"TRAIN method={method} update={update}/{final_update} "
                      f"action={values['action']:.3f} trajectory={values['trajectory_cumulative']:.3f} "
                      f"dino={values['dino']:.1f}", flush=True)
            if update in set(map(int, cfg["checkpoint_updates"])):
                save_patch(method, update, patch, metadata)
        frame = pd.DataFrame(rows)
        require(len(frame) == final_update and not frame.trajectory_update.duplicated().any(),
                "Training row contract failed")
        atomic_parquet(frame, final_output)
        atomic_json({"status": "COMPLETE", "method": method, "trajectory_updates": final_update,
                     "state_backward_exposures": final_update * 6,
                     "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                     "patch_sha256": sha256_file(patch_path(method, final_update)),
                     **metadata}, final_report)
        resume_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    run(args.method)


if __name__ == "__main__":
    main()

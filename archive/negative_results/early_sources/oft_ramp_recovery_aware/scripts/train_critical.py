#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch

from common import (ROOT, action_scale, atomic_json, atomic_parquet, atomic_torch,
                    load_json, require, require_gpu, sha256_file)
from critical_utils import (BRANCH_CONFIG_PATH, CRITICAL_ARTIFACT_PATH, METHODS,
                            critical_patch_path, critical_scores, direction_map,
                            load_critical_artifact, margin_loss)
from ramp_utils import (action_quantities, group_trajectories,
                        normalized_phase_weights, project_and_scale, training_index)
from runtime import differentiable, load_wrapper
from train_ramp import load_patch_payload


def load_training(cfg: dict):
    bank_path = Path(cfg["paths"]["training_bank"])
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    require(list(bank["selected_xy"]) == cfg["patch_xy"], "Patch location differs from bank")
    return bank_path, group_trajectories(bank["records"], int(cfg["training_trajectory_count"]))


def save_patch(method: str, update: int, patch: torch.Tensor, metadata: dict) -> None:
    atomic_torch({"method": method, "trajectory_update": update,
                  "patch": patch.detach().float().cpu(), **metadata},
                 critical_patch_path(method, update))


def trajectory_update(wrapper, patch, trajectory, method, cfg, scale, weights,
                      directions, artifact):
    xy = tuple(map(int, cfg["patch_xy"]))
    base_sum = torch.zeros_like(patch)
    critical_sum = torch.zeros_like(patch)
    totals = {"action": 0.0, "persistent": 0.0, "dino": 0.0,
              "signed_action": 0.0, "signed_dino": 0.0,
              "action_margin_loss": 0.0, "dino_margin_loss": 0.0}
    for index, record in enumerate(trajectory):
        wrapper.vla.zero_grad(set_to_none=True)
        action, dino = differentiable(wrapper, record["observation"], record["task_name"], patch, xy)
        clean_action = record["clean_action"].to(wrapper.device).float()
        clean_dino = record["clean_dino"].to(wrapper.device).float()
        delta_action, d_action, persistent, state_score, _mean = action_quantities(
            action, clean_action, scale, float(cfg["beta_persistent"]))
        delta_dino = dino.float() - clean_dino
        dino_norm = torch.linalg.vector_norm(delta_dino)
        direction = directions[(int(record["task_id"]), int(record["state_index"]),
                                int(record["decision_index"]))]
        signed_action, signed_dino = critical_scores(
            delta_action, delta_dino, record, direction, artifact)
        action_margin = margin_loss(signed_action, cfg["critical_margin"], cfg["critical_temperature"])
        dino_margin = margin_loss(signed_dino, cfg["critical_margin"], cfg["critical_temperature"])
        base_loss = -(state_score + float(cfg["lambda_dino"]) * dino_norm) / 6.0
        if method == "action_critical_b":
            auxiliary_loss = weights[index] * action_margin / 6.0
        elif method == "dino_critical_b":
            auxiliary_loss = (weights[index] *
                              (action_margin + float(cfg["critical_dino_weight"]) * dino_margin)
                              / (1.0 + float(cfg["critical_dino_weight"])) / 6.0)
        else:
            auxiliary_loss = None
        base_gradient = torch.autograd.grad(base_loss, patch, retain_graph=auxiliary_loss is not None)[0]
        base_sum.add_(base_gradient.detach())
        if auxiliary_loss is not None:
            critical_gradient = torch.autograd.grad(auxiliary_loss, patch)[0]
            critical_sum.add_(critical_gradient.detach())
        for key, value in (("action", d_action), ("persistent", persistent), ("dino", dino_norm),
                           ("signed_action", signed_action), ("signed_dino", signed_dino),
                           ("action_margin_loss", action_margin), ("dino_margin_loss", dino_margin)):
            totals[key] += float(value.detach()) / 6.0
        del action, dino, delta_dino, base_gradient

    total = base_sum.clone()
    scale_value = effective = 0.0
    conflict = False
    if method != "dino_continue_b":
        scaled, scale_value, conflict, effective = project_and_scale(
            base_sum, critical_sum, float(cfg["critical_gradient_ratio"]))
        total.add_(scaled)
    require(torch.isfinite(total).all(), "Non-finite Branch-B gradient")
    patch.grad = total
    return {**totals, "base_grad_l2": float(torch.linalg.vector_norm(base_sum)),
            "critical_grad_l2": float(torch.linalg.vector_norm(critical_sum)),
            "critical_scale": scale_value, "critical_conflict": int(conflict),
            "critical_effective_ratio": effective,
            "total_grad_l2": float(torch.linalg.vector_norm(total))}


def run(method: str) -> None:
    print(json.dumps(require_gpu()), flush=True)
    cfg = load_json(BRANCH_CONFIG_PATH)
    require(tuple(cfg["train_methods"]) == METHODS, "Branch-B method contract drift")
    bank_path, trajectories = load_training(cfg)
    artifact = load_critical_artifact()
    directions = direction_map(artifact)
    wrapper = load_wrapper(cfg)
    patch = load_patch_payload(cfg["paths"]["initial_patch"]).to(wrapper.device).requires_grad_(True)
    optimizer = torch.optim.AdamW([patch], lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scale = torch.as_tensor(action_scale(cfg), device=wrapper.device, dtype=torch.float32)
    weights = normalized_phase_weights(cfg["phase_weights"], wrapper.device)
    output_dir = ROOT / "outputs" / "branch_b" / "train"
    final_output = output_dir / f"{method}.parquet"
    final_report = ROOT / "reports" / f"BRANCH_B_TRAIN_{method.upper()}.json"
    final_update = int(cfg["trajectory_updates"])
    if final_output.is_file() and final_report.is_file() and critical_patch_path(method, final_update).is_file():
        require(len(pd.read_parquet(final_output)) == final_update, "Completed Branch-B output invalid")
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
    metadata = {"experiment_id": cfg["experiment_id"], "patch_xy": cfg["patch_xy"],
                "initial_patch_sha256": sha256_file(cfg["paths"]["initial_patch"]),
                "training_bank_sha256": sha256_file(bank_path),
                "critical_artifact_sha256": sha256_file(CRITICAL_ARTIFACT_PATH),
                "config_sha256": sha256_file(BRANCH_CONFIG_PATH)}
    if start == 1:
        save_patch(method, 0, patch, metadata)
    started = time.time()
    try:
        for update in range(start, final_update + 1):
            index = training_index(update, len(trajectories), int(cfg["seed"]))
            optimizer.zero_grad(set_to_none=True)
            values = trajectory_update(wrapper, patch, trajectories[index], method, cfg, scale,
                                       weights, directions, artifact)
            torch.nn.utils.clip_grad_norm_([patch], float(cfg["gradient_clip_norm"]))
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
                saturation = float(((patch <= 0.01) | (patch >= 0.99)).float().mean())
            first = trajectories[index][0]
            rows.append({"method": method, "trajectory_update": update,
                         "trajectory_index": index, "task_id": int(first["task_id"]),
                         "trajectory_rank": int(first["trajectory_rank"]), **values,
                         "patch_saturation_fraction": saturation,
                         "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                         "elapsed_s": time.time() - started})
            if update % 10 == 0:
                atomic_parquet(pd.DataFrame(rows), partial_path)
                atomic_torch({"method": method, "update": update, "patch": patch.detach(),
                              "optimizer": optimizer.state_dict(), "rows": rows}, resume_path)
            if update % 25 == 0:
                print(f"BRANCH_B_TRAIN method={method} update={update}/{final_update} "
                      f"action={values['action']:.3f} signed_action={values['signed_action']:.3f} "
                      f"signed_dino={values['signed_dino']:.3f}", flush=True)
            if update in set(map(int, cfg["checkpoint_updates"])):
                save_patch(method, update, patch, metadata)
        frame = pd.DataFrame(rows)
        require(len(frame) == final_update and not frame.trajectory_update.duplicated().any(),
                "Branch-B training-row contract failed")
        atomic_parquet(frame, final_output)
        atomic_json({"status": "COMPLETE", "method": method,
                     "trajectory_updates": final_update,
                     "state_backward_exposures": final_update * 6,
                     "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                     "patch_sha256": sha256_file(critical_patch_path(method, final_update)),
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

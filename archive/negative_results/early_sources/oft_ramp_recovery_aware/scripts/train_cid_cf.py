#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cid_cf_utils import (CID_CONFIG_PATH, METHODS, METHOD_OBJECTIVES, OBJECTIVES,
                          cosine, eot_dict, patch_path, rms_displacement,
                          sample_eot, scale_auxiliary, standardized_action_l2,
                          task_pair_indices, unit_displacement)
from common import (ROOT, action_scale, atomic_json, atomic_parquet, atomic_torch,
                    load_json, require, require_gpu, sha256_file)
from ramp_utils import group_trajectories
from runtime import differentiable_visual, load_wrapper
from train_ramp import load_patch_payload


def load_training(cfg: dict[str, Any]):
    path = Path(cfg["paths"]["training_bank"])
    bank = torch.load(path, map_location="cpu", weights_only=False)
    require(list(bank["selected_xy"]) == cfg["patch_xy"], "CID-CF patch location differs from bank")
    trajectories = group_trajectories(
        bank["records"], int(cfg["task_count"]) * int(cfg["trajectories_per_task"]))
    require(all(len(item) == int(cfg["states_per_trajectory"]) for item in trajectories),
            "CID-CF trajectory phase coverage drift")
    return path, trajectories


def save_patch(method: str, update: int, patch: torch.Tensor, metadata: dict[str, Any]) -> None:
    atomic_torch({"method": method, "optimizer_update": update,
                  "patch": patch.detach().float().cpu(), **metadata}, patch_path(method, update))


def _clean_siglip(wrapper, record: dict[str, Any], xy: tuple[int, int],
                  cache: dict[tuple[int, int, int], torch.Tensor]) -> torch.Tensor:
    key = (int(record["task_id"]), int(record["state_index"]), int(record["decision_index"]))
    if key not in cache:
        with torch.no_grad():
            _action, live_dino, siglip = differentiable_visual(
                wrapper, record["observation"], record["task_name"], None, xy)
        require(siglip is not None, "Clean SigLIP capture missing")
        stored = record["clean_dino"].to(wrapper.device).float()
        drift = torch.sqrt(torch.mean((live_dino.float() - stored).square()))
        require(float(drift) < 1e-3, f"Training-bank clean DINO drift: {float(drift):.6g}")
        # Preserve the model dtype exactly; the counterfactual branch must use
        # clean SigLIP rather than a separately quantized approximation.
        cache[key] = siglip.detach().to(device="cpu", dtype=wrapper.dtype)
    return cache[key].to(wrapper.device, dtype=wrapper.dtype)


def _prepass(wrapper, patch: torch.Tensor, contexts: list[dict[str, Any]], eot_values,
             cfg: dict[str, Any], clean_cache: dict) -> tuple[list[int], torch.Tensor, dict[str, float]]:
    xy = tuple(map(int, cfg["patch_xy"]))
    magnitudes, units = [], []
    for record, eot in zip(contexts, eot_values, strict=True):
        _clean_siglip(wrapper, record, xy, clean_cache)
        with torch.no_grad():
            _action, dino, _siglip = differentiable_visual(
                wrapper, record["observation"], record["task_name"], patch, xy, eot=eot)
        clean_dino = record["clean_dino"].to(wrapper.device).float()
        magnitudes.append(float(rms_displacement(dino, clean_dino)))
        units.append(unit_displacement(dino, clean_dino))
    count = max(1, int(round(len(contexts) * float(cfg["tail_fraction"]))))
    require(count == 3 and len(contexts) == 12, "CID-CF lowest-25% contract drift")
    tail = sorted(range(len(magnitudes)), key=magnitudes.__getitem__)[:count]
    prototype_sum = torch.stack(units).sum(dim=0)
    require(float(torch.linalg.vector_norm(prototype_sum)) > 0, "Degenerate consistency prototype")
    prototype = (prototype_sum / torch.linalg.vector_norm(prototype_sum)).detach()
    return tail, prototype, {
        "prepass_dino_rms_mean": float(sum(magnitudes) / len(magnitudes)),
        "prepass_dino_rms_min": float(min(magnitudes)),
        "prepass_dino_rms_max": float(max(magnitudes)),
    }


def cid_cf_update(wrapper, patch: torch.Tensor, contexts: list[dict[str, Any]],
                  eot_values, method: str, cfg: dict[str, Any], scale: torch.Tensor,
                  clean_cache: dict) -> dict[str, Any]:
    """Perform the two-pass, 12-context CID-CF gradient construction."""
    require(method in METHODS and tuple(cfg["train_methods"]) == METHODS, "CID-CF method contract drift")
    require(len(contexts) == len(eot_values) == int(cfg["contexts_per_update"]) == 12,
            "CID-CF update requires exactly 12 contexts")
    xy = tuple(map(int, cfg["patch_xy"]))
    tail_indices, prototype, prepass = _prepass(
        wrapper, patch, contexts, eot_values, cfg, clean_cache)
    tail_set = set(tail_indices)
    gradients = {name: torch.zeros_like(patch) for name in OBJECTIVES}
    totals = {name: 0.0 for name in OBJECTIVES}
    totals["dino_rms"] = 0.0

    context_count, tail_count = len(contexts), len(tail_indices)
    for index, (record, eot) in enumerate(zip(contexts, eot_values, strict=True)):
        wrapper.vla.zero_grad(set_to_none=True)
        clean_action = record["clean_action"].to(wrapper.device).float()
        clean_dino = record["clean_dino"].to(wrapper.device).float()
        clean_siglip = _clean_siglip(wrapper, record, xy, clean_cache)

        full_action, patched_dino, _patched_siglip = differentiable_visual(
            wrapper, record["observation"], record["task_name"], patch, xy, eot=eot)
        cf_action, _cf_dino, _unused = differentiable_visual(
            wrapper, record["observation"], record["task_name"], patch, xy,
            eot=eot, clean_siglip=clean_siglip)
        cf_objective = standardized_action_l2(cf_action, clean_action, scale)
        tail_objective = rms_displacement(patched_dino, clean_dino)
        direction = unit_displacement(patched_dino, clean_dino)
        consistency_objective = torch.dot(direction, prototype)
        full_objective = standardized_action_l2(full_action, clean_action, scale)

        gradients["cf"].add_(torch.autograd.grad(
            -cf_objective / context_count, patch)[0].detach())
        full_terms: list[tuple[str, torch.Tensor, float]] = []
        if index in tail_set:
            full_terms.append(("tail", tail_objective, float(tail_count)))
        full_terms.extend((("consistency", consistency_objective, float(context_count)),
                           ("full_action", full_objective, float(context_count))))
        for term_index, (name, objective, denominator) in enumerate(full_terms):
            gradient = torch.autograd.grad(
                -objective / denominator, patch,
                retain_graph=term_index < len(full_terms) - 1)[0]
            gradients[name].add_(gradient.detach())

        totals["cf"] += float(cf_objective.detach()) / context_count
        if index in tail_set:
            totals["tail"] += float(tail_objective.detach()) / tail_count
        totals["consistency"] += float(consistency_objective.detach()) / context_count
        totals["full_action"] += float(full_objective.detach()) / context_count
        totals["dino_rms"] += float(tail_objective.detach()) / context_count
        del full_action, cf_action, patched_dino, cf_objective, tail_objective

    require(all(bool(torch.isfinite(value).all()) for value in gradients.values()),
            "Non-finite CID-CF objective gradient")
    raw_norms = {name: float(torch.linalg.vector_norm(value)) for name, value in gradients.items()}
    require(all(value > 0 for value in raw_norms.values()), "Degenerate CID-CF objective gradient")
    pairwise = {f"grad_cos_{left}_{right}": cosine(gradients[left], gradients[right])
                for i, left in enumerate(OBJECTIVES) for right in OBJECTIVES[i + 1:]}

    active = METHOD_OBJECTIVES[method]
    scales = {name: 0.0 for name in OBJECTIVES}
    if method == "action_eot":
        total_gradient = gradients["full_action"].clone()
        scales["full_action"] = 1.0
    else:
        total_gradient = gradients["cf"].clone()
        scales["cf"] = 1.0
        ratios = cfg["auxiliary_gradient_ratios"]
        for name in active:
            if name == "cf":
                continue
            ratio_key = "consistency" if name == "consistency" else name
            scaled, factor = scale_auxiliary(
                gradients["cf"], gradients[name], float(ratios[ratio_key]))
            total_gradient.add_(scaled)
            scales[name] = factor
    require(bool(torch.isfinite(total_gradient).all()), "Non-finite CID-CF total gradient")
    patch.grad = total_gradient
    return {
        **{f"objective_{key}": value for key, value in totals.items()},
        **{f"raw_grad_l2_{key}": value for key, value in raw_norms.items()},
        **{f"gradient_scale_{key}": value for key, value in scales.items()},
        **pairwise,
        **prepass,
        "tail_context_indices": json.dumps(tail_indices),
        "active_objectives": json.dumps(active),
        "total_grad_l2_preclip": float(torch.linalg.vector_norm(total_gradient)),
        "clean_siglip_cache_entries": len(clean_cache),
    }


def run(method: str) -> None:
    print(json.dumps(require_gpu()), flush=True)
    cfg = load_json(CID_CONFIG_PATH)
    require(tuple(cfg["train_methods"]) == METHODS, "CID-CF method ordering drift")
    require(cfg["confirmation_suite"] == "libero_10" and bool(cfg["confirmation_locked"]),
            "Long confirmation lock is not active")
    bank_path, trajectories = load_training(cfg)
    wrapper = load_wrapper(cfg)
    patch = load_patch_payload(cfg["paths"]["initial_patch"]).to(wrapper.device).requires_grad_(True)
    require(tuple(patch.shape) == tuple(cfg["patch_shape"]), "CID-CF initial patch shape drift")
    optimizer = torch.optim.AdamW([patch], lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scale = torch.as_tensor(action_scale(cfg), device=wrapper.device, dtype=torch.float32)
    clean_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    output_dir = ROOT / "outputs" / "cid_cf" / "train"
    final_output = output_dir / f"{method}.parquet"
    final_report = ROOT / "reports" / f"CID_CF_TRAIN_{method.upper()}.json"
    final_update = int(cfg["trajectory_updates"])
    if final_output.is_file() and final_report.is_file() and patch_path(method, final_update).is_file():
        require(len(pd.read_parquet(final_output)) == final_update, "Completed CID-CF output invalid")
        print(f"CID_CF_ALREADY_COMPLETE method={method}")
        return

    resume_path = output_dir / f"{method}.resume.pt"
    partial_path = output_dir / f"{method}.partial.parquet"
    rows, start = [], 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=wrapper.device, weights_only=False)
        require(resume["method"] == method, "CID-CF resume method mismatch")
        patch.data.copy_(resume["patch"])
        optimizer.load_state_dict(resume["optimizer"])
        rows, start = resume["rows"], int(resume["update"]) + 1
    metadata = {
        "experiment_id": cfg["experiment_id"], "patch_xy": cfg["patch_xy"],
        "initial_patch_sha256": sha256_file(cfg["paths"]["initial_patch"]),
        "training_bank_sha256": sha256_file(bank_path),
        "config_sha256": sha256_file(CID_CONFIG_PATH),
    }
    if start == 1:
        save_patch(method, 0, patch, metadata)
    started = time.time()
    try:
        for update in range(start, final_update + 1):
            indices = task_pair_indices(update, trajectories, int(cfg["seed"]))
            selected = [trajectories[index] for index in indices]
            contexts = [record for trajectory in selected for record in trajectory]
            eot_values = [sample_eot(cfg, update, index) for index in range(len(contexts))]
            optimizer.zero_grad(set_to_none=True)
            values = cid_cf_update(
                wrapper, patch, contexts, eot_values, method, cfg, scale, clean_cache)
            preclip = torch.nn.utils.clip_grad_norm_([patch], float(cfg["gradient_clip_norm"]))
            values["clip_reported_preclip_l2"] = float(preclip)
            values["total_grad_l2_postclip"] = float(torch.linalg.vector_norm(patch.grad))
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
                saturation = float(((patch <= 0.01) | (patch >= 0.99)).float().mean())
            task_ids = [int(trajectory[0]["task_id"]) for trajectory in selected]
            ranks = [int(trajectory[0]["trajectory_rank"]) for trajectory in selected]
            eot_rows = [eot_dict(value) for value in eot_values]
            rows.append({
                "method": method, "optimizer_update": update,
                "trajectory_indices": json.dumps(indices), "task_ids": json.dumps(task_ids),
                "trajectory_ranks": json.dumps(ranks), **values,
                **{f"eot_{key}_min": min(row[key] for row in eot_rows) for key in eot_rows[0]},
                **{f"eot_{key}_max": max(row[key] for row in eot_rows) for key in eot_rows[0]},
                "patch_saturation_fraction": saturation,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "elapsed_s": time.time() - started,
            })
            if update % 10 == 0:
                atomic_parquet(pd.DataFrame(rows), partial_path)
                atomic_torch({"method": method, "update": update, "patch": patch.detach(),
                              "optimizer": optimizer.state_dict(), "rows": rows}, resume_path)
            if update % 25 == 0:
                print(f"CID_CF_TRAIN method={method} update={update}/{final_update} "
                      f"cf={values['objective_cf']:.3f} tail={values['objective_tail']:.3f} "
                      f"cons={values['objective_consistency']:.3f} "
                      f"full={values['objective_full_action']:.3f}", flush=True)
            if update in set(map(int, cfg["checkpoint_updates"])):
                save_patch(method, update, patch, metadata)
        frame = pd.DataFrame(rows)
        require(len(frame) == final_update and not frame.optimizer_update.duplicated().any(),
                "CID-CF training-row contract failed")
        atomic_parquet(frame, final_output)
        atomic_json({
            "status": "COMPLETE", "method": method, "optimizer_updates": final_update,
            "context_backward_exposures": final_update * int(cfg["contexts_per_update"]),
            "source_suite_only": cfg["suite"], "confirmation_suite_accessed": False,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "patch_sha256": sha256_file(patch_path(method, final_update)), **metadata,
        }, final_report)
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

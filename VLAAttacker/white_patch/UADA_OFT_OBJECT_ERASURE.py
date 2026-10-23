"""Exploring-OFT patch attack with a DINO object-erasure-chosen action target.

The attacker still updates one Wrist RGB patch through the complete OFT action
loss.  A clean-input DINO counterfactual only chooses the sign of each action
endpoint; no feature loss, branch-only patch forward, or oracle mask is used at
deployment.  The background-erasure arm is a matched training control.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_DINO import (
    ExploringOFTOpenVLAAttacker, atomic_json, atomic_parquet, atomic_torch,
    patch_metrics, require, save_patch, sha256_file,
)
from white_patch.UADA_OFT_JOINT500 import persistent6d_loss
from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, init_uada_patch, step_uada_patch


METHODS = ("exploring_oft_dino_object_erasure", "exploring_oft_dino_background_erasure")


def token_mask(mask: np.ndarray) -> torch.Tensor:
    """Match the frozen OFT mask readout's resize/crop to its 16x16 DINO grid."""
    require(mask.shape == (256, 256), f"Unexpected oracle mask shape: {mask.shape}")
    x = torch.as_tensor(mask.astype(np.float32))[None, None]
    x = F.interpolate(x, size=(224, 224), mode="nearest")
    axis = torch.linspace(-math.sqrt(0.9), math.sqrt(0.9), 224)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = F.grid_sample(x, torch.stack((xx, yy), dim=-1)[None], mode="nearest",
                      padding_mode="border", align_corners=True)
    return F.avg_pool2d(x, kernel_size=14)[0, 0].reshape(-1).clamp(0, 1)


def counterfactual_dino(clean_dino: torch.Tensor, target_mask: torch.Tensor,
                        background_mask: torch.Tensor, method: str, seed: int) -> torch.Tensor:
    """Replace target-object tokens, or equally many background tokens, by background mean."""
    require(method in METHODS, f"Unknown method: {method}")
    require(clean_dino.shape == (1, 256, 1024), "Unexpected DINO feature shape")
    target = target_mask.float().reshape(-1).clamp(0, 1)
    background = background_mask.float().reshape(-1).clamp(0, 1)
    require(target.numel() == background.numel() == 256, "Token mask shape drift")
    require(float(target.sum()) > 0 and float(background.sum()) > 0, "Empty target/background mask")
    background = background * (1.0 - target)
    require(float(background.sum()) > 0, "No uncontaminated background tokens")
    if method == METHODS[0]:
        weights = target
    else:
        # The sham changes the same token mass, outside the target, without
        # using action/DINO response to choose the background locations.
        candidates = torch.nonzero(background > 0, as_tuple=False).reshape(-1)
        count = min(int(torch.count_nonzero(target)), int(candidates.numel()))
        require(count > 0, "No background sham candidates")
        chosen = candidates[torch.randperm(candidates.numel(), generator=torch.Generator().manual_seed(seed))[:count]]
        weights = torch.zeros_like(target)
        weights[chosen] = target.sum() / count
        weights = weights.clamp(0, 1)
    anchor = (clean_dino.float()[0] * background[:, None]).sum(0) / background.sum()
    return (clean_dino.float() * (1.0 - weights[None, :, None])
            + anchor[None, None, :] * weights[None, :, None]).to(clean_dino.dtype)


def dino_chosen_endpoint(clean: torch.Tensor, counterfactual: torch.Tensor,
                         low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use object-erasure action direction; revert exact ties to the UADA endpoint."""
    require(clean.shape == counterfactual.shape and clean.shape[-1] == 7,
            "Clean/counterfactual action shape drift")
    delta = counterfactual[..., :6].mean(1, keepdim=True) - clean[..., :6].mean(1, keepdim=True)
    opposite = clean[..., :6].mean(1, keepdim=True) < (low[..., :6] + high[..., :6]) / 2
    toward_high = torch.where(delta > 0, torch.ones_like(opposite),
                              torch.where(delta < 0, torch.zeros_like(opposite), opposite))
    endpoint = torch.where(toward_high, high[..., :6], low[..., :6]).detach()
    return endpoint.expand(-1, clean.shape[1], -1), delta.detach()


class ExploringOFTObjectErasureAttacker(ExploringOFTOpenVLAAttacker):
    def train_patch(self, records: list[dict[str, Any]], masks: dict[tuple[int, int, int], Path],
                    cfg: dict[str, Any], method: str) -> None:
        require(method in METHODS, f"Unknown method: {method}")
        output_dir = Path(cfg["output_root"]) / method
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "status.json"
        if status_path.is_file():
            require(json.loads(status_path.read_text()).get("status") != "COMPLETE",
                    "Refusing to overwrite completed patch")

        torch.manual_seed(int(cfg["seed"]))
        np.random.seed(int(cfg["seed"]))
        random.seed(int(cfg["seed"]))
        patch = init_uada_patch(tuple(cfg["patch_shape"]), self.wrapper.device)
        initial = patch.detach().clone()
        optimizer, scheduler = build_uada_adamw(
            patch, float(cfg["learning_rate"]), int(cfg["warmup_outer_iterations"]),
            int(cfg["outer_iterations"]), weight_decay=float(cfg["weight_decay"]))
        low = torch.as_tensor(cfg["action_head_q01"], device=self.wrapper.device).reshape(1, 1, 7)
        high = torch.as_tensor(cfg["action_head_q99"], device=self.wrapper.device).reshape(1, 1, 7)
        metadata = {
            "experiment_id": cfg["experiment_id"], "method": method,
            "source_trainer": "read-only UADA.py::patchattack_unconstrained",
            "source_uada_sha256": sha256_file(Path(__file__).with_name("UADA.py")),
            "source_copy_sha256": sha256_file(Path(__file__).with_name("UADA_OFT_SOURCE_COPY.py")),
            "training_bank_sha256": sha256_file(cfg["training_bank"]),
            "mask_lineage": "exact Object bank replay, 300 states",
            "seed": int(cfg["seed"]), "data_seed": int(cfg["data_seed"]),
            "outer_iterations": int(cfg["outer_iterations"]), "inner_loop": int(cfg["inner_loop"]),
        }
        rows: list[dict[str, Any]] = []
        start_outer = 1
        resume_path = output_dir / "resume.pt"
        if resume_path.is_file():
            resume = torch.load(resume_path, map_location=self.wrapper.device, weights_only=False)
            require(resume["experiment_id"] == cfg["experiment_id"] and resume["method"] == method,
                    "Resume identity mismatch")
            patch.data.copy_(resume["patch"])
            initial = resume["initial"].to(self.wrapper.device)
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            rows = resume["rows"]
            start_outer = int(resume["outer"]) + 1
        else:
            save_patch(output_dir / "outer_0000.pt", patch, {**metadata, **patch_metrics(patch, initial)})

        started = time.time()
        for outer in range(start_outer, int(cfg["outer_iterations"]) + 1):
            record_index = self._training_index(outer, len(records), int(cfg["data_seed"]))
            record = records[record_index]
            key = (int(record["task_id"]), int(record["trajectory_rank"]), int(record["decision_index"]))
            require(key in masks, f"Missing exact Object mask {key}")
            with np.load(masks[key], allow_pickle=False) as arrays:
                target_mask = token_mask(arrays["wrist_pixel_mask_target_object"])
                background_mask = token_mask(arrays["wrist_pixel_mask_matched_background"])
            inputs, proprio = self._base_inputs(record)
            observation = record["observation"]
            with torch.no_grad():
                clean, clean_dino, _ = self._forward(inputs, proprio, observation, None, False)
                altered_dino = counterfactual_dino(
                    clean_dino, target_mask, background_mask, method,
                    int(cfg["sham_seed"]) + record_index)
                altered_action, _, _ = self._forward(
                    inputs, proprio, observation, None, False, clean_dino=altered_dino)
                target, delta = dino_chosen_endpoint(clean, altered_action, low, high)
            sums = {"loss": 0.0, "grad": 0.0}
            for inner in range(int(cfg["inner_loop"])):
                transform_seed = int(cfg["seed"]) * 10_000_000 + outer * 10_000 + inner
                random.seed(transform_seed)
                np.random.seed(transform_seed % (2**32 - 1))
                torch.manual_seed(transform_seed)
                optimizer.zero_grad(set_to_none=True)
                self.wrapper.vla.zero_grad(set_to_none=True)
                action, _, _ = self._forward(
                    inputs, proprio, observation, patch, bool(cfg["geometry"]))
                loss = persistent6d_loss(action, target, low, high, float(cfg["persistent_beta"]))
                loss.backward()
                require(patch.grad is not None and torch.isfinite(patch.grad).all(),
                        "Invalid patch gradient")
                sums["loss"] += float(loss.detach()) / int(cfg["inner_loop"])
                sums["grad"] += float(torch.linalg.vector_norm(patch.grad)) / int(cfg["inner_loop"])
                step_uada_patch(patch, optimizer, self.wrapper.vla)
            scheduler.step()
            structure = patch_metrics(patch, initial)
            rows.append({
                "method": method, "outer_iteration": outer,
                "optimizer_updates": outer * int(cfg["inner_loop"]),
                "record_index": record_index, "task_id": key[0],
                "trajectory_rank": key[1], "decision_index": key[2],
                "main_loss": sums["loss"], "patch_grad_l2": sums["grad"],
                "dino_teacher_action_l2": float(torch.linalg.vector_norm(altered_action - clean)),
                "dino_teacher_nonzero_dofs": int((delta != 0).sum()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": time.time() - started, **structure,
            })
            if outer % 5 == 0 or outer in cfg["checkpoint_outer_iterations"]:
                print(f"EXPLORING_OFT_OBJECT_ERASURE method={method} outer={outer}/{cfg['outer_iterations']} "
                      f"updates={outer * int(cfg['inner_loop'])} main={sums['loss']:.5f} "
                      f"teacher={rows[-1]['dino_teacher_action_l2']:.5f} "
                      f"tv_ratio={structure['tv_ratio']:.4f}", flush=True)
            if outer in cfg["checkpoint_outer_iterations"]:
                save_patch(output_dir / f"outer_{outer:04d}.pt", patch, {**metadata, **structure})
            if str(outer) in cfg["structure_gates"]:
                gate = cfg["structure_gates"][str(outer)]
                passed = (structure["init_correlation"] < float(gate["max_init_correlation"])
                          and structure["tv_ratio"] < float(gate["max_tv_ratio"]))
                atomic_json({"outer_iteration": outer, "passed": passed, **gate, **structure},
                            output_dir / f"structure_gate_{outer:04d}.json")
                if not passed:
                    atomic_json({"status": "STRUCTURE_GATE_FAIL", "outer_iteration": outer,
                                 **metadata, **structure}, status_path)
                    atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
                    return
            if outer % int(cfg["resume_every_outer_iterations"]) == 0:
                atomic_torch({"outer": outer, "experiment_id": cfg["experiment_id"], "method": method,
                              "patch": patch.detach(), "initial": initial.detach(),
                              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                              "rows": rows}, resume_path)
        atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
        atomic_json({"status": "COMPLETE", **metadata, **patch_metrics(patch, initial),
                     "elapsed_s": time.time() - started}, status_path)
        resume_path.unlink(missing_ok=True)

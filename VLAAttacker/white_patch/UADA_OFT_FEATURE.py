"""Exploring UADA on OFT with an additive Wrist visual-feature objective.

This deliberately inherits the existing source-aligned OFT adapter.  The
original UADA.py and UADA_OFT_DINO.py are not changed.  A single attacked
forward supplies both the full-action UADA loss and the selected branch's
feature loss; only the external RGB patch is optimized.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, init_uada_patch, step_uada_patch
from white_patch.UADA_OFT_DINO import (
    ExploringOFTOpenVLAAttacker, atomic_json, atomic_parquet, atomic_torch,
    patch_metrics, require, save_patch, sha256_file,
)


METHOD_BRANCH = {
    "exploring_oft_dino_feature": "dino",
    "exploring_oft_siglip_feature": "siglip",
}


def feature_disruption_loss(attacked: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Minimizing this maximizes mean per-token cosine distance from clean.

    Both branches are compared in their own normalized feature space, not by
    raw activation magnitude.  This is not an object-localization objective.
    """
    require(attacked.shape == clean.shape and attacked.ndim == 3,
            f"Feature shape mismatch: attacked={tuple(attacked.shape)}, clean={tuple(clean.shape)}")
    require(torch.isfinite(attacked).all() and torch.isfinite(clean).all(),
            "Non-finite visual features")
    return F.cosine_similarity(attacked.float(), clean.float().detach(), dim=-1).mean() - 1.0


def combine_gradients(action_grad: torch.Tensor, feature_grad: torch.Tensor,
                      ratio: float, frozen_weight: float | None) -> tuple[torch.Tensor, float, float, float]:
    """Freeze auxiliary scale on the first update, matching prior OFT arms."""
    action_norm = torch.linalg.vector_norm(action_grad.float())
    feature_norm = torch.linalg.vector_norm(feature_grad.float())
    require(float(action_norm) > 1e-12 and float(feature_norm) > 1e-12,
            "Zero action/feature gradient; cannot freeze meaningful scale")
    weight = float(frozen_weight) if frozen_weight is not None else ratio * float(action_norm / feature_norm)
    require(np.isfinite(weight) and weight > 0, "Invalid feature gradient weight")
    cosine = float(F.cosine_similarity(action_grad.reshape(-1).float(),
                                       feature_grad.reshape(-1).float(), dim=0))
    scaled_ratio = weight * float(feature_norm / action_norm)
    return action_grad + weight * feature_grad, weight, cosine, scaled_ratio


class ExploringOFTFeatureAttacker(ExploringOFTOpenVLAAttacker):
    def train_feature_patch(self, records: list[dict[str, Any]], cfg: dict[str, Any], method: str) -> None:
        require(method in METHOD_BRANCH, f"Unknown feature method: {method}")
        branch = METHOD_BRANCH[method]
        output_dir = Path(cfg["output_root"]) / method
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            require(status.get("status") != "COMPLETE", "Refusing to overwrite completed feature patch")

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
        maskidx = list(map(int, cfg["maskidx"]))
        checkpoints = set(map(int, cfg["checkpoint_outer_iterations"]))
        resume_path = output_dir / "resume.pt"
        rows: list[dict[str, Any]] = []
        start_outer = 1
        weight: float | None = None
        if resume_path.exists():
            resume = torch.load(resume_path, map_location=self.wrapper.device, weights_only=False)
            require(resume["experiment_id"] == cfg["experiment_id"] and resume["method"] == method,
                    "Resume identity mismatch")
            patch.data.copy_(resume["patch"])
            initial = resume["initial"].to(self.wrapper.device)
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            rows = resume["rows"]
            weight = resume["feature_weight"]
            start_outer = int(resume["outer"]) + 1
            print(f"RESUME method={method} outer={start_outer}", flush=True)

        metadata = {
            "experiment_id": cfg["experiment_id"], "method": method,
            "source_trainer": "VLAAttacker/white_patch/UADA.py::patchattack_unconstrained",
            "source_uada_sha256": sha256_file(Path(__file__).with_name("UADA.py")),
            "parent_oft_adapter_sha256": sha256_file(Path(__file__).with_name("UADA_OFT_DINO.py")),
            "adaptation": "OFT continuous action; Front+Wrist input; Wrist-only RGB patch",
            "auxiliary_objective": "maximize mean per-token clean/attacked cosine distance",
            "feature_branch": branch,
            "seed": int(cfg["seed"]), "data_seed": int(cfg["data_seed"]),
            "maskidx": maskidx, "outer_iterations": int(cfg["outer_iterations"]),
            "inner_loop": int(cfg["inner_loop"]), "batch_size": int(cfg["batch_size"]),
            "training_bank_sha256": sha256_file(cfg["training_bank"]),
            "baseline_patch_sha256": cfg["baseline_patch_sha256"],
        }
        if start_outer == 1:
            save_patch(output_dir / "outer_0000.pt", patch, {**metadata, **patch_metrics(patch, initial)})
        started = time.time()
        try:
            for outer in range(start_outer, int(cfg["outer_iterations"]) + 1):
                record_index = self._training_index(outer, len(records), int(cfg["data_seed"]))
                record = records[record_index]
                inputs, proprio = self._base_inputs(record)
                observation = record["observation"]
                with torch.no_grad():
                    clean_action, clean_dino, clean_siglip = self._forward(
                        inputs, proprio, observation, None, False, capture_siglip=True)
                require(clean_siglip is not None, "Clean Wrist-SigLIP anchor missing")
                target = torch.where(clean_action >= (low + high) / 2.0, low, high).detach()
                sums = {"action": 0.0, "feature_distance": 0.0, "grad": 0.0,
                        "cosine": 0.0, "scaled_ratio": 0.0}
                for inner in range(int(cfg["inner_loop"])):
                    transform_seed = int(cfg["seed"]) * 10_000_000 + outer * 10_000 + inner
                    random.seed(transform_seed)
                    np.random.seed(transform_seed % (2**32 - 1))
                    torch.manual_seed(transform_seed)
                    optimizer.zero_grad(set_to_none=True)
                    self.wrapper.vla.zero_grad(set_to_none=True)
                    action, dino, siglip = self._forward(
                        inputs, proprio, observation, patch, bool(cfg["geometry"]), capture_siglip=True)
                    require(siglip is not None, "Attacked Wrist-SigLIP feature missing")
                    action_loss = F.mse_loss(action[..., maskidx], target[..., maskidx])
                    selected, clean_selected = (dino, clean_dino) if branch == "dino" else (siglip, clean_siglip)
                    feature_loss = feature_disruption_loss(selected, clean_selected)
                    action_grad = torch.autograd.grad(action_loss, patch, retain_graph=True)[0]
                    feature_grad = torch.autograd.grad(feature_loss, patch)[0]
                    patch.grad, weight, cosine, scaled_ratio = combine_gradients(
                        action_grad, feature_grad, float(cfg["feature_gradient_ratio"]), weight)
                    require(torch.isfinite(patch.grad).all(), "Non-finite patch gradient")
                    grad_norm = float(torch.linalg.vector_norm(patch.grad))
                    step_uada_patch(patch, optimizer, self.wrapper.vla)
                    scale = 1.0 / int(cfg["inner_loop"])
                    sums["action"] += float(action_loss.detach()) * scale
                    sums["feature_distance"] += -float(feature_loss.detach()) * scale
                    sums["grad"] += grad_norm * scale
                    sums["cosine"] += cosine * scale
                    sums["scaled_ratio"] += scaled_ratio * scale
                scheduler.step()
                structure = patch_metrics(patch, initial)
                rows.append({
                    "method": method, "outer_iteration": outer,
                    "optimizer_updates": outer * int(cfg["inner_loop"]),
                    "record_index": record_index, "task_id": int(record["task_id"]),
                    "state_index": int(record["state_index"]),
                    "action_boundary_mse": sums["action"],
                    "feature_cosine_distance": sums["feature_distance"],
                    "patch_grad_l2": sums["grad"], "feature_gradient_weight": weight,
                    "action_feature_gradient_cosine": sums["cosine"],
                    "feature_scaled_gradient_ratio": sums["scaled_ratio"],
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_s": time.time() - started, **structure,
                })
                if outer % 5 == 0 or outer in checkpoints:
                    print(f"EXPLORING_OFT_FEATURE method={method} outer={outer}/{cfg['outer_iterations']} "
                          f"updates={outer * int(cfg['inner_loop'])} action_mse={sums['action']:.5f} "
                          f"feature_distance={sums['feature_distance']:.5f} "
                          f"corr={structure['init_correlation']:.4f} tv_ratio={structure['tv_ratio']:.4f}", flush=True)
                if outer in checkpoints:
                    save_patch(output_dir / f"outer_{outer:04d}.pt", patch,
                               {**metadata, **structure, "feature_gradient_weight": weight})
                    if outer in set(map(int, cfg.get("mechanism_probe_outer_iterations", []))):
                        probe = self._mechanism_probe(records, cfg, method, patch, outer)
                        atomic_parquet(probe, output_dir / f"mechanism_probe_{outer:04d}.parquet")
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
                                  "rows": rows, "feature_weight": weight}, resume_path)
            atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
            atomic_json({"status": "COMPLETE", **metadata, **patch_metrics(patch, initial),
                         "feature_gradient_weight": weight, "elapsed_s": time.time() - started}, status_path)
            resume_path.unlink(missing_ok=True)
        except BaseException:
            if rows:
                atomic_parquet(pd.DataFrame(rows), output_dir / "training.partial.parquet")
            raise

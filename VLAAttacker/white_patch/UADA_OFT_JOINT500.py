"""Six matched Exploring-OFT patch arms built outside read-only UADA.py.

Every arm copies Exploring's random patch, EOT, AdamW/cosine, clamp and sample
order.  The arms change only the declared action/visual objective.
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

from white_patch.UADA_OFT_DINO import (
    ExploringOFTOpenVLAAttacker, atomic_json, atomic_parquet, atomic_torch,
    patch_metrics, require, save_patch, sha256_file,
)
from white_patch.UADA_OFT_SOURCE_COPY import (
    build_uada_adamw, init_uada_patch, step_uada_patch,
)


METHODS = (
    "exploring_oft_dino_feature_uniform",
    "exploring_oft_siglip_feature_uniform",
    "exploring_oft_dino_signed_geometry",
    "exploring_oft_dino_signed_geometry_sham",
    "exploring_oft_full_persistent6d",
    "exploring_oft_dino_persistent6d",
)

FEATURE_METHODS = {
    "exploring_oft_dino_feature_uniform": "dino",
    "exploring_oft_siglip_feature_uniform": "siglip",
}
SIGNED_METHODS = {
    "exploring_oft_dino_signed_geometry",
    "exploring_oft_dino_signed_geometry_sham",
}


def feature_disruption_loss(attacked: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    require(attacked.shape == clean.shape and attacked.ndim == 3,
            f"Feature shape mismatch: attacked={tuple(attacked.shape)}, clean={tuple(clean.shape)}")
    return F.cosine_similarity(attacked.float(), clean.float().detach(), dim=-1).mean() - 1.0


def opposite_boundary_target(clean: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.where(clean >= (low + high) / 2.0, low, high).detach()


def persistent6d_target(clean: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """One opposite direction per DoF, held constant over the eight-step chunk."""
    clean_mean = clean[..., :6].mean(dim=1, keepdim=True)
    endpoint = torch.where(clean_mean >= (low[..., :6] + high[..., :6]) / 2.0,
                           low[..., :6], high[..., :6]).detach()
    return endpoint.expand(-1, clean.shape[1], -1)


def persistent6d_loss(action: torch.Tensor, target: torch.Tensor,
                      low: torch.Tensor, high: torch.Tensor, beta: float) -> torch.Tensor:
    require(action.ndim == 3 and action.shape[-1] == 7 and target.shape == action[..., :6].shape,
            "Persistent-6D action/target shape drift")
    scale = (high[..., :6] - low[..., :6]).clamp_min(1e-6)
    step = torch.mean(((action[..., :6] - target) / scale) ** 2)
    persistent = torch.mean(((action[..., :6].mean(dim=1, keepdim=True)
                              - target[:, :1]) / scale) ** 2)
    return step + float(beta) * persistent


def combine_gradients(main_grad: torch.Tensor, auxiliary_grad: torch.Tensor,
                      ratio: float, frozen_weight: float | None) -> tuple[torch.Tensor, float, float, float]:
    main_norm = torch.linalg.vector_norm(main_grad.float())
    auxiliary_norm = torch.linalg.vector_norm(auxiliary_grad.float())
    require(float(main_norm) > 1e-12 and float(auxiliary_norm) > 1e-12,
            "Zero main/auxiliary gradient")
    weight = float(frozen_weight) if frozen_weight is not None else ratio * float(main_norm / auxiliary_norm)
    cosine = float(F.cosine_similarity(main_grad.reshape(-1).float(),
                                       auxiliary_grad.reshape(-1).float(), dim=0))
    effective_ratio = weight * float(auxiliary_norm / main_norm)
    return main_grad + weight * auxiliary_grad, weight, cosine, effective_ratio


def direction_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(record["task_id"]), int(record["state_index"]),
            int(record["trajectory_rank"]), int(record["decision_index"]))


def signed_direction_loss(dino: torch.Tensor, clean_dino: torch.Tensor, direction: torch.Tensor,
                          sigma_v: float, sham_seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    vector = direction.to(dino.device).float().reshape_as(dino)
    if sham_seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(sham_seed))
        permutation = torch.randperm(vector.shape[-2], generator=generator).to(vector.device)
        vector = vector[:, permutation, :]
    score = torch.dot((dino.float() - clean_dino.float()).reshape(-1), vector.reshape(-1)) / float(sigma_v)
    return -score, score


class ExploringOFTJoint500Attacker(ExploringOFTOpenVLAAttacker):
    def train_joint_patch(self, records: list[dict[str, Any]], cfg: dict[str, Any], method: str) -> None:
        require(method in METHODS, f"Unknown joint method: {method}")
        output_dir = Path(cfg["output_root"]) / method
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            require(status.get("status") != "COMPLETE", "Refusing to overwrite completed patch")

        artifact = torch.load(cfg["signed_direction_artifact"], map_location="cpu", weights_only=False)
        directions = {direction_key(row): row["v_star"] for row in artifact["records"]}
        require(len(directions) == len(records), "Signed-direction coverage drift")
        sigma_v = float(artifact["sigma_v"])
        require(sigma_v > 0, "Invalid signed-direction normalization")

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
        checkpoints = set(map(int, cfg["checkpoint_outer_iterations"]))
        resume_path = output_dir / "resume.pt"
        rows: list[dict[str, Any]] = []
        start_outer = 1
        auxiliary_weight: float | None = None
        if resume_path.is_file():
            resume = torch.load(resume_path, map_location=self.wrapper.device, weights_only=False)
            require(resume["experiment_id"] == cfg["experiment_id"] and resume["method"] == method,
                    "Resume identity mismatch")
            patch.data.copy_(resume["patch"])
            initial = resume["initial"].to(self.wrapper.device)
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            rows = resume["rows"]
            auxiliary_weight = resume["auxiliary_weight"]
            start_outer = int(resume["outer"]) + 1

        metadata = {
            "experiment_id": cfg["experiment_id"], "method": method,
            "source_trainer": "read-only VLAAttacker/white_patch/UADA.py::patchattack_unconstrained",
            "source_uada_sha256": sha256_file(Path(__file__).with_name("UADA.py")),
            "source_copy_sha256": sha256_file(Path(__file__).with_name("UADA_OFT_SOURCE_COPY.py")),
            "signed_direction_sha256": sha256_file(cfg["signed_direction_artifact"]),
            "seed": int(cfg["seed"]), "data_seed": int(cfg["data_seed"]),
            "outer_iterations": int(cfg["outer_iterations"]), "inner_loop": int(cfg["inner_loop"]),
            "baseline_patch_sha256": cfg["baseline_patch_sha256"],
        }
        if start_outer == 1:
            save_patch(output_dir / "outer_0000.pt", patch, {**metadata, **patch_metrics(patch, initial)})

        started = time.time()
        for outer in range(start_outer, int(cfg["outer_iterations"]) + 1):
            record_index = self._training_index(outer, len(records), int(cfg["data_seed"]))
            record = records[record_index]
            inputs, proprio = self._base_inputs(record)
            observation = record["observation"]
            with torch.no_grad():
                clean, clean_dino, clean_siglip = self._forward(
                    inputs, proprio, observation, None, False, capture_siglip=True)
            require(clean_siglip is not None, "Clean Wrist-SigLIP anchor missing")
            old_target = opposite_boundary_target(clean, low, high)
            persistent_target = persistent6d_target(clean, low, high)
            sums = {"main": 0.0, "aux": 0.0, "signed": 0.0, "grad": 0.0,
                    "cosine": 0.0, "ratio": 0.0}

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

                if method in ("exploring_oft_full_persistent6d", "exploring_oft_dino_persistent6d"):
                    main_loss = persistent6d_loss(
                        action, persistent_target, low, high, float(cfg["persistent_beta"]))
                else:
                    main_loss = F.mse_loss(action[..., cfg["maskidx"]], old_target[..., cfg["maskidx"]])

                auxiliary_loss: torch.Tensor | None = None
                signed_score = torch.zeros((), device=self.wrapper.device)
                if method in FEATURE_METHODS:
                    branch = FEATURE_METHODS[method]
                    selected, anchor = (dino, clean_dino) if branch == "dino" else (siglip, clean_siglip)
                    auxiliary_loss = feature_disruption_loss(selected, anchor)
                elif method in SIGNED_METHODS:
                    sham = method.endswith("_sham")
                    auxiliary_loss, signed_score = signed_direction_loss(
                        dino, clean_dino, directions[direction_key(record)], sigma_v,
                        int(cfg["direction_sham_seed"]) + record_index if sham else None)
                elif method == "exploring_oft_dino_persistent6d":
                    random.seed(transform_seed)
                    np.random.seed(transform_seed % (2**32 - 1))
                    torch.manual_seed(transform_seed)
                    dino_route_action, _, _ = self._forward(
                        inputs, proprio, observation, patch, bool(cfg["geometry"]),
                        clean_siglip=clean_siglip)
                    auxiliary_loss = persistent6d_loss(
                        dino_route_action, persistent_target, low, high, float(cfg["persistent_beta"]))

                if auxiliary_loss is None:
                    main_loss.backward()
                    cosine = 0.0
                    effective_ratio = 0.0
                else:
                    main_grad = torch.autograd.grad(main_loss, patch, retain_graph=True)[0]
                    auxiliary_grad = torch.autograd.grad(auxiliary_loss, patch)[0]
                    patch.grad, auxiliary_weight, cosine, effective_ratio = combine_gradients(
                        main_grad, auxiliary_grad, float(cfg["auxiliary_gradient_ratio"]), auxiliary_weight)
                require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Invalid patch gradient")
                grad_norm = float(torch.linalg.vector_norm(patch.grad))
                step_uada_patch(patch, optimizer, self.wrapper.vla)
                scale = 1.0 / int(cfg["inner_loop"])
                sums["main"] += float(main_loss.detach()) * scale
                sums["aux"] += (float(auxiliary_loss.detach()) if auxiliary_loss is not None else 0.0) * scale
                sums["signed"] += float(signed_score.detach()) * scale
                sums["grad"] += grad_norm * scale
                sums["cosine"] += cosine * scale
                sums["ratio"] += effective_ratio * scale

            scheduler.step()
            structure = patch_metrics(patch, initial)
            rows.append({
                "method": method, "outer_iteration": outer,
                "optimizer_updates": outer * int(cfg["inner_loop"]),
                "record_index": record_index, "task_id": int(record["task_id"]),
                "state_index": int(record["state_index"]), "main_loss": sums["main"],
                "auxiliary_loss": sums["aux"], "signed_dino_score": sums["signed"],
                "patch_grad_l2": sums["grad"], "auxiliary_weight": auxiliary_weight,
                "main_aux_gradient_cosine": sums["cosine"],
                "effective_aux_gradient_ratio": sums["ratio"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": time.time() - started, **structure,
            })
            if outer % 5 == 0 or outer in checkpoints:
                print(f"EXPLORING_OFT_JOINT method={method} outer={outer}/{cfg['outer_iterations']} "
                      f"updates={outer * int(cfg['inner_loop'])} main={sums['main']:.5f} "
                      f"aux={sums['aux']:.5f} corr={structure['init_correlation']:.4f} "
                      f"tv_ratio={structure['tv_ratio']:.4f}", flush=True)
            if outer in checkpoints:
                save_patch(output_dir / f"outer_{outer:04d}.pt", patch,
                           {**metadata, **structure, "auxiliary_weight": auxiliary_weight})
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
                              "rows": rows, "auxiliary_weight": auxiliary_weight}, resume_path)

        atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
        atomic_json({"status": "COMPLETE", **metadata, **patch_metrics(patch, initial),
                     "auxiliary_weight": auxiliary_weight, "elapsed_s": time.time() - started}, status_path)
        resume_path.unlink(missing_ok=True)

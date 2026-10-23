"""Action-sensitive Wrist-DINO dense-token guidance for Exploring-OFT UADA.

This trainer keeps the audited Exploring patch optimizer: random RGB patch,
random Wrist EOT placement, AdamW/cosine schedule, one clamp per update, and
the OFT opposite-boundary action loss as the primary objective.  The DINO
addition is not a branch-only attack.  It uses the clean Wrist-DINO dense-token
Jacobian of the action-boundary loss as a local teacher, then trains the pixel
patch to push patch-covered Wrist-DINO tokens in that action-worsening direction.
"""

from __future__ import annotations

import json
import random
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_DINO import (
    atomic_json,
    atomic_parquet,
    atomic_torch,
    patch_metrics,
    require,
    save_patch,
    sha256_file,
)
from white_patch.UADA_OFT_JOINT500 import combine_gradients, opposite_boundary_target
from white_patch.UADA_OFT_OBJECT_DECOY import ExploringOFTObjectDecoyAttacker
from white_patch.UADA_OFT_SOURCE_COPY import (
    build_uada_adamw,
    init_uada_patch,
    step_uada_patch,
)


METHODS = (
    "exploring_oft_dino_jacobian_dense_r03",
    "exploring_oft_dino_jacobian_dense_r10",
)


def dino_action_descent_direction(
    action: torch.Tensor,
    dino: torch.Tensor,
    target: torch.Tensor,
    maskidx: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the dense-token direction that locally moves action to target."""
    require(action.ndim == 3 and action.shape[-1] == 7, "Unexpected action shape")
    require(dino.ndim == 3 and dino.shape[0] == 1, "Unexpected Wrist-DINO shape")
    loss = F.mse_loss(action[..., maskidx], target[..., maskidx])
    grad = torch.autograd.grad(loss, dino, retain_graph=False)[0].detach().float()
    token_norm = torch.linalg.vector_norm(grad, dim=-1).clamp_min(1e-12)
    direction = -grad / token_norm[..., None]
    return direction.detach(), token_norm.detach(), loss.detach()


def sensitivity_support(
    patch_tokens: torch.Tensor,
    token_sensitivity: torch.Tensor,
    *,
    min_mass: float = 0.5,
) -> torch.Tensor:
    """Weight the current patch footprint by clean action-DINO sensitivity."""
    support = patch_tokens.float().reshape(-1).clamp(0, 1)
    sensitivity = token_sensitivity.float().reshape(-1)
    sensitivity = sensitivity / sensitivity.mean().clamp_min(1e-12)
    weights = support * sensitivity.clamp(0.05, 20.0)
    if float(weights.sum()) < float(min_mass):
        weights = support
    require(float(weights.sum()) >= float(min_mass), "Patch footprint has no DINO-token mass")
    return weights


def dino_directional_loss(
    attacked_dino: torch.Tensor,
    clean_dino: torch.Tensor,
    direction: torch.Tensor,
    patch_tokens: torch.Tensor,
    token_sensitivity: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Encourage attacked Wrist-DINO tokens to move along the action-Jacobian."""
    require(attacked_dino.shape == clean_dino.shape == direction.shape,
            "DINO clean/attacked/direction shape drift")
    weights = sensitivity_support(patch_tokens, token_sensitivity).to(attacked_dino.device)
    delta = attacked_dino.float() - clean_dino.detach().float()
    score_per_token = (delta * direction.to(attacked_dino.device).float()).sum(dim=-1).reshape(-1)
    score = (score_per_token * weights).sum() / weights.sum()
    return -score, {
        "dino_direction_score": float(score.detach()),
        "dino_weight_mass": float(weights.detach().sum()),
        "dino_weighted_token_sensitivity": float(
            (token_sensitivity.reshape(-1).to(weights.device).float() * weights).sum().detach()
            / weights.sum().detach().clamp_min(1e-12)
        ),
    }


class ExploringOFTDinoJacobianAttacker(ExploringOFTObjectDecoyAttacker):
    """Fresh-start Wrist-only patch training with online DINO-Jacobian teachers."""

    def _forward_action_and_raw_dino(
        self,
        inputs: dict[str, torch.Tensor],
        proprio: np.ndarray,
        observation: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from src.hooks.manager import HookSpec, TensorHooks

        model_inputs = dict(inputs)
        model_inputs["pixel_values"] = self._patched_pixels(observation, None, False)
        dino_spec = HookSpec(path="vla.vision_backbone.featurizer", hook="output", call_index=1)
        roots = {"vla": self.wrapper.vla, "action_head": self.wrapper.action_head}
        with ExitStack() as stack:
            hooks = stack.enter_context(TensorHooks(roots, [dino_spec], mode="capture", retain_grad=True))
            _unused, hidden = self.wrapper.vla.predict_action(
                **model_inputs,
                unnorm_key=self.wrapper.gen_cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.wrapper.proprio_projector,
                noisy_action_projector=self.wrapper.noisy_action_projector,
                action_head=self.wrapper.action_head,
                use_film=False,
            )
        head = getattr(self.wrapper.action_head, "module", self.wrapper.action_head)
        action = head.predict_action(hidden).reshape(1, 8, 7).float()
        dino = hooks.captured[0]
        require(dino is not None and torch.isfinite(dino).all(), "Invalid clean Wrist-DINO capture")
        require(torch.isfinite(action).all(), "Invalid clean OFT action")
        return action, dino

    def train_jacobian_patch(self, records: list[dict[str, Any]], cfg: dict[str, Any], method: str) -> None:
        require(method in METHODS, f"Unknown DINO-Jacobian method: {method}")
        ratio = float(cfg["jacobian_gradient_ratios"][method])
        output_dir = Path(cfg["output_root"]) / method
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            require(status.get("status") != "COMPLETE", "Refusing to overwrite completed patch")

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
            print(f"RESUME method={method} outer={start_outer}", flush=True)

        metadata = {
            "experiment_id": cfg["experiment_id"],
            "method": method,
            "source_trainer": "read-only VLAAttacker/white_patch/UADA.py::patchattack_unconstrained",
            "source_uada_sha256": sha256_file(Path(__file__).with_name("UADA.py")),
            "source_copy_sha256": sha256_file(Path(__file__).with_name("UADA_OFT_SOURCE_COPY.py")),
            "adaptation": "OFT UADA action loss plus action-sensitive Wrist-DINO dense-token directional guidance",
            "seed": int(cfg["seed"]),
            "data_seed": int(cfg["data_seed"]),
            "maskidx": maskidx,
            "outer_iterations": int(cfg["outer_iterations"]),
            "inner_loop": int(cfg["inner_loop"]),
            "jacobian_gradient_ratio": ratio,
            "training_bank_sha256": sha256_file(cfg["training_bank"]),
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
            self.wrapper.vla.zero_grad(set_to_none=True)
            clean_action_graph, clean_dino_graph = self._forward_action_and_raw_dino(
                inputs, proprio, observation)
            target = opposite_boundary_target(clean_action_graph.detach(), low, high)
            direction, token_sensitivity, clean_jacobian_loss = dino_action_descent_direction(
                clean_action_graph, clean_dino_graph, target, maskidx)
            clean_action = clean_action_graph.detach()
            clean_dino = clean_dino_graph.detach().float()
            sums = {
                "action": 0.0,
                "dino": 0.0,
                "direction_score": 0.0,
                "weight_mass": 0.0,
                "weighted_sensitivity": 0.0,
                "grad": 0.0,
                "cosine": 0.0,
                "ratio": 0.0,
                "guided_steps": 0.0,
            }

            for inner in range(int(cfg["inner_loop"])):
                transform_seed = int(cfg["seed"]) * 10_000_000 + outer * 10_000 + inner
                random.seed(transform_seed)
                np.random.seed(transform_seed % (2**32 - 1))
                torch.manual_seed(transform_seed)
                optimizer.zero_grad(set_to_none=True)
                self.wrapper.vla.zero_grad(set_to_none=True)
                action, dino, patch_tokens = self._forward_with_patch_tokens(
                    inputs, proprio, observation, patch, bool(cfg["geometry"]))
                action_loss = F.mse_loss(action[..., maskidx], target[..., maskidx])
                dino_loss, dino_stats = dino_directional_loss(
                    dino, clean_dino, direction, patch_tokens, token_sensitivity)
                main_grad = torch.autograd.grad(action_loss, patch, retain_graph=True)[0]
                dino_grad = torch.autograd.grad(dino_loss, patch)[0]
                if float(torch.linalg.vector_norm(dino_grad.float())) > 1e-12:
                    patch.grad, auxiliary_weight, cosine, effective_ratio = combine_gradients(
                        main_grad, dino_grad, ratio, auxiliary_weight)
                    sums["guided_steps"] += 1.0
                else:
                    patch.grad = main_grad
                    cosine = 0.0
                    effective_ratio = 0.0
                require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Invalid patch gradient")
                grad_norm = float(torch.linalg.vector_norm(patch.grad))
                step_uada_patch(patch, optimizer, self.wrapper.vla)

                scale = 1.0 / int(cfg["inner_loop"])
                sums["action"] += float(action_loss.detach()) * scale
                sums["dino"] += float(dino_loss.detach()) * scale
                sums["direction_score"] += dino_stats["dino_direction_score"] * scale
                sums["weight_mass"] += dino_stats["dino_weight_mass"] * scale
                sums["weighted_sensitivity"] += dino_stats["dino_weighted_token_sensitivity"] * scale
                sums["grad"] += grad_norm * scale
                sums["cosine"] += cosine * scale
                sums["ratio"] += effective_ratio * scale

            scheduler.step()
            structure = patch_metrics(patch, initial)
            rows.append({
                "method": method,
                "outer_iteration": outer,
                "optimizer_updates": outer * int(cfg["inner_loop"]),
                "record_index": record_index,
                "task_id": int(record["task_id"]),
                "state_index": int(record["state_index"]),
                "trajectory_rank": int(record["trajectory_rank"]),
                "decision_index": int(record["decision_index"]),
                "clean_jacobian_action_loss": float(clean_jacobian_loss),
                "action_boundary_mse": sums["action"],
                "dino_directional_loss": sums["dino"],
                "dino_direction_score": sums["direction_score"],
                "dino_weight_mass": sums["weight_mass"],
                "dino_weighted_token_sensitivity": sums["weighted_sensitivity"],
                "guided_inner_steps": int(sums["guided_steps"]),
                "patch_grad_l2": sums["grad"],
                "auxiliary_weight": auxiliary_weight,
                "main_aux_gradient_cosine": sums["cosine"],
                "effective_aux_gradient_ratio": sums["ratio"],
                "clean_action_mean_dim0": float(clean_action[..., maskidx].mean()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": time.time() - started,
                **structure,
            })
            if outer % 5 == 0 or outer in checkpoints:
                print(f"EXPLORING_OFT_DINO_JACOBIAN method={method} outer={outer}/{cfg['outer_iterations']} "
                      f"updates={outer * int(cfg['inner_loop'])} action={sums['action']:.5f} "
                      f"dino_score={sums['direction_score']:.5f} ratio={sums['ratio']:.3f} "
                      f"corr={structure['init_correlation']:.4f} tv_ratio={structure['tv_ratio']:.4f}",
                      flush=True)
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

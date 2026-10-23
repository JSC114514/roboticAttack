"""Wrist-only, action-screened spatial DINO guidance for an Exploring UADA copy.

The read-only ``UADA.py`` is never imported for mutation.  Patch initialization,
AdamW schedule, EOT, boundary action target, and projected RGB update remain the
audited OFT copy of Exploring UADA.  The new guidance transports local Wrist-DINO
tokens by one grid cell; the direction is frozen by a separate action-head pilot.
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
import torchvision

from white_patch.UADA_OFT_DINO import (
    atomic_json, atomic_parquet, atomic_torch, patch_metrics, require,
    save_patch, sha256_file,
)
from white_patch.UADA_OFT_JOINT500 import opposite_boundary_target
from white_patch.UADA_OFT_OBJECT_DECOY import (
    ExploringOFTObjectDecoyAttacker, image_mask_to_dino_tokens,
    load_object_masks, record_key,
)
from white_patch.UADA_OFT_SOURCE_COPY import (
    build_uada_adamw, init_uada_patch, step_uada_patch,
)

METHOD = "exploring_oft_wrist_dino_spatial"
SHIFTS = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
GRID = 16


def site_token_mask(site: tuple[int, int], patch_size: int = 50) -> torch.Tensor:
    x, y = site
    require(0 <= x <= 256 - patch_size and 0 <= y <= 256 - patch_size,
            "Wrist patch outside 256x256 input")
    image_mask = torch.zeros(256, 256)
    image_mask[y:y + patch_size, x:x + patch_size] = 1.0
    return image_mask_to_dino_tokens(image_mask)


def shift_mask(mask: torch.Tensor, shift: tuple[int, int]) -> torch.Tensor:
    """Translate a DINO token mask with zero, rather than circular, padding."""
    require(mask.numel() == GRID * GRID and max(abs(v) for v in shift) == 1,
            "Expected one-token DINO shift")
    dx, dy = shift
    source = mask.reshape(GRID, GRID)
    result = torch.zeros_like(source)
    y_dst = slice(max(dy, 0), GRID + min(dy, 0))
    x_dst = slice(max(dx, 0), GRID + min(dx, 0))
    y_src = slice(max(-dy, 0), GRID - max(dy, 0))
    x_src = slice(max(-dx, 0), GRID - max(dx, 0))
    result[y_dst, x_dst] = source[y_src, x_src]
    return result.reshape(-1)


def spatial_teacher(clean_dino: torch.Tensor, patch_tokens: torch.Tensor,
                    shift: tuple[int, int]) -> torch.Tensor:
    """Replace *only* patch-footprint Wrist-DINO tokens by nearby clean tokens."""
    require(clean_dino.ndim == 3 and clean_dino.shape[:2] == (1, GRID * GRID),
            "Unexpected Wrist-DINO token shape")
    require(patch_tokens.numel() == GRID * GRID, "Patch/DINO grid mismatch")
    dx, dy = shift
    require(max(abs(dx), abs(dy)) == 1 and min(abs(dx), abs(dy)) == 0,
            "Only four cardinal one-token shifts are allowed")
    roi = patch_tokens.reshape(GRID, GRID) >= 0.5
    ys, xs = torch.where(roi)
    require(ys.numel() >= 4, "Too few DINO tokens in patch footprint")
    sy, sx = ys - dy, xs - dx
    require(bool(((sy >= 0) & (sy < GRID) & (sx >= 0) & (sx < GRID)).all()),
            "Shift would sample outside DINO grid")
    teacher = clean_dino.detach().clone().reshape(1, GRID, GRID, -1)
    source = clean_dino.detach().reshape(1, GRID, GRID, -1)
    teacher[:, ys, xs, :] = source[:, sy, sx, :]
    return teacher.reshape_as(clean_dino)


def target_support(patch_tokens: torch.Tensor, target_tokens: torch.Tensor,
                   shift: tuple[int, int]) -> torch.Tensor:
    require(patch_tokens.shape == target_tokens.shape == (GRID * GRID,),
            "Patch/target mask shape drift")
    return patch_tokens.float().clamp(0, 1) * torch.maximum(
        target_tokens.float().clamp(0, 1), shift_mask(target_tokens.float(), shift))


def spatial_loss(attacked_dino: torch.Tensor, teacher: torch.Tensor,
                 support: torch.Tensor) -> torch.Tensor:
    require(attacked_dino.shape == teacher.shape and support.numel() == GRID * GRID,
            "Spatial feature loss shape drift")
    weights = support.to(attacked_dino.device).float().reshape(1, -1)
    require(float(weights.sum()) >= 0.5, "No target-object token support")
    cosine = F.cosine_similarity(attacked_dino.float(), teacher.detach().float(), dim=-1)
    return ((1.0 - cosine) * weights).sum() / weights.sum()


def action_safe_gradient(main: torch.Tensor, auxiliary: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Keep UADA's first-order component; cap orthogonal DINO guidance to its norm.

    This is a parameter-free, main-task-preserving variant of gradient surgery.
    It does *not* guarantee an AdamW step improves the nonlinear action loss.
    """
    main_f, aux_f = main.float(), auxiliary.float()
    main_sq = torch.sum(main_f * main_f)
    require(float(main_sq) > 1e-20, "Zero UADA gradient")
    dot = torch.sum(main_f * aux_f)
    if float(dot) < 0.0:
        aux_f = aux_f - dot / main_sq * main_f
    aux_norm = torch.linalg.vector_norm(aux_f)
    main_norm = torch.sqrt(main_sq)
    if float(aux_norm) > 1e-20:
        aux_f = aux_f * torch.clamp(main_norm / aux_norm, max=1.0)
    cosine = float(dot / (main_norm * torch.linalg.vector_norm(auxiliary.float()).clamp_min(1e-20)))
    return (main_f + aux_f).to(main.dtype), cosine


class ExploringOFTDinoSpatialAttacker(ExploringOFTObjectDecoyAttacker):
    def __init__(self, wrapper: Any, save_dir: str | Path, site: tuple[int, int]):
        super().__init__(wrapper, save_dir, resize_patch=False)
        self.site = site

    def _paste_patch_tensor_and_mask(self, image: Any, patch: torch.Tensor,
                                     geometry: bool) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = torchvision.transforms.ToTensor()(
            self.randomPatchTransform._to_rgb_pil(image)).to(self.wrapper.device)
        _, height, width = image_tensor.shape
        _, patch_height, patch_width = patch.shape
        x, y = self.site
        require(0 <= x <= width - patch_width and 0 <= y <= height - patch_height,
                "Fixed Wrist site outside image")
        canvas = torch.full_like(image_tensor, -100.0)
        mask = torch.zeros((1, height, width), device=self.wrapper.device)
        canvas[:, y:y + patch_height, x:x + patch_width] = patch
        mask[:, y:y + patch_height, x:x + patch_width] = 1.0
        if geometry:
            matrix = self.randomPatchTransform.combined_transform_matrix().to(self.wrapper.device)
            canvas = self.randomPatchTransform.apply_affine_transform(canvas, matrix)
            if canvas.ndim == 4:
                canvas = canvas.squeeze(0)
            grid = F.affine_grid(matrix[:2, :].unsqueeze(0), mask.unsqueeze(0).size(),
                                 align_corners=False)
            mask = F.grid_sample(mask.unsqueeze(0), grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=False).squeeze(0)
        return torch.where(canvas < -20, image_tensor, canvas).clamp(0, 1), mask.clamp(0, 1)

    def train_spatial_patch(self, records: list[dict[str, Any]], cfg: dict[str, Any],
                            pilot: dict[str, Any]) -> None:
        output_dir = Path(cfg["output_root"]) / METHOD
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "status.json"
        if status_path.is_file():
            require(json.loads(status_path.read_text()).get("status") != "COMPLETE",
                    "Refusing to overwrite completed patch")
        masks = load_object_masks(cfg, records)
        selected = {int(task): SHIFTS[name] for task, name in pilot["selected_shifts"].items()}
        enabled = set(map(int, pilot["enabled_tasks"]))
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
        metadata = {
            "experiment_id": cfg["experiment_id"], "method": METHOD,
            "source_trainer": "read-only UADA.py::patchattack_unconstrained",
            "source_uada_sha256": sha256_file(Path(__file__).with_name("UADA.py")),
            "source_copy_sha256": sha256_file(Path(__file__).with_name("UADA_OFT_SOURCE_COPY.py")),
            "adaptation": "fixed Wrist patch; action-screened local DINO token transport; UADA main loss",
            "site_xy": list(self.site), "seed": int(cfg["seed"]), "data_seed": int(cfg["data_seed"]),
            "maskidx": maskidx, "outer_iterations": int(cfg["outer_iterations"]),
            "inner_loop": int(cfg["inner_loop"]), "pilot_sha256": sha256_file(cfg["pilot_receipt"]),
            "training_bank_sha256": sha256_file(cfg["training_bank"]),
        }
        resume_path = output_dir / "resume.pt"
        rows: list[dict[str, Any]] = []
        start_outer = 1
        if resume_path.is_file():
            resume = torch.load(resume_path, map_location=self.wrapper.device, weights_only=False)
            require(all(resume.get(k) == metadata[k] for k in (
                "experiment_id", "method", "source_uada_sha256", "source_copy_sha256",
                "pilot_sha256", "training_bank_sha256", "site_xy")), "Resume lineage drift")
            patch.data.copy_(resume["patch"])
            initial = resume["initial"].to(self.wrapper.device)
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            rows = resume["rows"]
            start_outer = int(resume["outer"]) + 1
        else:
            save_patch(output_dir / "outer_0000.pt", patch,
                       {**metadata, **patch_metrics(patch, initial)})

        started = time.time()
        for outer in range(start_outer, int(cfg["outer_iterations"]) + 1):
            index = self._training_index(outer, len(records), int(cfg["data_seed"]))
            record = records[index]
            inputs, proprio = self._base_inputs(record)
            observation = record["observation"]
            task_id = int(record["task_id"])
            target_tokens = masks[record_key(record)]["target_object"].to(self.wrapper.device)
            shift = selected[task_id]
            with torch.no_grad():
                clean_action, clean_dino, _ = self._forward(
                    inputs, proprio, observation, None, False)
            boundary = opposite_boundary_target(clean_action, low, high)
            action_total = dino_total = cosine_total = grad_total = 0.0
            guided_steps = 0
            for inner in range(int(cfg["inner_loop"])):
                transform_seed = int(cfg["seed"]) * 10_000_000 + outer * 10_000 + inner
                random.seed(transform_seed)
                np.random.seed(transform_seed % (2**32 - 1))
                torch.manual_seed(transform_seed)
                optimizer.zero_grad(set_to_none=True)
                self.wrapper.vla.zero_grad(set_to_none=True)
                action, dino, patch_tokens = self._forward_with_patch_tokens(
                    inputs, proprio, observation, patch, bool(cfg["geometry"]))
                action_loss = F.mse_loss(action[..., maskidx], boundary[..., maskidx])
                support = target_support(patch_tokens, target_tokens, shift)
                if task_id in enabled and float(support.sum()) >= 0.5:
                    teacher = spatial_teacher(clean_dino, patch_tokens, shift)
                    feature_loss = spatial_loss(dino, teacher, support)
                    main_grad = torch.autograd.grad(action_loss, patch, retain_graph=True)[0]
                    feature_grad = torch.autograd.grad(feature_loss, patch)[0]
                    patch.grad, cosine = action_safe_gradient(main_grad, feature_grad)
                    dino_total += float(feature_loss.detach()) / int(cfg["inner_loop"])
                    cosine_total += cosine / int(cfg["inner_loop"])
                    guided_steps += 1
                else:
                    action_loss.backward()
                require(patch.grad is not None and torch.isfinite(patch.grad).all(),
                        "Invalid patch gradient")
                action_total += float(action_loss.detach()) / int(cfg["inner_loop"])
                grad_total += float(torch.linalg.vector_norm(patch.grad)) / int(cfg["inner_loop"])
                step_uada_patch(patch, optimizer, self.wrapper.vla)
            scheduler.step()
            structure = patch_metrics(patch, initial)
            rows.append({"method": METHOD, "outer_iteration": outer,
                         "optimizer_updates": outer * int(cfg["inner_loop"]),
                         "record_index": index, "task_id": task_id,
                         "trajectory_rank": int(record["trajectory_rank"]),
                         "decision_index": int(record["decision_index"]),
                         "action_boundary_mse": action_total, "dino_spatial_loss": dino_total,
                         "guided_inner_steps": guided_steps,
                         "main_aux_gradient_cosine": cosine_total,
                         "patch_grad_l2": grad_total,
                         "learning_rate": float(optimizer.param_groups[0]["lr"]),
                         "elapsed_s": time.time() - started, **structure})
            if outer % 5 == 0 or outer in checkpoints:
                print(f"EXPLORING_OFT_DINO_SPATIAL outer={outer}/{cfg['outer_iterations']} "
                      f"action={action_total:.5f} dino={dino_total:.5f} "
                      f"guided={guided_steps} corr={structure['init_correlation']:.4f} "
                      f"tv_ratio={structure['tv_ratio']:.4f}", flush=True)
            if outer in checkpoints:
                save_patch(output_dir / f"outer_{outer:04d}.pt", patch, {**metadata, **structure})
            if str(outer) in cfg["structure_gates"]:
                gate = cfg["structure_gates"][str(outer)]
                passed = (structure["init_correlation"] < float(gate["max_init_correlation"])
                          and structure["tv_ratio"] < float(gate["max_tv_ratio"]))
                atomic_json({"outer_iteration": outer, "passed": passed, **gate, **structure},
                            output_dir / f"structure_gate_{outer:04d}.json")
                if not passed:
                    atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
                    atomic_json({"status": "STRUCTURE_GATE_FAIL", "outer_iteration": outer,
                                 **metadata, **structure}, status_path)
                    return
            if outer % int(cfg["resume_every_outer_iterations"]) == 0:
                atomic_torch({"outer": outer, **metadata, "patch": patch.detach(),
                              "initial": initial.detach(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "rows": rows}, resume_path)
        atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
        atomic_json({"status": "COMPLETE", **metadata, **patch_metrics(patch, initial),
                     "elapsed_s": time.time() - started}, status_path)
        resume_path.unlink(missing_ok=True)

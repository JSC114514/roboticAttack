"""Object-local DINO decoy patch training on top of Exploring UADA.

This file intentionally lives next to, but does not edit, the read-only
``UADA.py`` source.  It keeps the Exploring control flow (random RGB patch,
random EOT paste, AdamW/cosine schedule, clamp after every update) and adds a
Wrist-DINO dense-feature objective using the audited object masks.
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

from white_patch.UADA_OFT_DINO import (
    ExploringOFTOpenVLAAttacker,
    atomic_json,
    atomic_parquet,
    atomic_torch,
    patch_metrics,
    require,
    save_patch,
    sha256_file,
)
from white_patch.UADA_OFT_JOINT500 import combine_gradients, opposite_boundary_target
from white_patch.UADA_OFT_SOURCE_COPY import (
    build_uada_adamw,
    init_uada_patch,
    step_uada_patch,
)


METHODS = (
    "exploring_oft_dino_object_decoy_r03",
    "exploring_oft_dino_object_decoy_r10",
)

MASK_ROLES = ("target_object", "matched_background")


def record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (int(record["task_id"]), int(record["trajectory_rank"]), int(record["decision_index"]))


def image_mask_to_dino_tokens(mask: torch.Tensor, image_size: int = 224, grid: int = 16) -> torch.Tensor:
    """Match the processor resize/crop used for DINO Wrist tokens."""
    require(mask.ndim in (2, 3), f"Unexpected mask rank: {tuple(mask.shape)}")
    if mask.ndim == 2:
        mask = mask[None]
    x = mask.float()[None]
    x = F.interpolate(x, size=(image_size, image_size), mode="nearest")
    scale = math.sqrt(0.9)
    axis = torch.linspace(-scale, scale, image_size, device=x.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = F.grid_sample(x, torch.stack((xx, yy), dim=-1)[None], mode="nearest",
                      padding_mode="border", align_corners=True)
    return F.avg_pool2d(x, kernel_size=image_size // grid)[0, 0].reshape(-1)


def np_mask_to_tokens(mask: np.ndarray) -> torch.Tensor:
    require(mask.shape == (256, 256), f"Unexpected oracle mask shape: {mask.shape}")
    return image_mask_to_dino_tokens(torch.as_tensor(mask.astype(np.float32)))


def weighted_proto(features: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    require(features.ndim == 3 and features.shape[0] == 1, f"Unexpected feature shape: {tuple(features.shape)}")
    require(weights.numel() == features.shape[1], "Token-mask/feature mismatch")
    weights = weights.to(device=features.device, dtype=features.dtype).reshape(1, -1)
    mass = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    proto = (features.float() * weights.float()[..., None]).sum(dim=1) / mass.float()
    return F.normalize(proto, dim=-1), mass.squeeze(1).float()


def object_decoy_loss(
    attacked_dino: torch.Tensor,
    clean_dino: torch.Tensor,
    masks: dict[str, torch.Tensor],
    patch_tokens: torch.Tensor,
    *,
    object_weight: float,
    patch_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Make the real target look less like itself and the patch look target-like."""
    target = masks["target_object"].to(attacked_dino.device).float()
    background = masks["matched_background"].to(attacked_dino.device).float()
    patch = patch_tokens.to(attacked_dino.device).float().clamp(0, 1)
    visible_target = target * (1.0 - patch.clamp(0, 1))
    if float(visible_target.sum()) < 1e-3:
        visible_target = target

    clean_target, clean_target_mass = weighted_proto(clean_dino.detach(), target)
    clean_background, clean_background_mass = weighted_proto(clean_dino.detach(), background)
    attacked_target, attacked_target_mass = weighted_proto(attacked_dino, visible_target)
    attacked_patch, attacked_patch_mass = weighted_proto(attacked_dino, patch)

    object_clean_cos = F.cosine_similarity(attacked_target, clean_target, dim=-1).mean()
    object_bg_cos = F.cosine_similarity(attacked_target, clean_background, dim=-1).mean()
    patch_target_cos = F.cosine_similarity(attacked_patch, clean_target, dim=-1).mean()
    patch_bg_cos = F.cosine_similarity(attacked_patch, clean_background, dim=-1).mean()

    object_loss = object_clean_cos - object_bg_cos
    patch_loss = -patch_target_cos + patch_bg_cos
    loss = float(object_weight) * object_loss + float(patch_weight) * patch_loss
    return loss, {
        "object_clean_cos": float(object_clean_cos.detach()),
        "object_background_cos": float(object_bg_cos.detach()),
        "patch_target_cos": float(patch_target_cos.detach()),
        "patch_background_cos": float(patch_bg_cos.detach()),
        "clean_target_mass": float(clean_target_mass.detach()),
        "clean_background_mass": float(clean_background_mass.detach()),
        "attacked_target_mass": float(attacked_target_mass.detach()),
        "attacked_patch_mass": float(attacked_patch_mass.detach()),
    }


def load_object_masks(cfg: dict[str, Any], records: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, torch.Tensor]]:
    root = Path(cfg["object_mask_root"])
    require(root.is_dir(), f"Missing object mask root: {root}")
    training_hash = sha256_file(cfg["training_bank"])
    output: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
    for task_id in range(10):
        receipt_path = root / f"task_{task_id:02d}" / "RECEIPT.json"
        require(receipt_path.is_file(), f"Missing object mask receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(receipt.get("status") == "EXACT_OBJECT_MASK_SHARD_COMPLETE",
                f"Incomplete object mask shard: task={task_id}")
        require(receipt.get("bank_sha256") == training_hash,
                f"Object mask/training bank hash mismatch: task={task_id}")
        for sample in receipt["samples"]:
            key = (task_id, int(sample["trajectory_rank"]), int(sample["decision_index"]))
            array_path = Path(sample["array_path"])
            require(array_path.is_file(), f"Missing mask array: {array_path}")
            require(sha256_file(array_path) == sample["array_sha256"],
                    f"Object mask SHA256 drift: {array_path}")
            with np.load(array_path, allow_pickle=False) as arrays:
                output[key] = {
                    role: np_mask_to_tokens(np.asarray(arrays[f"wrist_pixel_mask_{role}"]))
                    for role in MASK_ROLES
                }
            for role in MASK_ROLES:
                require(float(output[key][role].sum()) > 0, f"Empty token mask role={role} key={key}")
    missing = [record_key(record) for record in records if record_key(record) not in output]
    require(not missing, f"Object mask coverage missing for {len(missing)} records; first={missing[:3]}")
    return output


class ExploringOFTObjectDecoyAttacker(ExploringOFTOpenVLAAttacker):
    def _paste_patch_tensor_and_mask(
        self,
        image: Any,
        patch: torch.Tensor,
        geometry: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        im = torchvision.transforms.ToTensor()(self.randomPatchTransform._to_rgb_pil(image)).to(self.wrapper.device)
        channels, image_h, image_w = im.shape
        patch_to_apply = patch
        _, patch_h, patch_w = patch_to_apply.shape
        if patch_h > image_h or patch_w > image_w:
            patch_h = min(patch_h, image_h)
            patch_w = min(patch_w, image_w)
            patch_to_apply = torchvision.transforms.Resize((patch_h, patch_w))(patch_to_apply)

        canvas = torch.ones(channels, image_h, image_w, device=self.wrapper.device) * -100
        mask = torch.zeros(1, image_h, image_w, device=self.wrapper.device)
        x = random.randint(0, image_w - patch_w)
        y = random.randint(0, image_h - patch_h)
        canvas[:, y:y + patch_h, x:x + patch_w] = patch_to_apply
        mask[:, y:y + patch_h, x:x + patch_w] = 1.0

        if geometry:
            matrix = self.randomPatchTransform.combined_transform_matrix().to(self.wrapper.device)
            canvas = self.randomPatchTransform.apply_affine_transform(canvas, matrix)
            if canvas.ndim == 4:
                canvas = canvas.squeeze(0)
            affine = matrix[:2, :].unsqueeze(0)
            grid = F.affine_grid(affine, mask.unsqueeze(0).size(), align_corners=False)
            mask = F.grid_sample(mask.unsqueeze(0), grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=False).squeeze(0)

        patched = torch.where(canvas < -20, im, canvas).clamp(0, 1)
        return patched, mask.clamp(0, 1)

    def _patched_pixels_and_tokens(
        self,
        observation: dict[str, Any],
        patch: torch.Tensor,
        geometry: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor

        front = torch.as_tensor(np.asarray(observation["full_image"]), device=self.wrapper.device, dtype=torch.float32)
        front = front.permute(2, 0, 1).unsqueeze(0).div(255.0)
        wrist, mask = self._paste_patch_tensor_and_mask(observation["wrist_image"], patch, geometry)
        wrist = wrist.unsqueeze(0)
        processor = self.wrapper.processor.image_processor
        front_values = _processor_normalize_tensor(front, processor, center_crop=True)
        wrist_values = _processor_normalize_tensor(wrist, processor, center_crop=True)
        patch_tokens = image_mask_to_dino_tokens(mask.squeeze(0)).to(self.wrapper.device)
        return torch.cat((front_values, wrist_values), dim=1).to(dtype=self.wrapper.dtype), patch_tokens

    def _forward_with_patch_tokens(
        self,
        inputs: dict[str, torch.Tensor],
        proprio: np.ndarray,
        observation: dict[str, Any],
        patch: torch.Tensor,
        geometry: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from src.hooks.manager import HookSpec, TensorHooks

        model_inputs = dict(inputs)
        model_inputs["pixel_values"], patch_tokens = self._patched_pixels_and_tokens(observation, patch, geometry)
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
        dino = hooks.captured[0].float()
        require(dino.ndim == 3 and dino.shape[1] == patch_tokens.numel(),
                f"DINO/token-mask shape drift: dino={tuple(dino.shape)} mask={tuple(patch_tokens.shape)}")
        require(torch.isfinite(action).all() and torch.isfinite(dino).all(), "Invalid forward output")
        return action, dino, patch_tokens

    def train_object_decoy_patch(
        self,
        records: list[dict[str, Any]],
        cfg: dict[str, Any],
        method: str,
    ) -> None:
        require(method in METHODS, f"Unknown object-decoy method: {method}")
        output_dir = Path(cfg["output_root"]) / method
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            require(status.get("status") != "COMPLETE", "Refusing to overwrite completed patch")

        object_masks = load_object_masks(cfg, records)
        ratio = float(cfg["decoy_gradient_ratios"][method])
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
            "adaptation": "Exploring OFT action loss plus Wrist-DINO object-to-background and patch-to-object decoy loss",
            "seed": int(cfg["seed"]),
            "data_seed": int(cfg["data_seed"]),
            "maskidx": maskidx,
            "outer_iterations": int(cfg["outer_iterations"]),
            "inner_loop": int(cfg["inner_loop"]),
            "decoy_gradient_ratio": ratio,
            "object_mask_root": str(Path(cfg["object_mask_root"]).resolve()),
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
            with torch.no_grad():
                clean_action, clean_dino, _ = self._forward(inputs, proprio, observation, None, False)
            target = opposite_boundary_target(clean_action, low, high)
            masks = object_masks[record_key(record)]
            sums = {
                "action": 0.0,
                "decoy": 0.0,
                "grad": 0.0,
                "cosine": 0.0,
                "ratio": 0.0,
                "object_clean_cos": 0.0,
                "object_background_cos": 0.0,
                "patch_target_cos": 0.0,
                "patch_background_cos": 0.0,
                "patch_mass": 0.0,
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
                decoy_loss, decoy_stats = object_decoy_loss(
                    dino, clean_dino, masks, patch_tokens,
                    object_weight=float(cfg["decoy_object_weight"]),
                    patch_weight=float(cfg["decoy_patch_weight"]))
                main_grad = torch.autograd.grad(action_loss, patch, retain_graph=True)[0]
                decoy_grad = torch.autograd.grad(decoy_loss, patch)[0]
                patch.grad, auxiliary_weight, cosine, effective_ratio = combine_gradients(
                    main_grad, decoy_grad, ratio, auxiliary_weight)
                require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Invalid patch gradient")
                grad_norm = float(torch.linalg.vector_norm(patch.grad))
                step_uada_patch(patch, optimizer, self.wrapper.vla)

                scale = 1.0 / int(cfg["inner_loop"])
                sums["action"] += float(action_loss.detach()) * scale
                sums["decoy"] += float(decoy_loss.detach()) * scale
                sums["grad"] += grad_norm * scale
                sums["cosine"] += cosine * scale
                sums["ratio"] += effective_ratio * scale
                sums["object_clean_cos"] += decoy_stats["object_clean_cos"] * scale
                sums["object_background_cos"] += decoy_stats["object_background_cos"] * scale
                sums["patch_target_cos"] += decoy_stats["patch_target_cos"] * scale
                sums["patch_background_cos"] += decoy_stats["patch_background_cos"] * scale
                sums["patch_mass"] += decoy_stats["attacked_patch_mass"] * scale

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
                "action_boundary_mse": sums["action"],
                "dino_object_decoy_loss": sums["decoy"],
                "patch_grad_l2": sums["grad"],
                "auxiliary_weight": auxiliary_weight,
                "main_aux_gradient_cosine": sums["cosine"],
                "effective_aux_gradient_ratio": sums["ratio"],
                "object_clean_cos": sums["object_clean_cos"],
                "object_background_cos": sums["object_background_cos"],
                "patch_target_cos": sums["patch_target_cos"],
                "patch_background_cos": sums["patch_background_cos"],
                "patch_token_mass": sums["patch_mass"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": time.time() - started,
                **structure,
            })
            if outer % 5 == 0 or outer in checkpoints:
                print(f"EXPLORING_OFT_OBJECT_DECOY method={method} outer={outer}/{cfg['outer_iterations']} "
                      f"updates={outer * int(cfg['inner_loop'])} action={sums['action']:.5f} "
                      f"decoy={sums['decoy']:.5f} patch_target={sums['patch_target_cos']:.3f} "
                      f"obj_bg={sums['object_background_cos']:.3f} corr={structure['init_correlation']:.4f} "
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

"""Minimal OFT adapter of Exploring's UADA training loop.

The control flow intentionally follows ``UADA.py::patchattack_unconstrained``:
random RGB patch, one new training sample per outer iteration, repeated random
geometric patch transforms in an inner loop, one optimizer update per inner
iteration, clamp, and an outer-iteration cosine schedule.  OFT requires only the
model/input adapter and a continuous opposite-boundary loss.  The optional DINO
term is additive and never replaces the UADA action objective.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from white_patch.UADA import OpenVLAAttacker
from white_patch.UADA_OFT_SOURCE_COPY import (
    build_uada_adamw,
    init_uada_patch,
    step_uada_patch,
)


METHODS = (
    "exploring_oft_uada",
    "exploring_oft_dino_cf",
    "exploring_oft_dino_cf_worst_chunk",
    "exploring_oft_full_worst_chunk",
    "exploring_oft_dino_restore",
    "exploring_oft_siglip_cf",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def atomic_torch(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)


def atomic_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)


def patch_metrics(patch: torch.Tensor, initial: torch.Tensor) -> dict[str, float]:
    value = patch.detach().float()
    reference = initial.detach().float()
    tv = (value[:, :, 1:] - value[:, :, :-1]).abs().mean()
    tv = tv + (value[:, 1:, :] - value[:, :-1, :]).abs().mean()
    initial_tv = (reference[:, :, 1:] - reference[:, :, :-1]).abs().mean()
    initial_tv = initial_tv + (reference[:, 1:, :] - reference[:, :-1, :]).abs().mean()
    x = value.reshape(-1) - value.mean()
    y = reference.reshape(-1) - reference.mean()
    correlation = torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)).clamp_min(1e-12)
    return {
        "tv": float(tv),
        "initial_tv": float(initial_tv),
        "tv_ratio": float(tv / initial_tv.clamp_min(1e-12)),
        "init_correlation": float(correlation),
        "init_rms_delta": float(torch.mean((value - reference) ** 2).sqrt()),
        "saturation_fraction": float(((value <= 0.01) | (value >= 0.99)).float().mean()),
    }


def save_patch(path: Path, patch: torch.Tensor, payload: dict[str, Any]) -> None:
    atomic_torch({"patch": patch.detach().float().cpu(), **payload}, path)
    rgb = patch.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    Image.fromarray(rgb).save(path.with_suffix(".png"))


class ExploringOFTOpenVLAAttacker(OpenVLAAttacker):
    """Exploring UADA control flow with the smallest OFT-specific surface."""

    def __init__(self, wrapper: Any, save_dir: str | Path, resize_patch: bool = False):
        # Do not call the OpenVLA-token parent initializer: OFT has a continuous
        # action head and no ActionTokenizer.  Reuse the original transform and
        # keep the same public training role.
        self.wrapper = wrapper
        self.vla = wrapper.vla.eval()
        self.save_dir = Path(save_dir)
        self.optimizer = "adamW"
        from white_patch.appply_random_transform import RandomPatchTransform
        self.randomPatchTransform = RandomPatchTransform(wrapper.device, resize_patch)

    def _normalized_proprio(self, observation: dict[str, Any]) -> np.ndarray:
        stats = self.wrapper.vla.norm_stats[self.wrapper.gen_cfg.unnorm_key]["proprio"]
        return self.wrapper._official_openvla_utils.normalize_proprio(np.asarray(observation["state"]), stats)

    def _base_inputs(self, record: dict[str, Any]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        inputs = dict(self.wrapper.prepare_hf_inputs(record["observation"], record["task_name"]))
        proprio = self._normalized_proprio(record["observation"])
        return inputs, proprio

    def _patched_pixels(self, observation: dict[str, Any], patch: torch.Tensor | None, geometry: bool) -> torch.Tensor:
        from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor

        front = torch.as_tensor(np.asarray(observation["full_image"]), device=self.wrapper.device, dtype=torch.float32)
        front = front.permute(2, 0, 1).unsqueeze(0).div(255.0)
        if patch is None:
            wrist = torch.as_tensor(np.asarray(observation["wrist_image"]), device=self.wrapper.device,
                                    dtype=torch.float32).permute(2, 0, 1).div(255.0)
        else:
            wrist = self.randomPatchTransform._paste_patch_tensor(observation["wrist_image"], patch, geometry)
        wrist = wrist.unsqueeze(0)
        processor = self.wrapper.processor.image_processor
        front_values = _processor_normalize_tensor(front, processor, center_crop=True)
        wrist_values = _processor_normalize_tensor(wrist, processor, center_crop=True)
        return torch.cat((front_values, wrist_values), dim=1).to(dtype=self.wrapper.dtype)

    def _forward(self, inputs: dict[str, torch.Tensor], proprio: np.ndarray,
                 observation: dict[str, Any], patch: torch.Tensor | None, geometry: bool, *,
                 clean_siglip: torch.Tensor | None = None,
                 clean_dino: torch.Tensor | None = None,
                 capture_siglip: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from src.hooks.manager import HookSpec, TensorHooks

        model_inputs = dict(inputs)
        model_inputs["pixel_values"] = self._patched_pixels(observation, patch, geometry)
        # Checkpoint timm_model_ids order is validated before training.  The
        # generic OFT source comments suggest the reverse order, but this
        # particular checkpoint has DINO first and SigLIP second.
        dino_spec = HookSpec(path="vla.vision_backbone.featurizer", hook="output", call_index=1)
        siglip_spec = HookSpec(path="vla.vision_backbone.fused_featurizer", hook="output", call_index=1)
        roots = {"vla": self.wrapper.vla, "action_head": self.wrapper.action_head}
        with ExitStack() as stack:
            if clean_dino is not None:
                stack.enter_context(TensorHooks(
                    roots, [dino_spec], mode="replace",
                    values=[clean_dino.to(self.wrapper.device, dtype=self.wrapper.dtype)]))
            if clean_siglip is not None:
                stack.enter_context(TensorHooks(
                    roots, [siglip_spec], mode="replace",
                    values=[clean_siglip.to(self.wrapper.device, dtype=self.wrapper.dtype)]))
            capture_specs = []
            if clean_dino is None:
                capture_specs.append(dino_spec)
            capture_siglip_internal = clean_siglip is None and (capture_siglip or clean_dino is not None)
            if capture_siglip_internal:
                capture_specs.append(siglip_spec)
            hooks = stack.enter_context(TensorHooks(roots, capture_specs, mode="capture", retain_grad=True))
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
        captured_index = 0
        if clean_dino is None:
            dino = hooks.captured[captured_index]
            captured_index += 1
        else:
            dino = clean_dino
        captured_siglip = hooks.captured[captured_index] if capture_siglip_internal else None
        siglip = captured_siglip if capture_siglip else None
        require(dino is not None and torch.isfinite(dino).all(), "Invalid Wrist-DINO activation")
        require(torch.isfinite(action).all(), "Invalid OFT action")
        if siglip is not None:
            require(torch.isfinite(siglip).all(), "Invalid Wrist-SigLIP activation")
        return action, dino.float(), siglip

    @staticmethod
    def _training_index(outer: int, size: int, seed: int) -> int:
        epoch, offset = divmod(outer - 1, size)
        order = list(range(size))
        random.Random(seed + epoch).shuffle(order)
        return order[offset]

    def _mechanism_probe(self, records: list[dict[str, Any]], cfg: dict[str, Any],
                         method: str, patch: torch.Tensor, outer: int) -> pd.DataFrame:
        """Identical held-out-style route measurements for every training arm.

        These are descriptive training-bank probes, not held-out CASR or proof
        that a branch is necessary.  The same first record per task and EOT
        draw are used for UADA, DINO-CF, and all other arms.
        """
        first_by_task: dict[int, int] = {}
        for index, record in enumerate(records):
            first_by_task.setdefault(int(record["task_id"]), index)
        require(len(first_by_task) == 10, "Mechanism probe requires all ten Object tasks")
        maskidx = list(map(int, cfg["maskidx"]))
        low = torch.as_tensor(cfg["action_head_q01"], device=self.wrapper.device).reshape(1, 1, 7)
        high = torch.as_tensor(cfg["action_head_q99"], device=self.wrapper.device).reshape(1, 1, 7)
        rows = []
        with torch.no_grad():
            for task_id, index in sorted(first_by_task.items()):
                record = records[index]
                inputs, proprio = self._base_inputs(record)
                observation = record["observation"]
                clean, clean_dino, clean_siglip = self._forward(
                    inputs, proprio, observation, None, False, capture_siglip=True)
                require(clean_siglip is not None, "Probe clean SigLIP anchor missing")
                target = torch.where(clean >= (low + high) / 2.0, low, high)
                seed = int(cfg["seed"]) * 10_000_000 + 9_000_000 + task_id
                def route(*, dino: torch.Tensor | None = None,
                          siglip: torch.Tensor | None = None) -> torch.Tensor:
                    random.seed(seed)
                    np.random.seed(seed % (2**32 - 1))
                    torch.manual_seed(seed)
                    action, _, _ = self._forward(
                        inputs, proprio, observation, patch, bool(cfg["geometry"]),
                        clean_dino=dino, clean_siglip=siglip)
                    return action[..., maskidx]
                full = route()
                dino_only = route(siglip=clean_siglip)
                siglip_only = route(dino=clean_dino)
                clean_action = clean[..., maskidx]
                target_action = target[..., maskidx]
                rows.append({
                    "method": method, "outer_iteration": outer,
                    "task_id": task_id, "record_index": index,
                    "full_boundary_mse": float(F.mse_loss(full, target_action)),
                    "dino_only_boundary_mse": float(F.mse_loss(dino_only, target_action)),
                    "siglip_only_boundary_mse": float(F.mse_loss(siglip_only, target_action)),
                    "full_clean_shift_mse": float(F.mse_loss(full, clean_action)),
                    "dino_only_clean_shift_mse": float(F.mse_loss(dino_only, clean_action)),
                    "siglip_only_clean_shift_mse": float(F.mse_loss(siglip_only, clean_action)),
                })
        return pd.DataFrame(rows)

    def patchattack_unconstrained_oft(self, records: list[dict[str, Any]], cfg: dict[str, Any], method: str) -> None:
        require(method in METHODS, f"Unknown method: {method}")
        route_objective = {
            "exploring_oft_uada": "none",
            "exploring_oft_dino_cf": "attacked_dino_clean_siglip_mean_boundary",
            "exploring_oft_dino_cf_worst_chunk": "attacked_dino_clean_siglip_worst_chunk_boundary",
            "exploring_oft_full_worst_chunk": "attacked_full_worst_chunk_boundary_control",
            "exploring_oft_dino_restore": "clean_dino_attacked_siglip_restore_to_clean_action",
            "exploring_oft_siglip_cf": "clean_dino_attacked_siglip_mean_boundary_control",
        }[method]
        output_dir = self.save_dir / method
        output_dir.mkdir(parents=True, exist_ok=True)
        final_status = output_dir / "status.json"
        if final_status.is_file():
            status = json.loads(final_status.read_text(encoding="utf-8"))
            if status.get("status") == "COMPLETE":
                print(f"ALREADY_COMPLETE method={method}")
                return

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
        lambda_dino_cf: float | None = None
        if resume_path.is_file():
            resume = torch.load(resume_path, map_location=self.wrapper.device, weights_only=False)
            patch.data.copy_(resume["patch"])
            initial = resume["initial"].to(self.wrapper.device)
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            rows = resume["rows"]
            lambda_dino_cf = resume["lambda_dino_cf"]
            start_outer = int(resume["outer"]) + 1
            print(f"RESUME method={method} outer={start_outer}", flush=True)

        metadata = {
            "experiment_id": cfg["experiment_id"],
            "method": method,
            "source_trainer": "VLAAttacker/white_patch/UADA.py::patchattack_unconstrained",
            "source_uada_sha256": sha256_file(Path(__file__).with_name("UADA.py")),
            "adaptation": "OFT continuous action head; Front+Wrist input; Wrist patch only",
            "route_objective": route_objective,
            "seed": int(cfg["seed"]),
            "data_seed": int(cfg["data_seed"]),
            "maskidx": maskidx,
            "outer_iterations": int(cfg["outer_iterations"]),
            "inner_loop": int(cfg["inner_loop"]),
            "batch_size": int(cfg["batch_size"]),
            "training_bank_sha256": sha256_file(cfg["training_bank"]),
        }
        if start_outer == 1:
            save_patch(output_dir / "outer_0000.pt", patch, {**metadata, **patch_metrics(patch, initial)})

        started = time.time()
        final_outer = int(cfg["outer_iterations"])
        try:
            for outer in range(start_outer, final_outer + 1):
                record_index = self._training_index(outer, len(records), int(cfg["data_seed"]))
                record = records[record_index]
                base_inputs, proprio = self._base_inputs(record)
                # The legacy bank supplies only clean source observations and task
                # text.  Recompute both anchors with the current checkpoint so no
                # cached output from the superseded optimizer lineage enters loss.
                with torch.no_grad():
                    clean, clean_dino, clean_siglip = self._forward(
                        base_inputs, proprio, record["observation"], None, False, capture_siglip=True)
                require(clean_siglip is not None, "Clean Wrist-SigLIP anchor missing")
                midpoint = (low + high) / 2.0
                target = torch.where(clean >= midpoint, low, high).detach()
                outer_action_loss = 0.0
                outer_dino_norm = 0.0
                outer_dino_cf_loss = 0.0
                outer_grad_norm = 0.0
                outer_route_grad_cosine = 0.0
                outer_route_scaled_norm_ratio = 0.0
                for inner in range(int(cfg["inner_loop"])):
                    # Paired randomization: both methods receive the exact same
                    # patch placement and affine draw at each outer/inner index.
                    transform_seed = int(cfg["seed"]) * 10_000_000 + outer * 10_000 + inner
                    random.seed(transform_seed)
                    np.random.seed(transform_seed % (2**32 - 1))
                    torch.manual_seed(transform_seed)
                    optimizer.zero_grad(set_to_none=True)
                    self.wrapper.vla.zero_grad(set_to_none=True)
                    action, dino, _siglip = self._forward(
                        base_inputs, proprio, record["observation"], patch, bool(cfg["geometry"]))
                    action_loss = F.mse_loss(action[..., maskidx], target[..., maskidx])
                    dino_norm = torch.linalg.vector_norm((dino - clean_dino).reshape(-1))
                    if method == "exploring_oft_uada":
                        action_loss.backward()
                        dino_cf_loss = action_loss.detach()
                    else:
                        # Full-model UADA is always the primary objective.  A
                        # method-specific second counterfactual forward tests
                        # DINO sufficiency, DINO necessity, temporal worst-chunk
                        # routing, full-route temporal control, or SigLIP control.
                        # Reset RNG so both forwards see the same patch transform.
                        g_action = torch.autograd.grad(action_loss, patch)[0]
                        random.seed(transform_seed)
                        np.random.seed(transform_seed % (2**32 - 1))
                        torch.manual_seed(transform_seed)
                        if method in ("exploring_oft_dino_cf", "exploring_oft_dino_cf_worst_chunk"):
                            route_action, _route_dino, _ = self._forward(
                                base_inputs, proprio, record["observation"], patch,
                                bool(cfg["geometry"]), clean_siglip=clean_siglip)
                            route_error = F.mse_loss(
                                route_action[..., maskidx], target[..., maskidx], reduction="none")
                            if method == "exploring_oft_dino_cf_worst_chunk":
                                dino_cf_loss = route_error.reshape(route_error.shape[0], route_error.shape[1], -1).mean(-1).max()
                            else:
                                dino_cf_loss = route_error.mean()
                        elif method == "exploring_oft_full_worst_chunk":
                            route_action, _route_dino, _ = self._forward(
                                base_inputs, proprio, record["observation"], patch,
                                bool(cfg["geometry"]))
                            route_error = F.mse_loss(
                                route_action[..., maskidx], target[..., maskidx], reduction="none")
                            dino_cf_loss = route_error.reshape(route_error.shape[0], route_error.shape[1], -1).mean(-1).max()
                        elif method == "exploring_oft_dino_restore":
                            route_action, _route_dino, _ = self._forward(
                                base_inputs, proprio, record["observation"], patch,
                                bool(cfg["geometry"]), clean_dino=clean_dino)
                            dino_cf_loss = F.mse_loss(
                                route_action[..., maskidx], clean[..., maskidx])
                        elif method == "exploring_oft_siglip_cf":
                            route_action, _route_dino, _ = self._forward(
                                base_inputs, proprio, record["observation"], patch,
                                bool(cfg["geometry"]), clean_dino=clean_dino)
                            dino_cf_loss = F.mse_loss(
                                route_action[..., maskidx], target[..., maskidx])
                        else:
                            raise AssertionError(method)
                        g_dino_cf = torch.autograd.grad(dino_cf_loss, patch)[0]
                        action_grad_norm = torch.linalg.vector_norm(g_action).clamp_min(1e-12)
                        dino_cf_grad_norm = torch.linalg.vector_norm(g_dino_cf).clamp_min(1e-12)
                        if lambda_dino_cf is None:
                            lambda_dino_cf = float(cfg["dino_cf_gradient_ratio"]) * float(
                                action_grad_norm / dino_cf_grad_norm)
                        route_cosine = F.cosine_similarity(
                            g_action.reshape(-1), g_dino_cf.reshape(-1), dim=0)
                        route_ratio = float(lambda_dino_cf) * dino_cf_grad_norm / action_grad_norm
                        patch.grad = g_action + float(lambda_dino_cf) * g_dino_cf
                    require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Invalid patch gradient")
                    grad_norm = float(torch.linalg.vector_norm(patch.grad))
                    step_uada_patch(patch, optimizer, self.wrapper.vla)
                    outer_action_loss += float(action_loss.detach()) / int(cfg["inner_loop"])
                    outer_dino_norm += float(dino_norm.detach()) / int(cfg["inner_loop"])
                    outer_dino_cf_loss += float(dino_cf_loss.detach()) / int(cfg["inner_loop"])
                    outer_grad_norm += grad_norm / int(cfg["inner_loop"])
                    if method != "exploring_oft_uada":
                        outer_route_grad_cosine += float(route_cosine.detach()) / int(cfg["inner_loop"])
                        outer_route_scaled_norm_ratio += float(route_ratio.detach()) / int(cfg["inner_loop"])
                scheduler.step()
                structure = patch_metrics(patch, initial)
                rows.append({
                    "method": method,
                    "route_objective": route_objective,
                    "outer_iteration": outer,
                    "optimizer_updates": outer * int(cfg["inner_loop"]),
                    "record_index": record_index,
                    "task_id": int(record["task_id"]),
                    "state_index": int(record["state_index"]),
                    "action_boundary_mse": outer_action_loss,
                    "training_route_objective_loss": (
                        outer_dino_cf_loss if method != "exploring_oft_uada" else None),
                    "wrist_dino_delta_l2": outer_dino_norm,
                    "patch_grad_l2": outer_grad_norm,
                    "lambda_dino_cf": float(lambda_dino_cf or 0.0),
                    "route_grad_cosine": outer_route_grad_cosine if method != "exploring_oft_uada" else None,
                    "route_scaled_norm_ratio": outer_route_scaled_norm_ratio if method != "exploring_oft_uada" else None,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_s": time.time() - started,
                    **structure,
                })
                if outer % 5 == 0 or outer in checkpoints:
                    print(
                        f"EXPLORING_OFT method={method} outer={outer}/{final_outer} "
                        f"updates={outer * int(cfg['inner_loop'])} action_mse={outer_action_loss:.5f} "
                        f"dino={outer_dino_norm:.2f} corr={structure['init_correlation']:.4f} "
                        f"tv_ratio={structure['tv_ratio']:.4f}", flush=True)
                if outer in checkpoints:
                    save_patch(output_dir / f"outer_{outer:04d}.pt", patch, {
                        **metadata, **structure, "lambda_dino_cf": lambda_dino_cf})
                    if outer in set(map(int, cfg.get("mechanism_probe_outer_iterations", []))):
                        probe = self._mechanism_probe(records, cfg, method, patch, outer)
                        atomic_parquet(probe, output_dir / f"mechanism_probe_{outer:04d}.parquet")
                if str(outer) in cfg["structure_gates"]:
                    gate = cfg["structure_gates"][str(outer)]
                    passed = structure["init_correlation"] < float(gate["max_init_correlation"])
                    passed = passed and structure["tv_ratio"] < float(gate["max_tv_ratio"])
                    atomic_json({"outer_iteration": outer, "passed": passed, **gate, **structure},
                                output_dir / f"structure_gate_{outer:04d}.json")
                    if not passed:
                        atomic_json({"status": "STRUCTURE_GATE_FAIL", "outer_iteration": outer, **metadata, **structure}, final_status)
                        atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
                        print(f"STRUCTURE_GATE_FAIL method={method} outer={outer}", flush=True)
                        return
                if outer % int(cfg["resume_every_outer_iterations"]) == 0:
                    atomic_torch({
                        "outer": outer,
                        "patch": patch.detach(),
                        "initial": initial.detach(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rows": rows,
                        "lambda_dino_cf": lambda_dino_cf,
                    }, resume_path)
            atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
            atomic_json({"status": "COMPLETE", **metadata, **patch_metrics(patch, initial),
                         "lambda_dino_cf": lambda_dino_cf, "elapsed_s": time.time() - started}, final_status)
            resume_path.unlink(missing_ok=True)
        except BaseException:
            if rows:
                atomic_parquet(pd.DataFrame(rows), output_dir / "training.partial.parquet")
            raise

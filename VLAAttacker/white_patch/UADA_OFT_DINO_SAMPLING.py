"""Exploring UADA copy with a position-matched DINO sampling comparison.

The action target, optimizer, inner-loop updates, and clamp come from the
read-only Exploring UADA port. DINO changes only which Object-bank state is
drawn; it is not an extra patch loss or a runtime oracle.
"""

from __future__ import annotations

import hashlib
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
    ExploringOFTOpenVLAAttacker, atomic_json, atomic_parquet, atomic_torch,
    patch_metrics, require, save_patch, sha256_file,
)
from white_patch.UADA_OFT_JOINT500 import opposite_boundary_target
from white_patch.UADA_OFT_SOURCE_COPY import (
    build_uada_adamw, init_uada_patch, step_uada_patch,
)


METHODS = ("exploring_oft_site_uada", "exploring_oft_site_dino_sample")


def key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (int(record["task_id"]), int(record["trajectory_rank"]), int(record["decision_index"]))


def dino_sampling_scores(records: list[dict[str, Any]], site: tuple[int, int]) -> dict[tuple[int, int, int], float]:
    # This imports the audited mask/feature alignment and fits prototypes only
    # on the previously frozen four discovery trajectories per Object task.
    from plan_dino_guided_placement import (
        PATCH_SIZE, candidate_masks, fit_prototypes, prepare,
    )
    from readout_object_bank_spatial_cpu import load_manifest

    require(len(records) == 300, "Expected 300 Object-bank states")
    manifest = load_manifest({"records": records, "config_sha256": _bank_config_sha()}, verify_arrays=True)
    prepared = prepare(manifest)
    prototypes = fit_prototypes(prepared)
    sites, footprints = candidate_masks()
    require(site in sites, "Selected patch site is outside predeclared candidate grid")
    footprint = footprints[sites.index(site)]
    require(PATCH_SIZE == 50, "Patch size drift")
    result = {}
    for item, prepared_item in zip(manifest, prepared, strict=True):
        record = item["row"]
        task_proto = prototypes[item["task_id"]]
        features = F.normalize(prepared_item["features"], dim=-1)
        logits = features @ (task_proto["target_object"] - task_proto["matched_background"])
        probability = torch.softmax(logits / 0.05, dim=0)
        result[key(record)] = max(float(footprint @ probability), 1e-6)
    require(len(result) == len(records), "DINO sampling score coverage drift")
    return result


def _bank_config_sha() -> str:
    from collect_object_bank_spatial_masks import CONFIG_PATH
    return sha256_file(CONFIG_PATH)


def draw_record_index(outer: int, records: list[dict[str, Any]], scores: dict,
                      data_seed: int, method: str) -> int:
    require(method in METHODS, "Unknown DINO-sampling method")
    # The frozen UADA permutation fixes the task at each outer step. Both arms
    # therefore receive identical task frequencies and optimizer/EOT seeds.
    base = ExploringOFTOpenVLAAttacker._training_index(outer, len(records), data_seed)
    task_id = int(records[base]["task_id"])
    candidates = [i for i, record in enumerate(records) if int(record["task_id"]) == task_id]
    require(len(candidates) == 30, "Unexpected per-task bank size")
    if method == METHODS[0]:
        return base
    weights = [scores[key(records[i])] for i in candidates]
    return random.Random(data_seed * 1_000_000 + outer).choices(candidates, weights=weights, k=1)[0]


class ExploringOFTDinoSamplingAttacker(ExploringOFTOpenVLAAttacker):
    def __init__(self, wrapper: Any, save_dir: str | Path, site: tuple[int, int]):
        super().__init__(wrapper, save_dir, resize_patch=False)
        self.site = site

    def _patched_pixels(self, observation: dict[str, Any], patch: torch.Tensor | None,
                        geometry: bool) -> torch.Tensor:
        from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor

        front = torch.as_tensor(np.asarray(observation["full_image"]), device=self.wrapper.device,
                                dtype=torch.float32).permute(2, 0, 1)[None].div(255)
        wrist = torchvision.transforms.ToTensor()(self.randomPatchTransform._to_rgb_pil(
            observation["wrist_image"])).to(self.wrapper.device)
        if patch is not None:
            _, image_h, image_w = wrist.shape
            _, patch_h, patch_w = patch.shape
            x, y = self.site
            require(0 <= x <= image_w - patch_w and 0 <= y <= image_h - patch_h,
                    "Anchored training patch outside Wrist image")
            canvas = torch.ones_like(wrist) * -100
            canvas[:, y:y + patch_h, x:x + patch_w] = patch
            if geometry:
                canvas = self.randomPatchTransform.apply_affine_transform(
                    canvas, self.randomPatchTransform.combined_transform_matrix().to(self.wrapper.device))
                if canvas.ndim == 4:
                    canvas = canvas.squeeze(0)
            wrist = torch.where(canvas < -20, wrist, canvas).clamp(0, 1)
        processor = self.wrapper.processor.image_processor
        front_values = _processor_normalize_tensor(front, processor, center_crop=True)
        wrist_values = _processor_normalize_tensor(wrist[None], processor, center_crop=True)
        return torch.cat((front_values, wrist_values), dim=1).to(dtype=self.wrapper.dtype)

    def train(self, records: list[dict[str, Any]], cfg: dict[str, Any], method: str,
              scores: dict[tuple[int, int, int], float]) -> None:
        require(method in METHODS, "Unknown method")
        output_dir = Path(cfg["output_root"]) / method
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "status.json"
        require(not status_path.exists(), f"Refusing to overwrite training status: {status_path}")

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
        score_bytes = json.dumps({str(k): scores[k] for k in sorted(scores)}, sort_keys=True).encode()
        score_sha = hashlib.sha256(score_bytes).hexdigest()
        metadata = {"experiment_id": cfg["experiment_id"], "method": method,
                    "source_trainer": "read-only UADA.py::patchattack_unconstrained",
                    "source_uada_sha256": sha256_file(Path(__file__).with_name("UADA.py")),
                    "source_copy_sha256": sha256_file(Path(__file__).with_name("UADA_OFT_SOURCE_COPY.py")),
                    "adaptation": "Wrist fixed DINO-selected site; original action MSE; DINO only reweights training states",
                    "training_bank_sha256": sha256_file(cfg["training_bank"]),
                    "dino_score_sha256": score_sha, "site_xy": list(self.site),
                    "seed": int(cfg["seed"]), "data_seed": int(cfg["data_seed"]),
                    "maskidx": maskidx, "outer_iterations": int(cfg["outer_iterations"]),
                    "inner_loop": int(cfg["inner_loop"]), "sampling": "uniform" if method == METHODS[0] else "dino_probability_mass"}
        resume_path = output_dir / "resume.pt"
        rows: list[dict[str, Any]] = []
        start_outer = 1
        if resume_path.is_file():
            resume = torch.load(resume_path, map_location=self.wrapper.device, weights_only=False)
            require(all(resume.get(name) == metadata[name] for name in (
                "experiment_id", "method", "training_bank_sha256", "dino_score_sha256", "site_xy")),
                "Resume lineage mismatch")
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
            record_index = draw_record_index(outer, records, scores, int(cfg["data_seed"]), method)
            record = records[record_index]
            inputs, proprio = self._base_inputs(record)
            observation = record["observation"]
            with torch.no_grad():
                clean_action, _clean_dino, _ = self._forward(inputs, proprio, observation, None, False)
            target = opposite_boundary_target(clean_action, low, high)
            action_total = grad_total = 0.0
            for inner in range(int(cfg["inner_loop"])):
                transform_seed = int(cfg["seed"]) * 10_000_000 + outer * 10_000 + inner
                random.seed(transform_seed)
                np.random.seed(transform_seed % (2**32 - 1))
                torch.manual_seed(transform_seed)
                optimizer.zero_grad(set_to_none=True)
                self.wrapper.vla.zero_grad(set_to_none=True)
                action, _dino, _ = self._forward(inputs, proprio, observation, patch, bool(cfg["geometry"]))
                action_loss = F.mse_loss(action[..., maskidx], target[..., maskidx])
                action_loss.backward()
                require(patch.grad is not None and torch.isfinite(patch.grad).all(), "Invalid patch gradient")
                grad_total += float(torch.linalg.vector_norm(patch.grad)) / int(cfg["inner_loop"])
                action_total += float(action_loss.detach()) / int(cfg["inner_loop"])
                step_uada_patch(patch, optimizer, self.wrapper.vla)
            scheduler.step()
            structure = patch_metrics(patch, initial)
            rows.append({"method": method, "outer_iteration": outer,
                         "optimizer_updates": outer * int(cfg["inner_loop"]),
                         "record_index": record_index, "task_id": int(record["task_id"]),
                         "trajectory_rank": int(record["trajectory_rank"]),
                         "decision_index": int(record["decision_index"]),
                         "dino_sample_score": scores[key(record)],
                         "action_boundary_mse": action_total, "patch_grad_l2": grad_total,
                         "learning_rate": float(optimizer.param_groups[0]["lr"]),
                         "elapsed_s": time.time() - started, **structure})
            if outer % 5 == 0 or outer in checkpoints:
                print(f"EXPLORING_OFT_DINO_SAMPLING method={method} outer={outer}/{cfg['outer_iterations']} "
                      f"action={action_total:.5f} score={scores[key(record)]:.3f} "
                      f"corr={structure['init_correlation']:.4f} tv_ratio={structure['tv_ratio']:.4f}", flush=True)
            if outer in checkpoints:
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
                atomic_torch({"outer": outer, **metadata, "patch": patch.detach(),
                              "initial": initial.detach(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "rows": rows}, resume_path)
        atomic_parquet(pd.DataFrame(rows), output_dir / "training.parquet")
        atomic_json({"status": "COMPLETE", **metadata, **patch_metrics(patch, initial),
                     "elapsed_s": time.time() - started}, status_path)
        resume_path.unlink(missing_ok=True)

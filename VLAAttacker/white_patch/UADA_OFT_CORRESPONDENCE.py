"""Independent Exploring-derived OFT trainer: full-RGB correspondence/TV arms.

No upstream edits. The RGB EOT operation is copied from the existing adapter;
the new footprint follows its actual threshold branch, not a separate affine.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

from white_patch.UADA_OFT_DINO import (
    ExploringOFTOpenVLAAttacker, atomic_json, atomic_parquet, atomic_torch,
    patch_metrics, require, save_patch, sha256_file,
)
from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, init_uada_patch, step_uada_patch
from white_patch.oft_correspondence_objectives import (
    appearance_loss, calibrated_gradient, correspondence_loss, feature_loss,
    make_teacher, token_coverage,
)

METHODS = ("uada_tv", "correspondence", "correspondence_tv", "feature_tv")
TV_METHODS = {"uada_tv", "correspondence_tv", "feature_tv"}


def record_key(record):
    return (int(record["task_id"]), int(record["trajectory_rank"]), int(record["decision_index"]))


def load_masks(cfg, records=None):
    masks, hashes = {}, {}
    root = Path(cfg["object_mask_root"])
    for task_id in range(10):
        receipt_path = root / f"task_{task_id:02d}" / "RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        require(receipt["status"] == "EXACT_OBJECT_MASK_SHARD_COMPLETE", "Incomplete mask receipt")
        require(receipt["bank_sha256"] == cfg["training_bank_sha256"], "Mask/bank identity drift")
        hashes[str(receipt_path)] = sha256_file(receipt_path)
        for sample in receipt["samples"]:
            key = (task_id, int(sample["trajectory_rank"]), int(sample["decision_index"]))
            require(key not in masks, f"Duplicate mask: {key}")
            path = Path(sample["array_path"])
            digest = sha256_file(path)
            require(digest == sample["array_sha256"], f"Mask changed: {path}")
            hashes[str(path)] = digest
            with np.load(path, allow_pickle=False) as arrays:
                mask = np.asarray(arrays["wrist_pixel_mask_target_object"])
                require(mask.shape == (256, 256), "Oracle mask shape drift")
                masks[key] = token_coverage(torch.as_tensor(mask.astype(np.float32)))
    require(len(masks) == 300, f"Expected 300 masks; got {len(masks)}")
    if records is not None:
        require(len(records) == 300 and len({record_key(r) for r in records}) == 300,
                "Duplicate/missing training records")
        require(set(masks) == {record_key(r) for r in records}, "Training/mask key mismatch")
    for key, mask in masks.items():
        # No model required to verify geometric query support on all 300 records.
        eligible = (mask >= cfg["min_object_coverage"]) & (torch.arange(256) % 16 < 15)
        require(bool(eligible.any()), f"No usable spatial query: {key}")
    return masks, hashes


def run_signature(cfg, method, data_hashes, source_hashes):
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "transformers", "timm", "numpy", "pandas")}
    payload = {"config": cfg, "method": method, "data": data_hashes, "source": source_hashes,
               "packages": packages, "cuda_version": torch.version.cuda}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_manifest(cfg, wrapper=None):
    """Hash actual local dependencies, including OFT model/processor/hook trees."""
    project = Path(__file__).resolve().parents[2]
    roots = [project / "VLAAttacker", project / "prismatic",
             Path(cfg["openvla_oft_root"]) / "prismatic",
             Path(cfg["openvla_oft_root"]) / "experiments",
             Path(cfg["attribution_root"]) / "src"]
    paths = set()
    for root in roots:
        require(root.is_dir(), f"Missing source dependency: {root}")
        paths.update(p.resolve() for p in root.rglob("*.py"))
    for relative in ("submit_exploring_oft_correspondence500.sh",
                     "slurm/exploring_oft_dino/50_train_correspondence500_v1_3090.sbatch"):
        paths.add((project / relative).resolve())
    checkpoint = Path(cfg["checkpoint"])
    paths.update(p.resolve() for p in checkpoint.glob("*.json"))
    paths.update(p.resolve() for p in checkpoint.glob("*.py"))
    if wrapper is not None:
        for obj in (wrapper, wrapper.vla, wrapper.processor, wrapper.processor.image_processor,
                    wrapper.action_head, wrapper.vla.vision_backbone):
            module_file = Path(inspect.getfile(type(obj))).resolve()
            paths.add(module_file)
            # Include siblings of dynamically loaded checkpoint model/processor classes.
            paths.update(module_file.parent.glob("*.py"))
    return {str(path): sha256_file(path) for path in sorted(paths)}


def validate_resume(resume, signature, method, inner_loop):
    require(resume["signature"] == signature and resume["method"] == method,
            "Resume rejected: config/source/data signature changed")
    require(len(resume["rows"]) == resume["outer"], "Resume row count mismatch")
    if resume["rows"]:
        require(resume["rows"][-1]["optimizer_updates"] == resume["outer"] * inner_loop,
                "Resume update count mismatch")


class ExploringOFTCorrespondenceAttacker(ExploringOFTOpenVLAAttacker):
    def _paste_with_support(self, image, patch, geometry):
        require(not self.randomPatchTransform.resize_patch, "This experiment forbids patch resizing")
        im = torchvision.transforms.ToTensor()(self.randomPatchTransform._to_rgb_pil(image)).to(self.wrapper.device)
        channels, h, w = im.shape
        require(tuple(patch.shape) == (3, 50, 50) and h >= 50 and w >= 50, "Patch/image shape drift")
        canvas = torch.ones(channels, h, w, device=self.wrapper.device) * -100
        x, y = random.randint(0, w - 50), random.randint(0, h - 50)
        canvas[:, y:y + 50, x:x + 50] = patch
        if geometry:
            matrix = self.randomPatchTransform.combined_transform_matrix().to(self.wrapper.device)
            canvas = self.randomPatchTransform.apply_affine_transform(canvas, matrix).squeeze(0)
        # Includes clipped/artificial boundary pixels, not just differentiable ones.
        support = (canvas.detach() >= -20).any(dim=0).float()
        return torch.where(canvas < -20, im, canvas).clamp(0, 1), support

    def _patched_pixels(self, observation, patch, geometry):
        if patch is None:
            return super()._patched_pixels(observation, None, False)
        from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor
        front = torch.as_tensor(np.asarray(observation["full_image"]), device=self.wrapper.device,
                                dtype=torch.float32).permute(2, 0, 1)[None] / 255
        wrist, footprint = self._paste_with_support(observation["wrist_image"], patch, geometry)
        processor = self.wrapper.processor.image_processor
        self.patch_tokens = token_coverage(footprint)
        return torch.cat((_processor_normalize_tensor(front, processor, center_crop=True),
                          _processor_normalize_tensor(wrist[None], processor, center_crop=True)), dim=1).to(self.wrapper.dtype)

    def train(self, records, masks, cfg, method, source_hashes, data_hashes):
        require(method in METHODS, "Unknown correspondence arm")
        out = Path(cfg["output_root"]) / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        manifest_path = out / "RUN_MANIFEST.json"
        metadata = {"experiment_id": cfg["experiment_id"], "method": method, "signature": signature,
                    "config": cfg, "source": source_hashes, "data": data_hashes,
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "adaptation": "Exploring-derived OFT; full RGB, correspondence and/or threshold TV",
                    "parameter_count": 7500, "visual_review": "REQUIRED"}
        metadata["packages"] = {name: importlib.metadata.version(name)
                                for name in ("torch", "torchvision", "transformers", "timm", "numpy", "pandas")}
        if manifest_path.exists():
            require(json.loads(manifest_path.read_text())["signature"] == signature,
                    "Existing run signature differs; use a new experiment directory")
        else:
            atomic_json(metadata, manifest_path)
        if status_path.exists():
            require(json.loads(status_path.read_text()).get("status") != "COMPLETE",
                    "Refusing to overwrite completed training")
        snapshot = out / "source_snapshot.zip"
        if not snapshot.exists():
            with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in source_hashes:
                    archive.write(path, arcname=path.lstrip("/"))
                archive.writestr("RUN_MANIFEST.json", json.dumps(metadata, indent=2))
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])
        patch = init_uada_patch(tuple(cfg["patch_shape"]), self.wrapper.device)
        initial = patch.detach().clone()
        optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"], cfg["warmup_outer_iterations"],
                                               cfg["outer_iterations"], cfg["weight_decay"])
        rows, weight, start, prior_elapsed = [], None, 1, 0.0
        if resume_path.exists():
            resume = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_resume(resume, signature, method, cfg["inner_loop"])
            with torch.no_grad():
                patch.copy_(resume["patch"].to(patch.device))
            initial = resume["initial"].to(patch.device)
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            rows, weight, start = resume["rows"], resume["weight"], resume["outer"] + 1
            prior_elapsed = resume["elapsed_s"]
            torch.set_rng_state(resume["torch_rng"])
            if patch.is_cuda:
                torch.cuda.set_rng_state_all(resume["cuda_rng"])
            np.random.set_state(resume["numpy_rng"])
            random.setstate(resume["python_rng"])
        else:
            save_patch(out / "outer_0000.pt", patch, {"signature": signature, **patch_metrics(patch, initial)})
        low = torch.tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        started = time.monotonic()
        outer = start - 1
        try:
            for outer in range(start, cfg["outer_iterations"] + 1):
                index = self._training_index(outer, len(records), cfg["data_seed"])
                record = records[index]
                inputs, proprio = self._base_inputs(record)
                with torch.no_grad():
                    clean_action, clean_dino, _ = self._forward(inputs, proprio, record["observation"], None, False)
                require(tuple(clean_action.shape) == (1, 8, 7), "Action layout drift")
                require(clean_dino.shape[:2] == (1, 256), "DINO dense grid drift")
                target = torch.where(clean_action >= (low + high) / 2, low, high).detach()
                teacher = make_teacher(clean_dino, masks[record_key(record)].to(patch.device),
                                       max_queries=cfg["max_queries"], min_coverage=cfg["min_object_coverage"],
                                       radius=cfg["neighborhood_radius"], dx=cfg["shift_tokens"],
                                       temperature=cfg["temperature"])
                sums = dict(action_mse=0., auxiliary_loss=0., appearance_loss=0., aux_ratio=0.,
                            aux_cosine=0., aux_active=0., visible_queries=0., visible_candidates=0.,
                            matching_dx=0., target_dx=0., tv_gradient_ratio=0., skipped_aux_steps=0.)
                for inner in range(cfg["inner_loop"]):
                    transform_seed = cfg["seed"] * 10_000_000 + outer * 10_000 + inner
                    random.seed(transform_seed)
                    np.random.seed(transform_seed % (2**32 - 1))
                    torch.manual_seed(transform_seed)
                    optimizer.zero_grad(set_to_none=True)
                    self.wrapper.vla.zero_grad(set_to_none=True)
                    action, dino, _ = self._forward(inputs, proprio, record["observation"], patch, cfg["geometry"])
                    action_loss = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
                    match_loss, diagnostics = correspondence_loss(dino, teacher, self.patch_tokens)
                    if method == "uada_tv":
                        aux = None
                    elif method == "feature_tv":
                        aux = feature_loss(dino, clean_dino, teacher, self.patch_tokens)
                    else:
                        aux = match_loss
                    main_grad = torch.autograd.grad(action_loss, patch, retain_graph=aux is not None)[0]
                    if outer == start and inner == 0:
                        require(float(main_grad.norm()) > 1e-12, "Real model action-to-patch gradient is zero")
                    grad, grad_stats = main_grad, {"aux_ratio": 0., "aux_cosine": 0., "aux_active": 0.}
                    if aux is not None:
                        aux_grad = torch.autograd.grad(aux, patch)[0]
                        grad, weight, grad_stats = calibrated_gradient(main_grad, aux_grad, weight,
                                                                       cfg["aux_initial_gradient_ratio"])
                    elif method != "uada_tv":
                        sums["skipped_aux_steps"] += 1
                    tv_loss = appearance_loss(patch, cfg["tv_threshold"]) if method in TV_METHODS else patch.sum() * 0
                    tv_grad = torch.autograd.grad(tv_loss * cfg["tv_weight"], patch)[0]
                    patch.grad = grad + tv_grad
                    require(bool(torch.isfinite(patch.grad).all()), "Nonfinite patch gradient")
                    tv_ratio = float(tv_grad.norm() / main_grad.norm().clamp_min(1e-12))
                    step_uada_patch(patch, optimizer, self.wrapper.vla)
                    stats = {"action_mse": float(action_loss.detach()),
                             "auxiliary_loss": float(aux.detach()) if aux is not None else 0.,
                             "appearance_loss": float(tv_loss.detach()), "tv_gradient_ratio": tv_ratio,
                             **grad_stats, **diagnostics}
                    for key, value in stats.items():
                        sums[key] += value / cfg["inner_loop"]
                scheduler.step()
                if outer == 5 and method != "uada_tv":
                    require(weight is not None, "First 250 updates had no nonzero DINO guidance; stop before wasting budget")
                structure = patch_metrics(patch, initial)
                elapsed = prior_elapsed + time.monotonic() - started
                rows.append({"method": method, "outer_iteration": outer, "record_index": index,
                             "optimizer_updates": outer * cfg["inner_loop"], "task_id": record["task_id"],
                             "trajectory_rank": record["trajectory_rank"], "decision_index": record["decision_index"],
                             "auxiliary_weight": weight, "learning_rate": optimizer.param_groups[0]["lr"],
                             "elapsed_s": elapsed, **structure, **sums})
                if outer in cfg["checkpoint_outer_iterations"]:
                    save_patch(out / f"outer_{outer:04d}.pt", patch,
                               {"method": method, "signature": signature, "auxiliary_weight": weight, **structure})
                if outer % cfg["resume_every_outer_iterations"] == 0 or outer == cfg["outer_iterations"]:
                    atomic_torch({"signature": signature, "method": method, "outer": outer,
                                  "patch": patch.detach().cpu(), "initial": initial.detach().cpu(),
                                  "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                  "rows": rows, "weight": weight, "elapsed_s": elapsed,
                                  "torch_rng": torch.get_rng_state(),
                                  "cuda_rng": torch.cuda.get_rng_state_all() if patch.is_cuda else [],
                                  "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, resume_path)
                    atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
                    atomic_json({"status": "RUNNING", "signature": signature, "method": method,
                                 "outer": outer, "elapsed_s": elapsed, **structure}, status_path)
                    print(f"CORRESPONDENCE500 method={method} outer={outer} action={sums['action_mse']:.5f} "
                          f"dx={sums['matching_dx']:.3f} tv={structure['tv']:.4f} aux_ratio={sums['aux_ratio']:.3f}", flush=True)
            if method != "uada_tv":
                require(weight is not None and sum(r["aux_active"] for r in rows) > 0,
                        "No nonzero DINO guidance during training")
            atomic_json({"status": "COMPLETE", "signature": signature, "method": method,
                         "outer": cfg["outer_iterations"], "optimizer_updates": cfg["outer_iterations"] * cfg["inner_loop"],
                         "elapsed_s": prior_elapsed + time.monotonic() - started, "visual_review": "REQUIRED",
                         "final_patch_sha256": sha256_file(out / f"outer_{cfg['outer_iterations']:04d}.pt"),
                         **patch_metrics(patch, initial)}, status_path)
            # Keep the final resumable state as a compact auditable optimizer record.
        except BaseException as error:
            atomic_json({"status": "FAILED", "signature": signature, "method": method, "attempted_outer": outer,
                         "error": repr(error), "resume": str(resume_path),
                         "elapsed_s": prior_elapsed + time.monotonic() - started}, status_path)
            raise

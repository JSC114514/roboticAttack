"""Independent Exploring-derived four-arm ILPD full-RGB patch refinement.

Copy of the audited alternating/ILA training scaffolding; inner objective is
the original continuous OFT UADA on every step, with temporary feature decay.
"""
from __future__ import annotations

import importlib.metadata
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_CORRESPONDENCE import (
    run_signature, source_manifest as base_source_manifest, validate_resume,
)
from white_patch.UADA_OFT_DINO import (
    atomic_json, atomic_parquet, atomic_torch, patch_metrics, require, save_patch, sha256_file,
)
from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, step_uada_patch
from white_patch.UADA_OFT_ILA import ExploringOFTILAAttacker, seed_transform
from white_patch.oft_correspondence_objectives import appearance_loss
from white_patch.oft_ilpd_features import (
    METHODS, box_metrics, project_rgb_, transform_dense, wrist_feature_transform,
)


def source_manifest(cfg, wrapper=None):
    sources = base_source_manifest(cfg, wrapper)
    project = Path(__file__).resolve().parents[2]
    for relative in ("submit_exploring_oft_dino_ilpd_refine200.sh",
                     "slurm/exploring_oft_dino/60_train_dino_ilpd_refine200_3090.sbatch",
                     "configs/exploring_oft_dino/object_dino_ilpd_refine200_v1.json",
                     "tests/test_exploring_oft_ilpd.py"):
        path = project / relative
        sources[str(path.resolve())] = sha256_file(path)
    return sources


class ExploringOFTILPDAttacker(ExploringOFTILAAttacker):
    def _modified_forward(self, inputs, proprio, observation, patch, geometry, method, anchors, alpha):
        with wrist_feature_transform(self.wrapper.vla.vision_backbone, method, anchors, alpha) as capture:
            result = self._forward(inputs, proprio, observation, patch, geometry, capture_siglip=True)
        return result, capture

    def _validate_ilpd_runtime(self, record, reference, cfg, method):
        # Preserve existing audited actual-pixel/full-OFT/direct-DINO checks.
        validation = self._validate_runtime(record, reference, cfg)
        validation["status"] = "CHECKING"
        validation["ilpd"] = {"method": method, "checks": []}
        inputs, proprio = self._base_inputs(record)
        obs = record["observation"]
        seed = cfg["seed"] * 10_000_000 + (cfg["base_outer_iterations"] + 1) * 10_000
        low = torch.tensor(cfg["action_head_q01"], device=reference.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=reference.device).reshape(1, 1, 7)
        with torch.no_grad():
            validation["full_forward_calls"] += 1
            clean_action, clean_dino, clean_siglip = self._forward(inputs, proprio, obs, None, False, capture_siglip=True)
        anchors = {"dino": clean_dino.detach(), "siglip": clean_siglip.detach()}
        target = torch.where(clean_action >= (low + high) / 2, low, high)
        patch = reference.detach().clone().requires_grad_(True)
        seed_transform(seed)
        validation["full_forward_calls"] += 1
        normal, base_capture = self._modified_forward(
            inputs, proprio, obs, patch, cfg["geometry"], METHODS[0], anchors, 1.)
        loss = F.mse_loss(normal[0][..., cfg["maskidx"]], target[..., cfg["maskidx"]])
        base_gradient = torch.autograd.grad(loss, patch)[0]
        require(bool(torch.isfinite(base_gradient).all()) and float(base_gradient.norm()) > 1e-12,
                "Real-model action-to-RGB gradient is zero/nonfinite")
        base_action = normal[0].detach()
        baseline_features = {b: [v["raw"].detach() for v in base_capture[b]] for b in base_capture}
        del loss, normal, base_capture
        for alpha in (1., cfg["feature_alpha"]):
            seed_transform(seed)
            validation["full_forward_calls"] += 1
            result, capture = self._modified_forward(
                inputs, proprio, obs, patch, cfg["geometry"], method, anchors, alpha)
            loss = F.mse_loss(result[0][..., cfg["maskidx"]], target[..., cfg["maskidx"]])
            selected = "siglip" if method == METHODS[3] else "dino"
            modified = method != METHODS[0] and alpha != 1.
            if modified:
                raw, forwarded = capture[selected][1]["raw"], capture[selected][1]["forward"]
                gradient, raw_grad, forward_grad = torch.autograd.grad(loss, (patch, raw, forwarded))
                require(torch.allclose(raw_grad.float(), alpha * forward_grad.float(), rtol=1e-5, atol=1e-6),
                        "Actual encoder local gradient does not match alpha")
            else:
                gradient = torch.autograd.grad(loss, patch)[0]
            require(bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 1e-12,
                    "Transformed action-to-RGB gradient is zero/nonfinite")
            errors = {}
            for branch, entries in capture.items():
                for index, entry in enumerate(entries):
                    raw, forwarded = entry["raw"], entry["forward"]
                    require(torch.equal(raw.detach(), baseline_features[branch][index]),
                            "EOT pairing or raw Front/encoder output drift")
                    expected = (transform_dense(raw.detach(), anchors[branch], alpha, method == METHODS[1])
                                if modified and branch == selected and index == 1 else raw.detach())
                    require(torch.equal(forwarded.detach(), expected), "Actual forwarded feature formula mismatch")
                    errors[f"{branch}_{index}_max_abs_error"] = float((forwarded.detach().float() - expected.float()).abs().max())
            if alpha == 1. or method in (METHODS[0], METHODS[1]):
                require(torch.equal(result[0].detach(), base_action), "Identity/backward-only forward changed actions")
            if alpha == 1. or method == METHODS[0]:
                require(torch.allclose(gradient, base_gradient, rtol=1e-5, atol=1e-6),
                        "Identity feature transform changed RGB gradient")
            validation["ilpd"]["checks"].append(
                {"alpha": alpha, "formula_exact": True, "front_and_other_encoder_unchanged": True,
                 "patch_grad_norm": float(gradient.norm()), **errors})
            del loss, result, capture, gradient
        validation["status"] = "PASS"
        self.runtime_alignment = validation
        return validation

    def train(self, records, cfg, method, source_hashes, data_hashes, model_load_s=0.):
        require(method in METHODS, "Unknown ILPD refinement method")
        out = Path(cfg["output_root"]) / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        require(previous.get("status") != "COMPLETE", "Refusing to overwrite completed training")
        if previous:
            require(previous["signature"] == signature, "Existing status signature changed")
        manifest_path = out / "RUN_MANIFEST.json"
        metadata = {"experiment_id": cfg["experiment_id"], "method": method, "signature": signature,
                    "config": cfg, "source": source_hashes, "data": data_hashes,
                    "parameter_count": 7500, "base_outer_iterations": cfg["base_outer_iterations"],
                    "base_optimizer_updates": cfg["base_outer_iterations"] * cfg["inner_loop"],
                    "visual_review": "REQUIRED", "optimizer_restarted": True,
                    "feature_main_objective": False, "objective_schedule": "UADA on every update",
                    "reference_kind": "same-state clean features, detached",
                    "feature_alpha": cfg["feature_alpha"], "rgb_radius": cfg["rgb_radius"],
                    "evaluation_forward": "normal model; RGB patch only; no feature hooks"}
        metadata["packages"] = {name: importlib.metadata.version(name)
                                for name in ("torch", "torchvision", "transformers", "timm", "numpy", "pandas")}
        metadata["cuda_version"] = torch.version.cuda
        if manifest_path.exists():
            require(json.loads(manifest_path.read_text())["signature"] == signature,
                    "Run manifest signature changed")
        else:
            atomic_json(metadata, manifest_path)
        snapshot = out / "source_snapshot.zip"
        if not snapshot.exists():
            with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in source_hashes:
                    archive.write(path, arcname=path.lstrip("/"))
                archive.writestr("RUN_MANIFEST.json", json.dumps(metadata, indent=2))
        require(sha256_file(cfg["baseline_patch"]) == cfg["baseline_patch_sha256"], "Reference patch changed")
        reference = torch.load(cfg["baseline_patch"], map_location="cpu", weights_only=False)["patch"]
        require(tuple(reference.shape) == (3, 50, 50) and bool(torch.isfinite(reference).all())
                and bool((reference >= 0).all() and (reference <= 1).all()), "Invalid reference RGB")
        reference = reference.detach().float().to(self.wrapper.device)
        initial = reference.clone()
        patch = initial.clone().requires_grad_(True)
        optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"], cfg["warmup_outer_iterations"],
                                               cfg["outer_iterations"], cfg["weight_decay"])
        rows, start, prior_elapsed = [], 1, float(previous.get("elapsed_s", 0.))
        costs = dict(previous.get("costs", {}))
        if resume_path.exists():
            resume = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_resume(resume, signature, method, cfg["inner_loop"])
            require(0 <= resume["outer"] <= cfg["outer_iterations"], "Resume outer exceeds refinement budget")
            require(torch.equal(resume["initial"], reference.cpu()), "Resume initial/reference mismatch")
            require(tuple(resume["patch"].shape) == tuple(reference.shape)
                    and bool(torch.isfinite(resume["patch"]).all())
                    and bool((resume["patch"] >= 0).all() and (resume["patch"] <= 1).all()),
                    "Invalid resumed RGB patch")
            with torch.no_grad():
                patch.copy_(resume["patch"].to(patch.device))
            box_metrics(patch, initial, cfg["rgb_radius"])
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            rows, start = resume["rows"], resume["outer"] + 1
            prior_elapsed = max(prior_elapsed, float(resume["elapsed_s"]))
            # Status can include failed/replayed work beyond the last checkpoint;
            # preserve it, and retain checkpoint costs if status was lost.
            for key, value in resume.get("costs", {}).items():
                costs[key] = max(float(value), costs.get(key, 0.))
            torch.set_rng_state(resume["torch_rng"])
            if patch.is_cuda:
                torch.cuda.set_rng_state_all(resume["cuda_rng"])
            np.random.set_state(resume["numpy_rng"])
            random.setstate(resume["python_rng"])
        else:
            require(not previous, "Existing status without resume; use a new output directory")
            save_patch(out / "outer_0000.pt", patch,
                       {"signature": signature, "method": method, "total_outer": cfg["base_outer_iterations"],
                        **patch_metrics(patch, initial)})
        costs["model_load_s"] = costs.get("model_load_s", 0.) + model_load_s
        low = torch.tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        started = time.monotonic()
        outer = start - 1
        def save_resume():
            atomic_torch({"signature": signature, "method": method, "outer": outer,
                          "patch": patch.detach().cpu(), "initial": initial.cpu(),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "rows": rows, "costs": dict(costs),
                          "elapsed_s": prior_elapsed + time.monotonic() - started,
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if patch.is_cuda else [],
                          "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, resume_path)
        def progress(status, **extra):
            completed = len(rows)
            return {"status": status, "signature": signature, "method": method,
                    "outer": completed, "total_outer": cfg["base_outer_iterations"] + completed,
                    "optimizer_updates": completed * cfg["inner_loop"],
                    "action_optimizer_updates": completed * cfg["inner_loop"],
                    "feature_optimizer_updates": 0,
                    "total_optimizer_updates": (cfg["base_outer_iterations"] + completed) * cfg["inner_loop"],
                    "elapsed_s": prior_elapsed + time.monotonic() - started, "costs": dict(costs),
                    "base_elapsed_s": cfg["baseline_elapsed_s"],
                    "cost_boundary": "baseline logged training + refinement elapsed + refinement model loading; "
                                     "historical baseline model loading unavailable",
                    "cost_including_base_s": cfg["baseline_elapsed_s"] + prior_elapsed
                                              + time.monotonic() - started + costs.get("model_load_s", 0.),
                    "visual_review": "REQUIRED", **extra}
        try:
            # Even a startup/early failure can resume from the immutable start.
            if not resume_path.exists():
                save_resume()
            runtime_forward_before = costs.get("runtime_full_forward_attempts", 0)
            validation = self._timed(costs, "runtime_validation_s",
                                     lambda: self._validate_ilpd_runtime(records[0], reference, cfg, method))
            costs["runtime_full_forward_attempts"] = runtime_forward_before + validation["full_forward_calls"]
            atomic_json(validation, out / "RUNTIME_ALIGNMENT.json")
            for outer in range(start, cfg["outer_iterations"] + 1):
                total_outer = cfg["base_outer_iterations"] + outer
                index = self._training_index(total_outer, len(records), cfg["data_seed"])
                record = records[index]
                inputs, proprio, target, anchors = None, None, None, None
                def prepare_clean():
                    nonlocal inputs, proprio, target, anchors
                    with torch.no_grad():
                        inputs, proprio = self._base_inputs(record)
                        costs["training_full_forward_attempts"] = costs.get("training_full_forward_attempts", 0) + 1
                        action, dino, siglip = self._forward(
                            inputs, proprio, record["observation"], None, False, capture_siglip=True)
                        require(siglip is not None, "Clean SigLIP anchor missing")
                        anchors = {"dino": dino.detach(), "siglip": siglip.detach()}
                        target = torch.where(action >= (low + high) / 2, low, high).detach()
                self._timed(costs, "clean_prepare_s", prepare_clean)
                sums = dict(main_loss=0., appearance_loss=0., main_grad_norm=0.,
                            dino_raw_delta_rms=0., dino_forward_delta_rms=0.,
                            siglip_raw_delta_rms=0., siglip_forward_delta_rms=0.)
                for inner in range(cfg["inner_loop"]):
                    seed = cfg["seed"] * 10_000_000 + total_outer * 10_000 + inner
                    def update():
                        seed_transform(seed)
                        optimizer.zero_grad(set_to_none=True)
                        costs["training_full_forward_attempts"] = costs.get("training_full_forward_attempts", 0) + 1
                        (action, _, _), capture = self._modified_forward(
                            inputs, proprio, record["observation"], patch, cfg["geometry"],
                            method, anchors, cfg["feature_alpha"])
                        loss = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
                        tv = appearance_loss(patch, cfg["tv_threshold"]) * cfg["tv_weight"]
                        gradient = torch.autograd.grad(loss, patch)[0]
                        norm = float(gradient.norm())
                        if outer == start and inner == 0:
                            require(norm > 1e-12, "Real-model action-to-patch gradient is zero")
                        patch.grad = gradient + torch.autograd.grad(tv, patch)[0]
                        require(bool(torch.isfinite(patch.grad).all()), "Nonfinite RGB patch gradient")
                        step_uada_patch(patch, optimizer, self.wrapper.vla)
                        project_rgb_(patch, initial, cfg["rgb_radius"])
                        stats = dict(main_loss=float(loss.detach()), appearance_loss=float(tv.detach()), main_grad_norm=norm)
                        with torch.no_grad():
                            for branch in ("dino", "siglip"):
                                for kind in ("raw", "forward"):
                                    delta = capture[branch][1][kind].float() - anchors[branch].float()
                                    stats[f"{branch}_{kind}_delta_rms"] = float(delta.square().mean().sqrt())
                        return stats
                    metrics = self._timed(costs, "full_vla_update_s", update)
                    for key, value in metrics.items():
                        sums[key] += value / cfg["inner_loop"]
                scheduler.step()
                structure = {**patch_metrics(patch, initial), **box_metrics(patch, initial, cfg["rgb_radius"])}
                rows.append({"method": method, "outer_iteration": outer, "total_outer_iteration": total_outer,
                             "record_index": index, "task_id": int(record["task_id"]),
                             "action_steps": cfg["inner_loop"], "feature_steps": 0,
                             "action_optimizer_updates": outer * cfg["inner_loop"], "feature_optimizer_updates": 0,
                             "optimizer_updates": outer * cfg["inner_loop"],
                             "total_optimizer_updates": total_outer * cfg["inner_loop"],
                             "learning_rate": optimizer.param_groups[0]["lr"],
                             "elapsed_s": prior_elapsed + time.monotonic() - started, **structure, **sums})
                if outer in cfg["checkpoint_outer_iterations"]:
                    save_patch(out / f"outer_{outer:04d}.pt", patch,
                               {"method": method, "signature": signature, "total_outer": total_outer,
                                "total_optimizer_updates": total_outer * cfg["inner_loop"], **structure})
                if outer % cfg["resume_every_outer_iterations"] == 0 or outer == cfg["outer_iterations"]:
                    save_resume()
                    atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
                    atomic_json(progress("RUNNING", **structure), status_path)
                    print(f"DINO_ILPD200 method={method} outer={outer}/{cfg['outer_iterations']} "
                          f"total_outer={total_outer} loss={sums['main_loss']:.5f} "
                          f"max_rgb_delta={structure['rgb_max_abs_delta']:.5f} tv={structure['tv']:.4f}", flush=True)
            atomic_json(progress("COMPLETE", **patch_metrics(patch, initial), **box_metrics(patch, initial, cfg["rgb_radius"]),
                                 final_patch_sha256=sha256_file(out / f"outer_{cfg['outer_iterations']:04d}.pt")),
                        status_path)
        except BaseException as error:
            if hasattr(self, "runtime_alignment"):
                alignment = dict(self.runtime_alignment)
                costs["runtime_full_forward_attempts"] = runtime_forward_before + alignment.get("full_forward_calls", 0)
                alignment.update(status="FAILED" if alignment["status"] != "PASS" else "PASS", error=repr(error))
                atomic_json(alignment, out / "RUNTIME_ALIGNMENT.json")
            atomic_json(progress("FAILED", attempted_outer=outer, error=repr(error), resume=str(resume_path)),
                        status_path)
            raise


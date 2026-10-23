"""Exploring-derived UADA/DINO alternating full-RGB patch refinement.

Only this new copy adds phase scheduling and local reference snapshots. The
audited ILA V2 input construction and original RGB EOT are inherited unchanged.
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
from white_patch.oft_alternating_objectives import alternating_projection

METHODS = ("alternating_fixed_tv", "alternating_rolling_tv")
FEATURE_METHODS = set(METHODS)


def source_manifest(cfg, wrapper=None):
    sources = base_source_manifest(cfg, wrapper)
    project = Path(__file__).resolve().parents[2]
    for relative in ("submit_exploring_oft_dino_alternating_refine200.sh",
                     "slurm/exploring_oft_dino/57_train_dino_alternating_refine200_3090.sbatch",
                     "configs/exploring_oft_dino/object_dino_alternating_refine200_v1.json"):
        path = project / relative
        sources[str(path.resolve())] = sha256_file(path)
    return sources


class ExploringOFTAlternatingAttacker(ExploringOFTILAAttacker):
    def train(self, records, cfg, method, source_hashes, data_hashes, model_load_s=0.):
        require(method in METHODS, "Unknown alternating refinement method")
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
                    "feature_main_objective": method in FEATURE_METHODS,
                    "reference_kind": "fixed UADA500-clean" if method == METHODS[0] else "current outer post-UADA minus pre-UADA",
                    "objective_schedule": "UADA then DINO; shared optimizer",
                    "action_steps_per_outer": cfg["action_steps_per_outer"],
                    "feature_steps_per_outer": cfg["feature_steps_per_outer"]}
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
                    "action_optimizer_updates": completed * cfg["action_steps_per_outer"],
                    "feature_optimizer_updates": completed * cfg["feature_steps_per_outer"],
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
            validation = self._timed(costs, "runtime_validation_s",
                                     lambda: self._validate_runtime(records[0], reference, cfg))
            atomic_json(validation, out / "RUNTIME_ALIGNMENT.json")
            require(cfg["action_steps_per_outer"] + cfg["feature_steps_per_outer"] == cfg["inner_loop"]
                    and min(cfg["action_steps_per_outer"], cfg["feature_steps_per_outer"]) > 0,
                    "Alternating phase budget drift")
            for outer in range(start, cfg["outer_iterations"] + 1):
                total_outer = cfg["base_outer_iterations"] + outer
                index = self._training_index(total_outer, len(records), cfg["data_seed"])
                record = records[index]
                inputs, proprio, target, clean = None, None, None, None
                def prepare_clean():
                    nonlocal inputs, proprio, target, clean
                    with torch.no_grad():
                        inputs, proprio = self._base_inputs(record)
                        action, _, _ = self._forward(inputs, proprio, record["observation"], None, False)
                        target = torch.where(action >= (low + high) / 2, low, high).detach()
                        if method == METHODS[0]:
                            clean, _ = self._features(record["observation"], None, False)
                self._timed(costs, "clean_prepare_s", prepare_clean)
                before = patch.detach().clone()
                after = None
                action_steps = feature_steps = 0
                sums = dict(main_loss=0., appearance_loss=0., main_grad_norm=0., feature_grad_active=0.,
                            action_loss=0., feature_loss=0., skipped_feature_steps=0., visible_tokens=0.,
                            reference_rms=0., projection_score=0., direction_cosine=0., delta_rms=0.,
                            feature_active=0., support_disagreement_tokens=0.)
                for inner in range(cfg["inner_loop"]):
                    is_feature = inner >= cfg["action_steps_per_outer"]
                    seed = cfg["seed"] * 10_000_000 + total_outer * 10_000 + inner
                    anchor, anchor_coverage, teacher, teacher_coverage = None, None, None, None
                    if is_feature:
                        if after is None:
                            after = patch.detach().clone()
                        def prepare_teacher():
                            with torch.no_grad():
                                if method == METHODS[0]:
                                    origin, origin_coverage = clean, None
                                    teacher_patch = reference
                                else:
                                    seed_transform(seed)
                                    origin, origin_coverage = self._features(record["observation"], before, cfg["geometry"])
                                    teacher_patch = after
                                seed_transform(seed)
                                endpoint, endpoint_coverage = self._features(record["observation"], teacher_patch, cfg["geometry"])
                            return origin, origin_coverage, endpoint, endpoint_coverage
                        anchor, anchor_coverage, teacher, teacher_coverage = self._timed(
                            costs, "reference_teacher_s", prepare_teacher)
                    def update():
                        seed_transform(seed)
                        optimizer.zero_grad(set_to_none=True)
                        if not is_feature:
                            action, _, _ = self._forward(inputs, proprio, record["observation"], patch, cfg["geometry"])
                            loss = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
                            stats = {}
                        else:
                            current, coverage = self._features(record["observation"], patch, cfg["geometry"])
                            loss, stats = alternating_projection(
                                current, anchor, teacher, coverage, anchor_coverage, teacher_coverage,
                                cfg["coverage_threshold"], cfg["min_reference_rms"], cfg["projection_eps"])
                        tv = appearance_loss(patch, cfg["tv_threshold"]) * cfg["tv_weight"]
                        if loss is None:
                            gradient = torch.zeros_like(patch)
                            stats["skipped_feature_steps"] = 1.
                        else:
                            gradient = torch.autograd.grad(loss, patch)[0]
                        norm = float(gradient.norm())
                        if outer == start and inner == 0:
                            require(norm > 1e-12, "Real-model action-to-patch gradient is zero")
                        patch.grad = gradient + torch.autograd.grad(tv, patch)[0]
                        require(bool(torch.isfinite(patch.grad).all()), "Nonfinite RGB patch gradient")
                        step_uada_patch(patch, optimizer, self.wrapper.vla)
                        stats.update(main_loss=float(loss.detach()) if loss is not None else 0.,
                                     appearance_loss=float(tv.detach()), main_grad_norm=norm,
                                     feature_grad_active=float(is_feature and norm > 1e-12))
                        return stats
                    metrics = self._timed(costs, "dino_update_s" if is_feature else "full_vla_update_s", update)
                    if is_feature:
                        feature_steps += 1
                        sums["feature_loss"] += metrics["main_loss"] / cfg["feature_steps_per_outer"]
                    else:
                        action_steps += 1
                        sums["action_loss"] += metrics["main_loss"] / cfg["action_steps_per_outer"]
                    for key, value in metrics.items():
                        divisor = cfg["inner_loop"] if key in ("main_loss", "appearance_loss", "main_grad_norm") else cfg["feature_steps_per_outer"]
                        sums[key] += value / divisor
                require(action_steps == cfg["action_steps_per_outer"] and feature_steps == cfg["feature_steps_per_outer"],
                        "Actual phase update count drift")
                scheduler.step()
                if method in FEATURE_METHODS and outer == min(5, cfg["outer_iterations"]):
                    require(sum(row["feature_grad_active"] for row in rows) + sums["feature_grad_active"] > 0,
                            "No nonzero DINO-to-patch gradient in first five alternating outer iterations")
                structure = patch_metrics(patch, initial)
                rows.append({"method": method, "outer_iteration": outer, "total_outer_iteration": total_outer,
                             "record_index": index, "task_id": int(record["task_id"]),
                             "action_steps": action_steps, "feature_steps": feature_steps,
                             "action_optimizer_updates": outer * cfg["action_steps_per_outer"],
                             "feature_optimizer_updates": outer * cfg["feature_steps_per_outer"],
                             "rolling_rgb_reference_rms": float((after - before).square().mean().sqrt()) if method == METHODS[1] else 0.,
                             "reference_refreshes": int(method == METHODS[1]),
                             "optimizer_updates": outer * cfg["inner_loop"],
                             "total_optimizer_updates": total_outer * cfg["inner_loop"],
                             "learning_rate": optimizer.param_groups[0]["lr"],
                             "elapsed_s": prior_elapsed + time.monotonic() - started,
                             **structure, **sums})
                if outer in cfg["checkpoint_outer_iterations"]:
                    save_patch(out / f"outer_{outer:04d}.pt", patch,
                               {"method": method, "signature": signature, "total_outer": total_outer,
                                "total_optimizer_updates": total_outer * cfg["inner_loop"], **structure})
                if outer % cfg["resume_every_outer_iterations"] == 0 or outer == cfg["outer_iterations"]:
                    save_resume()
                    atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
                    atomic_json(progress("RUNNING", **structure), status_path)
                    print(f"DINO_ALTERNATING200 method={method} outer={outer}/{cfg['outer_iterations']} "
                          f"total_outer={total_outer} loss={sums['main_loss']:.5f} "
                          f"projection={sums['projection_score']:.3f} tv={structure['tv']:.4f}", flush=True)
            atomic_json(progress("COMPLETE", **patch_metrics(patch, initial),
                                 final_patch_sha256=sha256_file(out / f"outer_{cfg['outer_iterations']:04d}.pt")),
                        status_path)
        except BaseException as error:
            if hasattr(self, "runtime_alignment"):
                alignment = dict(self.runtime_alignment)
                alignment.update(status="FAILED" if alignment["status"] != "PASS" else "PASS", error=repr(error))
                atomic_json(alignment, out / "RUNTIME_ALIGNMENT.json")
            atomic_json(progress("FAILED", attempted_outer=outer, error=repr(error), resume=str(resume_path)),
                        status_path)
            raise

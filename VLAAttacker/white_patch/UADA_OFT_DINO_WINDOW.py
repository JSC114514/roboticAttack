"""Exploring-derived RGB UADA continuation over three-decision replay clips.

The original opposite-boundary action loss and RGB optimizer are retained.
DINO changes only which on-policy clip is selected.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

from white_patch.UADA_OFT_CORRESPONDENCE import (run_signature,
    source_manifest as base_source_manifest, validate_resume)
from white_patch.UADA_OFT_DINO import (atomic_json, atomic_parquet, atomic_torch,
    patch_metrics, require, save_patch, sha256_file)
from white_patch.UADA_OFT_ILA import ExploringOFTILAAttacker, seed_transform
from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, step_uada_patch
from white_patch.oft_correspondence_objectives import appearance_loss


def source_manifest(cfg, wrapper=None):
    sources = base_source_manifest(cfg, wrapper)
    project = Path(__file__).resolve().parents[2]
    for relative in ("configs/exploring_oft_dino/object_dino_window_refine200_v1.json",
                     "scripts/dino_window_common.py",
                     "scripts/collect_exploring_oft_dino_window.py",
                     "scripts/build_exploring_oft_dino_window_bank.py",
                     "VLAAttacker/UADA_OFT_DINO_WINDOW_wrapper.py",
                     "slurm/exploring_oft_dino/70_collect_dino_window_3090.sbatch",
                     "slurm/exploring_oft_dino/71_build_dino_window_bank_cpu.sbatch",
                     "slurm/exploring_oft_dino/72_train_dino_window_4090.sbatch",
                     "submit_exploring_oft_dino_window_train.sh",
                     "configs/exploring_oft_dino/object_dino_replay_refine200_v1.json",
                     "scripts/dino_replay_common.py",
                     "scripts/collect_exploring_oft_dino_replay.py",
                     "scripts/build_exploring_oft_dino_replay_bank.py",
                     "VLAAttacker/UADA_OFT_DINO_REPLAY_wrapper.py",
                     "slurm/exploring_oft_dino/64_collect_dino_replay_4090.sbatch",
                     "slurm/exploring_oft_dino/69_collect_dino_replay_3090.sbatch",
                     "slurm/exploring_oft_dino/65_build_dino_replay_bank_cpu.sbatch",
                     "slurm/exploring_oft_dino/66_train_dino_replay_4090.sbatch",
                     "submit_exploring_oft_dino_replay_train.sh"):
        path = project / relative
        sources[str(path.resolve())] = sha256_file(path)
    return sources


class ExploringOFTDINOWindowAttacker(ExploringOFTILAAttacker):
    def __init__(self, wrapper, save_dir, site):
        super().__init__(wrapper, save_dir, resize_patch=False)
        self.site = tuple(site)

    def _paste_with_support(self, image, patch, geometry):
        """The Exploring affine EOT, anchored to the actual Wrist test site."""
        im = torchvision.transforms.ToTensor()(self.randomPatchTransform._to_rgb_pil(image)).to(self.wrapper.device)
        _, height, width = im.shape
        _, patch_h, patch_w = patch.shape
        x, y = self.site
        require(tuple(patch.shape) == (3, 50, 50) and 0 <= x <= width - patch_w
                and 0 <= y <= height - patch_h, "Fixed Wrist site outside image")
        canvas = torch.ones_like(im) * -100
        canvas[:, y:y + patch_h, x:x + patch_w] = patch
        if geometry:
            canvas = self.randomPatchTransform.apply_affine_transform(
                canvas, self.randomPatchTransform.combined_transform_matrix().to(self.wrapper.device))
            if canvas.ndim == 4:
                canvas = canvas.squeeze(0)
        support = (canvas.detach() >= -20).any(dim=0).float()
        return torch.where(canvas < -20, im, canvas).clamp(0, 1), support

    def train(self, clean, replay, cfg, method, source_hashes, data_hashes, model_load_s=0.):
        from dino_window_common import METHODS, draw_record
        kind = method.split("_seed", 1)[0]
        require(kind in METHODS and len(clean) == 300 and len(replay) > 0,
                "Window training data drift")
        out = Path(cfg["output_root"]) / "training" / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        require(previous.get("status") != "COMPLETE", "Completed training is immutable")
        if previous:
            require(previous["signature"] == signature, "Existing training signature drift")
        manifest = {"experiment_id": cfg["experiment_id"], "method": method,
                    "signature": signature, "config": cfg, "source": source_hashes,
                    "data": data_hashes, "parameter_count": 7500,
                    "optimizer_restarted": True, "action_main_objective": "Exploring OFT UADA dim0 opposite-boundary MSE",
                    "feature_loss": False, "feature_hook_during_training": False,
                    "visual_review": "REQUIRED", "fixed_wrist_site": list(self.site),
                    "evaluation_forward": "normal OFT; RGB patch only"}
        manifest_path = out / "RUN_MANIFEST.json"
        if manifest_path.exists():
            require(json.loads(manifest_path.read_text()) == manifest, "Training manifest drift")
        else:
            atomic_json(manifest, manifest_path)
        reference = torch.load(cfg["baseline_patch"], map_location="cpu", weights_only=False)["patch"].float()
        require(tuple(reference.shape) == (3, 50, 50) and bool(torch.isfinite(reference).all()),
                "Invalid warm-start RGB patch")
        initial = reference.to(self.wrapper.device)
        patch = initial.detach().clone().requires_grad_(True)
        optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"],
            cfg["warmup_outer_iterations"], cfg["outer_iterations"], cfg["weight_decay"])
        rows, start = [], 1
        prior_elapsed = float(previous.get("elapsed_s", 0.))
        costs = dict(previous.get("costs", {}))
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_resume(saved, signature, method, cfg["inner_loop"])
            require(0 <= saved["outer"] <= cfg["outer_iterations"]
                    and torch.equal(saved["initial"], reference), "Invalid resume anchor/budget")
            with torch.no_grad():
                patch.copy_(saved["patch"].to(patch.device))
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            rows, start = saved["rows"], saved["outer"] + 1
            prior_elapsed = max(prior_elapsed, float(saved["elapsed_s"]))
            for key, value in saved.get("costs", {}).items():
                costs[key] = max(float(value), float(costs.get(key, 0.)))
            torch.set_rng_state(saved["torch_rng"])
            if patch.is_cuda:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
        else:
            require(not previous, "Existing status without resume")
            save_patch(out / "outer_0000.pt", patch, {"signature": signature,
                       "method": method, "total_outer": cfg["base_outer_iterations"]})
        costs["model_load_s"] = costs.get("model_load_s", 0.) + model_load_s
        low = torch.tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        started = time.monotonic()
        outer = start - 1

        def elapsed():
            return prior_elapsed + time.monotonic() - started

        def save_resume():
            atomic_torch({"signature": signature, "method": method, "outer": outer,
                "patch": patch.detach().cpu(), "initial": reference,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rows": rows, "costs": dict(costs), "elapsed_s": elapsed(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if patch.is_cuda else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, resume_path)

        def progress(status, **extra):
            return {"status": status, "method": method, "signature": signature,
                    "outer": len(rows), "total_outer": cfg["base_outer_iterations"] + len(rows),
                    "optimizer_updates": len(rows) * cfg["inner_loop"],
                    "total_optimizer_updates": (cfg["base_outer_iterations"] + len(rows)) * cfg["inner_loop"],
                    "elapsed_s": elapsed(), "costs": dict(costs),
                    "known_cost_including_base_s": cfg["baseline_elapsed_s"] + elapsed()
                    + costs.get("model_load_s", 0.),
                    "cost_boundary": "logged historical baseline training + refinement including model loading; "
                                     "historical baseline model loading unavailable",
                    "visual_review": "REQUIRED", **extra}

        try:
            if not resume_path.exists():
                save_resume()
            alignment = self._timed(costs, "runtime_validation_s",
                                    lambda: self._validate_runtime(clean[0], initial, cfg))
            atomic_json(alignment, out / "RUNTIME_ALIGNMENT.json")
            for outer in range(start, cfg["outer_iterations"] + 1):
                total_outer = cfg["base_outer_iterations"] + outer
                base_index = self._training_index(total_outer, len(clean), cfg["data_seed"] + cfg["seed"])
                record, origin, index = draw_record(clean, replay, kind, outer, base_index,
                                                    cfg["data_seed"] + cfg["seed"])
                observations = (record["observations"] if origin == "replay"
                                else [record["observation"]] * cfg["clip_length"])
                require(len(observations) == 3, "Three-decision clip drift")
                prepared, targets = [], []
                for observation in observations:
                    inputs, proprio = self._base_inputs(
                        {"observation": observation, "task_name": record["task_name"]})
                    with torch.no_grad():
                        action, _, _ = self._timed(costs, "clean_prepare_s", lambda:
                            self._forward(inputs, proprio, observation, None, False))
                        target = torch.where(action >= (low + high) / 2, low, high).detach()
                    prepared.append((inputs, proprio, observation))
                    targets.append(target)
                losses = []
                for inner in range(cfg["inner_loop"]):
                    seed_transform(cfg["seed"] * 10_000_000 + total_outer * 10_000 + inner)
                    optimizer.zero_grad(set_to_none=True)
                    geometry = inner >= cfg["inner_loop"] // 2
                    inputs, proprio, observation = prepared[inner % cfg["clip_length"]]
                    target = targets[inner % cfg["clip_length"]]
                    action, _, _ = self._timed(costs, "patched_forward_s", lambda:
                        self._forward(inputs, proprio, observation, patch, geometry))
                    main = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
                    tv = appearance_loss(patch, cfg["tv_threshold"]) * cfg["tv_weight"]
                    patch.grad = torch.autograd.grad(main + tv, patch)[0]
                    require(bool(torch.isfinite(patch.grad).all()), "Nonfinite RGB gradient")
                    if outer == start and inner == 0:
                        require(float(patch.grad.norm()) > 1e-12, "Zero real-model patch gradient")
                    step_uada_patch(patch, optimizer, self.wrapper.vla)
                    losses.append(float(main.detach()))
                scheduler.step()
                structure = patch_metrics(patch, initial)
                rows.append({"method": method, "outer_iteration": outer,
                             "total_outer_iteration": total_outer, "origin": origin,
                             "record_index": index, "base_record_index": base_index,
                             "task_id": int(record["task_id"]), "state_index": int(record["state_index"]),
                             "dino_score": float(record.get("dino_score", 0.)),
                             "optimizer_updates": outer * cfg["inner_loop"],
                             "total_optimizer_updates": total_outer * cfg["inner_loop"],
                             "main_loss": float(np.mean(losses)), "elapsed_s": elapsed(), **structure})
                if outer in cfg["checkpoint_outer_iterations"]:
                    save_patch(out / f"outer_{outer:04d}.pt", patch,
                               {"method": method, "signature": signature, "total_outer": total_outer,
                                "total_optimizer_updates": total_outer * cfg["inner_loop"], **structure})
                if outer % cfg["resume_every_outer_iterations"] == 0 or outer == cfg["outer_iterations"]:
                    save_resume()
                    atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
                    atomic_json(progress("RUNNING", **structure), status_path)
                    print(f"DINO_WINDOW method={method} outer={outer}/200 loss={np.mean(losses):.5f} "
                          f"tv={structure['tv']:.5f} origin={origin}", flush=True)
            final = out / f"outer_{cfg['outer_iterations']:04d}.pt"
            atomic_json(progress("COMPLETE", final_patch_sha256=sha256_file(final),
                                 final_png_sha256=sha256_file(final.with_suffix(".png")),
                                 **patch_metrics(patch, initial)), status_path)
        except BaseException as error:
            atomic_json(progress("FAILED", attempted_outer=outer, error=repr(error),
                                 resume=str(resume_path)), status_path)
            raise


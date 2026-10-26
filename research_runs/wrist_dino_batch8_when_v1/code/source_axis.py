"""Fresh Exploring-OFT RGB training with state/axis replay priority.

The action optimizer is still the original UADA-style opposite-boundary MSE on a
single continuous action coordinate. DINO is used only to score replay candidates:
for each Object replay center and each XYZ translation axis, we ask whether
restoring clean Wrist-DINO removes progress toward the wrong action boundary.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_DINO_TRANSFER import ExploringOFTDINOTransferAttacker
from white_patch.UADA_OFT_CORRESPONDENCE import run_signature
from white_patch.UADA_OFT_DINO import (
    atomic_json, atomic_parquet, atomic_torch, patch_metrics, require,
    save_patch, sha256_file,
)
from white_patch.UADA_OFT_DINO_SAMPLING_CLEAR import (
    _discard_partial_artifacts, _terminal_float32_rounding_evidence,
    canonical_quantized_patch, patch_tensor_sha256, verify_loaded_checkpoint_source,
)
from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, init_uada_patch, step_uada_patch
from white_patch.UADA_OFT_ILA import seed_transform


def source_manifest(cfg, wrapper=None):
    from dino_state_axis_common import CONFIG

    root = Path(__file__).resolve().parents[2]
    directories = [
        root / "VLAAttacker",
        root / "prismatic",
        Path(cfg["openvla_oft_root"]) / "prismatic",
        Path(cfg["openvla_oft_root"]) / "experiments",
        Path(cfg["attribution_root"]) / "src",
    ]
    libero = Path(cfg["libero_root"]) / "libero"
    if libero.is_dir():
        directories.append(libero)
    paths = {
        Path(CONFIG).resolve(),
        root / "research_runs/wrist_dino_state_axis_v1/PLAN.md",
        root / "scripts/dino_state_axis_common.py",
        root / "scripts/run_exploring_oft_dino_state_axis_cross_suite.py",
        root / "scripts/analyze_exploring_oft_dino_state_axis_cross_suite.py",
        root / "scripts/run_exploring_oft_scale500_cross_suite.py",
        root / "VLAAttacker/UADA_OFT_DINO_STATE_AXIS_wrapper.py",
        root / "VLAAttacker/white_patch/UADA_OFT_DINO_STATE_AXIS.py",
        root / "VLAAttacker/white_patch/UADA_OFT_DINO_TRANSFER.py",
        root / "VLAAttacker/white_patch/UADA_OFT_SOURCE_COPY.py",
        root / "VLAAttacker/white_patch/UADA_OFT_DINO.py",
        root / "VLAAttacker/white_patch/UADA.py",
        root.parent / "oft_ramp_recovery_aware/scripts/common.py",
        root.parent / "oft_ramp_recovery_aware/scripts/runtime.py",
        root.parent / "oft_ramp_recovery_aware/scripts/run_continuous_cldev.py",
        root.parent / "oft_mgrb_lite_low_cost/scripts/model_runtime.py",
    }
    for directory in directories:
        require(directory.is_dir(), f"Missing runtime source tree: {directory}")
        paths.update(path.resolve() for path in directory.rglob("*.py"))
    paths.update(path.resolve() for path in (root / "scripts").glob("*state_axis*.py"))
    paths.update(path.resolve() for path in (root / "slurm/exploring_oft_dino").glob("*state_axis*.sbatch"))
    paths.update(path.resolve() for path in root.glob("*state_axis*.sh"))
    paths.update(path.resolve() for pattern in ("*.json", "*.py")
                 for path in Path(cfg["checkpoint"]).glob(pattern))
    reference = json.loads(Path(cfg["eval_reference"]).read_text())
    for name in reference["checkpoints"].values():
        checkpoint = Path(name)
        require((checkpoint / "config.json").is_file(), f"Missing evaluation metadata: {checkpoint}")
        paths.update(path.resolve() for pattern in ("*.json", "*.py") for path in checkpoint.glob(pattern))
    for pattern in ("*.bddl", "*.init", "*.pruned_init"):
        paths.update(path.resolve() for path in Path(cfg["libero_root"]).rglob(pattern))
    libero_config = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))) / "config.yaml"
    require(libero_config.is_file(), f"Missing LIBERO runtime configuration: {libero_config}")
    paths.add(libero_config.resolve())
    for name, digest in cfg.get("frozen_eval_sources", {}).items():
        require(sha256_file(name) == digest, f"Frozen evaluation source changed: {name}")
        paths.add(Path(name).resolve())
    if wrapper is not None:
        objects = [
            wrapper, wrapper.vla, wrapper.processor, wrapper.processor.image_processor,
            wrapper.action_head, wrapper.vla.vision_backbone,
            wrapper.vla.vision_backbone.featurizer,
            wrapper.vla.vision_backbone.fused_featurizer,
            wrapper.proprio_projector, wrapper.noisy_action_projector,
        ]
        official = getattr(wrapper, "_official_openvla_utils", None)
        if official is not None and getattr(official, "__file__", None):
            paths.add(Path(official.__file__).resolve())
        for obj in objects:
            if obj is None:
                continue
            actual = getattr(obj, "module", obj)
            filename = Path(inspect.getfile(type(actual))).resolve()
            verify_loaded_checkpoint_source(filename, cfg)
            paths.add(filename)
            paths.update(path.resolve() for path in filename.parent.glob("*.py"))
    return {str(path): sha256_file(path) for path in sorted(paths)}


def _rng_state(patch):
    return {
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if patch.is_cuda else [],
    }


def _restore_rng(saved, patch):
    torch.set_rng_state(saved["torch_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    if patch.is_cuda:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def validate_state(saved, cfg):
    require(saved["config"] == cfg, "State-axis resume config drift")
    outer = saved["outer"]
    require(isinstance(outer, int) and 0 <= outer <= cfg["outer_iterations"]
            and len(saved["rows"]) == outer, "Resume outer/row mismatch")
    expected = outer * cfg["inner_loop"]
    require(all(row["outer_iteration"] == i and row["optimizer_updates"] == i * cfg["inner_loop"]
                for i, row in enumerate(saved["rows"], 1)), "Resume update sequence drift")
    states = saved["optimizer"]["state"]
    if expected:
        require(len(states) == 1 and all(int(state["step"]) == expected
                and "exp_avg" in state and "exp_avg_sq" in state
                and bool(torch.isfinite(state["exp_avg"]).all())
                and bool(torch.isfinite(state["exp_avg_sq"]).all()) for state in states.values()),
                "Adam moments/step missing, reset, or nonfinite")
    else:
        require(not states, "Fresh outer-zero Adam state must be empty")
    require(saved["scheduler"]["last_epoch"] == outer, "Scheduler reset/drift")
    for name in ("patch", "initial", "initial_raw"):
        value = saved[name]
        require(tuple(value.shape) == tuple(cfg["patch_shape"])
                and bool(torch.isfinite(value).all())
                and bool(((value >= 0) & (value <= 1)).all()),
                f"Invalid saved {name} RGB")
    require(torch.equal(saved["initial"], saved["initial_raw"])
            and saved["raw_random_patch_sha256"] == patch_tensor_sha256(saved["initial_raw"])
            and saved["initial_patch_sha256"] == patch_tensor_sha256(saved["initial"]),
            "Initialization tensor identity drift")
    require(saved.keys() >= {"torch_rng", "numpy_rng", "python_rng", "cuda_rng", "source", "data"},
            "Missing complete resume RNG/identity")
    history = saved["score_refresh_history"]
    require([item["outer"] for item in history] == [i for i in cfg["refresh_after_outer"] if i < outer],
            "Resume score-refresh sequence drift")
    require(saved["score_after"] == (history[-1]["outer"] if history else 0),
            "Resume score boundary drift")
    terminal = saved.get("terminal_rounding_history", [])
    unchanged = [
        row for row in saved["rows"]
        if row["learning_rate"] != 0. and row["rgb_outer_max_abs"] == 0.
    ]
    require([item["outer"] for item in terminal] == [row["outer_iteration"] for row in unchanged],
            "Resume terminal-rounding evidence sequence drift")


def _structure(patch, initial):
    result = patch_metrics(patch, initial)
    delta = patch.detach() - initial
    result.update(
        rgb_init_mean_abs=float(delta.abs().mean()),
        rgb_init_changed_pixel_fraction=float((delta.abs().amax(0) > 0).float().mean()),
    )
    return result


def _save_checkpoint(path, patch, payload):
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.pending.pt")
    save_patch(temporary, patch, payload)
    os.replace(temporary, path)
    os.replace(temporary.with_suffix(".png"), path.with_suffix(".png"))


class ExploringOFTDINOStateAxisAttacker(ExploringOFTDINOTransferAttacker):
    def _patched_pixels(self, observation, patch, geometry):
        cached = getattr(self, "_scoring_pixels", None)
        if cached is not None:
            return cached
        return super()._patched_pixels(observation, patch, geometry)

    def _axis_score_record(self, record, displayed, cfg, method, refresh_outer, record_index):
        observation = record["observation"]
        inputs, proprio = self._base_inputs(record)
        low = torch.tensor(cfg["action_head_q01"], device=displayed.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=displayed.device).reshape(1, 1, 7)
        ranges = (high - low).clamp_min(1e-12)
        with torch.no_grad():
            clean_action, clean_dino, _ = self._forward(inputs, proprio, observation, None, False)
            target = torch.where(clean_action >= (low + high) / 2, low, high)
            per_view = []
            original_site, original_diverse = self.site, getattr(self, "transfer_diverse", False)
            try:
                for view in range(2):
                    seed = cfg["data_seed"] * 1_000_000 + refresh_outer * 10_000 + record_index * 2 + view
                    seed_transform(seed)
                    self.transfer_diverse = bool(view)
                    geometry = bool(view)
                    pixels = super()._patched_pixels(observation, displayed, geometry)
                    self._scoring_pixels = pixels
                    seed_transform(seed)
                    normal, normal_dino, siglip = self._forward(
                        inputs, proprio, observation, displayed, geometry, capture_siglip=True)
                    restored = restored_siglip = None
                    if method == "dino_remaining":
                        seed_transform(seed)
                        restored, _, restored_siglip = self._forward(
                            inputs, proprio, observation, displayed, geometry,
                            clean_dino=clean_dino, capture_siglip=True)
                        require(siglip is not None and restored_siglip is not None
                                and torch.equal(siglip, restored_siglip),
                                "DINO scoring changed the current Wrist-SigLIP value")
                    patched_error = (((normal - target) / ranges).square().mean(dim=1)).squeeze(0)
                    restored_error = None
                    if restored is not None:
                        restored_error = (((restored - target) / ranges).square().mean(dim=1)).squeeze(0)
                    per_view.append({
                        "view": view,
                        "seed": seed,
                        "geometry": geometry,
                        "diverse": bool(view),
                        "pixels_sha256": patch_tensor_sha256(pixels),
                        "patched_error": [float(patched_error[j]) for j in cfg["train_axes"]],
                        "restored_error": (
                            [float(restored_error[j]) for j in cfg["train_axes"]]
                            if restored_error is not None else None
                        ),
                        "dino_delta_rms": float((normal_dino - clean_dino).square().mean().sqrt()),
                        "patched_siglip_retained": method == "dino_remaining",
                    })
                    self._scoring_pixels = None
            finally:
                self._scoring_pixels = None
                self.site, self.transfer_diverse = original_site, original_diverse
        axis_scores = []
        for offset, axis in enumerate(cfg["train_axes"]):
            patched = [view["patched_error"][offset] for view in per_view]
            patched_mean = float(np.mean(patched))
            if method == "dino_remaining":
                restored = [view["restored_error"][offset] for view in per_view]
                dino_score = float(np.mean([
                    max(0.0, r - p) * p for p, r in zip(patched, restored, strict=True)
                ]))
                restored_mean = float(np.mean(restored))
            else:
                dino_score = 0.0
                restored_mean = None
            axis_scores.append({
                "axis": int(axis),
                "patched_error": patched_mean,
                "restored_error": restored_mean,
                "action_remaining_score": patched_mean,
                "dino_remaining_score": dino_score,
            })
        require(all(np.isfinite(item["patched_error"]) and np.isfinite(item["action_remaining_score"])
                    and np.isfinite(item["dino_remaining_score"]) for item in axis_scores),
                "Nonfinite state-axis score")
        return {
            "axis_scores": axis_scores,
            "score_views": per_view,
            "score_view_count": 2,
            "score_full_forward_calls": 5 if method == "dino_remaining" else 3,
            "score_route": "two_legal_eot_axis_priority",
            "dino_score": float(np.mean([item["dino_remaining_score"] for item in axis_scores])),
            "patched_target_mse": float(np.mean([item["patched_error"] for item in axis_scores])),
            "restored_target_mse": (
                float(np.mean([item["restored_error"] for item in axis_scores]))
                if method == "dino_remaining" else None
            ),
            "dino_delta_rms": float(np.mean([view["dino_delta_rms"] for view in per_view])),
        }

    def _refresh_axis_scores(self, replay, patch, cfg, method, refresh_outer):
        state = _rng_state(patch)
        original_site, original_diverse = self.site, getattr(self, "transfer_diverse", False)
        try:
            self.transfer_diverse = False
            displayed = canonical_quantized_patch(patch).to(patch.device)
            return [
                dict(record, **self._axis_score_record(record, displayed, cfg, method, refresh_outer, index))
                for index, record in enumerate(replay)
            ]
        finally:
            self._scoring_pixels = None
            self.site, self.transfer_diverse = original_site, original_diverse
            _restore_rng(state, patch)

    def train_state_axis(self, clean, replay, cfg, method, source_hashes, data_hashes, model_load_s=0.):
        from dino_state_axis_common import METHODS, draw_record

        require(method in METHODS and len(clean) == 300 and len(replay) == 258,
                "State-axis method/source data drift")
        out = Path(cfg["output_root"]) / "training" / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        require(not previous or previous["signature"] == signature, "Existing run identity drift")
        if previous.get("status") == "COMPLETE":
            final = out / f"outer_{cfg['outer_iterations']:04d}.pt"
            require(previous["final_patch_sha256"] == sha256_file(final)
                    and previous["final_png_sha256"] == sha256_file(final.with_suffix(".png"))
                    and previous["resume_sha256"] == sha256_file(resume_path)
                    and previous["source_snapshot_sha256"] == sha256_file(out / "source_snapshot.zip"),
                    "Completed training artifacts changed")
            return
        seed_transform(cfg["seed"])
        patch = init_uada_patch(tuple(cfg["patch_shape"]), "cpu")
        initial_raw = patch.detach().clone()
        patch = patch.to(self.wrapper.device).detach().clone().requires_grad_(True)
        initial = patch.detach().clone()
        raw_hash, initial_hash = patch_tensor_sha256(initial_raw), patch_tensor_sha256(initial)
        manifest = {
            "experiment_id": cfg["experiment_id"],
            "method": method,
            "signature": signature,
            "config": cfg,
            "source": source_hashes,
            "data": data_hashes,
            "parameter_count": int(patch.numel()),
            "fresh_start_outer": 0,
            "initial_optimizer_state_empty": True,
            "optimizer_restarted_at_branch": False,
            "initialization": "seed42 random RGB; no feasible TV/HF correction",
            "raw_random_patch_sha256": raw_hash,
            "initial_patch_sha256": initial_hash,
            "feature_loss": False,
            "feature_replacement_hook_during_training": False,
            "appearance_regularization": False,
            "training_read_only_dense_capture": True,
            "training_action_forward": "inherited normal OFT; no feature replacement",
            "active_gradient_clipping": False,
            "scheduler_total_outer": cfg["outer_iterations"],
            "state_axis_policy": cfg["score_priority"],
            "evaluation_forward": "normal OFT, quantized single Wrist RGB patch",
        }
        manifest_path = out / "RUN_MANIFEST.json"
        if manifest_path.exists():
            require(json.loads(manifest_path.read_text()) == manifest, "Training manifest drift")
        else:
            atomic_json(manifest, manifest_path)
        snapshot = out / "source_snapshot.zip"
        if not snapshot.exists():
            temporary = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.tmp")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, digest in source_hashes.items():
                    require(sha256_file(name) == digest, f"Source changed before snapshot: {name}")
                    archive.write(name, arcname=name.lstrip("/"))
                archive.writestr("RUN_MANIFEST.json", json.dumps(manifest, indent=2))
            os.replace(temporary, snapshot)
        snapshot_hash = sha256_file(snapshot)
        optimizer, scheduler = build_uada_adamw(
            patch, cfg["learning_rate"], cfg["warmup_outer_iterations"],
            cfg["outer_iterations"], cfg["weight_decay"])
        rows, start, prior_elapsed, costs, history, score_after = [], 1, 0., {}, [], 0
        terminal_rounding_history = []
        replay = [dict(record) for record in replay]
        score_fields = (
            "axis_scores", "score_views", "score_view_count", "score_full_forward_calls",
            "score_route", "dino_score", "patched_target_mse", "restored_target_mse",
            "dino_delta_rms",
        )
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            require(saved["signature"] == signature and saved["method"] == method
                    and saved["source"] == source_hashes and saved["data"] == data_hashes,
                    "Resume source/data/method signature drift")
            validate_state(saved, cfg)
            require(torch.equal(saved["initial"], initial.cpu())
                    and torch.equal(saved["initial_raw"], initial_raw.cpu()),
                    "Resume initial RGB drift")
            with torch.no_grad():
                patch.copy_(saved["patch"].to(patch.device))
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            rows, start = saved["rows"], saved["outer"] + 1
            history, score_after = saved["score_refresh_history"], saved["score_after"]
            terminal_rounding_history = list(saved.get("terminal_rounding_history", []))
            prior_elapsed = max(float(saved["elapsed_s"]), float(previous.get("elapsed_s", 0.)))
            costs = dict(saved["costs"])
            for key, value in previous.get("costs", {}).items():
                costs[key] = max(costs.get(key, 0.), value)
            require(len(saved["replay_scoring"]) == len(replay), "Resume score coverage drift")
            replay = [dict(record, **score)
                      for record, score in zip(replay, saved["replay_scoring"], strict=True)]
            _restore_rng(saved, patch)
            _discard_partial_artifacts(out, saved)
            atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
        else:
            require(not previous, "Existing status without atomic resume")
            require(not optimizer.state, "Fresh Adam state must be empty")
            _save_checkpoint(out / "initial_raw.pt", initial_raw, {
                "method": method,
                "signature": signature,
                "raw_random_patch_sha256": raw_hash,
                "scientific_feasibility_required": False,
            })
            _save_checkpoint(out / "outer_0000.pt", initial, {
                "method": method,
                "signature": signature,
                "total_outer": 0,
                "total_optimizer_updates": 0,
                "raw_random_patch_sha256": raw_hash,
                "initial_patch_sha256": initial_hash,
                **_structure(initial, initial),
            })
        costs["model_load_s"] = costs.get("model_load_s", 0.) + model_load_s
        low = torch.tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        started = time.monotonic()
        completed_outer, attempted_outer = start - 1, start - 1

        def elapsed():
            return prior_elapsed + model_load_s + time.monotonic() - started

        def bump(key, number=1):
            costs[key] = costs.get(key, 0) + number

        def scoring_payload():
            return [{key: record[key] for key in score_fields if key in record} for record in replay]

        def save_resume():
            payload = {
                "signature": signature,
                "config": cfg,
                "source": source_hashes,
                "data": data_hashes,
                "method": method,
                "outer": completed_outer,
                "patch": patch.detach().cpu(),
                "initial": initial.cpu(),
                "initial_raw": initial_raw.cpu(),
                "raw_random_patch_sha256": raw_hash,
                "initial_patch_sha256": initial_hash,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rows": rows,
                "costs": dict(costs),
                "elapsed_s": elapsed(),
                "score_after": score_after,
                "score_refresh_history": history,
                "terminal_rounding_history": list(terminal_rounding_history),
                "replay_scoring": scoring_payload(),
                **_rng_state(patch),
            }
            validate_state(payload, cfg)
            atomic_torch(payload, resume_path)

        def progress(status, **extra):
            return {
                "status": status,
                "method": method,
                "signature": signature,
                "outer": completed_outer,
                "optimizer_updates": completed_outer * cfg["inner_loop"],
                "elapsed_s": elapsed(),
                "costs": dict(costs),
                "score_refresh_outer": [item["outer"] for item in history],
                "raw_random_patch_sha256": raw_hash,
                "initial_patch_sha256": initial_hash,
                "source_snapshot_sha256": snapshot_hash,
                "terminal_rounding_history": list(terminal_rounding_history),
                "cost_boundary": "all executed training, scoring, validation and model loads",
                "visual_review": "NOT_HUMAN_APPROVED",
                **extra,
            }

        try:
            if not resume_path.exists():
                save_resume()
            rng = _rng_state(patch)
            try:
                self.transfer_diverse = False
                alignment = self._timed(costs, "runtime_validation_s", lambda:
                    self._validate_runtime(clean[0], patch.detach(), cfg))
            finally:
                _restore_rng(rng, patch)
                self.transfer_diverse = False
            atomic_json(alignment, out / "RUNTIME_ALIGNMENT.json")
            for outer in range(start, cfg["outer_iterations"] + 1):
                attempted_outer = outer
                if outer - 1 in cfg["refresh_after_outer"]:
                    refresh_outer = outer - 1
                    score_started = time.monotonic()
                    replay = self._timed(costs, "score_refresh_s", lambda:
                        self._refresh_axis_scores(replay, patch, cfg, method, refresh_outer))
                    calls = sum(int(record.get("score_full_forward_calls", 0)) for record in replay)
                    bump("score_refresh_count")
                    bump("score_records", len(replay))
                    bump("score_full_forward_calls", calls)
                    score_after = refresh_outer
                    history.append({
                        "outer": refresh_outer,
                        "patch_tensor_sha256": patch_tensor_sha256(patch),
                        "quantized_patch_sha256": patch_tensor_sha256(canonical_quantized_patch(patch)),
                        "records": len(replay),
                        "full_forward_calls": calls,
                        "elapsed_s": time.monotonic() - score_started,
                    })
                    atomic_json({**history[-1], "method": method, "records": scoring_payload()},
                                out / f"scores_after_{refresh_outer:04d}.json")
                before_outer = patch.detach().clone()
                record, origin, index, base_index, axis = draw_record(clean, replay, method, outer, cfg)
                observations = record["observations"] if origin == "replay" else [record["observation"]]
                require(origin != "replay" or len(observations) == 3,
                        "Three-decision replay clip drift")
                prepared = []
                for observation in observations:
                    inputs, proprio = self._base_inputs({
                        "observation": observation,
                        "task_name": record["task_name"],
                    })
                    with torch.no_grad():
                        action, _, _ = self._timed(costs, "clean_prepare_s", lambda:
                            self._forward(inputs, proprio, observation, None, False))
                        bump("training_clean_forward_calls")
                        target = torch.where(action >= (low + high) / 2, low, high).detach()
                    prepared.append((inputs, proprio, observation, target))
                lr = float(optimizer.param_groups[0]["lr"])
                sums = {
                    "main_loss": 0., "action_gradient_norm": 0., "patch_gradient_norm": 0.,
                    "rgb_step_mean_abs_sum": 0., "rgb_step_max_abs": 0.,
                    "effective_rgb_updates": 0,
                }
                terminal_rounding_evidence = []
                for inner in range(cfg["inner_loop"]):
                    seed_transform(cfg["seed"] * 10_000_000 + outer * 10_000 + inner)
                    optimizer.zero_grad(set_to_none=True)
                    inputs, proprio, observation, target = prepared[inner % len(prepared)]
                    geometry = cfg["geometry"] and inner >= cfg["inner_loop"] // 2
                    self.transfer_diverse = bool(outer > cfg["bootstrap_outer"] and geometry)
                    try:
                        action, _, _ = self._timed(costs, "patched_forward_s", lambda:
                            self._forward(inputs, proprio, observation, patch, geometry))
                    finally:
                        self.transfer_diverse = False
                    bump("training_patched_forward_calls")
                    main = F.mse_loss(action[..., [axis]], target[..., [axis]])
                    require(bool(torch.isfinite(main)), "Nonfinite training loss")
                    grad = self._timed(costs, "action_backward_s", lambda:
                        torch.autograd.grad(main, patch)[0])
                    bump("training_action_backward_calls")
                    require(bool(torch.isfinite(grad).all())
                            and (outer != 1 or inner != 0 or float(grad.norm()) > 1e-12),
                            "Nonfinite/zero real-model action RGB gradient")
                    patch.grad = grad
                    before_step = patch.detach().clone()
                    self._timed(costs, "optimizer_s", lambda:
                        step_uada_patch(patch, optimizer, self.wrapper.vla))
                    bump("actual_optimizer_updates")
                    step_delta = (patch.detach() - before_step).abs()
                    step_max = float(step_delta.max())
                    sums["rgb_step_mean_abs_sum"] += float(step_delta.mean())
                    sums["rgb_step_max_abs"] = max(sums["rgb_step_max_abs"], step_max)
                    sums["effective_rgb_updates"] += int(step_max > 0.)
                    if cfg["outer_iterations"] - outer + 1 <= 2 and step_max == 0.:
                        evidence = _terminal_float32_rounding_evidence(before_step, optimizer, grad, grad)
                        if evidence is not None:
                            terminal_rounding_evidence.append(evidence)
                    sums["main_loss"] += float(main.detach()) / cfg["inner_loop"]
                    norm = float(grad.norm())
                    sums["action_gradient_norm"] += norm / cfg["inner_loop"]
                    sums["patch_gradient_norm"] += norm / cfg["inner_loop"]
                scheduler.step()
                structure = _structure(patch, initial)
                delta = (patch.detach() - before_outer).abs()
                movement = {
                    "rgb_outer_mean_abs": float(delta.mean()),
                    "rgb_outer_max_abs": float(delta.max()),
                    "rgb_outer_changed_pixel_fraction": float((delta.amax(0) > 0).float().mean()),
                    "zero_lr_warmup_outer": lr == 0.,
                    "zero_lr_first_step": outer == 1 and lr == 0.,
                }
                if lr != 0. and movement["rgb_outer_max_abs"] == 0.:
                    require(cfg["outer_iterations"] - outer + 1 <= 2
                            and sums["effective_rgb_updates"] == 0
                            and len(terminal_rounding_evidence) == cfg["inner_loop"],
                            "Nonzero LR but unchanged complete outer RGB")
                    terminal_rounding_history.append({
                        "outer": outer,
                        "learning_rate": lr,
                        "evidence_count": len(terminal_rounding_evidence),
                        "max_intended_update_abs": max(
                            item["max_intended_update_abs"] for item in terminal_rounding_evidence),
                        "reason": "terminal_float32_rounding",
                        "effective_rgb_updates": 0,
                    })
                # Fixed schema avoids NaN when clean and replay rows share Parquet.
                # Availability flags distinguish finite placeholders from scores.
                selected = {
                    "selected_score_available": False,
                    "selected_restored_error_available": False,
                    "selected_action_remaining_score": 0.,
                    "selected_dino_remaining_score": 0.,
                    "selected_patched_error": 0.,
                    "selected_restored_error": 0.,
                }
                if "axis_scores" in record:
                    for item in record["axis_scores"]:
                        if int(item["axis"]) == int(axis):
                            selected.update({
                                "selected_score_available": True,
                                "selected_action_remaining_score": float(item["action_remaining_score"]),
                                "selected_dino_remaining_score": float(item["dino_remaining_score"]),
                                "selected_patched_error": float(item["patched_error"]),
                            })
                            if item["restored_error"] is not None:
                                selected["selected_restored_error"] = float(item["restored_error"])
                                selected["selected_restored_error_available"] = True
                            break
                rows.append({
                    "method": method,
                    "outer_iteration": outer,
                    "optimizer_updates": outer * cfg["inner_loop"],
                    "origin": origin,
                    "record_index": index,
                    "base_record_index": base_index,
                    "task_id": int(record["task_id"]),
                    "state_index": int(record["state_index"]),
                    "action_axis": int(axis),
                    "dino_score": float(record.get("dino_score", 0.)),
                    "learning_rate": lr,
                    "score_after_outer": score_after,
                    "elapsed_s": elapsed(),
                    **selected,
                    **sums,
                    **movement,
                    **structure,
                })
                if outer in cfg["checkpoint_outer_iterations"]:
                    _save_checkpoint(out / f"outer_{outer:04d}.pt", patch, {
                        "method": method,
                        "signature": signature,
                        "total_outer": outer,
                        "total_optimizer_updates": outer * cfg["inner_loop"],
                        "raw_random_patch_sha256": raw_hash,
                        "initial_patch_sha256": initial_hash,
                        **structure,
                    })
                completed_outer = outer
                save_resume()
                atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
                atomic_json(progress("RUNNING", **structure), status_path)
                if outer % 50 == 0 or outer == cfg["outer_iterations"]:
                    print(f"STATE_AXIS method={method} outer={outer} axis={axis} "
                          f"loss={sums['main_loss']:.6f} tv={structure['tv']:.6f}", flush=True)
            final = out / f"outer_{cfg['outer_iterations']:04d}.pt"
            atomic_json(progress("COMPLETE", final_patch_sha256=sha256_file(final),
                                 final_png_sha256=sha256_file(final.with_suffix(".png")),
                                 resume_sha256=sha256_file(resume_path), **_structure(patch, initial)),
                        status_path)
        except BaseException as error:
            if getattr(self, "runtime_alignment", {}).get("status") == "CHECKING":
                atomic_json({**self.runtime_alignment, "status": "FAILED", "error": repr(error)},
                            out / "RUNTIME_ALIGNMENT.json")
            if resume_path.exists():
                committed = torch.load(resume_path, map_location="cpu", weights_only=False)
                completed_outer = committed["outer"]
            atomic_json(progress("FAILED", attempted_outer=attempted_outer, error=repr(error),
                                 resume=str(resume_path)), status_path)
            raise

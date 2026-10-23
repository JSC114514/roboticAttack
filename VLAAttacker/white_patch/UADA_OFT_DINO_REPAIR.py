"""Isolated Exploring-derived Wrist-DINO repair and RGB appearance training.

The complete shared 250-outer prefix is counted in the 1000-outer budget;
Adam moments and the original 2000-outer cosine schedule are carried forward.
Only this trainer installs causal SigLIP hooks. Deployment uses normal OFT.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_DINO_TRANSFER import ExploringOFTDINOTransferAttacker
from white_patch.UADA_OFT_CORRESPONDENCE import run_signature, source_manifest as base_sources
from white_patch.UADA_OFT_DINO import (atomic_json, atomic_torch, atomic_parquet,
    save_patch, patch_metrics, require, sha256_file)
from white_patch.UADA_OFT_SOURCE_COPY import init_uada_patch, build_uada_adamw, step_uada_patch
from white_patch.UADA_OFT_ILA import seed_transform
from white_patch.oft_dino_repair_objectives import (
    charbonnier_tv, high_frequency_energy, repair_rgb)

CAUSAL_METHODS = {"dino_causal_clear", "dino_repair_clear"}
SCORE_FIELDS = ("dino_score", "dino_delta_rms", "patched_target_mse",
                "restored_target_mse", "dino_weight", "shuffled_weight")


def source_manifest(cfg, wrapper=None):
    from dino_repair_common import CONFIG
    sources = base_sources(cfg, wrapper)
    root = Path(__file__).resolve().parents[2]
    paths = [CONFIG, root / "scripts/dino_transfer_common.py",
             root / "scripts/dino_window_common.py", root / "scripts/dino_replay_common.py",
             root / "scripts/collect_exploring_oft_dino_replay.py",
             root.parent / "oft_ramp_recovery_aware/scripts/runtime.py",
             root.parent / "oft_ramp_recovery_aware/scripts/common.py",
             root.parent / "oft_mgrb_lite_low_cost/scripts/model_runtime.py"]
    paths += list((root / "scripts").glob("*dino_repair*.py"))
    paths += list((root / "slurm/exploring_oft_dino").glob("*repair*.sbatch"))
    paths += list(root.glob("submit_exploring_oft_dino_repair*.sh"))
    paths += [root / "research_runs/wrist_dino_repair_v1/PLAN.md"]
    for path in paths:
        sources[str(path.resolve())] = sha256_file(path)
    # Preserve the actual normal-evaluation and bootstrap dependencies in the
    # source archive as well as pinning them in the independent CPU preflight.
    for name, digest in cfg["frozen_eval_sources"].items():
        require(sha256_file(name) == digest, f"Frozen evaluation source changed: {name}")
        sources[str(Path(name).resolve())] = digest
    return sources


def _config_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def validate_state(saved, cfg, *, is_prefix=False):
    # prefix_state() checks its config against the immutable OLD config. A new
    # branch intentionally has a different identity/config; Adam still matches.
    require(is_prefix or saved["config"] == cfg, "Repair resume config drift")
    require(len(saved["rows"]) == saved["outer"], "Resume row/outer mismatch")
    expected = saved["outer"] * cfg["inner_loop"]
    require(not saved["rows"] or saved["rows"][-1]["optimizer_updates"] == expected,
            "Wrong cumulative patch update count")
    states = saved["optimizer"]["state"]
    if expected:
        require(len(states) == 1 and all(int(value["step"]) == expected for value in states.values()),
                "Missing/reset Adam moments or update step")
    require(saved["scheduler"]["last_epoch"] == saved["outer"], "Scheduler reset/drift")
    value = saved["patch"]
    require(tuple(value.shape) == tuple(cfg["patch_shape"]) and bool(torch.isfinite(value).all())
            and bool((value >= 0).all() and (value <= 1).all()), "Invalid resumed RGB")


@contextmanager
def frozen_model(wrapper):
    """Avoid model-weight graphs during eta PGD, restoring flags on failure."""
    unique = {}
    for name in ("vla", "action_head", "proprio_projector", "noisy_action_projector"):
        module = getattr(wrapper, name, None)
        if module is not None and hasattr(module, "parameters"):
            for parameter in module.parameters():
                unique[id(parameter)] = parameter
    flags = [(parameter, parameter.requires_grad) for parameter in unique.values()]
    try:
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


class ExploringOFTDINORepairAttacker(ExploringOFTDINOTransferAttacker):
    def _sample_rgb(self, observation, patch, geometry):
        # Exactly one Exploring EOT draw. The same live RGB graph is retained
        # for the outer gradient and a detached view is used by eta PGD.
        self.transfer_diverse = bool(geometry)
        try:
            rgb, support = self._paste_with_support(observation["wrist_image"], patch, geometry)
            require(tuple(rgb.shape) == (3, 224, 224) and tuple(support.shape) == (224, 224),
                    "This frozen renderer requires 224 RGB/support")
            return rgb, support
        finally:
            self.transfer_diverse = False

    def _pixels_from_rgb(self, observation, wrist_rgb):
        from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor
        front = torch.as_tensor(np.asarray(observation["full_image"]), device=self.wrapper.device,
            dtype=torch.float32).permute(2, 0, 1)[None] / 255
        processor = self.wrapper.processor.image_processor
        front_values = _processor_normalize_tensor(front, processor, center_crop=True)
        wrist_values = _processor_normalize_tensor(wrist_rgb[None], processor, center_crop=True)
        require(tuple(wrist_values.shape) == (1, 6, 224, 224), "Wrist processor layout drift")
        full = torch.cat((front_values, wrist_values), dim=1).to(self.wrapper.dtype)
        require(tuple(full.shape) == (1, 12, 224, 224), "Front/Wrist layout drift")
        return full

    def _features_rgb(self, observation, rgb, costs=None):
        if costs is not None:
            costs["direct_dino_forward_calls"] = costs.get("direct_dino_forward_calls", 0) + 1
        # Same full allocation, dtype and non-contiguous view as full OFT.
        return self._direct_dino(self._pixels_from_rgb(observation, rgb)[:, 6:9])

    def _forward_rgb(self, inputs, proprio, observation, rgb, *, clean_siglip=None,
                     capture_siglip=False, costs=None):
        from src.hooks.manager import HookSpec, TensorHooks
        model_inputs = dict(inputs)
        model_inputs["pixel_values"] = self._pixels_from_rgb(observation, rgb)
        roots = {"vla": self.wrapper.vla, "action_head": self.wrapper.action_head}
        dino = HookSpec(path="vla.vision_backbone.featurizer", hook="output", call_index=1)
        siglip = HookSpec(path="vla.vision_backbone.fused_featurizer", hook="output", call_index=1)
        with ExitStack() as stack:
            if clean_siglip is not None:
                stack.enter_context(TensorHooks(roots, [siglip], mode="replace",
                    values=[clean_siglip.detach().to(self.wrapper.device, dtype=self.wrapper.dtype)]))
            # Capture follows replacement: validation sees the consumed value.
            specs = [dino, siglip] if capture_siglip else [dino]
            hooks = stack.enter_context(TensorHooks(roots, specs, mode="capture", retain_grad=True))
            if costs is not None:
                costs["action_forward_calls"] = costs.get("action_forward_calls", 0) + 1
                costs["full_model_dino_forward_calls"] = costs.get("full_model_dino_forward_calls", 0) + 2
            _, hidden = self.wrapper.vla.predict_action(**model_inputs,
                unnorm_key=self.wrapper.gen_cfg.unnorm_key, do_sample=False, proprio=proprio,
                proprio_projector=self.wrapper.proprio_projector,
                noisy_action_projector=self.wrapper.noisy_action_projector,
                action_head=self.wrapper.action_head, use_film=False)
        head = getattr(self.wrapper.action_head, "module", self.wrapper.action_head)
        action = head.predict_action(hidden).reshape(1, 8, 7).float()
        dense = hooks.captured[0].float()
        require(tuple(action.shape) == (1, 8, 7) and bool(torch.isfinite(action).all())
                and bool(torch.isfinite(dense).all()), "Invalid RGB OFT output")
        return action, dense, hooks.captured[1] if capture_siglip else None

    def _repair_eta(self, observation, rgb, support, clean_dino, cfg, costs):
        with frozen_model(self.wrapper):
            eta, stats = repair_rgb(rgb, support, clean_dino,
                lambda value: self._features_rgb(observation, value, costs),
                steps=cfg["repair_steps"], epsilon=cfg["repair_epsilon"],
                step_size=cfg["repair_step_size"])
        costs["repair_pgd_updates"] = costs.get("repair_pgd_updates", 0) + stats["repair_pgd_updates"]
        costs["repair_dino_forward_calls"] = costs.get("repair_dino_forward_calls", 0) + stats["repair_dino_forward_calls"]
        return eta.detach(), stats

    def _validate_repair_runtime(self, record, reference, cfg, method, costs):
        """Actual checkpoint interfaces, hook cleanup and bounded CUDA checks."""
        inputs, proprio = self._base_inputs(record)
        obs = record["observation"]
        low = torch.as_tensor(cfg["action_head_q01"], device=reference.device).reshape(1, 1, 7)
        high = torch.as_tensor(cfg["action_head_q99"], device=reference.device).reshape(1, 1, 7)
        modules = [self.wrapper.vla.vision_backbone.featurizer,
                   self.wrapper.vla.vision_backbone.fused_featurizer]
        hook_counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]
        with torch.no_grad():
            clean_action, clean_dino, clean_siglip = self._forward(
                inputs, proprio, obs, None, False, capture_siglip=True)
        costs["runtime_action_forward_calls"] = costs.get("runtime_action_forward_calls", 0) + 1
        require(clean_siglip is not None, "Missing clean Wrist SigLIP")
        patch = reference.detach().clone().requires_grad_(True)
        seed_transform(cfg["seed"] * 10_000_000 + 8_100_000)
        rgb, support = self._sample_rgb(obs, patch, True)
        actual_inputs = []
        module = self.wrapper.vla.vision_backbone.featurizer
        calls = 0
        def capture_wrist(_module, args):
            nonlocal calls
            if calls == 1:
                actual_inputs.append(args[0].detach())
            calls += 1
        handle = module.register_forward_pre_hook(capture_wrist)
        try:
            action, dense, restored = self._forward_rgb(inputs, proprio, obs, rgb,
                clean_siglip=clean_siglip if method in CAUSAL_METHODS else None,
                capture_siglip=True, costs=costs)
        finally:
            handle.remove()
        pixels = self._pixels_from_rgb(obs, rgb)[:, 6:9]
        require(calls == 2 and len(actual_inputs) == 1 and torch.equal(actual_inputs[0], pixels.detach()),
                "Direct DINO input differs from actual OFT wrist input")
        with torch.no_grad():
            direct = self._features_rgb(obs, rgb, costs)
        require(torch.allclose(direct, dense.detach(), rtol=1e-4, atol=1e-4),
                "Direct DINO output differs from actual OFT wrist output")
        target = torch.where(clean_action >= (low + high) / 2, low, high).detach()
        gradient = torch.autograd.grad(F.mse_loss(action[..., cfg["maskidx"]],
            target[..., cfg["maskidx"]]), patch, retain_graph=True)[0]
        require(bool(torch.isfinite(gradient).all()), "Nonfinite OFT interface action gradient")
        training_gradient_norm = float(gradient.norm())
        probe_kind = "training_target"
        if training_gradient_norm <= 1e-12:
            # BF16 can represent a reached q-boundary exactly. The opposite
            # boundary tests the same computational interface without changing
            # any training target, calibration sample or training objective.
            target = torch.where(clean_action >= (low+high)/2, high, low).detach()
            gradient = torch.autograd.grad(F.mse_loss(action[..., cfg["maskidx"]],
                target[..., cfg["maskidx"]]), patch, retain_graph=True)[0]
            probe_kind = "opposite_boundary_interface_probe"
        require(bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 1e-12,
                "Both runtime action interface probes have zero/nonfinite RGB gradient")
        report = {"status": "PASS", "pixel_exact": True, "feature_close": True,
                  "gradient_norm": float(gradient.norm()), "single_eot": True,
                  "training_gradient_norm": training_gradient_norm, "gradient_probe": probe_kind,
                  "actual_pixel_shape": list(pixels.shape), "actual_pixel_stride": list(pixels.stride()),
                  "actual_pixel_storage_offset": pixels.storage_offset(), "method": method}
        if method in CAUSAL_METHODS:
            require(torch.allclose(restored, clean_siglip.to(restored), rtol=1e-5, atol=1e-5),
                    "Clean Wrist SigLIP restoration failed")
            zero = torch.zeros_like(rgb).detach()
            action_zero, _, restored_zero = self._forward_rgb(inputs, proprio, obs,
                (rgb + zero).clamp(0, 1), clean_siglip=clean_siglip,
                capture_siglip=True, costs=costs)
            gradient_zero = torch.autograd.grad(F.mse_loss(action_zero[..., cfg["maskidx"]],
                target[..., cfg["maskidx"]]), patch)[0]
            relative = float((gradient-gradient_zero).norm() / gradient.norm().clamp_min(1e-12))
            require(torch.allclose(action_zero, action, rtol=.05, atol=1e-4)
                    and bool(torch.isfinite(gradient_zero).all()) and float(gradient_zero.norm()) > 1e-12
                    and relative <= .1, "Eta=0 causal forward/gradient mismatch")
            require(torch.allclose(restored_zero, clean_siglip.to(restored_zero), rtol=1e-5, atol=1e-5),
                    "Eta=0 SigLIP closure failed")
            report.update(clean_siglip_restored=True, eta_zero_action_max_error=float(
                (action_zero-action).abs().max()), eta_zero_gradient_relative_l2=relative)
        if method == "dino_repair_clear":
            eta, repair = self._repair_eta(obs, rgb, support, clean_dino, cfg, costs)
            require(not eta.requires_grad and not bool((eta * (support > 0)[None]).any()),
                    "Eta leaked into patch support or outer gradient")
            report.update(repair)
        require(hook_counts == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules],
                "Leaked DINO/SigLIP runtime hook")
        # A subsequent normal call must be independent of clean-SigLIP hooks.
        with torch.no_grad():
            normal, _, _ = self._forward_rgb(inputs, proprio, obs, rgb.detach(), costs=costs)
            normal_repeat, _, _ = self._forward_rgb(inputs, proprio, obs, rgb.detach(), costs=costs)
        require(torch.allclose(normal, normal_repeat, rtol=.05, atol=1e-4),
                "Normal OFT deployment forward changed after hook removal")
        report.update(training_hooks_removed=True, normal_deployment_repeat_close=True)
        return report

    def calibrate_appearance(self, clean, cfg, source_hashes, data_hashes, model_load_s=0.):
        from dino_repair_common import prefix_state
        path = Path(cfg["output_root"]) / "reports/APPEARANCE_CALIBRATION.json"
        if path.exists():
            report = json.loads(path.read_text())
            require(report["status"] == "COMPLETE" and report["config"] == cfg
                    and report["source"] == source_hashes and report["data"] == data_hashes,
                    "Immutable appearance calibration identity drift")
            return report
        saved, digest, _ = prefix_state(cfg)
        require(data_hashes["prefix_resume"] == digest, "Calibration prefix hash changed")
        validate_state(saved, cfg, is_prefix=True)
        first_by_task = {}
        for index, record in enumerate(clean):
            first_by_task.setdefault(int(record["task_id"]), (index, record))
        require(set(first_by_task) == set(range(10)), "Calibration needs ten Object tasks")
        patch = saved["patch"].to(self.wrapper.device).detach().clone().requires_grad_(True)
        low = torch.tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        costs, gradient_norms = {"model_load_s": model_load_s}, []
        started = time.monotonic()
        alignment = self._validate_repair_runtime(first_by_task[0][1], patch, cfg, "clear", costs)
        for task_id, (index, record) in sorted(first_by_task.items()):
            inputs, proprio = self._base_inputs(record)
            obs = record["observation"]
            with torch.no_grad():
                action, _, _ = self._forward(inputs, proprio, obs, None, False)
                target = torch.where(action >= (low+high)/2, low, high).detach()
            costs["calibration_clean_action_forward_calls"] = costs.get("calibration_clean_action_forward_calls", 0) + 1
            seed_transform(cfg["seed"] * 10_000_000 + 8_200_000 + task_id)
            rgb, _ = self._sample_rgb(obs, patch, False)
            action, _, _ = self._forward_rgb(inputs, proprio, obs, rgb, costs=costs)
            main = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
            tv = charbonnier_tv(patch)
            hf = high_frequency_energy(patch, cfg["hf_sigma"], cfg["hf_kernel"])
            norms = [float(torch.autograd.grad(term, patch, retain_graph=True)[0].norm())
                     for term in (main, tv, hf)]
            require(all(np.isfinite(value) for value in norms),
                    "Nonfinite calibration action or appearance gradient")
            gradient_norms.append({"task_id": task_id, "record_index": index,
                "state_index": int(record["state_index"]), "action": norms[0], "tv": norms[1], "hf": norms[2]})
        medians = {key: float(np.median([row[key] for row in gradient_norms]))
                   for key in ("action", "tv", "hf")}
        require(all(value > 1e-12 for value in medians.values()),
                "Zero calibration median action or appearance gradient")
        ratio = cfg["appearance_grad_ratio"]
        report = {"status": "COMPLETE", "experiment_id": cfg["experiment_id"], "config": cfg,
            "config_sha256": _config_sha(cfg), "source": source_hashes, "data": data_hashes,
            "prefix_resume_sha256": digest, "calibration_outer": saved["outer"],
            "record_policy": "first clean source record per Object task_id 0..9",
            "gradient_norms": gradient_norms, "median_gradient_norms": medians,
            "zero_action_record_count": sum(row["action"] <= 1e-12 for row in gradient_norms),
            "lambda_tv": ratio * medians["action"] / medians["tv"],
            "lambda_hf": ratio * medians["action"] / medians["hf"],
            "appearance_grad_ratio": ratio, "costs": costs,
            "calibration_elapsed_s": time.monotonic()-started+model_load_s,
            "prefix_elapsed_s": float(saved["elapsed_s"]), "runtime_alignment": alignment}
        require(not path.exists(), "Calibration already exists; refusing overwrite")
        atomic_json(report, path)
        return report

    def train_repair(self, clean, replay, cfg, method, source_hashes, data_hashes,
                     model_load_s=0.):
        from dino_repair_common import METHODS, draw_record, prefix_state, load_calibration
        require(method in METHODS and len(clean) == 300 and len(replay) > 0,
                "Repair method or Object bank drift")
        end = cfg["outer_iterations"]
        calibration = load_calibration(cfg)
        calibration_path = Path(cfg["output_root"]) / "reports/APPEARANCE_CALIBRATION.json"
        require(calibration["config"] == cfg and calibration["source"] == source_hashes
                and calibration["data"] == data_hashes and calibration["status"] == "COMPLETE",
                "Training/calibration identity drift")
        lambdas = [float(calibration[key]) for key in ("lambda_tv", "lambda_hf")]
        require(all(np.isfinite(value) and value > 0 for value in lambdas), "Invalid calibrated appearance weights")
        out = Path(cfg["output_root"]) / "training" / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        require(not previous or previous["signature"] == signature, "Existing repair signature drift")
        if previous.get("status") == "COMPLETE":
            require(previous["final_patch_sha256"] == sha256_file(out / f"outer_{end:04d}.pt")
                and previous["final_png_sha256"] == sha256_file(out / f"outer_{end:04d}.png")
                and previous["resume_sha256"] == sha256_file(resume_path), "Completed repair output drift")
            print(f"ALREADY_COMPLETE method={method}", flush=True)
            return
        manifest = {"experiment_id": cfg["experiment_id"], "config": cfg, "method": method,
            "signature": signature, "source": source_hashes, "data": data_hashes,
            "calibration_sha256": sha256_file(calibration_path), "lambda_tv": lambdas[0], "lambda_hf": lambdas[1],
            "initialization": "shared seeded-random RGB 250-outer prefix, fully counted",
            "parameter_count": 7500, "optimizer_restarted_at_branch": False,
            "scheduler_total_outer": cfg["scheduler_total_outer"], "feature_loss": False,
            "causal_training_forward": method in CAUSAL_METHODS, "repair_inner_loop": method == "dino_repair_clear",
            "evaluation_forward": "normal OFT, quantized Wrist RGB patch only"}
        manifest_path = out / "RUN_MANIFEST.json"
        if manifest_path.exists():
            require(json.loads(manifest_path.read_text()) == manifest, "Repair manifest drift")
        else:
            atomic_json(manifest, manifest_path)
        snapshot = out / "source_snapshot.zip"
        if not snapshot.exists():
            with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for path, digest in source_hashes.items():
                    require(sha256_file(path) == digest, f"Source changed during snapshot: {path}")
                    archive.write(path, arcname=path.lstrip("/"))
                archive.writestr("RUN_MANIFEST.json", json.dumps(manifest, indent=2))
        snapshot_sha = sha256_file(snapshot)
        seed_transform(cfg["seed"])
        patch = init_uada_patch(tuple(cfg["patch_shape"]), self.wrapper.device)
        initial = patch.detach().clone()
        optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"],
            cfg["warmup_outer_iterations"], cfg["scheduler_total_outer"], cfg["weight_decay"])
        is_prefix = not resume_path.exists()
        if not is_prefix:
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            require(saved["signature"] == signature and saved["method"] == method
                    and saved["source_snapshot_sha256"] == snapshot_sha, "Resume identity/source snapshot drift")
        else:
            require(not previous, "Existing status has no complete resume")
            saved, digest, _ = prefix_state(cfg)
            require(data_hashes["prefix_resume"] == digest, "Prefix hash drift after manifest")
        validate_state(saved, cfg, is_prefix=is_prefix)
        require(torch.equal(saved["initial"], initial.cpu()) and cfg["bootstrap_outer"] <= saved["outer"] <= end,
                "Initial RGB/prefix/budget drift")
        with torch.no_grad():
            patch.copy_(saved["patch"].to(patch.device))
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rows, start = copy.deepcopy(saved["rows"]), saved["outer"] + 1
        prior_elapsed = max(float(saved["elapsed_s"]), float(previous.get("elapsed_s", 0.)))
        prefix_elapsed = float(saved["elapsed_s"]) if is_prefix else float(saved["prefix_elapsed_s"])
        costs = dict(saved["costs"])
        for key, value in previous.get("costs", {}).items():
            costs[key] = max(costs.get(key, 0), value)
        if is_prefix:
            costs["prefix_patch_updates"] = saved["outer"] * cfg["inner_loop"]
        costs["model_load_s"] = costs.get("model_load_s", 0.) + model_load_s
        replay = [dict(record) for record in replay]
        score_after = saved.get("score_after", cfg["bootstrap_outer"])
        scoring = saved.get("replay_scoring", [])
        if scoring:
            require(len(scoring) == len(replay), "Resume replay-score coverage drift")
            replay = [dict(record, **score) for record, score in zip(replay, scoring, strict=True)]
        torch.set_rng_state(saved["torch_rng"])
        if patch.is_cuda:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if is_prefix:
            for checkpoint, value in ((0, initial), (cfg["bootstrap_outer"], patch)):
                save_patch(out / f"outer_{checkpoint:04d}.pt", value,
                    {"method": method, "signature": signature, "total_outer": checkpoint,
                     "total_optimizer_updates": checkpoint * cfg["inner_loop"]})
        low = torch.tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        started = time.monotonic()
        outer, completed = start - 1, start - 1
        def elapsed():
            return prior_elapsed + model_load_s + time.monotonic()-started
        def save_resume():
            payload = {"signature": signature, "config": cfg, "method": method, "outer": completed,
                "source_snapshot_sha256": snapshot_sha,
                "patch": patch.detach().cpu(), "initial": initial.cpu(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rows": rows, "costs": dict(costs), "elapsed_s": elapsed(), "prefix_elapsed_s": prefix_elapsed,
                "score_after": score_after, "replay_scoring": [
                    {key: record[key] for key in SCORE_FIELDS if key in record} for record in replay],
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if patch.is_cuda else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
            validate_state(payload, cfg)
            atomic_torch(payload, resume_path)
        def progress(status, **extra):
            return {"status": status, "method": method, "signature": signature, "outer": completed,
                "optimizer_updates": completed * cfg["inner_loop"], "elapsed_s": elapsed(),
                "prefix_elapsed_s": prefix_elapsed, "costs": dict(costs),
                "calibration_elapsed_s": calibration.get("calibration_elapsed_s", 0),
                "source_snapshot_sha256": snapshot_sha,
                "cost_boundary": "training includes shared prefix, model loads and failed attempts; calibration separate",
                "visual_review": "AUTO_VALIDATION", **extra}
        try:
            if is_prefix:
                save_resume()
            # Validation must not perturb restored RNG/sample/EOT order.
            rng = (torch.get_rng_state(), np.random.get_state(), random.getstate(),
                   torch.cuda.get_rng_state_all() if patch.is_cuda else [])
            try:
                alignment = self._timed(costs, "runtime_validation_s", lambda:
                    self._validate_repair_runtime(clean[0], patch.detach(), cfg, method, costs))
            finally:
                torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
                if patch.is_cuda:
                    torch.cuda.set_rng_state_all(rng[3])
            atomic_json(alignment, out / "RUNTIME_ALIGNMENT.json")
            for outer in range(start, end + 1):
                if method != "uniform_clear" and outer-1 in cfg["refresh_after_outer"]:
                    replay = self._timed(costs, "score_refresh_s", lambda:
                        self._refresh_scores(replay, patch, cfg))
                    score_after = outer-1
                    costs["score_refresh_records"] = costs.get("score_refresh_records", 0)+len(replay)
                    costs["score_refresh_action_forward_calls"] = costs.get("score_refresh_action_forward_calls", 0)+3*len(replay)
                    costs["full_model_dino_forward_calls"] = costs.get("full_model_dino_forward_calls", 0)+6*len(replay)
                    atomic_json({"score_after_outer": score_after,
                        "patch_sha256": sha256_file(out / f"outer_{score_after:04d}.pt"),
                        "records": [{key: record[key] for key in SCORE_FIELDS if key in record}
                                    for record in replay]}, out / f"scores_after_{score_after:04d}.json")
                before_outer = patch.detach().clone()
                record, origin, index, base_index = draw_record(clean, replay, method, outer, cfg)
                observations = record["observations"] if origin == "replay" else [record["observation"]]
                prepared = []
                for obs in observations:
                    inputs, proprio = self._base_inputs({"observation": obs, "task_name": record["task_name"]})
                    with torch.no_grad():
                        action, clean_dino, clean_siglip = self._timed(costs, "clean_prepare_s", lambda:
                            self._forward(inputs, proprio, obs, None, False,
                                          capture_siglip=method in CAUSAL_METHODS))
                        target = torch.where(action >= (low+high)/2, low, high).detach()
                        clean_dino = clean_dino.detach()
                        clean_siglip = clean_siglip.detach() if clean_siglip is not None else None
                    costs["clean_action_forward_calls"] = costs.get("clean_action_forward_calls", 0)+1
                    costs["full_model_dino_forward_calls"] = costs.get("full_model_dino_forward_calls", 0)+2
                    prepared.append((inputs, proprio, obs, target, clean_dino, clean_siglip))
                sums, monitor_stats = {}, {}
                lr = optimizer.param_groups[0]["lr"]
                for inner in range(cfg["inner_loop"]):
                    seed_transform(cfg["seed"] * 10_000_000 + outer*10_000 + inner)
                    optimizer.zero_grad(set_to_none=True)
                    inputs, proprio, obs, target, clean_dino, clean_siglip = prepared[inner % len(prepared)]
                    geometry = bool(cfg["geometry"] and inner >= cfg["inner_loop"]//2)
                    rgb, support = self._sample_rgb(obs, patch, geometry)
                    stats = {}
                    if method == "dino_repair_clear":
                        eta, stats = self._timed(costs, "repair_s", lambda:
                            self._repair_eta(obs, rgb, support, clean_dino, cfg, costs))
                        attacked_rgb = (rgb+eta).clamp(0, 1)
                    else:
                        attacked_rgb = rgb
                    action, _, _ = self._timed(costs, "patched_forward_s", lambda:
                        self._forward_rgb(inputs, proprio, obs, attacked_rgb,
                            clean_siglip=clean_siglip if method in CAUSAL_METHODS else None, costs=costs))
                    costs["patch_action_forward_calls"] = costs.get("patch_action_forward_calls", 0)+1
                    main = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
                    tv = charbonnier_tv(patch)
                    hf = high_frequency_energy(patch, cfg["hf_sigma"], cfg["hf_kernel"])
                    appearance = lambdas[0]*tv + lambdas[1]*hf
                    monitor = inner == 0 and (outer == start or outer % 50 == 0)
                    if monitor:
                        gradients = [torch.autograd.grad(term, patch, retain_graph=True)[0]
                                     for term in (main, lambdas[0]*tv, lambdas[1]*hf)]
                        require(all(bool(torch.isfinite(g).all()) for g in gradients), "Nonfinite monitored RGB gradient")
                        monitor_stats = {key: float(value.norm()) for key, value in zip(
                            ("action_gradient_norm", "tv_gradient_norm", "hf_gradient_norm"), gradients)}
                    patch.grad = torch.autograd.grad(main+appearance, patch)[0]
                    require(bool(torch.isfinite(patch.grad).all()), "Nonfinite total RGB gradient")
                    total_gradient_norm = float(patch.grad.norm())
                    before_inner = patch.detach().clone()
                    step_uada_patch(patch, optimizer, self.wrapper.vla)
                    costs["new_patch_updates"] = costs.get("new_patch_updates", 0)+1
                    stats.update(main_loss=float(main.detach()), appearance_loss=float(appearance.detach()),
                        total_gradient_norm=total_gradient_norm,
                        tv_charbonnier=float(tv.detach()), hf_energy=float(hf.detach()),
                        rgb_step_mean_abs=float((patch.detach()-before_inner).abs().mean()),
                        rgb_step_max_abs=float((patch.detach()-before_inner).abs().max()))
                    for key, value in stats.items():
                        if not isinstance(value, list):
                            sums[key] = sums.get(key, 0.)+value
                scheduler.step()
                structure = patch_metrics(patch, initial)
                delta = patch.detach()-before_outer
                if outer == start:
                    require(lr == 0 or float(delta.abs().max()) > 0,
                            "Nonzero first LR but no startup RGB update")
                row = {"method": method, "outer_iteration": outer, "optimizer_updates": outer*cfg["inner_loop"],
                    "origin": origin, "record_index": index, "base_record_index": base_index,
                    "task_id": int(record["task_id"]), "state_index": int(record["state_index"]),
                    "dino_score": float(record.get("dino_score", 0)), "learning_rate": lr,
                    "elapsed_s": elapsed(), "score_after_outer": score_after,
                    "rgb_outer_mean_abs": float(delta.abs().mean()), "rgb_outer_max_abs": float(delta.abs().max()),
                    "zero_update_outer": bool(float(delta.abs().max()) == 0),
                    "rgb_std": float(patch.detach().std()), **monitor_stats, **structure,
                    **{key: value/cfg["inner_loop"] for key, value in sums.items()}}
                rows.append(row)
                completed = outer
                if outer in cfg["checkpoint_outer_iterations"]:
                    save_patch(out / f"outer_{outer:04d}.pt", patch,
                        {"method": method, "signature": signature, "total_outer": outer,
                         "total_optimizer_updates": outer*cfg["inner_loop"], **structure})
                # Commit every complete outer. Exception handling never writes
                # a partial patch/Adam state paired with completed row counters.
                save_resume()
                if outer % cfg["resume_every_outer_iterations"] == 0 or outer == end:
                    atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
                    atomic_json(progress("RUNNING", **structure), status_path)
                    print(f"DINO_REPAIR method={method} outer={outer}/{end} loss={row['main_loss']:.5f} "
                          f"tv={structure['tv']:.5f} origin={origin}", flush=True)
            final = out / f"outer_{end:04d}.pt"
            atomic_json(progress("COMPLETE", final_patch_sha256=sha256_file(final),
                final_png_sha256=sha256_file(final.with_suffix(".png")),
                resume_sha256=sha256_file(resume_path), **patch_metrics(patch, initial)), status_path)
        except BaseException as error:
            # A checkpoint/serialization error may follow rows.append; report
            # only the state actually committed on disk, never the partial RAM.
            if resume_path.exists():
                committed = torch.load(resume_path, map_location="cpu", weights_only=False)
                completed = committed["outer"]
                atomic_parquet(pd.DataFrame(committed["rows"]), out / "training.parquet")
            atomic_json(progress("FAILED", attempted_outer=outer, error=repr(error),
                resume=str(resume_path)), status_path)
            raise

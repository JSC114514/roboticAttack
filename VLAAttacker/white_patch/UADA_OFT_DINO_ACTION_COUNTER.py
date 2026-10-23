"""Fresh RGB UADA training with a Wrist-DINO action-counter inner update.

The inherited adapter supplies the established renderer and normal OFT forward.
Every counter forward recomputes SigLIP and detaches only its Wrist output; the
outer update consumes the ordinary complete OFT graph. No trained prefix is read.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import random
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_DINO_REPAIR import ExploringOFTDINORepairAttacker, frozen_model
from white_patch.UADA_OFT_CORRESPONDENCE import run_signature
from white_patch.UADA_OFT_DINO import (
    atomic_json, atomic_parquet, atomic_torch, patch_metrics, require,
    save_patch, sha256_file,
)
from white_patch.UADA_OFT_SOURCE_COPY import (
    build_uada_adamw, init_uada_patch, step_uada_patch,
)
from white_patch.UADA_OFT_ILA import seed_transform
from white_patch.oft_correspondence_objectives import total_variation
from white_patch.oft_dino_repair_objectives import high_frequency_energy
from white_patch.oft_dino_action_counter_objectives import (
    action_counter_eta, bounded_tv_loss, feasible_permute_eta,
)


def patch_tensor_sha256(patch):
    """Method-independent identity of the float32 RGB initialization."""
    value = patch.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def source_manifest(cfg, wrapper=None):
    """Pin runtime trees and this workflow without legacy launch/config coupling."""
    from dino_action_counter_common import CONFIG
    root = Path(__file__).resolve().parents[2]
    source_roots = [root / "VLAAttacker", root / "prismatic",
        Path(cfg["openvla_oft_root"]) / "prismatic",
        Path(cfg["openvla_oft_root"]) / "experiments",
        Path(cfg["attribution_root"]) / "src"]
    libero = Path(cfg["libero_root"]) / "libero"
    if libero.is_dir():
        source_roots.append(libero)
    paths = {Path(CONFIG).resolve(), root / "research_runs/wrist_dino_action_counter_v1/PLAN.md",
        root / "scripts/run_exploring_oft_scale500_cross_suite.py",
        root.parent / "oft_ramp_recovery_aware/scripts/common.py",
        root.parent / "oft_ramp_recovery_aware/scripts/runtime.py",
        root.parent / "oft_ramp_recovery_aware/scripts/run_continuous_cldev.py",
        root.parent / "oft_mgrb_lite_low_cost/scripts/model_runtime.py"}
    for directory in source_roots:
        require(directory.is_dir(), f"Missing runtime source tree: {directory}")
        paths.update(path.resolve() for path in directory.rglob("*.py"))
    paths.update(path.resolve() for path in (root / "scripts").glob("*action_counter*.py"))
    paths.update(path.resolve() for pattern in ("*action_counter*.sbatch", "*counter500*.sbatch")
                 for path in (root / "slurm/exploring_oft_dino").glob(pattern))
    paths.update(path.resolve() for path in root.glob("*action_counter*.sh"))
    checkpoint = Path(cfg["checkpoint"])
    paths.update(path.resolve() for pattern in ("*.json", "*.py") for path in checkpoint.glob(pattern))
    reference = json.loads(Path(cfg["eval_reference"]).read_text())
    for name in reference["checkpoints"].values():
        model = Path(name)
        require((model / "config.json").is_file(), f"Missing evaluation checkpoint metadata: {model}")
        paths.update(path.resolve() for pattern in ("*.json", "*.py") for path in model.glob(pattern))
    for pattern in ("*.bddl", "*.init", "*.pruned_init"):
        paths.update(path.resolve() for path in Path(cfg["libero_root"]).rglob(pattern))
    libero_config = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))) / "config.yaml"
    require(libero_config.is_file(), f"Missing LIBERO runtime paths: {libero_config}")
    paths.add(libero_config.resolve())
    for name, digest in cfg.get("frozen_eval_sources", {}).items():
        require(sha256_file(name) == digest, f"Frozen evaluation source changed: {name}")
        paths.add(Path(name).resolve())
    if wrapper is not None:
        objects = [wrapper, wrapper.vla, wrapper.processor, wrapper.processor.image_processor,
                   wrapper.action_head, wrapper.vla.vision_backbone,
                   wrapper.vla.vision_backbone.featurizer,
                   wrapper.vla.vision_backbone.fused_featurizer,
                   wrapper.proprio_projector, wrapper.noisy_action_projector]
        official = getattr(wrapper, "_official_openvla_utils", None)
        if official is not None and getattr(official, "__file__", None):
            paths.add(Path(official.__file__).resolve())
        for obj in objects:
            if obj is None:
                continue
            actual = getattr(obj, "module", obj)
            filename = Path(inspect.getfile(type(actual))).resolve()
            paths.add(filename)
            paths.update(path.resolve() for path in filename.parent.glob("*.py"))
    return {str(path): sha256_file(path) for path in sorted(paths)}


@contextmanager
def detach_wrist_siglip(wrapper):
    """Preserve fresh forward values and cut only call 1's SigLIP derivative."""
    module = wrapper.vla.vision_backbone.fused_featurizer
    calls = 0
    def detach(_module, _inputs, output):
        nonlocal calls
        index, calls = calls, calls + 1
        if index != 1:
            return output
        require(torch.is_tensor(output), "Expected the actual dense Wrist-SigLIP tensor")
        return output.detach()
    handle = module.register_forward_hook(detach)
    try:
        yield
        require(calls == 2, "OFT SigLIP call order drift: expected Front then Wrist")
    finally:
        handle.remove()


def validate_state(saved, cfg):
    require(saved["config"] == cfg, "Action-counter resume config drift")
    outer = saved["outer"]
    require(isinstance(outer, int) and 0 <= outer <= cfg["outer_iterations"]
            and len(saved["rows"]) == outer, "Resume outer/row mismatch")
    expected = outer * cfg["inner_loop"]
    require(all(row["outer_iteration"] == index and row["optimizer_updates"] == index * cfg["inner_loop"]
                for index, row in enumerate(saved["rows"], 1)), "Resume update sequence drift")
    states = saved["optimizer"]["state"]
    if expected:
        require(len(states) == 1 and all(int(state["step"]) == expected
            and "exp_avg" in state and "exp_avg_sq" in state
            and bool(torch.isfinite(state["exp_avg"]).all())
            and bool(torch.isfinite(state["exp_avg_sq"]).all()) for state in states.values()),
            "Adam moments/steps missing, reset or nonfinite")
    else:
        require(not states, "Fresh outer-zero Adam state must be empty")
    require(saved["scheduler"]["last_epoch"] == outer, "Scheduler reset/drift")
    for name in ("patch", "initial"):
        value = saved[name]
        require(tuple(value.shape) == tuple(cfg["patch_shape"]) and bool(torch.isfinite(value).all())
                and bool(((value >= 0) & (value <= 1)).all()), f"Invalid saved {name} RGB")
    require(saved.keys() >= {"torch_rng", "numpy_rng", "python_rng", "cuda_rng"}, "Missing resume RNG")
    require(saved["initial_patch_sha256"] == patch_tensor_sha256(saved["initial"]),
            "Initial patch identity drift")


def _rng_state(patch):
    return {"torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if patch.is_cuda else []}


def _restore_rng(saved, patch):
    torch.set_rng_state(saved["torch_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    if patch.is_cuda:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


class ExploringOFTDINOActionCounterAttacker(ExploringOFTDINORepairAttacker):
    def _sample_rgb(self, observation, patch, geometry):
        self.transfer_diverse = bool(geometry and getattr(self, "action_counter_outer", 0)
                                     > getattr(self, "diverse_after_outer", 250))
        try:
            rgb, support = self._paste_with_support(observation["wrist_image"], patch, geometry)
            require(tuple(rgb.shape) == (3, 224, 224) and tuple(support.shape) == (224, 224),
                    "Frozen renderer requires 224 RGB/support")
            return rgb, support
        finally:
            self.transfer_diverse = False

    def _counter_forward(self, inputs, proprio, obs, rgb, *, capture_siglip=False, costs=None):
        with detach_wrist_siglip(self.wrapper):
            return self._forward_rgb(inputs, proprio, obs, rgb,
                                     capture_siglip=capture_siglip, costs=costs)

    def _counter_eta(self, inputs, proprio, obs, rgb, support, target, cfg, method, costs):
        actions, losses = [], []
        def loss_fn(value):
            action, _, _ = self._timed(costs, "counter_forward_s", lambda:
                self._counter_forward(inputs, proprio, obs, value, costs=costs))
            loss = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
            actions.append(action.detach())
            losses.append(float(loss.detach()))
            return loss
        with frozen_model(self.wrapper):
            eta, stats = self._timed(costs, "counter_s", lambda:
                action_counter_eta(rgb, support, loss_fn, steps=cfg["counter_steps"],
                    epsilon=cfg["counter_epsilon"], step_size=cfg["counter_step_size"]))
        for key in ("counter_steps", "counter_forward_calls", "counter_backward_calls"):
            costs[key] = costs.get(key, 0) + stats[key]
        baseline_action = actions[0]
        selected_action = actions[max(range(len(losses)), key=losses.__getitem__)]
        if method.startswith("random_"):
            eta, random_stats = self._timed(costs, "random_permutation_s", lambda:
                feasible_permute_eta(rgb, support, eta, generator=None))
            stats.update(random_stats)
        eta = eta.detach()
        require(not eta.requires_grad and bool(torch.isfinite(eta).all()), "Invalid detached counter eta")
        require(float(eta.abs().max()) <= cfg["counter_epsilon"] + 2e-7
                and bool((eta[:, support > 0] == 0).all()), "Counter eta violates support/budget")
        require(bool(((rgb.detach() + eta >= -2e-7) & (rgb.detach() + eta <= 1 + 2e-7)).all()),
                "Counter eta violates RGB feasibility")
        return eta, stats, baseline_action, selected_action

    def _validate_action_counter_runtime(self, record, reference, cfg, method, costs):
        inputs, proprio = self._base_inputs(record)
        obs = record["observation"]
        modules = [self.wrapper.vla.vision_backbone.featurizer,
                   self.wrapper.vla.vision_backbone.fused_featurizer]
        counts = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]
        probe = reference.detach().clone().requires_grad_(True)
        rgb, support = self._sample_rgb(obs, probe, False)
        with torch.no_grad():
            clean_action, _, _ = self._forward(inputs, proprio, obs, None, False)
        costs["runtime_clean_forward_calls"] = costs.get("runtime_clean_forward_calls", 0) + 1
        low = torch.as_tensor(cfg["action_head_q01"], device=probe.device).reshape(1, 1, 7)
        high = torch.as_tensor(cfg["action_head_q99"], device=probe.device).reshape(1, 1, 7)
        target = torch.where(clean_action >= (low + high) / 2, low, high).detach()
        action, dense, siglip = self._forward_rgb(inputs, proprio, obs, rgb,
                                                capture_siglip=True, costs=costs)
        inner_rgb = rgb.detach().clone().requires_grad_(True)
        inner_action, inner_dense, inner_siglip = self._counter_forward(
            inputs, proprio, obs, inner_rgb, capture_siglip=True, costs=costs)
        require(torch.equal(siglip.detach(), inner_siglip.detach())
                and torch.equal(dense.detach(), inner_dense.detach()),
                "Inner detach changed freshly computed OFT visual values")
        rtol = cfg.get("runtime_action_rtol", .05 if self.wrapper.dtype == torch.bfloat16 else 1e-6)
        atol = cfg.get("runtime_action_atol", 1e-4 if self.wrapper.dtype == torch.bfloat16 else 1e-6)
        require(torch.allclose(action.detach(), inner_action.detach(), rtol=rtol, atol=atol),
                "Inner detach changed OFT forward action")
        require(not inner_siglip.requires_grad, "Wrist-SigLIP gradient survived inner detach")
        normal_loss = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
        normal_gradient, normal_rgb_gradient = torch.autograd.grad(normal_loss, (probe, rgb))
        inner_loss = F.mse_loss(inner_action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
        inner_gradient = torch.autograd.grad(inner_loss, inner_rgb)[0]
        require(bool(torch.isfinite(normal_gradient).all()) and bool(torch.isfinite(inner_gradient).all()),
                "Nonfinite outer/inner RGB gradient")
        zero_base = rgb.detach().clone().requires_grad_(True)
        zero_rgb = (zero_base + torch.zeros_like(rgb)).clamp(0, 1)
        zero_action, _, zero_siglip = self._forward_rgb(inputs, proprio, obs, zero_rgb,
                                                       capture_siglip=True, costs=costs)
        require(torch.allclose(action.detach(), zero_action.detach(), rtol=rtol, atol=atol)
                and torch.equal(siglip.detach(), zero_siglip.detach()), "Eta-zero normal OFT equivalence failed")
        require(zero_siglip.requires_grad, "Outer Wrist-SigLIP derivative was not restored")
        zero_gradient = torch.autograd.grad(F.mse_loss(zero_action[..., cfg["maskidx"]],
            target[..., cfg["maskidx"]]), zero_base)[0]
        require(bool(torch.isfinite(zero_gradient).all()), "Nonfinite eta-zero normal RGB gradient")
        relative_gradient_error = float((zero_gradient - normal_rgb_gradient).norm()
            / normal_rgb_gradient.norm().clamp_min(1e-12))
        require(relative_gradient_error <= cfg.get("runtime_gradient_relative_tol", .1),
                "Eta-zero normal OFT gradient equivalence failed")
        if method.startswith(("dino_", "random_")):
            eta, stats, _, _ = self._counter_eta(inputs, proprio, obs, rgb, support, target, cfg, method, costs)
        else:
            eta, stats = torch.zeros_like(rgb), {}
        require(counts == [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules],
                "Runtime forward left hooks installed")
        return {"status": "PASS", "method": method, "fresh_siglip_values_identical": True,
                "actual_dense_values_identical": True, "inner_wrist_siglip_detached": True,
                "outer_wrist_siglip_restored": True, "eta_zero_equivalent": True,
                "hooks_removed": True, "single_eot": True,
                "outer_gradient_norm": float(normal_gradient.norm()),
                "inner_gradient_norm": float(inner_gradient.norm()),
                "eta_zero_gradient_relative_error": relative_gradient_error,
                "zero_gradient_allowed_for_saturation": True, "eta_linf": float(eta.abs().max()),
                **{key: float(value) for key, value in stats.items() if np.isscalar(value)}}

    def train_action_counter(self, clean, replay, cfg, method, source_hashes, data_hashes, model_load_s=0.):
        from dino_action_counter_common import METHODS, draw_record
        require(method in METHODS and len(clean) == 300 and len(replay) == 258,
                "Action-counter method/Object source bank drift")
        require(tuple(cfg["patch_shape"]) == (3, 50, 50) and cfg["maskidx"] == [0],
                "Action-counter RGB/action threat drift")
        self.diverse_after_outer = cfg["diverse_after_outer"]
        end = cfg["outer_iterations"]
        threshold = cfg["tv_thresholds"][method.rsplit("_", 1)[1]]
        out = Path(cfg["output_root"]) / "training" / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        require(not previous or previous["signature"] == signature, "Existing action-counter signature drift")
        if previous.get("status") == "COMPLETE":
            require(previous["final_patch_sha256"] == sha256_file(out / f"outer_{end:04d}.pt")
                and previous["final_png_sha256"] == sha256_file(out / f"outer_{end:04d}.png")
                and previous["resume_sha256"] == sha256_file(resume_path), "Completed output drift")
            print(f"ALREADY_COMPLETE method={method}", flush=True)
            return
        seed_transform(cfg["seed"])
        patch = init_uada_patch(tuple(cfg["patch_shape"]), self.wrapper.device)
        initial = patch.detach().clone()
        initial_sha = patch_tensor_sha256(initial)
        optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"],
            cfg["warmup_outer_iterations"], cfg["scheduler_total_outer"], cfg["weight_decay"])
        require(not optimizer.state, "Fresh Adam must have no moments")
        manifest = {"experiment_id": cfg["experiment_id"], "config": cfg, "method": method,
            "signature": signature, "source": source_hashes, "data": data_hashes,
            "initialization": "fresh seed42 random RGB and empty Adam at outer zero",
            "initial_patch_sha256": initial_sha, "fresh_start_outer": 0,
            "initial_optimizer_state_empty": True, "optimizer_restarted_at_outer": 0,
            "parameter_count": patch.numel(), "scheduler_total_outer": cfg["scheduler_total_outer"],
            "tv_threshold": threshold, "tv_scale": cfg["tv_scale"], "tv_weight": cfg["tv_weight"],
            "feature_loss": False, "counter_from_outer": 1 if not method.startswith("none_") else None,
            "inner_siglip": "fresh Wrist output detached only for eta gradient",
            "outer_forward": "normal complete OFT", "evaluation_forward": "normal OFT, quantized Wrist RGB patch only"}
        manifest_path = out / "RUN_MANIFEST.json"
        if manifest_path.exists():
            require(json.loads(manifest_path.read_text()) == manifest, "Action-counter manifest drift")
        else:
            atomic_json(manifest, manifest_path)
        snapshot = out / "source_snapshot.zip"
        if not snapshot.exists():
            temporary = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.tmp")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path, digest in source_hashes.items():
                    require(sha256_file(path) == digest, f"Source changed during snapshot: {path}")
                    archive.write(path, arcname=str(Path(path).resolve()).lstrip("/"))
                archive.writestr("RUN_MANIFEST.json", json.dumps(manifest, sort_keys=True))
            os.replace(temporary, snapshot)
        snapshot_sha = sha256_file(snapshot)
        rows, start, prior_elapsed, costs = [], 1, 0., {}
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_state(saved, cfg)
            require(saved["signature"] == signature and saved["method"] == method
                    and saved["source_snapshot_sha256"] == snapshot_sha
                    and torch.equal(saved["initial"], initial.cpu()), "Resume identity/source/initial RGB drift")
            with torch.no_grad():
                patch.copy_(saved["patch"].to(patch.device))
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            rows, start = copy.deepcopy(saved["rows"]), saved["outer"] + 1
            prior_elapsed = max(float(saved["elapsed_s"]), float(previous.get("elapsed_s", 0.)))
            costs = dict(saved["costs"])
            for key, value in previous.get("costs", {}).items():
                costs[key] = max(costs.get(key, 0), value)
            _restore_rng(saved, patch)
        else:
            require(not previous, "Existing status has no complete outer resume")
            save_patch(out / "outer_0000.pt", initial, {"method": method, "signature": signature,
                "total_outer": 0, "total_optimizer_updates": 0, "initial_patch_sha256": initial_sha})
        costs["model_load_s"] = costs.get("model_load_s", 0.) + model_load_s
        low = torch.as_tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.as_tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        started = time.monotonic()
        completed = outer = start - 1
        def elapsed():
            return prior_elapsed + model_load_s + time.monotonic() - started
        def save_resume():
            payload = {"signature": signature, "config": cfg, "method": method, "outer": completed,
                "source_snapshot_sha256": snapshot_sha, "initial_patch_sha256": initial_sha,
                "patch": patch.detach().cpu(), "initial": initial.cpu(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rows": rows, "costs": dict(costs), "elapsed_s": elapsed(), **_rng_state(patch)}
            validate_state(payload, cfg)
            atomic_torch(payload, resume_path)
        def progress(status, **extra):
            return {"status": status, "method": method, "signature": signature, "outer": completed,
                "optimizer_updates": completed * cfg["inner_loop"], "elapsed_s": elapsed(),
                "initial_patch_sha256": initial_sha, "costs": dict(costs),
                "source_snapshot_sha256": snapshot_sha,
                "cost_boundary": "fresh training includes counter PGD for random, model loads and failed attempts",
                **extra}
        try:
            if not resume_path.exists():
                save_resume()
            rng = _rng_state(patch)
            try:
                self.action_counter_outer = completed
                alignment = self._timed(costs, "runtime_validation_s", lambda:
                    self._validate_action_counter_runtime(clean[0], patch.detach(), cfg, method, costs))
            finally:
                _restore_rng(rng, patch)
            atomic_json(alignment, out / "RUNTIME_ALIGNMENT.json")
            for outer in range(start, end + 1):
                self.action_counter_outer = outer
                before_outer = patch.detach().clone()
                record, origin, index, base_index = draw_record(clean, replay, method, outer, cfg)
                observations = record["observations"] if origin == "replay" else [record["observation"]]
                prepared = []
                for obs in observations:
                    inputs, proprio = self._base_inputs({"observation": obs, "task_name": record["task_name"]})
                    with torch.no_grad():
                        clean_action, _, _ = self._timed(costs, "clean_prepare_s", lambda:
                            self._forward(inputs, proprio, obs, None, False))
                        target = torch.where(clean_action >= (low + high) / 2, low, high).detach()
                    costs["clean_action_forward_calls"] = costs.get("clean_action_forward_calls", 0) + 1
                    costs["full_model_dino_forward_calls"] = costs.get("full_model_dino_forward_calls", 0) + 2
                    prepared.append((inputs, proprio, obs, target, clean_action.detach()))
                lr, sums = optimizer.param_groups[0]["lr"], {}
                for inner in range(cfg["inner_loop"]):
                    seed_transform(cfg["seed"] * 10_000_000 + outer * 10_000 + inner)
                    optimizer.zero_grad(set_to_none=True)
                    inputs, proprio, obs, target, clean_action = prepared[inner % len(prepared)]
                    geometry = bool(cfg["geometry"] and inner >= cfg["inner_loop"] // 2)
                    rgb, support = self._timed(costs, "render_s", lambda: self._sample_rgb(obs, patch, geometry))
                    costs["eot_draws"] = costs.get("eot_draws", 0) + 1
                    if method.startswith("none_"):
                        eta, stats, baseline_action, selected_action = torch.zeros_like(rgb), {}, None, None
                    else:
                        eta, stats, baseline_action, selected_action = self._counter_eta(
                            inputs, proprio, obs, rgb, support, target, cfg, method, costs)
                    action, _, _ = self._timed(costs, "outer_forward_s", lambda:
                        self._forward_rgb(inputs, proprio, obs, (rgb + eta.detach()).clamp(0, 1), costs=costs))
                    costs["outer_forward_calls"] = costs.get("outer_forward_calls", 0) + 1
                    main = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
                    appearance = cfg["tv_weight"] * bounded_tv_loss(patch, threshold, scale=cfg["tv_scale"])
                    tv = total_variation(patch)
                    patch.grad = self._timed(costs, "outer_backward_s", lambda:
                        torch.autograd.grad(main + appearance, patch)[0])
                    require(bool(torch.isfinite(patch.grad).all()), "Nonfinite full outer RGB gradient")
                    gradient_norm = float(patch.grad.norm())
                    before_inner = patch.detach().clone()
                    self._timed(costs, "optimizer_s", lambda: step_uada_patch(patch, optimizer, self.wrapper.vla))
                    costs["attempted_patch_updates"] = costs.get("attempted_patch_updates", 0) + 1
                    costs["outer_backward_calls"] = costs.get("outer_backward_calls", 0) + 1
                    baseline_action = action.detach() if baseline_action is None else baseline_action
                    baseline_loss = F.mse_loss(baseline_action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
                    delta = patch.detach() - before_inner
                    stats.update(main_loss=float(main.detach()), appearance_loss=float(appearance.detach()),
                        tv=float(tv.detach()), tv_violated=float(tv.detach() > threshold),
                        total_gradient_norm=gradient_norm, eta_linf=float(eta.abs().max()),
                        eta_rms=float(eta.square().mean().sqrt()), eta_mean_abs=float(eta.abs().mean()),
                        augmentation_loss_delta=float(main.detach() - baseline_loss),
                        clean_action_distance=float(F.mse_loss(action.detach(), clean_action)),
                        clean_action_distance_before_eta=float(F.mse_loss(baseline_action, clean_action)),
                        counter_selected_clean_action_distance=float(F.mse_loss(
                            baseline_action if selected_action is None else selected_action, clean_action)),
                        rgb_step_mean_abs=float(delta.abs().mean()), rgb_step_max_abs=float(delta.abs().max()),
                        rgb_step_zero=float(delta.abs().max() == 0))
                    for key, value in stats.items():
                        if np.isscalar(value):
                            sums[key] = sums.get(key, 0.) + float(value)
                scheduler.step()
                structure = patch_metrics(patch, initial)
                with torch.no_grad():
                    structure["hf_energy"] = float(high_frequency_energy(patch.detach()))
                delta = patch.detach() - before_outer
                before_png = before_outer.mul(255).round().to(torch.uint8)
                after_png = patch.detach().mul(255).round().to(torch.uint8)
                row = {"method": method, "outer_iteration": outer, "optimizer_updates": outer * cfg["inner_loop"],
                    "origin": origin, "record_index": index, "base_record_index": base_index,
                    "task_id": int(record["task_id"]), "state_index": int(record["state_index"]),
                    "phase_proxy": record.get("phase_proxy", "clean"), "learning_rate": lr,
                    "elapsed_s": elapsed(), "diverse_eot_enabled": outer > self.diverse_after_outer,
                    "tv_threshold": threshold, "rgb_outer_mean_abs": float(delta.abs().mean()),
                    "rgb_outer_max_abs": float(delta.abs().max()), "zero_update_outer": bool(delta.abs().max() == 0),
                    "png_changed_fraction": float((before_png != after_png).float().mean()),
                    "rgb_std": float(patch.detach().std()), **structure,
                    **{key: value / cfg["inner_loop"] for key, value in sums.items()}}
                # Keep structural end-of-outer TV distinct from the mean update TV.
                row["tv_update_mean"] = row["tv"]
                row["tv"] = structure["tv"]
                rows.append(row)
                completed = outer
                if outer in cfg["checkpoint_outer_iterations"] or outer == end:
                    self._timed(costs, "checkpoint_s", lambda:
                        save_patch(out / f"outer_{outer:04d}.pt", patch,
                            {"method": method, "signature": signature, "total_outer": outer,
                             "total_optimizer_updates": outer * cfg["inner_loop"], **structure}))
                save_resume()
                if outer % cfg["resume_every_outer_iterations"] == 0 or outer == end:
                    atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
                    atomic_json(progress("RUNNING", **structure), status_path)
                    if outer == start or outer % 10 == 0 or outer == end:
                        print(f"ACTION_COUNTER method={method} outer={outer}/{end} loss={row['main_loss']:.5f} "
                              f"tv={structure['tv']:.5f} origin={origin}", flush=True)
            final = out / f"outer_{end:04d}.pt"
            atomic_json(progress("COMPLETE", final_patch_sha256=sha256_file(final),
                final_png_sha256=sha256_file(final.with_suffix(".png")),
                resume_sha256=sha256_file(resume_path), **patch_metrics(patch, initial)), status_path)
        except BaseException as error:
            if resume_path.exists():
                committed = torch.load(resume_path, map_location="cpu", weights_only=False)
                completed = committed["outer"]
                atomic_parquet(pd.DataFrame(committed["rows"]), out / "training.parquet")
            atomic_json(progress("FAILED", attempted_outer=outer, error=repr(error), resume=str(resume_path)), status_path)
            raise

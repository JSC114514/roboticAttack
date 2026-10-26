"""Exploring-derived Wrist RGB action/attention/content matched controls.

The deployable artifact is still only a 3x50x50 RGB patch.  The DINO register
objective can intervene on four K/V tokens in an extra training-only clean
branch; the normal RGB/action branch and evaluation forward stay unmodified.
"""
from __future__ import annotations

import json
import os
import random
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_DINO import (
    atomic_json, atomic_parquet, atomic_torch, patch_metrics, require, save_patch,
    sha256_file,
)
from white_patch.UADA_OFT_DINO_STATE_AXIS import ExploringOFTDINOStateAxisAttacker
from white_patch.UADA_OFT_CORRESPONDENCE import run_signature
from white_patch.UADA_OFT_DINO_SAMPLING_CLEAR import patch_tensor_sha256, _terminal_float32_rounding_evidence
from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, init_uada_patch, step_uada_patch
from white_patch.UADA_OFT_ILA import seed_transform
from white_patch.UADA_OFT_DINO_STATE_AXIS import _save_checkpoint
from oft_dino_register_content_objectives import (
    auxiliary_active, combine_gradients, content_indices, extract_kv, replace_wrist_kv,
)


def source_manifest(cfg, wrapper=None):
    from dino_register_content_common import source_manifest as register_sources

    return register_sources(cfg, wrapper)


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
    require(saved["config"] == cfg and len(saved["rows"]) == saved["outer"],
            "Register resume config/row mismatch")
    expected = saved["outer"] * cfg["inner_loop"]
    require(saved["outer"] in range(cfg["outer_iterations"] + 1), "Register resume outer out of range")
    if expected:
        require(saved["rows"][-1]["optimizer_updates"] == expected, "Register resume update count drift")
    states = saved["optimizer"]["state"]
    if expected:
        require(len(states) == 1 and all(int(state["step"]) == expected
                and "exp_avg" in state and "exp_avg_sq" in state
                and bool(torch.isfinite(state["exp_avg"]).all())
                and bool(torch.isfinite(state["exp_avg_sq"]).all()) for state in states.values()),
                "Adam moments/step missing, reset, or nonfinite")
    else:
        require(not states, "Fresh register Adam state must be empty")
    require(saved["scheduler"]["last_epoch"] == saved["outer"], "Register scheduler drift")
    for name in ("patch", "initial"):
        value = saved[name]
        require(tuple(value.shape) == tuple(cfg["patch_shape"])
                and bool(torch.isfinite(value).all())
                and bool(((value >= 0) & (value <= 1)).all()),
                f"Invalid saved {name} RGB")
    require(saved.keys() >= {"torch_rng", "numpy_rng", "python_rng", "cuda_rng", "terminal_rounding_history"},
            "Delayed resume missing complete RNG/evidence")
    unchanged = [row["outer_iteration"] for row in saved["rows"]
                 if row["learning_rate"] != 0. and row["rgb_outer_max_abs"] == 0.]
    require(unchanged == [item["outer"] for item in saved["terminal_rounding_history"]],
            "Terminal rounding evidence/row mismatch")
    for item in saved["terminal_rounding_history"]:
        require(cfg["outer_iterations"] - item["outer"] + 1 <= 2
                and item["evidence_count"] == cfg["inner_loop"]
                and item["effective_rgb_updates"] == 0,
                "Unchanged RGB outside the proven final rounding boundary")


class ExploringOFTDINORegisterContentAttacker(ExploringOFTDINOStateAxisAttacker):
    @contextmanager
    def _capture_dino_qkv(self, cfg):
        featurizer = self.wrapper.vla.vision_backbone.featurizer
        blocks = getattr(featurizer, "blocks", None)
        require(blocks is not None and len(blocks) > max(cfg["register_attention_layers"]),
                "DINO block layout drift")
        captures = defaultdict(list)
        handles = []
        for layer in cfg["register_attention_layers"]:
            qkv = blocks[layer].attn.qkv

            def hook(_module, _inputs, output, *, layer=layer):
                captures[int(layer)].append(output)

            handles.append(qkv.register_forward_hook(hook))
        try:
            yield captures
        finally:
            for handle in handles:
                handle.remove()

    def _attention_masses(self, captures, cfg, target):
        featurizer = self.wrapper.vla.vision_backbone.featurizer
        prefix = int(cfg["dino_prefix_tokens"])
        register_queries = torch.tensor(cfg["register_query_tokens"], device=self.wrapper.device, dtype=torch.long)
        coverage = getattr(self, "patch_tokens", None)
        require(coverage is not None and tuple(coverage.shape) == (256,),
                "Actual Wrist patch token footprint was not captured")
        patch_dense = torch.nonzero(coverage.to(self.wrapper.device).float() > 0, as_tuple=True)[0]
        require(patch_dense.numel() > 0, "No DINO dense token intersects the current Wrist patch")
        patch_keys = patch_dense + prefix
        all_dense = torch.arange(256, device=self.wrapper.device, dtype=torch.long)
        nonpatch_dense = all_dense[coverage.to(self.wrapper.device).float() <= 0] + prefix
        require(nonpatch_dense.numel() > 0, "Patch covered every DINO dense token")
        spatial_queries = nonpatch_dense
        target_queries = register_queries if target == "register" else spatial_queries

        losses, register_masses, spatial_masses = [], [], []
        wrist_call_index = 1
        for layer in cfg["register_attention_layers"]:
            values = captures.get(int(layer), [])
            require(len(values) > wrist_call_index, f"DINO qkv hook call count drift at layer {layer}")
            qkv_out = values[wrist_call_index]
            block = featurizer.blocks[int(layer)]
            attn = block.attn
            batch, tokens, channels3 = qkv_out.shape
            require(batch == 1 and tokens == 256 + prefix and channels3 % 3 == 0,
                    f"DINO qkv shape drift at layer {layer}: {tuple(qkv_out.shape)}")
            qkv = qkv_out.reshape(batch, tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = attn.q_norm(q), attn.k_norm(k)
            logits = (q.float() * float(attn.scale)) @ k.float().transpose(-2, -1)

            def mass(queries):
                selected = logits[:, :, queries, :]
                log_patch = torch.logsumexp(selected.index_select(-1, patch_keys), dim=-1)
                log_all = torch.logsumexp(selected, dim=-1)
                value = (log_patch - log_all).exp()
                return value.mean(), -(log_patch - log_all).mean()

            target_mass, target_loss = mass(target_queries)
            register_mass, _ = mass(register_queries)
            spatial_mass, _ = mass(spatial_queries)
            losses.append(target_loss)
            register_masses.append(register_mass.detach())
            spatial_masses.append(spatial_mass.detach())
        aux_loss = torch.stack(losses).mean()
        return aux_loss, {
            "register_attention_mass": float(torch.stack(register_masses).mean()),
            "spatial_attention_mass": float(torch.stack(spatial_masses).mean()),
            "patch_token_count": int(patch_dense.numel()),
            "target_query_count": int(target_queries.numel()),
        }


    def _content_module(self, cfg):
        return self.wrapper.vla.vision_backbone.featurizer.blocks[int(cfg["content_layer"])].attn.qkv

    def _objective_step(self, inputs, proprio, observation, target, clean_action,
                        patch, geometry, axis, cfg, spec, enabled, costs):
        content = enabled and spec["target"].endswith("_content")
        capture_cfg = {**cfg, "register_attention_layers": [cfg["content_layer"]]} if content else cfg
        if enabled:
            with self._capture_dino_qkv(capture_cfg) as captures:
                outputs = self._timed(costs, "patched_forward_s", lambda:
                    self._forward(inputs, proprio, observation, patch, geometry))
        else:
            captures = {}
            outputs = self._timed(costs, "patched_forward_s", lambda:
                self._forward(inputs, proprio, observation, patch, geometry))
        self.transfer_diverse = False
        costs["training_patched_forward_calls"] = costs.get("training_patched_forward_calls", 0) + 1
        action = outputs[0]
        main = F.mse_loss(action[..., [axis]], target[..., [axis]])
        require(bool(torch.isfinite(main)), "Nonfinite normal action loss")
        main_value = float(main.detach())
        stats = {"register_attention_mass": 0., "spatial_attention_mass": 0.,
                 "patch_token_count": 0., "target_query_count": 0.,
                 "content_token_count": 0., "cf_clean_action_mae": 0.,
                 "cf_patched_action_mae": 0., "donor_kv_change": 0.}
        aux = None
        if enabled and not content:
            require(spec["target"] == "register_attention", "Unknown attention target")
            aux, attention_stats = self._attention_masses(captures, cfg, "register")
            stats.update(attention_stats)
        if content:
            values = captures[int(cfg["content_layer"])]
            require(len(values) == 2, "Content donor must be the second native Wrist call")
            indices = content_indices(spec["target"], self.patch_tokens, cfg)
            donor = extract_kv(values[1], indices)
            donor_leaf = donor.detach().clone().requires_grad_(True)
            patched_action = action.detach()
            stats.update(patch_token_count=int((self.patch_tokens > 0).sum()),
                         content_token_count=int(indices.numel()))
        action_grad = self._timed(costs, "action_backward_s", lambda:
            torch.autograd.grad(main, patch, retain_graph=enabled)[0])
        costs["training_action_backward_calls"] = costs.get("training_action_backward_calls", 0) + 1
        require(bool(torch.isfinite(action_grad).all()), "Nonfinite normal RGB gradient")
        if not enabled:
            return action_grad, None, main_value, 0., stats
        if not content:
            require(bool(torch.isfinite(aux)), "Nonfinite attention loss")
            aux_value = float(aux.detach())
            aux_grad = self._timed(costs, "aux_backward_s", lambda:
                torch.autograd.grad(aux, patch)[0])
        else:
            # Free the normal LLM graph before the clean LLM branch.
            # Only the upstream DINO donor graph is retained.
            captures.clear()
            del values, action, outputs, main
            with replace_wrist_kv(self._content_module(cfg), indices, donor_leaf) as audit:
                cf_outputs = self._timed(costs, "content_forward_s", lambda:
                    self._forward(inputs, proprio, observation, None, False))
            cf_action = cf_outputs[0]
            aux = F.mse_loss(cf_action[..., [axis]], target[..., [axis]])
            require(bool(torch.isfinite(aux)), "Nonfinite content action loss")
            aux_value = float(aux.detach())
            stats.update(
                cf_clean_action_mae=float((cf_action.detach() - clean_action).abs().mean()),
                cf_patched_action_mae=float((cf_action.detach() - patched_action).abs().mean()),
                donor_kv_change=float((donor_leaf.detach() - audit["clean_kv"]).float().abs().mean()))
            leaf_grad = self._timed(costs, "content_readout_backward_s", lambda:
                torch.autograd.grad(aux, donor_leaf)[0])
            del cf_action, cf_outputs, aux
            aux_grad = self._timed(costs, "content_rgb_backward_s", lambda:
                torch.autograd.grad(donor, patch, grad_outputs=leaf_grad)[0])
            costs["training_content_forward_calls"] = costs.get("training_content_forward_calls", 0) + 1
            costs["training_content_chain_backward_calls"] = costs.get("training_content_chain_backward_calls", 0) + 1
        costs["training_aux_backward_calls"] = costs.get("training_aux_backward_calls", 0) + 1
        require(bool(torch.isfinite(aux_grad).all()), "Nonfinite auxiliary RGB gradient")
        return action_grad, aux_grad, main_value, aux_value, stats

    def _validate_runtime(self, record, reference, cfg):
        alignment = super()._validate_runtime(record, reference, cfg)
        if not self.content_target.endswith("_content"):
            alignment["content_branch"] = "NOT_APPLICABLE"
            return alignment
        self.runtime_alignment["status"] = "CHECKING"
        inputs, proprio = self._base_inputs(record)
        capture_cfg = {**cfg, "register_attention_layers": [cfg["content_layer"]]}
        with torch.no_grad(), self._capture_dino_qkv(capture_cfg) as captures:
            clean_action, _, _ = self._forward(inputs, proprio, record["observation"], None, False)
        _, coverage = self._wrist_pixels(record["observation"], reference, False)
        indices = content_indices(self.content_target, coverage, cfg)
        require(len(captures[cfg["content_layer"]]) == 2, "Startup native call count changed")
        clean_donor = extract_kv(captures[cfg["content_layer"]][1], indices)
        with torch.no_grad(), replace_wrist_kv(self._content_module(cfg), indices, clean_donor):
            identity_action, _, _ = self._forward(inputs, proprio, record["observation"], None, False)
        identity_error = float((identity_action - clean_action).abs().max())
        require(torch.allclose(identity_action, clean_action, atol=1e-5, rtol=1e-5),
                "Clean donor intervention is not an identity on actual OFT")
        del captures, clean_donor, identity_action
        low = torch.tensor(cfg["action_head_q01"], device=reference.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=reference.device).reshape(1, 1, 7)
        target = torch.where(clean_action >= (low + high) / 2, low, high).detach()
        probe = reference.detach().clone().requires_grad_(True)
        spec = {"target": self.content_target, "gradient_ratio": .1, "project_conflict": False}
        costs = {}
        main_grad, content_grad, main_loss, content_loss, stats = self._objective_step(
            inputs, proprio, record["observation"], target, clean_action,
            probe, False, 0, cfg, spec, True, costs)
        require(float(content_grad.norm()) > 1e-12, "Actual content objective has no RGB gradient")
        alignment.update(status="PASS", content_branch={
            "clean_donor_identity_max_abs": identity_error,
            "clean_donor_identity_close": True, "normal_forward_feature_replacement": False,
            "layer": cfg["content_layer"], "target": self.content_target,
            "main_rgb_gradient_norm": float(main_grad.norm()),
            "content_rgb_gradient_norm": float(content_grad.norm()),
            "main_loss": main_loss, "content_loss": content_loss, **stats,
            "startup_costs": costs, "evaluation_intervention": False})
        if probe.is_cuda:
            alignment["content_branch"].update(
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(probe.device),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(probe.device))
        self.wrapper.vla.zero_grad(set_to_none=True)
        return alignment

    def train_register(self, clean, replay, cfg, method, source_hashes, data_hashes, model_load_s=0.):
        from dino_register_content_common import METHODS, method_spec
        from dino_state_axis_common import draw_record

        require(method in METHODS and len(clean) == 300 and len(replay) == 258,
                "Register method/source data drift")
        spec = method_spec(cfg, method)
        self.content_target = spec["target"]
        out = Path(cfg["output_root"]) / "training" / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        require(not previous or previous["signature"] == signature, "Existing register run identity drift")
        if previous.get("status") == "COMPLETE":
            final = out / f"outer_{cfg['outer_iterations']:04d}.pt"
            require(previous["final_patch_sha256"] == sha256_file(final)
                    and previous["final_png_sha256"] == sha256_file(final.with_suffix(".png"))
                    and previous["resume_sha256"] == sha256_file(resume_path),
                    "Completed register artifacts changed")
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
            "method_spec": spec,
            "signature": signature,
            "config": cfg,
            "source": source_hashes,
            "data": data_hashes,
            "parameter_count": int(patch.numel()),
            "fresh_start_outer": 0,
            "initial_optimizer_state_empty": True,
            "initialization": "seed42 random RGB; no historical patch warm start",
            "raw_random_patch_sha256": raw_hash,
            "initial_patch_sha256": initial_hash,
            "action_main_objective": "Exploring-OFT UADA dim0 opposite-boundary MSE",
            "auxiliary_objective": ("none" if spec["target"] == "none" else cfg["content_objective"]
                                    if spec["target"].endswith("_content") else cfg["attention_objective"]),
            "attention_gradient_combination": cfg["attention_gradient_combination"],
            "feature_replacement_hook_during_training": spec["target"].endswith("_content"),
            "normal_forward_feature_replacement": False,
            "content_layer": cfg["content_layer"],
            "content_token_count": cfg["content_token_count"],
            "feature_loss": False,
            "appearance_regularization": False,
            "scheduler_total_outer": cfg["scheduler_total_outer"],
            "auxiliary_start_outer": cfg["auxiliary_start_outer"],
            "evaluation_forward": "normal OFT, quantized single Wrist RGB patch",
        }
        manifest_path = out / "RUN_MANIFEST.json"
        if manifest_path.exists():
            require(json.loads(manifest_path.read_text()) == manifest, "Register manifest drift")
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
        optimizer, scheduler = build_uada_adamw(
            patch, cfg["learning_rate"], cfg["warmup_outer_iterations"],
            cfg["scheduler_total_outer"], cfg["weight_decay"])
        rows, start, prior_elapsed, costs = [], 1, 0., {}
        terminal_rounding_history = []
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            require(saved["signature"] == signature and saved["method"] == method
                    and saved["source"] == source_hashes and saved["data"] == data_hashes,
                    "Register resume source/data/method drift")
            validate_state(saved, cfg)
            require(torch.equal(saved["initial"], initial.cpu())
                    and torch.equal(saved["initial_raw"], initial_raw.cpu()),
                    "Register resume initial RGB drift")
            with torch.no_grad():
                patch.copy_(saved["patch"].to(patch.device))
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            rows, start = saved["rows"], saved["outer"] + 1
            prior_elapsed = max(float(saved["elapsed_s"]), float(previous.get("elapsed_s", 0.)))
            costs = dict(saved["costs"])
            terminal_rounding_history = list(saved["terminal_rounding_history"])
            _restore_rng(saved, patch)
            atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
        else:
            require(not previous, "Existing register status without atomic resume")
            require(not optimizer.state, "Fresh register Adam state must be empty")
            _save_checkpoint(out / "initial_raw.pt", initial_raw, {
                "method": method,
                "signature": signature,
                "raw_random_patch_sha256": raw_hash,
            })
            _save_checkpoint(out / "outer_0000.pt", initial, {
                "method": method,
                "signature": signature,
                "total_outer": 0,
                "total_optimizer_updates": 0,
                "raw_random_patch_sha256": raw_hash,
                "initial_patch_sha256": initial_hash,
                **patch_metrics(initial, initial),
            })
        costs["model_load_s"] = costs.get("model_load_s", 0.) + model_load_s
        low = torch.tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        started = time.monotonic()
        completed_outer, attempted_outer = start - 1, start - 1

        def elapsed():
            return prior_elapsed + model_load_s + time.monotonic() - started

        def bump(key, value=1):
            costs[key] = costs.get(key, 0) + value

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
                "terminal_rounding_history": list(terminal_rounding_history),
                "costs": dict(costs),
                "elapsed_s": elapsed(),
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
                "raw_random_patch_sha256": raw_hash,
                "initial_patch_sha256": initial_hash,
                "source_snapshot_sha256": sha256_file(snapshot),
                "visual_review": "NOT_HUMAN_APPROVED",
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(patch.device) if patch.is_cuda else 0,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(patch.device) if patch.is_cuda else 0,
                "cost_boundary": "content/attention/control training including model loads and runtime validation",
                "terminal_rounding_history": list(terminal_rounding_history),
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
                before_outer = patch.detach().clone()
                record, origin, index, base_index, axis = draw_record(clean, replay, "uniform_axis0", outer, cfg)
                require(axis == int(cfg["clean_axis"]), "Register training must use fixed axis0")
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
                    prepared.append((inputs, proprio, observation, target, action.detach()))
                lr = float(optimizer.param_groups[0]["lr"])
                aux_enabled = auxiliary_active(outer, cfg) and spec["target"] != "none"
                terminal_rounding_evidence = []
                sums = {
                    "main_loss": 0., "aux_loss": 0., "action_grad_norm": 0.,
                    "raw_gradient_cosine": 0., "projected_gradient_cosine": 0.,
                    "gradient_conflict_fraction": 0., "projection_applied_fraction": 0.,
                    "effective_aux_gradient_ratio": 0., "aux_zero_gradient_steps": 0.,
                    "aux_grad_norm": 0., "aux_grad_scale": 0.,
                    "combined_grad_norm": 0., "register_attention_mass": 0.,
                    "spatial_attention_mass": 0., "patch_token_count": 0.,
                    "target_query_count": 0., "content_token_count": 0.,
                    "cf_clean_action_mae": 0., "cf_patched_action_mae": 0.,
                    "donor_kv_change": 0., "rgb_step_mean_abs_sum": 0.,
                    "rgb_step_max_abs": 0., "effective_rgb_updates": 0,
                }
                for inner in range(cfg["inner_loop"]):
                    seed_transform(cfg["seed"] * 10_000_000 + outer * 10_000 + inner)
                    optimizer.zero_grad(set_to_none=True)
                    inputs, proprio, observation, target, clean_action = prepared[inner % len(prepared)]
                    geometry = cfg["geometry"] and inner >= cfg["inner_loop"] // 2
                    self.transfer_diverse = bool(outer > cfg["bootstrap_outer"] and geometry)
                    try:
                        action_grad, aux_grad, main_value, aux_value, aux_stats = self._objective_step(
                            inputs, proprio, observation, target, clean_action,
                            patch, geometry, axis, cfg, spec, aux_enabled, costs)
                    finally:
                        self.transfer_diverse = False
                    action_norm = action_grad.norm()
                    require(outer != 1 or inner != 0 or float(action_norm) > 1e-12,
                            "Initial action objective has no RGB gradient")
                    if aux_enabled:
                        require(outer != cfg["auxiliary_start_outer"] or inner != 0
                                or float(aux_grad.norm()) > 1e-12,
                                "First real auxiliary objective has no RGB gradient")
                    combined, gradient_stats = combine_gradients(
                        action_grad, aux_grad, float(spec["gradient_ratio"]), spec["project_conflict"])
                    patch.grad = combined
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
                        evidence = _terminal_float32_rounding_evidence(before_step, optimizer, action_grad, combined)
                        if evidence is not None:
                            terminal_rounding_evidence.append(evidence)
                    sums["main_loss"] += main_value / cfg["inner_loop"]
                    sums["aux_loss"] += aux_value / cfg["inner_loop"]
                    sums["action_grad_norm"] += float(action_norm.detach()) / cfg["inner_loop"]
                    for key, value in gradient_stats.items():
                        divisor = 1 if key == "aux_zero_gradient_steps" else cfg["inner_loop"]
                        sums[key] += value / divisor
                    sums["combined_grad_norm"] += float(combined.norm().detach()) / cfg["inner_loop"]
                    for key, value in aux_stats.items():
                        sums[key] += value / cfg["inner_loop"]
                scheduler.step()
                structure = patch_metrics(patch, initial)
                delta = (patch.detach() - before_outer).abs()
                movement = {
                    "rgb_outer_mean_abs": float(delta.mean()),
                    "rgb_outer_max_abs": float(delta.max()),
                    "rgb_outer_changed_pixel_fraction": float((delta.amax(0) > 0).float().mean()),
                    "zero_lr_warmup_outer": lr == 0.,
                    "aux_grad_finite": True,
                }
                if lr != 0. and movement["rgb_outer_max_abs"] == 0.:
                    require(cfg["outer_iterations"] - outer + 1 <= 2
                            and sums["effective_rgb_updates"] == 0
                            and len(terminal_rounding_evidence) == cfg["inner_loop"],
                            "Nonzero LR but unchanged complete RGB outer")
                    terminal_rounding_history.append({
                        "outer": outer, "learning_rate": lr,
                        "evidence_count": len(terminal_rounding_evidence),
                        "max_intended_update_abs": max(item["max_intended_update_abs"]
                                                       for item in terminal_rounding_evidence),
                        "effective_rgb_updates": 0, "reason": "terminal_float32_rounding",
                    })
                rows.append({
                    "method": method,
                    "target": spec["target"],
                    "gradient_ratio": float(spec["gradient_ratio"]),
                    "auxiliary_enabled": aux_enabled,
                    "project_conflict": spec["project_conflict"],
                    "outer_iteration": outer,
                    "optimizer_updates": outer * cfg["inner_loop"],
                    "origin": origin,
                    "record_index": index,
                    "base_record_index": base_index,
                    "task_id": int(record["task_id"]),
                    "state_index": int(record["state_index"]),
                    "action_axis": int(axis),
                    "learning_rate": lr,
                    "elapsed_s": elapsed(),
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
                    print(f"DINO_REGISTER_CONTENT method={method} outer={outer} "
                          f"loss={sums['main_loss']:.6f} aux={sums['aux_loss']:.6f} "
                          f"reg_mass={sums['register_attention_mass']:.5f} "
                          f"cf_clean_mae={sums['cf_clean_action_mae']:.6f} "
                          f"tv={structure['tv']:.6f}", flush=True)
            final = out / f"outer_{cfg['outer_iterations']:04d}.pt"
            atomic_json(progress(
                "COMPLETE",
                final_patch_sha256=sha256_file(final),
                final_png_sha256=sha256_file(final.with_suffix(".png")),
                resume_sha256=sha256_file(resume_path),
                **patch_metrics(patch, initial),
            ), status_path)
        except BaseException as error:
            if getattr(self, "runtime_alignment", {}).get("status") == "CHECKING":
                atomic_json({**self.runtime_alignment, "status": "FAILED", "error": repr(error)},
                            out / "RUNTIME_ALIGNMENT.json")
            if resume_path.exists():
                committed = torch.load(resume_path, map_location="cpu", weights_only=False)
                completed_outer = committed["outer"]
            atomic_json(progress("FAILED", attempted_outer=attempted_outer,
                                 error=repr(error), resume=str(resume_path)), status_path)
            raise

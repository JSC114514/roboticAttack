"""Exploring-derived Wrist RGB training with DINOv2 register attention guidance.

The deployable artifact is still only a 3x50x50 RGB patch.  The DINO register
objective is a training-time read-only signal computed from native timm Q/K
activations; it does not replace features or change the evaluation forward.
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
from white_patch.UADA_OFT_DINO_SAMPLING_CLEAR import patch_tensor_sha256
from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, init_uada_patch, step_uada_patch
from white_patch.UADA_OFT_ILA import seed_transform


def source_manifest(cfg, wrapper=None):
    from dino_register_common import source_manifest as register_sources

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
    require(saved.keys() >= {"torch_rng", "numpy_rng", "python_rng", "cuda_rng"},
            "Register resume missing complete RNG")


class ExploringOFTDINORegisterAttacker(ExploringOFTDINOStateAxisAttacker):
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

    def train_register(self, clean, replay, cfg, method, source_hashes, data_hashes, model_load_s=0.):
        from dino_register_common import METHODS, method_spec
        from dino_state_axis_common import draw_record

        require(method in METHODS and len(clean) == 300 and len(replay) == 258,
                "Register method/source data drift")
        spec = method_spec(cfg, method)
        out = Path(cfg["output_root"]) / "training" / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        require(not previous or previous["signature"] == signature, "Existing register run identity drift")
        if previous.get("status") == "COMPLETE":
            final = out / "outer_0500.pt"
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
            "auxiliary_objective": cfg["attention_objective"],
            "attention_gradient_combination": cfg["attention_gradient_combination"],
            "feature_replacement_hook_during_training": False,
            "feature_loss": False,
            "appearance_regularization": False,
            "scheduler_total_outer": cfg["scheduler_total_outer"],
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
            _restore_rng(saved, patch)
            atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
        else:
            require(not previous, "Existing register status without atomic resume")
            require(not optimizer.state, "Fresh register Adam state must be empty")
            save_patch(out / "initial_raw.pt", initial_raw, {
                "method": method,
                "signature": signature,
                "raw_random_patch_sha256": raw_hash,
            })
            save_patch(out / "outer_0000.pt", initial, {
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
                "cost_boundary": "register training including model loads and runtime validation",
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
                    prepared.append((inputs, proprio, observation, target))
                lr = float(optimizer.param_groups[0]["lr"])
                sums = {
                    "main_loss": 0., "aux_loss": 0., "action_grad_norm": 0.,
                    "aux_grad_norm": 0., "aux_grad_scale": 0.,
                    "combined_grad_norm": 0., "register_attention_mass": 0.,
                    "spatial_attention_mass": 0., "patch_token_count": 0.,
                    "target_query_count": 0., "rgb_step_mean_abs_sum": 0.,
                    "rgb_step_max_abs": 0., "effective_rgb_updates": 0,
                }
                for inner in range(cfg["inner_loop"]):
                    seed_transform(cfg["seed"] * 10_000_000 + outer * 10_000 + inner)
                    optimizer.zero_grad(set_to_none=True)
                    inputs, proprio, observation, target = prepared[inner % len(prepared)]
                    geometry = cfg["geometry"] and inner >= cfg["inner_loop"] // 2
                    self.transfer_diverse = bool(outer > cfg["bootstrap_outer"] and geometry)
                    try:
                        with self._capture_dino_qkv(cfg) as captures:
                            action, _, _ = self._timed(costs, "patched_forward_s", lambda:
                                self._forward(inputs, proprio, observation, patch, geometry))
                        bump("training_patched_forward_calls")
                    finally:
                        self.transfer_diverse = False
                    main = F.mse_loss(action[..., [axis]], target[..., [axis]])
                    aux, aux_stats = self._attention_masses(captures, cfg, spec["target"])
                    require(bool(torch.isfinite(main)) and bool(torch.isfinite(aux)),
                            "Nonfinite register training loss")
                    action_grad = self._timed(costs, "action_backward_s", lambda:
                        torch.autograd.grad(main, patch, retain_graph=True)[0])
                    aux_grad = self._timed(costs, "aux_backward_s", lambda:
                        torch.autograd.grad(aux, patch)[0])
                    bump("training_action_backward_calls")
                    bump("training_aux_backward_calls")
                    require(bool(torch.isfinite(action_grad).all()) and bool(torch.isfinite(aux_grad).all()),
                            "Nonfinite register RGB gradient")
                    action_norm = action_grad.norm()
                    aux_norm = aux_grad.norm()
                    require(float(action_norm) > 1e-12 and float(aux_norm) > 1e-12,
                            "Zero action or register gradient")
                    scale = float(spec["gradient_ratio"]) * action_norm / aux_norm.clamp_min(1e-12)
                    combined = action_grad + scale * aux_grad
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
                    sums["main_loss"] += float(main.detach()) / cfg["inner_loop"]
                    sums["aux_loss"] += float(aux.detach()) / cfg["inner_loop"]
                    sums["action_grad_norm"] += float(action_norm.detach()) / cfg["inner_loop"]
                    sums["aux_grad_norm"] += float(aux_norm.detach()) / cfg["inner_loop"]
                    sums["aux_grad_scale"] += float(scale.detach()) / cfg["inner_loop"]
                    sums["combined_grad_norm"] += float(combined.norm().detach()) / cfg["inner_loop"]
                    sums["register_attention_mass"] += aux_stats["register_attention_mass"] / cfg["inner_loop"]
                    sums["spatial_attention_mass"] += aux_stats["spatial_attention_mass"] / cfg["inner_loop"]
                    sums["patch_token_count"] += aux_stats["patch_token_count"] / cfg["inner_loop"]
                    sums["target_query_count"] += aux_stats["target_query_count"] / cfg["inner_loop"]
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
                require(lr == 0. or movement["rgb_outer_max_abs"] > 0.,
                        "Nonzero LR but unchanged register RGB")
                rows.append({
                    "method": method,
                    "target": spec["target"],
                    "gradient_ratio": float(spec["gradient_ratio"]),
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
                    save_patch(out / f"outer_{outer:04d}.pt", patch, {
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
                    print(f"DINO_REGISTER method={method} outer={outer} "
                          f"loss={sums['main_loss']:.6f} aux={sums['aux_loss']:.6f} "
                          f"reg_mass={sums['register_attention_mass']:.5f} "
                          f"tv={structure['tv']:.6f}", flush=True)
            final = out / "outer_0500.pt"
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

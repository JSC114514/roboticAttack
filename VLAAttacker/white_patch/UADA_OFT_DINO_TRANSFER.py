"""Independent Exploring-derived learning curve with a shared counted prefix.

Derived from UADA_OFT_DINO_WINDOW. The original source and completed runs are
unchanged. RGB updates, opposite-boundary action objective and EOT are reused.
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

from white_patch.UADA_OFT_DINO_WINDOW import ExploringOFTDINOWindowAttacker
from white_patch.UADA_OFT_CORRESPONDENCE import run_signature, source_manifest as base_sources
from white_patch.UADA_OFT_DINO import (atomic_json, atomic_torch, atomic_parquet,
                                      save_patch, patch_metrics, require, sha256_file)
from white_patch.UADA_OFT_SOURCE_COPY import init_uada_patch, build_uada_adamw, step_uada_patch
from white_patch.UADA_OFT_ILA import seed_transform
from white_patch.oft_correspondence_objectives import appearance_loss


def source_manifest(cfg, wrapper=None):
    from dino_transfer_common import CONFIG
    sources = base_sources(cfg)
    root = Path(__file__).resolve().parents[2]
    paths = [root / "scripts/dino_transfer_common.py", root / "scripts/dino_window_common.py",
             root / "scripts/dino_replay_common.py", root / "scripts/collect_exploring_oft_dino_replay.py",
             CONFIG]
    paths += list((root / "scripts").glob("*dino_transfer*.py"))
    paths += list((root / "slurm/exploring_oft_dino").glob("*transfer2000*.sbatch"))
    paths += list(root.glob("submit_exploring_oft_dino_transfer2000*.sh"))
    for path in paths:
        sources[str(path.resolve())] = sha256_file(path)
    return sources


def validate_state(saved, cfg):
    require(saved["config"] == cfg and len(saved["rows"]) == saved["outer"],
            "Resume config/row mismatch")
    expected = saved["outer"] * cfg["inner_loop"]
    require(not saved["rows"] or saved["rows"][-1]["optimizer_updates"] == expected,
            "Wrong cumulative updates")
    states = saved["optimizer"]["state"]
    if expected:
        require(len(states) == 1 and all(int(s["step"]) == expected for s in states.values()),
                "Adam moments/step missing or reset")
    require(saved["scheduler"]["last_epoch"] == saved["outer"],
            "Learning-rate scheduler reset or drift")


class ExploringOFTDINOTransferAttacker(ExploringOFTDINOWindowAttacker):
    def _paste_with_support(self, image, patch, geometry):
        if not getattr(self, "transfer_diverse", False):
            return super()._paste_with_support(image, patch, geometry)
        # WHEN-inspired EOT: vary training appearance and Wrist site only.
        # The underlying 7500 RGB parameters and deployment patch stay unchanged.
        original_site = self.site
        try:
            self.site = (random.randint(0, 174), random.randint(0, 174))
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            transformed = ((patch - 0.5) * contrast + 0.5) * brightness
            return super()._paste_with_support(image, transformed.clamp(0, 1), geometry)
        finally:
            self.site = original_site

    def _refresh_scores(self, replay, patch, cfg):
        from collect_exploring_oft_dino_replay import score_record
        from dino_transfer_common import score_and_weights
        # Snapshot RNG: scoring must not perturb subsequent EOT or sample order.
        tr, nr, pr = torch.get_rng_state(), np.random.get_state(), random.getstate()
        cr = torch.cuda.get_rng_state_all() if patch.is_cuda else []
        try:
            displayed = patch.detach().mul(255).round().div(255)
            rows = [dict(r, **score_record(self.wrapper, r["observation"], r["task_name"],
                                         displayed, cfg)) for r in replay]
            return score_and_weights(rows, cfg["data_seed"])
        finally:
            torch.set_rng_state(tr)
            np.random.set_state(nr)
            random.setstate(pr)
            if patch.is_cuda:
                torch.cuda.set_rng_state_all(cr)

    def train_transfer(self, clean, replay, cfg, method, source_hashes, data_hashes,
                      model_load_s=0.):
        from dino_transfer_common import METHODS, draw_record, prefix_state
        require(method in ("warmup", *METHODS) and len(clean) == 300,
                "Transfer method/clean bank drift")
        end = cfg["bootstrap_outer"] if method == "warmup" else cfg["outer_iterations"]
        out = Path(cfg["output_root"]) / "training" / method
        out.mkdir(parents=True, exist_ok=True)
        signature = run_signature(cfg, method, data_hashes, source_hashes)
        status_path, resume_path = out / "status.json", out / "resume.pt"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        require(not previous or previous["signature"] == signature, "Existing signature drift")
        if previous.get("status") == "COMPLETE":
            require(previous["final_patch_sha256"] == sha256_file(out / f"outer_{end:04d}.pt")
                    and previous["resume_sha256"] == sha256_file(resume_path), "Completed output drift")
            print(f"ALREADY_COMPLETE method={method}", flush=True)
            return
        manifest = {"experiment_id": cfg["experiment_id"], "config": cfg, "method": method,
                    "signature": signature, "source": source_hashes, "data": data_hashes,
                    "initialization": "seeded random RGB; shared 250-outer prefix counted",
                    "parameter_count": 7500, "optimizer_restarted_at_branch": False,
                    "scheduler_total_outer": cfg["outer_iterations"], "feature_loss": False,
                    "evaluation_forward": "normal OFT, single Wrist RGB patch"}
        manifest_path = out / "RUN_MANIFEST.json"
        if manifest_path.exists():
            require(json.loads(manifest_path.read_text()) == manifest, "Manifest drift")
        else:
            atomic_json(manifest, manifest_path)
        seed_transform(cfg["seed"])
        patch = init_uada_patch(tuple(cfg["patch_shape"]), self.wrapper.device)
        initial = patch.detach().clone()
        optimizer, scheduler = build_uada_adamw(patch, cfg["learning_rate"],
            cfg["warmup_outer_iterations"], cfg["outer_iterations"], cfg["weight_decay"])
        rows, start, prior_elapsed, prefix_elapsed, costs = [], 1, 0., 0., {}
        saved = None
        replay = [dict(r) for r in replay]
        score_after = cfg["bootstrap_outer"]
        score_fields = ("dino_score", "dino_delta_rms", "patched_target_mse",
                        "restored_target_mse", "dino_weight", "shuffled_weight")
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            require(saved["signature"] == signature and saved["method"] == method, "Resume identity drift")
        elif method != "warmup":
            saved, digest, _ = prefix_state(cfg)
            require(data_hashes["prefix_resume"] == digest, "Prefix changed after manifest")
        else:
            require(not previous, "Status exists without valid resume")
        if saved is not None:
            validate_state(saved, cfg)
            require(torch.equal(saved["initial"], initial.cpu()) and 0 <= saved["outer"] <= end,
                    "Initial RGB or budget changed")
            with torch.no_grad():
                patch.copy_(saved["patch"].to(patch.device))
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            rows, start = saved["rows"], saved["outer"] + 1
            prior_elapsed = max(float(saved["elapsed_s"]), float(previous.get("elapsed_s", 0.)))
            prefix_elapsed = (float(saved["elapsed_s"]) if saved["method"] == "warmup" and method != "warmup"
                              else float(saved.get("prefix_elapsed_s", 0.)))
            costs = dict(saved["costs"])
            score_after = saved.get("score_after", cfg["bootstrap_outer"])
            scoring = saved.get("replay_scoring", [])
            if scoring:
                require(len(scoring) == len(replay), "Resume score coverage drift")
                replay = [dict(r, **s) for r, s in zip(replay, scoring, strict=True)]
            torch.set_rng_state(saved["torch_rng"])
            if patch.is_cuda:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
        if not resume_path.exists():
            save_patch(out / "outer_0000.pt", initial, {"method": method, "signature": signature,
                                                       "total_outer": 0, "total_optimizer_updates": 0})
            if method != "warmup":
                save_patch(out / f"outer_{cfg['bootstrap_outer']:04d}.pt", patch,
                           {"method": method, "signature": signature, "total_outer": cfg["bootstrap_outer"],
                            "total_optimizer_updates": cfg["bootstrap_outer"] * cfg["inner_loop"]})
        costs["model_load_s"] = costs.get("model_load_s", 0.) + model_load_s
        low = torch.tensor(cfg["action_head_q01"], device=patch.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=patch.device).reshape(1, 1, 7)
        started = time.monotonic()
        outer = start - 1

        def elapsed():
            return prior_elapsed + model_load_s + time.monotonic() - started

        def save_resume():
            payload = {"signature": signature, "config": cfg, "method": method, "outer": outer,
                "patch": patch.detach().cpu(), "initial": initial.detach().cpu(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rows": rows, "costs": costs, "elapsed_s": elapsed(), "prefix_elapsed_s": prefix_elapsed,
                "score_after": score_after,
                "replay_scoring": [{k: r[k] for k in score_fields if k in r} for r in replay],
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if patch.is_cuda else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
            validate_state(payload, cfg)
            atomic_torch(payload, resume_path)

        def progress(status, **extra):
            return {"status": status, "method": method, "signature": signature,
                    "outer": len(rows), "optimizer_updates": len(rows) * cfg["inner_loop"],
                    "elapsed_s": elapsed(), "prefix_elapsed_s": prefix_elapsed, "costs": costs,
                    "cost_boundary": "training including shared prefix and model loads; collection reported separately",
                    "visual_review": ("AUTO_VALIDATION" if cfg.get("auto_eval_after_training", False)
                                      else "REQUIRED"), **extra}

        try:
            if not resume_path.exists():
                save_resume()
            alignment = self._timed(costs, "runtime_validation_s",
                                   lambda: self._validate_runtime(clean[0], patch.detach(), cfg))
            atomic_json(alignment, out / "RUNTIME_ALIGNMENT.json")
            for outer in range(start, end + 1):
                if method in ("dynamic", "diverse") and outer - 1 in cfg["refresh_after_outer"]:
                    replay = self._timed(costs, "score_refresh_s",
                                         lambda: self._refresh_scores(replay, patch, cfg))
                    score_after = outer - 1
                    atomic_json({"score_after_outer": score_after,
                                 "patch_sha256": sha256_file(out / f"outer_{score_after:04d}.pt"),
                                 "records": [{k: r[k] for k in score_fields if k in r} for r in replay]},
                                out / f"scores_after_{score_after:04d}.json")
                    print(f"DINO_REFRESH method={method} after={score_after}", flush=True)
                before_outer = patch.detach().clone()
                record, origin, index, base_index = draw_record(clean, replay, method, outer, cfg)
                observations = record["observations"] if origin == "replay" else [record["observation"]]
                prepared = []
                for obs in observations:
                    inputs, proprio = self._base_inputs({"observation": obs, "task_name": record["task_name"]})
                    with torch.no_grad():
                        action, _, _ = self._timed(costs, "clean_prepare_s",
                            lambda: self._forward(inputs, proprio, obs, None, False))
                        target = torch.where(action >= (low + high) / 2, low, high).detach()
                    prepared.append((inputs, proprio, obs, target))
                losses = []
                lr = optimizer.param_groups[0]["lr"]
                movement = {}
                for inner in range(cfg["inner_loop"]):
                    seed_transform(cfg["seed"] * 10_000_000 + outer * 10_000 + inner)
                    optimizer.zero_grad(set_to_none=True)
                    inputs, proprio, obs, target = prepared[inner % len(prepared)]
                    geometry = cfg["geometry"] and inner >= cfg["inner_loop"] // 2
                    self.transfer_diverse = method == "diverse" and geometry
                    try:
                        action, _, _ = self._timed(costs, "patched_forward_s",
                            lambda: self._forward(inputs, proprio, obs, patch, geometry))
                    finally:
                        self.transfer_diverse = False
                    main = F.mse_loss(action[..., cfg["maskidx"]], target[..., cfg["maskidx"]])
                    tv = appearance_loss(patch, cfg["tv_threshold"]) * cfg["tv_weight"]
                    monitor = inner == 0 and (outer == start or outer % 50 == 0)
                    if monitor:
                        action_grad = torch.autograd.grad(main, patch, retain_graph=True)[0]
                        tv_grad = torch.autograd.grad(tv, patch, retain_graph=True)[0]
                        require(bool(torch.isfinite(action_grad).all()), "Nonfinite action RGB gradient")
                        if outer == start:
                            require(float(action_grad.norm()) > 1e-12, "Zero real-model action gradient")
                        movement = {"action_gradient_norm": float(action_grad.norm()),
                                    "tv_gradient_norm": float(tv_grad.norm())}
                    patch.grad = torch.autograd.grad(main + tv, patch)[0]
                    require(bool(torch.isfinite(patch.grad).all()), "Nonfinite RGB gradient")
                    if outer == start and inner == 0:
                        require(float(patch.grad.norm()) > 1e-12, "Zero real-model patch gradient")
                    step_uada_patch(patch, optimizer, self.wrapper.vla)
                    losses.append(float(main.detach()))
                scheduler.step()
                structure = patch_metrics(patch, initial)
                delta = patch.detach() - before_outer
                movement.update(rgb_outer_mean_abs=float(delta.abs().mean()),
                                rgb_outer_max_abs=float(delta.abs().max()))
                require(lr == 0 or movement["rgb_outer_max_abs"] > 0,
                        "Nonzero learning rate but zero RGB update over entire outer")
                rows.append({"method": method, "outer_iteration": outer,
                             "optimizer_updates": outer * cfg["inner_loop"], "origin": origin,
                             "record_index": index, "base_record_index": base_index,
                             "task_id": int(record["task_id"]), "state_index": int(record["state_index"]),
                             "dino_score": float(record.get("dino_score", 0.)), "learning_rate": lr,
                             "main_loss": float(np.mean(losses)), "elapsed_s": elapsed(),
                             "score_after_outer": score_after, **movement, **structure})
                if outer in cfg["checkpoint_outer_iterations"]:
                    save_patch(out / f"outer_{outer:04d}.pt", patch,
                               {"method": method, "signature": signature, "total_outer": outer,
                                "total_optimizer_updates": outer * cfg["inner_loop"], **structure})
                if outer % cfg["resume_every_outer_iterations"] == 0 or outer == end:
                    save_resume()
                    atomic_parquet(pd.DataFrame(rows), out / "training.parquet")
                    atomic_json(progress("RUNNING", **structure), status_path)
                    print(f"DINO_TRANSFER method={method} outer={outer}/{end} "
                          f"loss={np.mean(losses):.5f} tv={structure['tv']:.5f} origin={origin}", flush=True)
            final = out / f"outer_{end:04d}.pt"
            atomic_json(progress("COMPLETE", final_patch_sha256=sha256_file(final),
                                 final_png_sha256=sha256_file(final.with_suffix(".png")),
                                 resume_sha256=sha256_file(resume_path),
                                 **patch_metrics(patch, initial)), status_path)
        except BaseException as error:
            atomic_json(progress("FAILED", attempted_outer=outer, error=repr(error), resume=str(resume_path)), status_path)
            raise

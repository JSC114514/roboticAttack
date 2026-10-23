"""Contracts for OFT-DINO-REGISTER-DELAYED2000-V2.

This experiment keeps the Exploring-derived UADA action objective and changes
only the training-time auxiliary gradient: DINOv2 register/query tokens are
encouraged to read the actual Wrist patch token footprint. Evaluation uses a
normal OFT forward with the learned RGB patch only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))

from dino_state_axis_common import (  # noqa: E402
    BANK_SHA, REPLAY_SHA, SOURCE_SHA, check_rgb, digest_json, frozen_digest,
    replay_records, source_records,
)
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file  # noqa: E402

CONFIG = Path(os.environ.get(
    "OFT_DINO_REGISTER_DELAYED_CONFIG",
    ROOT / "configs/exploring_oft_dino/object_dino_register_delayed2000_v2.json",
))
METHODS = TRAIN_ORDER = ("register_late_g10", "spatial_late_g10", "register_late_projected_g10")
REFERENCES = ("original_uada", "uniform_axis0")
CONDITIONS = (*METHODS, *REFERENCES)
CHECKPOINTS = (1000, 2000)
FINAL_OUTER = 2000
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def config(path: str | Path = CONFIG) -> dict:
    raw = _load_json(path)
    parent = _load_json(raw["parent_config"])
    cfg = {**parent, **raw}
    cfg["config_path"] = str(Path(path).resolve())
    cfg["parent_config_sha256"] = sha256_file(raw["parent_config"])
    require(cfg["experiment_id"] == "OFT-DINO-REGISTER-DELAYED2000-V2"
            and tuple(cfg["methods"]) == METHODS
            and cfg["seed"] == 42
            and cfg["data_seed"] == cfg["rollout_seed"] == 20260929,
            "Register experiment identity/seed changed")
    require(cfg["suite"] == "libero_object"
            and cfg["unnorm_key"] == "libero_object_no_noops"
            and cfg["training_bank_sha256"] == BANK_SHA
            and cfg["replay_bank_sha256"] == REPLAY_SHA
            and cfg["sampling_policy"] == "object_replay_uniform_axis0_with_delayed_register_attention_v2",
            "Register data/source-suite policy drift")
    require(cfg["base_outer_iterations"] == 0
            and cfg["bootstrap_outer"] == 250
            and cfg["diverse_after_outer"] == 250
            and cfg["outer_iterations"] == 2000
            and cfg["auxiliary_start_outer"] == 251
            and cfg["primary_checkpoint_outer"] == 2000
            and cfg["scheduler_total_outer"] == 2000
            and cfg["inner_loop"] == 50
            and cfg["checkpoint_outer_iterations"] == [0, 250, 500, 1000, 1500, 2000]
            and cfg["eval_checkpoint_outer_iterations"] == list(CHECKPOINTS)
            and cfg["refresh_after_outer"] == []
            and cfg["resume_every_outer_iterations"] == 1,
            "Register budget/checkpoint schedule drift")
    require(cfg["patch_shape"] == [3, 50, 50]
            and cfg["patch_xy"] == [26, 137]
            and cfg["maskidx"] == [0]
            and cfg["clean_axis"] == 0
            and cfg["batch_size"] == 1
            and cfg["clip_length"] == 3
            and cfg["learning_rate"] == .002
            and cfg["weight_decay"] == 0
            and cfg["warmup_outer_iterations"] == 20
            and cfg["geometry"] is True
            and cfg["resize_patch"] is False
            and cfg["dtype"] == "bfloat16"
            and cfg["num_open_loop_steps"] == 8,
            "Threat/optimizer/EOT drift")
    require(cfg["register_attention_layers"] == [8, 16]
            and cfg["register_query_tokens"] == [1, 2, 3, 4]
            and cfg["dino_prefix_tokens"] == 5
            and cfg["dino_grid_size"] == [16, 16]
            and cfg["dino_patch_size"] == 14,
            "DINO register-attention target changed")
    require(set(cfg["method_specs"]) == set(METHODS), "Missing register method specs")
    expected_specs = {
        "register_late_g10": {"target": "register", "gradient_ratio": .10, "project_conflict": False},
        "spatial_late_g10": {"target": "spatial", "gradient_ratio": .10, "project_conflict": False},
        "register_late_projected_g10": {"target": "register", "gradient_ratio": .10, "project_conflict": True},
    }
    require(cfg["method_specs"] == expected_specs, "Delayed target/weight/projection drift")
    require(Path(cfg["output_root"]).resolve()
            == (ROOT / "research_runs/wrist_dino_register_v2").resolve()
            and cfg["auto_eval_after_training"] is True
            and cfg["eval_hooks"] is False
            and cfg["training_gpu_limit"] == 3
            and cfg["evaluation_gpu_limit"] == 8,
            "Output/evaluation policy drift")
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(source.stat().st_mode & 0o222 == 0 and sha256_file(source) == SOURCE_SHA,
            "Read-only Exploring source changed")
    for key in ("training_bank", "replay_bank", "eval_reference"):
        require(frozen_digest(cfg, cfg[key]) == cfg[f"{key}_sha256"],
                f"{key} identity drift")
    return cfg


def method_spec(cfg: dict, method: str) -> dict:
    require(method in METHODS, f"Unknown register method: {method}")
    return dict(cfg["method_specs"][method])


def reference_patches(cfg: dict) -> dict[int, dict[str, dict]]:
    expected = {f"{name}_{outer}" for outer in CHECKPOINTS for name in REFERENCES}
    require(set(cfg["external_references"]) == expected, "Reference budget/coverage changed")
    result = {outer: {} for outer in CHECKPOINTS}
    for outer in CHECKPOINTS:
        for condition in REFERENCES:
            item = cfg["external_references"][f"{condition}_{outer}"]
            path = Path(item["pt"])
            require(path.name == f"outer_{outer:04d}.pt" and int(item["checkpoint_outer"]) == outer,
                    f"Reference budget changed: {condition}/{outer}")
            for file, key in ((path, "pt_sha256"), (path.with_suffix(".png"), "png_sha256"),
                              (Path(item["manifest"]), "manifest_sha256"),
                              (Path(item["history"]), "history_sha256")):
                require(file.is_file() and sha256_file(file) == item[key], f"Reference changed: {file}")
            history = pd.read_parquet(item["history"])
            row = history.loc[history.outer_iteration.eq(outer)]
            require(len(row) == 1 and int(row.iloc[0].optimizer_updates) == outer * cfg["inner_loop"],
                    f"Reference training budget missing: {condition}/{outer}")
            result[outer][condition] = {
                "patch": check_rgb(path, path.with_suffix(".png")),
                "pt": item["pt_sha256"], "png": item["png_sha256"],
                "checkpoint_outer": outer,
                "recorded_training_elapsed_s": float(row.iloc[0].elapsed_s),
            }
    return result


def source_manifest(cfg: dict, wrapper=None) -> dict[str, str]:
    from white_patch.UADA_OFT_DINO_STATE_AXIS import source_manifest as state_sources

    sources = state_sources(cfg, wrapper)
    extra = [
        CONFIG,
        ROOT / "research_runs/wrist_dino_register_v2/PLAN.md",
        ROOT / "scripts/dino_register_delayed_common.py",
        ROOT / "scripts/run_exploring_oft_dino_register_delayed_cross_suite.py",
        ROOT / "scripts/analyze_exploring_oft_dino_register_delayed_cross_suite.py",
        ROOT / "VLAAttacker/UADA_OFT_DINO_REGISTER_DELAYED_wrapper.py",
        ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO_REGISTER_DELAYED.py",
        ROOT / "submit_exploring_oft_dino_register_delayed2000_v2.sh",
        ROOT / "tests/test_dino_register_delayed_cpu.py",
    ]
    extra.extend(sorted((ROOT / "slurm/exploring_oft_dino").glob("*register_delayed*.sbatch")))
    for path in extra:
        require(Path(path).is_file(), f"Missing register source: {path}")
        sources[str(Path(path).resolve())] = sha256_file(path)
    return sources


def data_identity(cfg: dict) -> dict:
    return {
        "clean_bank": frozen_digest(cfg, cfg["training_bank"]),
        "window_bank": frozen_digest(cfg, cfg["replay_bank"]),
        "config": sha256_file(CONFIG),
        "parent_config": cfg["parent_config_sha256"],
    }


def source_preflight(cfg: dict) -> dict:
    for table in ("frozen_inputs", "frozen_eval_assets"):
        for name in cfg[table]:
            frozen_digest(cfg, name)
    sources = source_manifest(cfg)
    require(sources and all(Path(p).is_file() and sha256_file(p) == h for p, h in sources.items()),
            "Register source fingerprint failed")
    return {
        "status": "PASS",
        "source": sources,
        "data": data_identity(cfg),
        "model_loaded": False,
        "submitted": False,
        "original_uada_sha256": SOURCE_SHA,
    }


def pipeline_preflight(cfg: dict) -> dict:
    source = source_preflight(cfg)
    clean, (replay, replay_digest) = source_records(cfg), replay_records(cfg)
    refs = reference_patches(cfg)
    from white_patch.oft_dino_register_delayed_objectives import cpu_preflight
    objective_check = cpu_preflight(cfg)
    # Exercise the uniform replay schedule without loading an OFT model.
    from dino_state_axis_common import draw_record

    live_replay = [dict(r) for r in replay]
    for outer in range(1, cfg["outer_iterations"] + 1):
        for method in METHODS:
            draw_record(clean, live_replay, "uniform_axis0", outer, cfg)
            method_spec(cfg, method)
    from run_exploring_oft_scale500_cross_suite import frozen_clean

    eval_cfg = _load_json(cfg["eval_reference"])
    denominators = {suite: int(frozen_clean(eval_cfg, suite).success.sum())
                    for suite in eval_cfg["suites"]}
    require(denominators == DENOMINATORS, "Frozen clean denominators changed")
    return {
        **source,
        "clean_records": len(clean),
        "replay_records": len(replay),
        "replay_sha256": replay_digest,
        "fresh_start_outer": 0,
        "new_training_methods": list(METHODS),
        "conditions": list(CONDITIONS),
        "external_references": {
            str(outer): {name: {k: v for k, v in item.items() if k != "patch"}
                         for name, item in arms.items()}
            for outer, arms in refs.items()
        },
        "objective_cpu_check": objective_check,
        "auxiliary_start_outer": cfg["auxiliary_start_outer"],
        "evaluation_checkpoint_outer_iterations": list(CHECKPOINTS),
        "denominators": denominators,
        "maximum_attack_episodes": len(CONDITIONS) * len(CHECKPOINTS) * 400,
        "training_gpu_limit": cfg["training_gpu_limit"],
        "evaluation_gpu_limit": cfg["evaluation_gpu_limit"],
        "bootstrap_draws": cfg["bootstrap_draws"],
        "bootstrap_seed": cfg["bootstrap_seed"],
    }


def compact_preflight(report: dict) -> dict:
    result = dict(report)
    sources = result.pop("source", {})
    result.update(source_files=len(sources), source_manifest_digest=digest_json(sources))
    return result


def preparation_identity(cfg: dict, report: dict) -> dict:
    return {"config": cfg, "source": report["source"], "data": report["data"]}


def prepare(cfg: dict) -> dict:
    report = pipeline_preflight(cfg)
    identity = preparation_identity(cfg, report)
    result = {
        "status": "COMPLETE",
        "identity": identity,
        "signature": digest_json(identity),
        "experiment_id": cfg["experiment_id"],
        "preflight": compact_preflight(report),
    }
    path = Path(cfg["output_root"]) / "reports/PREPARATION.json"
    if path.exists():
        require(json.loads(path.read_text()) == result, "Existing preparation changed")
    else:
        atomic_json(result, path)
    return {"status": "COMPLETE", "signature": result["signature"], "preparation": str(path)}


def require_preparation(cfg: dict, report: dict | None = None) -> dict:
    report = report or source_preflight(cfg)
    path = Path(cfg["output_root"]) / "reports/PREPARATION.json"
    saved = json.loads(path.read_text())
    require(saved["status"] == "COMPLETE"
            and saved["signature"] == digest_json(saved["identity"])
            and saved["identity"] == preparation_identity(cfg, report),
            "CPU preparation signature/source/data changed")
    return saved


def validate_training(cfg: dict, methods: tuple[str, ...] = METHODS, *, write_report: bool = True) -> dict:
    require_preparation(cfg)
    refs = reference_patches(cfg)
    expected_data = data_identity(cfg)
    checkpoints, movement, metrics = {}, {}, []
    raw_expected = torch.rand(tuple(cfg["patch_shape"]), generator=torch.Generator().manual_seed(42))
    for method in methods:
        require(method in METHODS, f"Invalid method: {method}")
        folder = Path(cfg["output_root"]) / "training" / method
        status = _load_json(folder / "status.json")
        manifest = _load_json(folder / "RUN_MANIFEST.json")
        require(status["status"] == "COMPLETE" and status["method"] == method
                and status["outer"] == FINAL_OUTER
                and status["optimizer_updates"] == FINAL_OUTER * cfg["inner_loop"]
                and manifest["config"] == cfg and manifest["data"] == expected_data
                and manifest["signature"] == status["signature"]
                and manifest["fresh_start_outer"] == 0 and manifest["parameter_count"] == 7500
                and manifest["method_spec"] == method_spec(cfg, method)
                and manifest["scheduler_total_outer"] == cfg["scheduler_total_outer"]
                and manifest["feature_replacement_hook_during_training"] is False
                and manifest["appearance_regularization"] is False
                and manifest["auxiliary_start_outer"] == 251
                and manifest["evaluation_forward"] == "normal OFT, quantized single Wrist RGB patch",
                f"Delayed training lineage failed: {method}")
        for path, digest in manifest["source"].items():
            require(Path(path).is_file() and sha256_file(path) == digest, f"Completed source changed: {path}")
        require(status["source_snapshot_sha256"] == sha256_file(folder / "source_snapshot.zip")
                and status["resume_sha256"] == sha256_file(folder / "resume.pt"),
                f"Committed training artifacts changed: {method}")
        from white_patch.UADA_OFT_DINO_REGISTER_DELAYED import validate_state
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        validate_state(saved, cfg)
        require(saved["outer"] == FINAL_OUTER and saved["signature"] == status["signature"],
                "Final resume budget/identity mismatch")
        require(_load_json(folder / "RUNTIME_ALIGNMENT.json")["status"] == "PASS",
                "Actual Wrist-DINO alignment did not pass")
        initial = check_rgb(folder / "outer_0000.pt", folder / "outer_0000.png")
        final = check_rgb(folder / "outer_2000.pt", folder / "outer_2000.png")
        require(status["final_patch_sha256"] == sha256_file(folder / "outer_2000.pt")
                and status["final_png_sha256"] == sha256_file(folder / "outer_2000.png")
                and torch.equal(saved["patch"].float(), final),
                "Final checkpoint/status/resume RGB mismatch")
        require(torch.equal(initial, raw_expected), f"Fresh seed42 RGB changed: {method}")
        delta = (final - initial).abs()
        require(float(delta.max()) > 0 and float(delta.mean()) > 1e-7, f"Unchanged RGB: {method}")
        rows = pd.read_parquet(folder / "training.parquet")
        require(len(rows) == FINAL_OUTER and rows.outer_iteration.tolist() == list(range(1, FINAL_OUTER + 1))
                and rows.optimizer_updates.tolist() == list(range(cfg["inner_loop"], FINAL_OUTER * cfg["inner_loop"] + 1, cfg["inner_loop"]))
                and np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy()).all()
                and rows.aux_grad_finite.all()
                and rows.auxiliary_enabled.eq(rows.outer_iteration.ge(251)).all(),
                f"Training coverage/numerics/schedule failed: {method}")
        warm = rows.loc[~rows.auxiliary_enabled]
        require(len(warm) == 250 and warm.aux_grad_norm.eq(0).all()
                and warm.aux_loss.eq(0).all() and warm.effective_aux_gradient_ratio.eq(0).all(),
                "Auxiliary gradient leaked into the action-only prefix")
        require(rows.effective_aux_gradient_ratio.between(0, .10001).all(),
                "Auxiliary gradient exceeded the fixed 10% budget")
        if method_spec(cfg, method)["project_conflict"]:
            require(rows.loc[rows.auxiliary_enabled].projected_gradient_cosine.ge(-1e-5).all(),
                    "Projected auxiliary gradient still opposes the raw main gradient")
        else:
            require(rows.projection_applied_fraction.eq(0).all(), "Unexpected auxiliary projection")
        unchanged = rows.loc[rows.learning_rate.ne(0) & rows.rgb_outer_max_abs.eq(0)]
        require(unchanged.outer_iteration.tolist() == [r["outer"] for r in saved["terminal_rounding_history"]],
                "Unchanged RGB requires explicit terminal float32 evidence")
        checkpoints[method] = {}
        for outer in cfg["checkpoint_outer_iterations"]:
            path = folder / f"outer_{outer:04d}.pt"
            check_rgb(path, path.with_suffix(".png"))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(payload["total_outer"] == outer
                    and payload["total_optimizer_updates"] == outer * cfg["inner_loop"]
                    and payload["signature"] == status["signature"]
                    and payload["method"] == method,
                    "Checkpoint budget/method/signature mismatch")
            checkpoints[method][str(outer)] = {"pt": sha256_file(path), "png": sha256_file(path.with_suffix(".png"))}
        movement[method] = {"mae_from_initial": float(delta.mean()),
                           "changed_fraction_from_initial": float(final.mul(255).round().ne(initial.mul(255).round()).float().mean()),
                           "final_tv": float(rows.iloc[-1].tv)}
        for outer in CHECKPOINTS:
            history = rows.loc[rows.outer_iteration.le(outer)]
            active = history.loc[history.auxiliary_enabled]
            current = check_rgb(folder / f"outer_{outer:04d}.pt", folder / f"outer_{outer:04d}.png")
            metrics.append({
                "condition": method, "checkpoint_outer": outer,
                "gradient_ratio": .10, "target": method_spec(cfg, method)["target"],
                "project_conflict": method_spec(cfg, method)["project_conflict"],
                "mean_action_loss": float(history.main_loss.mean()),
                "mean_active_aux_loss": float(active.aux_loss.mean()),
                "mean_active_register_mass": float(active.register_attention_mass.mean()),
                "mean_active_spatial_mass": float(active.spatial_attention_mass.mean()),
                "mean_raw_gradient_cosine": float(active.raw_gradient_cosine.mean()),
                "conflict_fraction": float(active.gradient_conflict_fraction.mean()),
                "projection_applied_fraction": float(active.projection_applied_fraction.mean()),
                "mean_effective_aux_ratio": float(active.effective_aux_gradient_ratio.mean()),
                "rgb_mae_from_initial": float((current - initial).abs().mean()),
                "rgb_mae_from_outer500": float((current - check_rgb(folder / "outer_0500.pt", folder / "outer_0500.png")).abs().mean()),
                "quantized_changed_fraction_from_outer500": float(current.mul(255).round().ne(
                    check_rgb(folder / "outer_0500.pt", folder / "outer_0500.png").mul(255).round()).float().mean()),
                "final_tv": float(history.iloc[-1].tv),
            })
    result = {
        "status": "COMPLETE", "experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG),
        "checkpoints": checkpoints,
        "references": {str(outer): {name: {k: v for k, v in item.items() if k != "patch"}
                                   for name, item in arms.items()} for outer, arms in refs.items()},
        "movement_from_initial": movement, "fresh_start_outer": 0,
        "optimizer_updates": FINAL_OUTER * cfg["inner_loop"], "scheduler_total_outer": 2000,
        "evaluation_checkpoint_outer_iterations": list(CHECKPOINTS),
        "evaluation_ready": True, "evaluation_hooks": False,
    }
    if write_report:
        require(tuple(methods) == METHODS, "Final validation requires all delayed arms")
        root = Path(cfg["output_root"])
        atomic_json(result, root / "AUTO_VALIDATION.json")
        pd.DataFrame(metrics).to_csv(root / "REGISTER_TRAINING_METRICS.csv", index=False)
    return result


def submit_chain(cfg: dict) -> dict:
    root = Path(cfg["output_root"])
    final = root / "evaluation/four_suite/STATUS.json"
    if final.exists():
        status = json.loads(final.read_text())
        if status.get("status") == "COMPLETE":
            return {"status": "ALREADY_COMPLETE", "jobs": []}
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    with (root / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ",".join([
            "oft_reglate_prepare", "oft_reglate_train", "oft_reglate_validate",
            "oft_reglate_eval", "oft_reglate_stats",
        ])
        queue = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", names],
                               check=False, capture_output=True, text=True)
        require(queue.returncode == 0,
                "squeue query failed; duplicate-chain state unknown: " + queue.stderr.strip())
        require(not queue.stdout.strip(),
                "A register chain remains queued; inspect its receipt/logs before resubmitting")
        completed = [
            m for m in METHODS
            if (root / "training" / m / "status.json").exists()
            and json.loads((root / "training" / m / "status.json").read_text()).get("status") == "COMPLETE"
        ]
        if completed:
            validate_training(cfg, tuple(completed), write_report=False)
        pending = [i for i, m in enumerate(TRAIN_ORDER) if m not in completed]
        receipt_path = root / "reports/SUBMISSION_RECEIPT.json"
        receipt = {
            "status": "SUBMITTING",
            "experiment_id": cfg["experiment_id"],
            "config_sha256": sha256_file(CONFIG),
            "jobs": [],
        }
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            require(prior["config_sha256"] == receipt["config_sha256"],
                    "Existing submission config changed")
            receipt["previous_receipt"] = prior
        atomic_json(receipt, receipt_path)
        dependency = None
        stages = [("prepare", "132_register_delayed_prepare_cpu.sbatch", None)]
        if pending:
            stages.append(("train", "133_register_delayed_train_4090.sbatch",
                           ",".join(map(str, pending)) + f"%{min(3, len(pending))}"))
        stages.extend([
            ("validate", "134_register_delayed_validate_cpu.sbatch", None),
            ("eval", "135_register_delayed_eval_3090.sbatch", "0-9%8"),
            ("stats", "136_register_delayed_stats_cpu.sbatch", None),
        ])
        for stage, script, array in stages:
            command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
            if dependency:
                command.append(f"--dependency=afterok:{dependency}")
            if array:
                command.append(f"--array={array}")
            command.append(str(ROOT / "slurm/exploring_oft_dino" / script))
            response = subprocess.run(command, check=False, capture_output=True, text=True)
            job_id = response.stdout.strip().split(";")[0]
            if response.returncode or not re.fullmatch(r"[0-9]+", job_id):
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage,
                               error=response.stderr.strip() or response.stdout.strip())
                atomic_json(receipt, receipt_path)
                raise RuntimeError(f"Submission failed at {stage}: {receipt['error']}")
            receipt["jobs"].append({"stage": stage, "job_id": job_id,
                                    "dependency": dependency, "command": command})
            atomic_json(receipt, receipt_path)
            dependency = job_id
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, receipt_path)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    cfg = config()
    action = (
        submit_chain if args.submit
        else validate_training if args.validate
        else prepare if args.prepare
        else pipeline_preflight
    )
    report = action(cfg)
    print(json.dumps(compact_preflight(report) if args.preflight else report, indent=2))

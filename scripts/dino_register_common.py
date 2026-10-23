"""Contracts for OFT-DINO-REGISTER500-V1.

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
    "OFT_DINO_REGISTER_CONFIG",
    ROOT / "configs/exploring_oft_dino/object_dino_register500_v1.json",
))
METHODS = TRAIN_ORDER = ("register_g10", "register_g30", "spatial_g30")
REFERENCES = ("original_uada_500", "uniform_axis0_500")
CONDITIONS = (*METHODS, *REFERENCES)
CHECKPOINT = 500
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def config(path: str | Path = CONFIG) -> dict:
    raw = _load_json(path)
    parent = _load_json(raw["parent_config"])
    cfg = {**parent, **raw}
    cfg["config_path"] = str(Path(path).resolve())
    cfg["parent_config_sha256"] = sha256_file(raw["parent_config"])
    require(cfg["experiment_id"] == "OFT-DINO-REGISTER500-V1"
            and tuple(cfg["methods"]) == METHODS
            and cfg["seed"] == 42
            and cfg["data_seed"] == cfg["rollout_seed"] == 20260929,
            "Register experiment identity/seed changed")
    require(cfg["suite"] == "libero_object"
            and cfg["unnorm_key"] == "libero_object_no_noops"
            and cfg["training_bank_sha256"] == BANK_SHA
            and cfg["replay_bank_sha256"] == REPLAY_SHA
            and cfg["sampling_policy"] == "object_replay_uniform_axis0_with_internal_dino_register_attention_v1",
            "Register data/source-suite policy drift")
    require(cfg["base_outer_iterations"] == 0
            and cfg["bootstrap_outer"] == 250
            and cfg["diverse_after_outer"] == 250
            and cfg["outer_iterations"] == 500
            and cfg["scheduler_total_outer"] == 2000
            and cfg["inner_loop"] == 50
            and cfg["checkpoint_outer_iterations"] == [0, 250, 500]
            and cfg["eval_checkpoint_outer_iterations"] == [500]
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
    for method, spec in cfg["method_specs"].items():
        require(spec["target"] in {"register", "spatial"}
                and spec["gradient_ratio"] in {0.10, 0.30},
                f"Invalid register method spec: {method}")
    require(Path(cfg["output_root"]).resolve()
            == (ROOT / "research_runs/wrist_dino_register_v1").resolve()
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


def reference_patches(cfg: dict) -> dict[str, dict]:
    require(set(cfg["external_references"]) == set(REFERENCES), "Reference coverage changed")
    result = {}
    for condition, item in cfg["external_references"].items():
        path = Path(item["pt"])
        require(path.name == "outer_0500.pt" and int(item["checkpoint_outer"]) == CHECKPOINT,
                f"Reference budget changed: {condition}")
        for file, key in (
            (path, "pt_sha256"),
            (path.with_suffix(".png"), "png_sha256"),
            (Path(item["manifest"]), "manifest_sha256"),
            (Path(item["history"]), "history_sha256"),
        ):
            require(file.is_file() and sha256_file(file) == item[key],
                    f"Reference changed: {file}")
        patch = check_rgb(path, path.with_suffix(".png"))
        history = pd.read_parquet(item["history"])
        row = history.loc[history.outer_iteration.eq(CHECKPOINT)]
        require(len(row) == 1 and int(row.iloc[0].optimizer_updates) == CHECKPOINT * 50,
                f"Reference history row missing: {condition}")
        result[condition] = {
            "patch": patch,
            "pt": item["pt_sha256"],
            "png": item["png_sha256"],
            "checkpoint_outer": CHECKPOINT,
            "recorded_training_elapsed_s": float(row.iloc[0].elapsed_s),
        }
    return result


def source_manifest(cfg: dict, wrapper=None) -> dict[str, str]:
    from white_patch.UADA_OFT_DINO_STATE_AXIS import source_manifest as state_sources

    sources = state_sources(cfg, wrapper)
    extra = [
        CONFIG,
        ROOT / "research_runs/wrist_dino_register_v1/PLAN.md",
        ROOT / "scripts/dino_register_common.py",
        ROOT / "scripts/run_exploring_oft_dino_register_cross_suite.py",
        ROOT / "scripts/analyze_exploring_oft_dino_register_cross_suite.py",
        ROOT / "VLAAttacker/UADA_OFT_DINO_REGISTER_wrapper.py",
        ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO_REGISTER.py",
        ROOT / "submit_exploring_oft_dino_register500_v1.sh",
    ]
    extra.extend(sorted((ROOT / "slurm/exploring_oft_dino").glob("*register*.sbatch")))
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
    # Exercise the uniform replay schedule for every new method without loading a model.
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
            name: {k: v for k, v in item.items() if k != "patch"}
            for name, item in refs.items()
        },
        "denominators": denominators,
        "maximum_attack_episodes": len(CONDITIONS) * 400,
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
        status = json.loads((folder / "status.json").read_text())
        manifest = json.loads((folder / "RUN_MANIFEST.json").read_text())
        require(status["status"] == "COMPLETE"
                and status["method"] == method
                and status["outer"] == CHECKPOINT
                and status["optimizer_updates"] == CHECKPOINT * cfg["inner_loop"]
                and manifest["config"] == cfg
                and manifest["data"] == expected_data
                and manifest["signature"] == status["signature"]
                and manifest["fresh_start_outer"] == 0
                and manifest["parameter_count"] == 7500
                and manifest["scheduler_total_outer"] == cfg["scheduler_total_outer"]
                and manifest["feature_replacement_hook_during_training"] is False
                and manifest["evaluation_forward"] == "normal OFT, quantized single Wrist RGB patch",
                f"Register training lineage failed: {method}")
        for path, digest in manifest["source"].items():
            require(Path(path).is_file() and sha256_file(path) == digest,
                    f"Completed source changed: {path}")
        initial = check_rgb(folder / "outer_0000.pt", folder / "outer_0000.png")
        final = check_rgb(folder / "outer_0500.pt", folder / "outer_0500.png")
        require(torch.equal(initial, raw_expected), f"Fresh seed42 RGB changed: {method}")
        delta = (final - initial).abs()
        require(float(delta.max()) > 0 and float(delta.mean()) > 1e-7,
                f"No visible RGB update: {method}")
        rows = pd.read_parquet(folder / "training.parquet")
        require(len(rows) == CHECKPOINT
                and rows.outer_iteration.tolist() == list(range(1, CHECKPOINT + 1))
                and rows.optimizer_updates.tolist() == list(range(cfg["inner_loop"], CHECKPOINT * cfg["inner_loop"] + 1, cfg["inner_loop"]))
                and np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy()).all()
                and rows.aux_grad_finite.all()
                and rows.action_grad_norm.gt(0).all(),
                f"Training rows incomplete/nonfinite: {method}")
        checkpoints[method] = {
            "0": {"pt": sha256_file(folder / "outer_0000.pt"), "png": sha256_file(folder / "outer_0000.png")},
            "250": {"pt": sha256_file(folder / "outer_0250.pt"), "png": sha256_file(folder / "outer_0250.png")},
            "500": {"pt": sha256_file(folder / "outer_0500.pt"), "png": sha256_file(folder / "outer_0500.png")},
        }
        movement[method] = {
            "mae_from_initial": float(delta.mean()),
            "changed_fraction_from_initial": float(final.mul(255).round().ne(initial.mul(255).round()).float().mean()),
            "final_tv": float(rows.iloc[-1].tv),
            "mean_register_attention_mass": float(rows.register_attention_mass.mean()),
            "mean_spatial_attention_mass": float(rows.spatial_attention_mass.mean()),
        }
        metrics.append({
            "condition": method,
            "checkpoint_outer": CHECKPOINT,
            "gradient_ratio": float(manifest["method_spec"]["gradient_ratio"]),
            "target": manifest["method_spec"]["target"],
            "mean_action_loss": float(rows.main_loss.mean()),
            "mean_aux_loss": float(rows.aux_loss.mean()),
            "mean_aux_grad_scale": float(rows.aux_grad_scale.mean()),
            **movement[method],
        })
    result = {
        "status": "COMPLETE",
        "experiment_id": cfg["experiment_id"],
        "config_sha256": sha256_file(CONFIG),
        "checkpoints": checkpoints,
        "references": {k: {n: v for n, v in item.items() if n != "patch"} for k, item in refs.items()},
        "movement_from_initial": movement,
        "fresh_start_outer": 0,
        "optimizer_updates": CHECKPOINT * cfg["inner_loop"],
        "scheduler_total_outer": cfg["scheduler_total_outer"],
        "evaluation_ready": True,
        "evaluation_hooks": False,
    }
    if write_report:
        require(tuple(methods) == METHODS, "Final validation requires all register arms")
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
            "oft_register_prepare", "oft_register_train", "oft_register_validate",
            "oft_register_eval", "oft_register_stats",
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
        stages = [("prepare", "127_register_prepare_cpu.sbatch", None)]
        if pending:
            stages.append(("train", "128_register_train_4090.sbatch",
                           ",".join(map(str, pending)) + f"%{min(3, len(pending))}"))
        stages.extend([
            ("validate", "129_register_validate_cpu.sbatch", None),
            ("eval", "130_register_eval_3090.sbatch", "0-9%8"),
            ("stats", "131_register_stats_cpu.sbatch", None),
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

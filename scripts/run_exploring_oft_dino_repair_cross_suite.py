#!/usr/bin/env python3
"""Four normal-OFT suites per task; 1600 new episodes plus 800 frozen controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dino_repair_common import CONFIG, METHODS, config, frozen_digest, load_calibration, prefix_state, replay_records
from run_exploring_oft_scale500_cross_suite import frozen_clean, run_shard, validate
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}
EXPERIMENT_ID = "OFT-DINO-REPAIR1000-XSUITE-V1"
REFERENCES = {"diverse_reference_1000": "diverse_1000", "original_uada_1000": "original_uada_1000"}
CONDITIONS = (*REFERENCES, *METHODS)
STALE_KEYS = {"manual_visual_review", "required_manual_visual_review", "training_status",
              "patch_root", "patch_sources"}


def load_config():
    train = config()
    old = json.loads((Path(train["legacy_eval_root"]) / "EVAL_CONFIG.json").read_text())
    cfg = {k: v for k, v in old.items() if k not in STALE_KEYS}
    cfg.update(experiment_id=EXPERIMENT_ID, conditions=list(METHODS),
               output_root=str(Path(train["output_root"]) / "evaluation/four_suite"),
               log_tag="DINO_REPAIR1000_XSUITE", bootstrap_seed=20261002)
    require(cfg["suites"] == list(DENOMINATORS) and cfg["state_indices"] == list(range(40, 50))
            and cfg["task_count"] == 10 and cfg["patch_xy"] == train["patch_xy"]
            and cfg["patch_shape"] == train["patch_shape"], "Frozen test grid/site changed")
    return cfg


def source_preflight(train):
    for table in ("frozen_inputs", "frozen_eval_sources", "frozen_eval_assets"):
        for name, digest in train[table].items():
            require(Path(name).is_file() and sha256_file(name) == digest,
                    f"Frozen input/source/asset changed: {name}")
    for name, digest in train["legacy_recorded_eval_sources"].items():
        require(sha256_file(name) == digest,
                f"Historically recorded evaluation dependency changed: {name}")
    return {"verified_recorded_sources": len(train["legacy_recorded_eval_sources"]),
            "historical_source_hash_missing": train["unrecorded_legacy_sources"],
            "historical_caveat": "Missing old runtime fingerprints are frozen at preparation only; "
                                 "their historical equivalence is not claimed."}


def reference_patches(train):
    old = json.loads(Path(train["prefix_config"]).read_text())
    original = json.loads(Path(train["original_config"]).read_text())
    result = {}
    for name, folder in (
        ("diverse_reference_1000", Path(old["output_root"]) / "training/diverse"),
        ("original_uada_1000", Path(original["output_root"]) / "exploring_oft_uada"),
    ):
        path = folder / "outer_1000.pt"
        result[name] = {"path": str(path), "pt_sha256": frozen_digest(train, path),
                        "png_sha256": frozen_digest(train, path.with_suffix(".png"))}
    return result


def validate_metadata(rows, traces, suite, task, cfg, hashes, *, legacy=False):
    require(not rows.empty and rows.success.isin([0, 1]).all()
            and rows.failure.isin([0, 1]).all()
            and rows.failure.eq(1 - rows.success).all(), "Invalid success/failure data")
    expected_id = "OFT-DINO-TRANSFER2000-XSUITE-V1" if legacy else EXPERIMENT_ID
    require(rows.experiment_id.eq(expected_id).all() and traces.experiment_id.eq(expected_id).all(),
            "Evaluation experiment identity changed")
    require(rows.checkpoint.eq(cfg["checkpoints"][suite]).all()
            and traces.checkpoint.eq(cfg["checkpoints"][suite]).all()
            and rows.unnorm_key.eq(cfg["unnorm_keys"][suite]).all()
            and traces.unnorm_key.eq(cfg["unnorm_keys"][suite]).all()
            and rows.training_suite.eq("libero_object").all(), "Model/training suite metadata changed")
    clean = frozen_clean(cfg, suite).loc[lambda f: f.task_id.astype(int).eq(task)]
    for state in cfg["state_indices"]:
        block = rows.loc[rows.state_index.astype(int).eq(state)]
        expected = clean.loc[clean.state_index.astype(int).eq(state)].iloc[0]
        require(len(block) == len(hashes)
                and block.rollout_seed.eq(int(expected.rollout_seed)).all()
                and block.frozen_clean_success.eq(int(expected.success)).all(),
                "Frozen state/seed/clean eligibility changed")
        tb = traces.loc[traces.state_index.astype(int).eq(state)]
        require(tb.frozen_clean_success.eq(int(expected.success)).all(), "Trace clean eligibility changed")
        for condition, digest in hashes.items():
            require(block.loc[block.condition.eq(condition), "patch_sha256"].eq(digest).all(),
                    f"Patch bytes differ from frozen evaluation: {condition}")
    for _, row in rows.iterrows():
        subset = traces.loc[traces.state_index.eq(row.state_index) & traces.condition.eq(row.condition)]
        require(len(subset) == int(row.decision_blocks)
                and set(subset.decision_block.astype(int)) == set(range(int(row.decision_blocks))),
                "Incomplete episode trace")


def reference_frames(cfg, train=None):
    train = train or config()
    oldroot = Path(train["legacy_eval_root"])
    oldcfg = json.loads((oldroot / "EVAL_CONFIG.json").read_text())
    compatibility = ("training_suite", "suites", "task_count", "state_indices", "patch_shape",
                     "patch_xy", "num_steps_wait", "num_open_loop_steps", "image_resolution",
                     "dtype", "checkpoints", "unnorm_keys", "max_steps_by_suite",
                     "frozen_object_clean", "frozen_cross_suite_clean", "order_seed", "paths")
    require(all(oldcfg[key] == cfg[key] for key in compatibility),
            "Frozen reference model/runtime/seed/grid/site compatibility changed")
    status = json.loads((oldroot / "STATUS.json").read_text())
    require(status["status"] == "COMPLETE" and status["attack_episodes"] == 4000
            and status["eligible_rows"] == 3820, "Old evaluation is incomplete")
    refs = reference_patches(train)
    hashes = {REFERENCES[name]: item["pt_sha256"] for name, item in refs.items()}
    selected_rows, selected_traces = [], []
    for suite in cfg["suites"]:
        for task in range(10):
            rp = oldroot / suite / f"task_{task:02d}_rollouts.parquet"
            tp = oldroot / suite / f"task_{task:02d}_traces.parquet"
            frozen_digest(train, rp)
            frozen_digest(train, tp)
            oldrows, oldtraces = pd.read_parquet(rp), pd.read_parquet(tp)
            validate(oldrows, oldtraces, suite, task, oldcfg)
            rows = oldrows.loc[oldrows.condition.isin(hashes)].copy()
            traces = oldtraces.loc[oldtraces.condition.isin(hashes)].copy()
            validate_metadata(rows, traces, suite, task, cfg, hashes, legacy=True)
            for frame, source in ((rows, rp), (traces, tp)):
                frame["source_experiment_id"] = frame.experiment_id
                frame["source_condition"] = frame.condition
                frame["source_path"] = str(source)
                frame["source_sha256"] = frozen_digest(train, source)
                frame["reused"] = True
                frame["condition"] = frame.condition.replace({old: new for new, old in REFERENCES.items()})
                frame["experiment_id"] = EXPERIMENT_ID
            selected_rows.append(rows)
            selected_traces.append(traces)
    rows, traces = pd.concat(selected_rows, ignore_index=True), pd.concat(selected_traces, ignore_index=True)
    require(len(rows) == 800 and not rows.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(), "Incomplete frozen control grid")
    return rows, traces


def patches(cfg, device):
    train = config()
    _, prefix_digest, _ = prefix_state(train)
    _, bank_digest = replay_records(train)
    calibration = load_calibration(train)
    cal_path = Path(train["output_root"]) / "reports/APPEARANCE_CALIBRATION.json"
    result, hashes, movement = {}, {}, {}
    for method in METHODS:
        folder = Path(train["output_root"]) / "training" / method
        status = json.loads((folder / "status.json").read_text())
        manifest = json.loads((folder / "RUN_MANIFEST.json").read_text())
        alignment = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
        require(status["status"] == "COMPLETE" and status["method"] == method
                and status["outer"] == 1000 and status["optimizer_updates"] == 50000
                and manifest["config"] == train and manifest["data"]["prefix_resume"] == prefix_digest
                and manifest["data"]["window_bank"] == bank_digest
                and manifest["calibration_sha256"] == sha256_file(cal_path)
                and manifest["scheduler_total_outer"] == 2000
                and manifest["optimizer_restarted_at_branch"] is False
                and manifest["parameter_count"] == 7500
                and alignment["status"] == "PASS", f"Training lineage/alignment changed: {method}")
        for name, digest in manifest["source"].items():
            require(sha256_file(name) == digest, f"Training source changed: {name}")
        path = folder / "outer_1000.pt"
        png = path.with_suffix(".png")
        require(sha256_file(path) == status["final_patch_sha256"]
                and sha256_file(png) == status["final_png_sha256"]
                and sha256_file(folder / "source_snapshot.zip") == status["source_snapshot_sha256"]
                and sha256_file(folder / "resume.pt") == status["resume_sha256"],
                f"Completed training bytes changed: {method}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = payload["patch"].float()
        require(tuple(value.shape) == (3, 50, 50) and bool(torch.isfinite(value).all())
                and bool(((value >= 0) & (value <= 1)).all())
                and payload["method"] == method and payload["signature"] == status["signature"]
                and payload["total_outer"] == 1000 and payload["total_optimizer_updates"] == 50000,
                f"Invalid RGB checkpoint: {method}")
        with Image.open(png) as picture:
            require(picture.mode == "RGB" and picture.size == (50, 50)
                    and np.array_equal(np.asarray(picture),
                                       value.permute(1, 2, 0).mul(255).round().byte().numpy()),
                    "PNG differs from deployed RGB quantization")
        initial = torch.load(folder / "outer_0250.pt", map_location="cpu", weights_only=False)["patch"].float()
        movement[method] = float((value - initial).abs().mean())
        require(movement[method] > 1e-7, f"Patch did not move after prefix: {method}")
        digest = sha256_file(path)
        result[method] = (value.to(device), digest)
        hashes[method] = {"pt": digest, "png": sha256_file(png)}
    return result, hashes, movement


def preflight(cfg):
    train = config()
    source = source_preflight(train)
    saved, prefix_digest, _ = prefix_state(train)
    replay, bank_digest = replay_records(train)
    denom = {suite: int(frozen_clean(cfg, suite).success.sum()) for suite in cfg["suites"]}
    require(denom == DENOMINATORS, "Frozen clean denominators changed")
    reference_frames(cfg, train)
    complete = all((Path(train["output_root"]) / "training" / m / "status.json").is_file()
                   and json.loads((Path(train["output_root"]) / "training" / m / "status.json").read_text()
                       ).get("status") == "COMPLETE" for m in METHODS)
    if complete:
        patches(cfg, torch.device("cpu"))
    return {"status": "PASS" if complete else "WAITING_FOR_TRAINING", "inputs_ready": True,
            "conditions": CONDITIONS, "new_attack_episodes": 1600, "reused_attack_episodes": 800,
            "total_attack_episodes": 2400, "eligible_rows_expected": 2292,
            "denominators": denom, "prefix_outer": saved["outer"], "prefix_sha256": prefix_digest,
            "replay_sha256": bank_digest, "replay_records": len(replay), "source_checks": source,
            "training_gpu_limit": 4, "evaluation_gpu_limit": 8, "evaluation_log_pairs": 10,
            "model_loaded": False, "submitted": False}


def auto_validate():
    cfg = load_config()
    require(preflight(cfg)["status"] == "PASS", "Training incomplete")
    _, hashes, movement = patches(cfg, torch.device("cpu"))
    root = Path(config()["output_root"])
    report = {"status": "COMPLETE", "config_sha256": sha256_file(CONFIG),
              "policy": "automatic tensor/lineage/change check; visual review remains descriptive",
              "checkpoints": hashes, "movement_from_prefix": movement, "evaluation_ready": True,
              "evaluation_hooks": False}
    atomic_json(report, root / "AUTO_VALIDATION.json")
    return report


def validate_new_shard(rows, traces, suite, task, cfg, hashes):
    validate(rows, traces, suite, task, cfg)
    validate_metadata(rows, traces, suite, task, cfg, {m: v["pt"] for m, v in hashes.items()})


def run_checked(task):
    require(0 <= task < 10, "Task array index must be 0..9")
    cfg = load_config()
    require(preflight(cfg)["status"] == "PASS", "Training incomplete")
    train = config()
    validation = json.loads((Path(train["output_root"]) / "AUTO_VALIDATION.json").read_text())
    require(validation["status"] == "COMPLETE" and validation["config_sha256"] == sha256_file(CONFIG)
            and validation["evaluation_hooks"] is False, "Automatic validation changed")
    _, hashes, _ = patches(cfg, torch.device("cpu"))
    require(validation["checkpoints"] == hashes, "Validation patch bytes changed")
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    expanded = output / "EVAL_CONFIG.json"
    if expanded.exists():
        require(json.loads(expanded.read_text()) == cfg, "Existing evaluation config changed")
    else:
        atomic_json(cfg, expanded)
    for suite_index, suite in enumerate(cfg["suites"]):
        folder = output / suite
        rp, tp = folder / f"task_{task:02d}_rollouts.parquet", folder / f"task_{task:02d}_traces.parquet"
        if rp.exists() and tp.exists():
            validate_new_shard(pd.read_parquet(rp), pd.read_parquet(tp), suite, task, cfg, hashes)
            print(f"DINO_REPAIR_ALREADY_COMPLETE suite={suite} task={task}", flush=True)
            continue
        partial_rp = rp.with_name(f"task_{task:02d}_rollouts.partial.parquet")
        if partial_rp.exists():
            partial = pd.read_parquet(partial_rp)
            require(partial.experiment_id.eq(EXPERIMENT_ID).all()
                    and partial.checkpoint.eq(cfg["checkpoints"][suite]).all(),
                    "Partial shard config identity changed")
            for method, digest in hashes.items():
                require(partial.loc[partial.condition.eq(method), "patch_sha256"].eq(digest["pt"]).all(),
                        "Partial shard patch changed")
        run_shard(task + suite_index * 10, expanded,
                  patch_loader=lambda base, device: patches(base, device)[0])
        validate_new_shard(pd.read_parquet(rp), pd.read_parquet(tp), suite, task, cfg, hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--auto-validate", action="store_true")
    group.add_argument("--array-index", type=int, choices=range(10))
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(load_config()), indent=2))
    elif args.auto_validate:
        print(json.dumps(auto_validate(), indent=2))
    else:
        require(os.environ.get("SLURM_JOB_ID"), "User-submitted Slurm job required")
        require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "EGL environment missing")
        run_checked(args.array_index)

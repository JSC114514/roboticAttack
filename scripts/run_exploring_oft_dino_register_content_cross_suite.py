#!/usr/bin/env python3
"""Four-suite normal-OFT evaluation for OFT-DINO-REGISTER-CONTENT500-V1."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dino_register_content_common import (
    CHECKPOINTS, FINAL_OUTER, CONDITIONS, CONFIG, DENOMINATORS, METHODS, REFERENCES, compact_preflight,
    config, digest_json, pipeline_preflight, reference_patches, require_preparation,
    source_preflight, validate_training,
)
from run_exploring_oft_scale500_cross_suite import (
    frozen_clean, load_wrapper, make_env, renderer_metadata, run_episode, runtime_config,
)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet, atomic_torch, require, sha256_file

EXPERIMENT_ID = "OFT-DINO-REGISTER-CONTENT500-V1"
KEY = ["checkpoint_outer", "suite", "task_id", "state_index", "condition"]


def load_config(train=None):
    train = train or config()
    old = json.loads(Path(train["eval_reference"]).read_text())
    drop = {"manual_visual_review", "required_manual_visual_review", "training_status",
            "patch_root", "patch_sources"}
    cfg = {k: v for k, v in old.items() if k not in drop}
    cfg.update(
        experiment_id=EXPERIMENT_ID,
        conditions=list(CONDITIONS),
        output_root=str(Path(train["output_root"]) / "evaluation/four_suite"),
        log_tag="DINO_REGISTER_CONTENT500_XSUITE",
        bootstrap_seed=train["bootstrap_seed"],
        bootstrap_draws=train["bootstrap_draws"],
        eval_hooks=False,
        eval_eta=False,
        eval_quantization=train["eval_quantization"],
        training_config_sha256=sha256_file(CONFIG),
        evaluation_checkpoint_outer_iterations=list(CHECKPOINTS),
    )
    require(cfg["suites"] == list(DENOMINATORS)
            and cfg["task_count"] == 10
            and cfg["state_indices"] == list(range(40, 50))
            and cfg["patch_xy"] == [26, 137]
            and cfg["patch_shape"] == [3, 50, 50]
            and cfg["num_open_loop_steps"] == 8
            and cfg["training_suite"] == "libero_object",
            "Frozen register evaluation grid/site/seed changed")
    return cfg


def patches(cfg, device):
    train = config()
    report = json.loads((Path(train["output_root"]) / "AUTO_VALIDATION.json").read_text())
    require(report["status"] == "COMPLETE"
            and report["config_sha256"] == sha256_file(CONFIG)
            and report["evaluation_hooks"] is False
            and report["optimizer_updates"] == FINAL_OUTER * train["inner_loop"],
            "CPU training validation missing/changed")
    result, hashes = {}, {}
    from dino_state_axis_common import check_rgb

    refs = reference_patches(train)
    for outer in CHECKPOINTS:
        result[outer], hashes[str(outer)] = {}, {}
        for method in METHODS:
            folder = Path(train["output_root"]) / "training" / method
            path, png = folder / f"outer_{outer:04d}.pt", folder / f"outer_{outer:04d}.png"
            digest = {"pt": sha256_file(path), "png": sha256_file(png)}
            require(digest == report["checkpoints"][method][str(outer)],
                    "Validated delayed patch bytes changed")
            result[outer][method] = (check_rgb(path, png).to(device), digest["pt"])
            hashes[str(outer)][method] = digest
        for name, item in refs[outer].items():
            result[outer][name] = (item["patch"].to(device), item["pt"])
            hashes[str(outer)][name] = {"pt": item["pt"], "png": item["png"]}
        require(set(result[outer]) == set(CONDITIONS), "Delayed patch grid changed")
    return result, hashes


def preflight():
    train = config()
    report = pipeline_preflight(train)
    complete = all((Path(train["output_root"]) / "training" / m / "status.json").exists()
                   and json.loads((Path(train["output_root"]) / "training" / m / "status.json").read_text()
                                 ).get("status") == "COMPLETE" for m in METHODS)
    return {
        **report,
        "status": "WAITING_FOR_TRAINING" if not complete else "INPUTS_AND_TRAINING_PRESENT",
        "conditions": list(CONDITIONS),
        "maximum_attack_episodes": len(CHECKPOINTS) * len(CONDITIONS) * 400,
        "new_training_methods": len(METHODS),
        "external_references": list(REFERENCES),
        "evaluation_checkpoint_outer_iterations": list(CHECKPOINTS),
    }


def _order(cfg, suite, task, state, outer):
    order = list(CONDITIONS)
    seed = int(cfg["order_seed"]) + 1000000 * outer + 10000 * cfg["suites"].index(suite) + 100 * task + state
    random.Random(seed).shuffle(order)
    return order


def validate_metadata(rows, traces, suite, task, cfg, hashes, *, partial=False):
    require(not rows.empty and not traces.empty and set(rows.condition) <= set(CONDITIONS),
            "Empty/unknown-condition register shard")
    states = set(rows.state_index.astype(int))
    blocks = set(zip(rows.checkpoint_outer.astype(int), rows.state_index.astype(int)))
    expected = {(outer, state) for outer in CHECKPOINTS for state in cfg["state_indices"]}
    require(blocks <= expected and (partial or blocks == expected), "Frozen budget/state coverage changed")
    require(not rows.duplicated(KEY).any()
            and not traces.duplicated(KEY + ["decision_block"]).any(),
            "Duplicate rollout/trace key")
    for frame in (rows, traces):
        require(frame.experiment_id.eq(EXPERIMENT_ID).all()
                and frame.suite.eq(suite).all()
                and frame.task_id.eq(task).all()
                and set(zip(frame.checkpoint_outer.astype(int), frame.state_index.astype(int))) == blocks
                and frame.checkpoint.eq(cfg["checkpoints"][suite]).all()
                and frame.unnorm_key.eq(cfg["unnorm_keys"][suite]).all()
                and frame.training_suite.eq("libero_object").all()
                and frame.checkpoint_outer.isin(CHECKPOINTS).all()
                and np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(),
                "Register evaluation metadata changed")
    require(rows.success.isin([0, 1]).all()
            and rows.failure.eq(1 - rows.success).all()
            and rows.frozen_clean_success.isin([0, 1]).all()
            and rows.renderer_vendor.str.lower().str.contains("nvidia").all()
            and rows.decision_blocks.ge(1).all(),
            "Invalid success/renderer/decision data")
    clean = frozen_clean(cfg, suite).loc[lambda f: f.task_id.astype(int).eq(task)]
    for outer, state in sorted(blocks):
        block = rows.loc[rows.checkpoint_outer.eq(outer) & rows.state_index.eq(state)]
        expected_clean = clean.loc[clean.state_index.astype(int).eq(state)]
        require(len(expected_clean) == 1, "Missing frozen clean state")
        expected_clean = expected_clean.iloc[0]
        conditions = set(block.condition)
        require((conditions == set(CONDITIONS))
                and len(block) == len(conditions)
                and block.rollout_seed.eq(int(expected_clean.rollout_seed)).all()
                and block.frozen_clean_success.eq(int(expected_clean.success)).all(),
                "Frozen seed/clean eligibility/condition grid changed")
        tb = traces.loc[traces.checkpoint_outer.eq(outer) & traces.state_index.eq(state)]
        require(set(tb.condition) == conditions
                and tb.frozen_clean_success.eq(int(expected_clean.success)).all(),
                "Trace clean eligibility/condition changed")
        for row in block.itertuples():
            require(row.patch_sha256 == hashes[str(outer)][row.condition]["pt"],
                    f"Patch bytes differ during evaluation: {row.condition}")
            subset = tb.loc[tb.condition.eq(row.condition)]
            require(len(subset) == int(row.decision_blocks)
                    and set(subset.decision_block.astype(int)) == set(range(int(row.decision_blocks))),
                    "Incomplete episode trace")
            require(int(row.condition_order) == _order(cfg, suite, task, state, outer).index(row.condition) + 1,
                    "Frozen condition order changed")
    require(len(traces) == int(rows.decision_blocks.sum()), "Orphan trace row")


def validate_new_shard(rows, traces, suite, task, cfg, hashes):
    require(len(rows) == len(CHECKPOINTS) * len(CONDITIONS) * 10, "Expected all condition/state rows")
    validate_metadata(rows, traces, suite, task, cfg, hashes)


def validate_runtime(rows, traces, cfg, suite, task):
    path = Path(cfg["output_root"]) / f"task_{task:02d}_{suite}_RUNTIME.json"
    identity = json.loads(path.read_text())
    require(identity["config_signature"] == digest_json(cfg)
            and identity["evaluation_hooks"] is False
            and identity["evaluation_checkpoint_outer_iterations"] == list(CHECKPOINTS),
            "Actual evaluation runtime identity changed")
    for name, digest in identity["source"].items():
        require(Path(name).is_file() and sha256_file(name) == digest,
                f"Actual loaded evaluation runtime source changed: {name}")
    expected = digest_json(identity["source"])
    for frame in (rows, traces):
        fresh = frame.loc[~frame.reused.astype(bool)]
        require(fresh.empty or fresh.runtime_source_digest.eq(expected).all(),
                "Raw runtime digest changed")


def run_shard(index, cfg, patch_loader=patches):
    suite, task_id = cfg["suites"][index // 10], index % 10
    runtime = runtime_config(cfg, suite)
    folder = Path(cfg["output_root"]) / suite
    folder.mkdir(parents=True, exist_ok=True)
    rp, tp = [folder / f"task_{task_id:02d}_{kind}.parquet" for kind in ("rollouts", "traces")]
    pr, pt = [p.with_name(p.stem + ".partial.parquet") for p in (rp, tp)]
    joint = folder / f"task_{task_id:02d}_resume.pt"
    loaded, hashes = patch_loader(cfg, "cpu")
    rows, traces = [], []
    if joint.exists():
        tx = torch.load(joint, map_location="cpu", weights_only=False)
        require(tx["config_signature"] == digest_json(cfg) and tx["patches"] == hashes,
                "Register shard checkpoint source/config/patch identity changed")
        rf, tf = pd.DataFrame(tx["rows"]), pd.DataFrame(tx["traces"])
        validate_metadata(rf, tf, suite, task_id, cfg, hashes, partial=True)
        rows, traces = tx["rows"], tx["traces"]
    elif pr.exists() or pt.exists():
        require(pr.exists() and pt.exists(), "Incomplete partial file pair")
        rf, tf = pd.read_parquet(pr), pd.read_parquet(pt)
        validate_metadata(rf, tf, suite, task_id, cfg, hashes, partial=True)
        rows, traces = rf.to_dict("records"), tf.to_dict("records")
    complete = {(int(row["checkpoint_outer"]), int(row["state_index"])) for row in rows}
    expected = {(outer, state) for outer in CHECKPOINTS for state in cfg["state_indices"]}
    if complete == expected:
        rf, tf = pd.DataFrame(rows), pd.DataFrame(traces)
        validate_new_shard(rf, tf, suite, task_id, cfg, hashes)
        atomic_parquet(rf, rp)
        atomic_parquet(tf, tp)
        return
    clean = frozen_clean(cfg, suite).loc[lambda f: f.task_id.astype(int).eq(task_id)]
    wrapper = load_wrapper(runtime)
    env = None
    try:
        from dino_register_content_common import source_manifest as actual_manifest

        actual_source = actual_manifest(config(), wrapper)
        runtime_identity = {
            "source": actual_source,
            "config_signature": digest_json(cfg),
            "evaluation_hooks": False,
            "evaluation_eta": False,
            "evaluation_checkpoint_outer_iterations": list(CHECKPOINTS),
        }
        dynamic_path = Path(cfg["output_root"]) / f"task_{task_id:02d}_{suite}_RUNTIME.json"
        if dynamic_path.exists():
            require(json.loads(dynamic_path.read_text()) == runtime_identity,
                    "Loaded register evaluation runtime source changed")
        else:
            atomic_json(runtime_identity, dynamic_path)
        env, task, states = make_env(runtime, task_id)
        renderer, vendor = renderer_metadata(runtime)
        require(max(cfg["state_indices"]) < len(states), "Frozen state49 unavailable")
        loaded = {outer: {name: (value.to(wrapper.device), digest)
                          for name, (value, digest) in arms.items()}
                  for outer, arms in loaded.items()}
        for outer in CHECKPOINTS:
            for state in cfg["state_indices"]:
                if (outer, state) in complete:
                    continue
                match = clean.loc[clean.state_index.astype(int).eq(state)]
                require(len(match) == 1, "Missing clean eligibility/rollout seed")
                order = _order(cfg, suite, task_id, state, outer)
                state_rows, state_traces = [], []
                for condition in order:
                    patch, digest = loaded[outer][condition]
                    rollout, trace = run_episode(env, wrapper, task, states[state],
                                                 int(match.iloc[0].rollout_seed),
                                                 condition, patch, digest, runtime, renderer, vendor,
                                                 order.index(condition) + 1)
                    metadata = {
                        "experiment_id": EXPERIMENT_ID,
                        "training_suite": "libero_object",
                        "suite": suite,
                        "task_id": task_id,
                        "task_name": task.language,
                        "state_index": state,
                        "frozen_clean_success": int(match.iloc[0].success),
                        "checkpoint": cfg["checkpoints"][suite],
                        "unnorm_key": cfg["unnorm_keys"][suite],
                        "config_signature": digest_json(cfg),
                        "runtime_source_digest": digest_json(actual_source),
                        "source_experiment_id": EXPERIMENT_ID,
                        "source_condition": condition,
                        "checkpoint_outer": outer,
                        "source_condition_order": order.index(condition) + 1,
                        "reused": False,
                        "historical_source_hash_missing": "[]",
                    }
                    state_rows.append({**metadata, **rollout})
                    state_traces.extend({**metadata, **row} for row in trace)
                validate_metadata(pd.DataFrame([*rows, *state_rows]),
                                  pd.DataFrame([*traces, *state_traces]),
                                  suite, task_id, cfg, hashes, partial=True)
                rows.extend(state_rows)
                traces.extend(state_traces)
                atomic_torch({"config_signature": digest_json(cfg), "patches": hashes,
                              "rows": rows, "traces": traces}, joint)
                atomic_parquet(pd.DataFrame(rows), pr)
                atomic_parquet(pd.DataFrame(traces), pt)
                if state in (44, 49):
                    print(f"DINO_REGISTER_CONTENT_EVAL outer={outer} suite={suite} task={task_id} paired_states={state-39}/10", flush=True)
        rf, tf = pd.DataFrame(rows), pd.DataFrame(traces)
        validate_new_shard(rf, tf, suite, task_id, cfg, hashes)
        atomic_parquet(rf, rp)
        atomic_parquet(tf, tp)
        pr.unlink(missing_ok=True)
        pt.unlink(missing_ok=True)
    finally:
        if env is not None:
            env.close()
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


def run_checked(task):
    require(0 <= task < 10, "Task array index must be 0..9")
    train, cfg = config(), load_config()
    preparation = require_preparation(train, source_preflight(train))
    validation = json.loads((Path(train["output_root"]) / "AUTO_VALIDATION.json").read_text())
    require(validation["status"] == "COMPLETE"
            and validation["config_sha256"] == sha256_file(CONFIG)
            and validation["evaluation_hooks"] is False
            and preparation["status"] == "COMPLETE",
            "Register validation/preparation changed")
    _, hashes = patches(cfg, "cpu")
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"task_{task:02d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "config.lock").open("a") as config_lock:
            fcntl.flock(config_lock, fcntl.LOCK_EX)
            expanded = output / "EVAL_CONFIG.json"
            if expanded.exists():
                require(json.loads(expanded.read_text()) == cfg, "Existing evaluation config changed")
            else:
                atomic_json(cfg, expanded)
        receipt = {"status": "RUNNING", "task_id": task, "config_signature": digest_json(cfg),
                   "patches": hashes, "suites": {},
                   "evaluation_checkpoint_outer_iterations": list(CHECKPOINTS)}
        receipt_path = output / f"task_{task:02d}_STATUS.json"
        prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if prior:
            require(prior["config_signature"] == receipt["config_signature"]
                    and prior["patches"] == hashes, "Existing task receipt changed")
        for si, suite in enumerate(cfg["suites"]):
            rp, tp = [output / suite / f"task_{task:02d}_{kind}.parquet" for kind in ("rollouts", "traces")]
            if not (rp.exists() and tp.exists()):
                run_shard(task + si * 10, cfg)
            if receipt_path.exists() and suite in prior.get("suites", {}):
                require(sha256_file(rp) == prior["suites"][suite]["rollouts_sha256"]
                        and sha256_file(tp) == prior["suites"][suite]["traces_sha256"],
                        "Previously committed register shard bytes changed")
            rf, tf = pd.read_parquet(rp), pd.read_parquet(tp)
            validate_new_shard(rf, tf, suite, task, cfg, hashes)
            validate_runtime(rf, tf, cfg, suite, task)
            receipt["suites"][suite] = {
                "rollouts_sha256": sha256_file(rp),
                "traces_sha256": sha256_file(tp),
                "evaluation_checkpoint_outer_iterations": list(CHECKPOINTS),
            }
            atomic_json(receipt, receipt_path)
        receipt["status"] = "COMPLETE"
        atomic_json(receipt, receipt_path)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--auto-validate", action="store_true")
    group.add_argument("--array-index", type=int, choices=range(10))
    args = parser.parse_args()
    if args.preflight:
        result = compact_preflight(preflight())
    elif args.auto_validate:
        result = validate_training(config())
    else:
        require(os.environ.get("SLURM_JOB_ID"), "User-submitted Slurm job required")
        require(os.environ.get("MUJOCO_GL") == "egl"
                and os.environ.get("PYOPENGL_PLATFORM") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "EGL environment missing")
        from run_exploring_oft_scale500_cross_suite import require_gpu

        require_gpu()
        result = run_checked(args.array_index)
    print(json.dumps(result, indent=2))

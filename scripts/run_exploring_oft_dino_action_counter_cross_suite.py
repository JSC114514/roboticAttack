#!/usr/bin/env python3
"""Eight matched-500 RGB patches on normal OFT; four suites per task shard.

The deployment route uses only the established normal predict_action runner.
Counter perturbations and training feature hooks are never installed here.
"""
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

from dino_action_counter_common import (
    CONFIG, METHODS, ROOT, config, digest_json, external_baselines, pipeline_preflight,
    require_preparation, source_preflight, validate_training, check_rgb,
    compact_preflight,
)
from run_exploring_oft_scale500_cross_suite import (
    frozen_clean, runtime_config, run_episode, load_wrapper, make_env, renderer_metadata,
)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet, atomic_torch, require, sha256_file

DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}
EXPERIMENT_ID = "OFT-DINO-ACTION-COUNTER500-V1"
REFERENCES = ("original_uada_500", "diverse_reference_500")
CONDITIONS = (*METHODS, *REFERENCES)
STALE_KEYS = {"manual_visual_review", "required_manual_visual_review", "training_status",
              "patch_root", "patch_sources"}
KEY = ["suite", "task_id", "state_index", "condition"]


def load_config(train=None):
    train = train or config()
    old = json.loads(Path(train["eval_reference"]).read_text())
    cfg = {k: v for k, v in old.items() if k not in STALE_KEYS}
    cfg.update(experiment_id=EXPERIMENT_ID, conditions=list(CONDITIONS),
               output_root=str(Path(train["output_root"]) / "evaluation/four_suite"),
               log_tag="DINO_ACTION_COUNTER500_XSUITE", bootstrap_seed=20260907,
               bootstrap_draws=10000, eval_hooks=False, eval_eta=False,
               eval_quantization=train["eval_quantization"], training_config_sha256=sha256_file(CONFIG))
    require(cfg["suites"] == list(DENOMINATORS) and cfg["task_count"] == 10
            and cfg["state_indices"] == list(range(40, 50)) and cfg["patch_xy"] == [26, 137]
            and cfg["patch_shape"] == [3, 50, 50] and cfg["num_open_loop_steps"] == 8
            and cfg["training_suite"] == "libero_object" and cfg["bootstrap_seed"] == 20260907,
            "Frozen four-suite evaluation grid/site/seed changed")
    return cfg


def patches(cfg, device):
    train = config()
    report = json.loads((Path(train["output_root"]) / "AUTO_VALIDATION.json").read_text())
    require(report["status"] == "COMPLETE" and report["config_sha256"] == sha256_file(CONFIG)
            and report["evaluation_hooks"] is False and report["fresh_start_outer"] == 0,
            "CPU training validation missing/changed")
    result, hashes = {}, {}
    for method in METHODS:
        folder = Path(train["output_root"]) / "training" / method
        path, png = folder / "outer_0500.pt", folder / "outer_0500.png"
        digest = {"pt": sha256_file(path), "png": sha256_file(png)}
        require(digest == report["checkpoints"][method], "Validated patch bytes changed")
        value = check_rgb(path, png)
        # run_episode -> paste_numpy quantizes exactly as the frozen deployment policy.
        result[method] = (value.to(device), digest["pt"])
        hashes[method] = digest
    for method, item in external_baselines(train).items():
        result[method] = (item["patch"].to(device), item["pt"])
        hashes[method] = {"pt": item["pt"], "png": item["png"]}
    return result, hashes, report["movement_from_initial"]


def preflight(cfg=None):
    train = config()
    report = pipeline_preflight(train)
    complete = all((Path(train["output_root"]) / "training" / m / "status.json").exists()
                   and json.loads((Path(train["output_root"]) / "training" / m / "status.json").read_text()
                                 ).get("status") == "COMPLETE" for m in METHODS)
    return {**report, "status": "WAITING_FOR_TRAINING" if not complete else "INPUTS_AND_TRAINING_PRESENT",
            "conditions": list(CONDITIONS), "maximum_attack_episodes": 3200,
            "new_training_methods": 6, "external_checkpoint_evaluation_required":
                [m for m in REFERENCES if m not in train["legacy_same500_reuse"]]}


def _order(cfg, suite, task, state):
    order = list(CONDITIONS)
    seed = int(cfg["order_seed"]) + 10000 * cfg["suites"].index(suite) + 100 * task + state
    random.Random(seed).shuffle(order)
    return order


def validate_metadata(rows, traces, suite, task, cfg, hashes, *, partial=False, legacy_id=None):
    require(not rows.empty and not traces.empty and set(rows.condition) <= set(CONDITIONS),
            "Empty/unknown-condition shard")
    states = set(rows.state_index.astype(int))
    require(states <= set(cfg["state_indices"]) and (partial or states == set(cfg["state_indices"])),
            "Frozen state coverage changed")
    require(not rows.duplicated(KEY).any() and not traces.duplicated(KEY + ["decision_block"]).any(),
            "Duplicate rollout/trace key")
    expected = legacy_id or EXPERIMENT_ID
    for frame in (rows, traces):
        require(frame.experiment_id.eq(expected).all() and frame.suite.eq(suite).all()
                and frame.task_id.eq(task).all() and set(frame.state_index.astype(int)) == states
                and frame.checkpoint.eq(cfg["checkpoints"][suite]).all()
                and frame.unnorm_key.eq(cfg["unnorm_keys"][suite]).all()
                and frame.training_suite.eq("libero_object").all(),
                "Model/experiment/task metadata changed")
        require(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(),
                "Nonfinite rollout/trace value")
        integer_columns = [name for name in ("task_id", "state_index", "condition_order", "rollout_seed",
                           "decision_blocks", "decision_block", "step") if name in frame]
        require(all(np.equal(frame[name].to_numpy(float), np.floor(frame[name].to_numpy(float))).all()
                    for name in integer_columns), "Noninteger rollout/trace index")
        if legacy_id is None:
            require(frame.config_signature.eq(digest_json(cfg)).all(), "Shard configuration signature changed")
    require(rows.success.isin([0, 1]).all() and rows.failure.isin([0, 1]).all()
            and rows.failure.eq(1 - rows.success).all()
            and rows.frozen_clean_success.isin([0, 1]).all()
            and rows.renderer_vendor.str.lower().str.contains("nvidia").all()
            and rows.decision_blocks.ge(1).all() and rows.elapsed_s.ge(0).all(),
            "Invalid success/renderer/decision/cost data")
    clean = frozen_clean(cfg, suite).loc[lambda f: f.task_id.astype(int).eq(task)]
    for state in states:
        block = rows.loc[rows.state_index.eq(state)]
        expected_clean = clean.loc[clean.state_index.astype(int).eq(state)]
        require(len(expected_clean) == 1, "Missing frozen clean state")
        expected_clean = expected_clean.iloc[0]
        conditions = set(block.condition)
        require((partial or legacy_id is not None or conditions == set(CONDITIONS))
                and len(block) == len(conditions)
                and block.rollout_seed.eq(int(expected_clean.rollout_seed)).all()
                and block.frozen_clean_success.eq(int(expected_clean.success)).all(),
                "Frozen seed/clean eligibility/condition grid changed")
        tb = traces.loc[traces.state_index.eq(state)]
        require(set(tb.condition) == conditions
                and tb.frozen_clean_success.eq(int(expected_clean.success)).all(),
                "Trace clean eligibility/condition changed")
        for _, row in block.iterrows():
            require(row.patch_sha256 == hashes[row.condition]["pt"],
                    f"Patch bytes differ from frozen evaluation: {row.condition}")
            subset = tb.loc[tb.condition.eq(row.condition)]
            require(len(subset) == int(row.decision_blocks)
                    and set(subset.decision_block.astype(int)) == set(range(int(row.decision_blocks))),
                    "Incomplete episode trace")
            if legacy_id is None:
                require(int(row.condition_order) == _order(cfg, suite, task, state).index(row.condition) + 1,
                        "Frozen condition order changed")
    require(len(traces) == int(rows.decision_blocks.sum()), "Orphan episode trace")


def validate_new_shard(rows, traces, suite, task, cfg, hashes):
    require(len(rows) == 80, "Expected 8 conditions x 10 states")
    validate_metadata(rows, traces, suite, task, cfg, hashes)


def validate_runtime(rows, traces, cfg, suite, task):
    path = Path(cfg["output_root"]) / f"task_{task:02d}_{suite}_RUNTIME.json"
    identity = json.loads(path.read_text())
    require(identity["config_signature"] == digest_json(cfg)
            and identity["evaluation_hooks"] is False and identity["evaluation_eta"] is False,
            "Actual evaluation runtime identity changed")
    for name, digest in identity["source"].items():
        require(Path(name).is_file() and sha256_file(name) == digest,
                f"Actual loaded evaluation runtime source changed: {name}")
    expected = digest_json(identity["source"])
    for frame in (rows, traces):
        fresh = frame.loc[~frame.reused.astype(bool)]
        require(fresh.empty or fresh.runtime_source_digest.eq(expected).all(),
                "Raw actual-runtime digest changed")
    return identity


def reference_frames(cfg, train=None, task_filter=None):
    """Optional reuse requires pinned same-500 patch identities and complete raw traces.

    No 1000 checkpoint results are relabeled as 500. Empty default means the
    external 500 checkpoints are evaluated normally with the new patches.
    """
    train = train or config()
    reuse = train["legacy_same500_reuse"]
    require(set(reuse) <= set(REFERENCES), "Only external matched-500 reference reuse allowed")
    rows_out, traces_out = [], []
    compatibility = ("training_suite", "suites", "task_count", "state_indices", "patch_shape",
        "patch_xy", "num_steps_wait", "num_open_loop_steps", "image_resolution", "dtype",
        "checkpoints", "unnorm_keys", "max_steps_by_suite", "frozen_object_clean",
        "frozen_cross_suite_clean", "order_seed", "paths")
    for condition, item in reuse.items():
        require(item["checkpoint_outer"] == 500 and item["patch_sha256"] ==
                train["external_baselines"][condition]["pt_sha256"], "Legacy reuse is not the same 500 P")
        oldpath = Path(item["config_path"])
        require(sha256_file(oldpath) == item["config_sha256"], "Legacy same-500 config changed")
        oldcfg = json.loads(oldpath.read_text())
        require(all(oldcfg[k] == cfg[k] for k in compatibility)
                and oldcfg.get("eval_hooks", False) is False and oldcfg.get("eval_eta", False) is False,
                "Legacy same-500 normal-OFT model/grid/site/seed mismatch")
        for name, digest in item["recorded_runtime_sources"].items():
            require(sha256_file(name) == digest, "Recorded historical runtime source changed")
        for suite in cfg["suites"]:
            for task in range(10) if task_filter is None else [task_filter]:
                paths = [Path(item["root"]) / suite / f"task_{task:02d}_{kind}.parquet"
                         for kind in ("rollouts", "traces")]
                frames = []
                for path in paths:
                    require(sha256_file(path) == item["shard_hashes"][str(path)], "Legacy shard bytes changed")
                    frame = pd.read_parquet(path)
                    frame = frame.loc[frame.condition.eq(item["condition"])].copy()
                    frame["source_condition"] = frame.condition
                    frame["source_experiment_id"] = frame.experiment_id
                    frame["source_condition_order"] = frame.get("condition_order", 0)
                    frame["source_path"], frame["source_sha256"] = str(path), sha256_file(path)
                    frame["condition"] = condition
                    frames.append(frame)
                hashes = {condition: {"pt": item["patch_sha256"]}}
                require(len(frames[0]) == 10, "Incomplete same-500 historical episodes")
                validate_metadata(*frames, suite, task, cfg, hashes, legacy_id=item["experiment_id"])
                for frame in frames:
                    frame["experiment_id"] = EXPERIMENT_ID
                    frame["config_signature"] = digest_json(cfg)
                    frame["reused"] = True
                    frame["historical_source_hash_missing"] = json.dumps(train["historical_source_hash_missing"])
                frames[0]["condition_order"] = [_order(cfg, suite, task, int(s)).index(condition) + 1
                                                   for s in frames[0].state_index]
                rows_out.append(frames[0])
                traces_out.append(frames[1])
    return (pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame(),
            pd.concat(traces_out, ignore_index=True) if traces_out else pd.DataFrame())


def run_shard(index, cfg, patch_loader=patches, reused=None):
    suite, task_id = cfg["suites"][index // 10], index % 10
    runtime = runtime_config(cfg, suite)
    folder = Path(cfg["output_root"]) / suite
    folder.mkdir(parents=True, exist_ok=True)
    rp, tp = [folder / f"task_{task_id:02d}_{kind}.parquet" for kind in ("rollouts", "traces")]
    pr, pt = [p.with_name(p.stem + ".partial.parquet") for p in (rp, tp)]
    joint = folder / f"task_{task_id:02d}_resume.pt"
    loaded, hashes, _ = patch_loader(cfg, "cpu")
    rows, traces = [], []
    if joint.exists():
        transaction = torch.load(joint, map_location="cpu", weights_only=False)
        require(transaction["config_signature"] == digest_json(cfg) and transaction["patches"] == hashes,
                "Joint shard checkpoint source/config/patch identity changed")
        rf, tf = pd.DataFrame(transaction["rows"]), pd.DataFrame(transaction["traces"])
        validate_metadata(rf, tf, suite, task_id, cfg, hashes, partial=True)
        require(rf.groupby("state_index").condition.nunique().eq(8).all(),
                "Joint checkpoint contains an incomplete paired state")
        validate_runtime(rf, tf, cfg, suite, task_id)
        rows, traces = transaction["rows"], transaction["traces"]
    elif pr.exists() or pt.exists():
        require(pr.exists() and pt.exists(), "Incomplete partial file pair; preserve and inspect it")
        rf, tf = pd.read_parquet(pr), pd.read_parquet(pt)
        validate_metadata(rf, tf, suite, task_id, cfg, hashes, partial=True)
        require(rf.groupby("state_index").condition.nunique().eq(8).all(),
                "Only complete paired states may be committed to partial shards")
        rows, traces = rf.to_dict("records"), tf.to_dict("records")
    complete = {int(row["state_index"]) for row in rows}
    if complete == set(cfg["state_indices"]):
        rf, tf = pd.DataFrame(rows), pd.DataFrame(traces)
        validate_new_shard(rf, tf, suite, task_id, cfg, hashes)
        # Recover either half of the final pair from the atomic joint transaction.
        atomic_parquet(rf, rp)
        atomic_parquet(tf, tp)
        return
    reused_rows, reused_traces = reused or (pd.DataFrame(), pd.DataFrame())
    clean = frozen_clean(cfg, suite).loc[lambda f: f.task_id.astype(int).eq(task_id)]
    wrapper = load_wrapper(runtime)
    env = None
    try:
        # The normal loader has never installed training hooks. Record actual
        # loaded class definitions so a local runtime-cache change is auditable.
        from white_patch.UADA_OFT_DINO_ACTION_COUNTER import source_manifest
        actual_source = source_manifest(config(), wrapper)
        dynamic_path = Path(cfg["output_root"]) / f"task_{task_id:02d}_{suite}_RUNTIME.json"
        runtime_identity = {"source": actual_source, "config_signature": digest_json(cfg),
                            "evaluation_hooks": False, "evaluation_eta": False}
        if dynamic_path.exists():
            require(json.loads(dynamic_path.read_text()) == runtime_identity, "Loaded evaluation runtime source changed")
        else:
            atomic_json(runtime_identity, dynamic_path)
        env, task, states = make_env(runtime, task_id)
        renderer, vendor = renderer_metadata(runtime)
        require(max(cfg["state_indices"]) < len(states), "Frozen state49 unavailable")
        loaded = {name: (value.to(wrapper.device), digest) for name, (value, digest) in loaded.items()}
        for state in cfg["state_indices"]:
            if state in complete:
                continue
            match = clean.loc[clean.state_index.astype(int).eq(state)]
            require(len(match) == 1, "Missing clean eligibility/rollout seed")
            order = _order(cfg, suite, task_id, state)
            rr = reused_rows.loc[reused_rows.suite.eq(suite) & reused_rows.state_index.eq(state)] \
                if not reused_rows.empty else pd.DataFrame()
            rt = reused_traces.loc[reused_traces.suite.eq(suite) & reused_traces.state_index.eq(state)] \
                if not reused_traces.empty else pd.DataFrame()
            state_rows, state_traces = rr.to_dict("records"), rt.to_dict("records")
            for method in order:
                if not rr.empty and method in set(rr.condition):
                    continue
                patch, digest = loaded[method]
                rollout, trace = run_episode(env, wrapper, task, states[state], int(match.iloc[0].rollout_seed),
                    method, patch, digest, runtime, renderer, vendor, order.index(method) + 1)
                metadata = {"experiment_id": EXPERIMENT_ID, "training_suite": "libero_object",
                    "suite": suite, "task_id": task_id, "task_name": task.language, "state_index": state,
                    "frozen_clean_success": int(match.iloc[0].success),
                    "checkpoint": cfg["checkpoints"][suite], "unnorm_key": cfg["unnorm_keys"][suite],
                    "config_signature": digest_json(cfg), "runtime_source_digest": digest_json(actual_source),
                    "source_experiment_id": EXPERIMENT_ID, "source_condition": method,
                    "source_condition_order": order.index(method) + 1,
                    "reused": False, "historical_source_hash_missing": "[]"}
                state_rows.append({**metadata, **rollout})
                state_traces.extend({**metadata, **row} for row in trace)
            validate_metadata(pd.DataFrame(state_rows), pd.DataFrame(state_traces), suite, task_id,
                              cfg, hashes, partial=True)
            rows.extend(state_rows)
            traces.extend(state_traces)
            # Paired complete state is the resumable transaction, never one episode.
            atomic_torch({"config_signature": digest_json(cfg), "patches": hashes,
                          "rows": rows, "traces": traces}, joint)
            atomic_parquet(pd.DataFrame(rows), pr)
            atomic_parquet(pd.DataFrame(traces), pt)
            if state in (44, 49):
                print(f"COUNTER_EVAL suite={suite} task={task_id} paired_states={state-39}/10", flush=True)
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
    require(validation["status"] == "COMPLETE" and validation["config_sha256"] == sha256_file(CONFIG)
            and validation["preparation_signature"] == preparation["signature"]
            and validation["evaluation_hooks"] is False, "Automatic validation/preparation changed")
    _, hashes, _ = patches(cfg, "cpu")
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"task_{task:02d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The global config lock only protects the initial atomic write.
        with (output / "config.lock").open("a") as config_lock:
            fcntl.flock(config_lock, fcntl.LOCK_EX)
            expanded = output / "EVAL_CONFIG.json"
            if expanded.exists():
                require(json.loads(expanded.read_text()) == cfg, "Existing evaluation config changed")
            else:
                atomic_json(cfg, expanded)
        reused = reference_frames(cfg, train, task_filter=task)
        receipt = {"status": "RUNNING", "task_id": task, "config_signature": digest_json(cfg),
                   "patches": hashes, "suites": {}}
        receipt_path = output / f"task_{task:02d}_STATUS.json"
        prior = {}
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            require(prior["config_signature"] == receipt["config_signature"] and prior["patches"] == hashes,
                    "Existing task receipt/patch signature changed")
        for si, suite in enumerate(cfg["suites"]):
            rp, tp = [output / suite / f"task_{task:02d}_{kind}.parquet" for kind in ("rollouts", "traces")]
            if not (rp.exists() and tp.exists()):
                run_shard(task + si * 10, cfg, reused=reused)
            if receipt_path.exists() and suite in prior.get("suites", {}):
                require(sha256_file(rp) == prior["suites"][suite]["rollouts_sha256"]
                        and sha256_file(tp) == prior["suites"][suite]["traces_sha256"],
                        "Previously committed shard bytes changed")
            rf, tf = pd.read_parquet(rp), pd.read_parquet(tp)
            validate_new_shard(rf, tf, suite, task, cfg, hashes)
            validate_runtime(rf, tf, cfg, suite, task)
            receipt["suites"][suite] = {"rollouts_sha256": sha256_file(rp), "traces_sha256": sha256_file(tp)}
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
        require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "EGL environment missing")
        from run_exploring_oft_scale500_cross_suite import require_gpu
        require_gpu()
        result = run_checked(args.array_index)
    print(json.dumps(result, indent=2))

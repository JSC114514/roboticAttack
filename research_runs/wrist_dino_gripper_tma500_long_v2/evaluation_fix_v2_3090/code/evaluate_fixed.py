#!/usr/bin/env python3
"""NORMAL OFT + uint8 Wrist RGB ONLY. This module does not import the training engine."""
from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import traceback
from pathlib import Path
import hashlib
import random
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from eval_common import (ROOT, ORIGIN_ROOT, REPAIR_ID, CONFIG, METHODS, CONDITIONS, DENOMINATORS, atomic_json, atomic_parquet,
                            atomic_torch, config, condition_metadata, digest, load_json, preparation, reference_patch,
                            require, require_stage_gpu, runtime_config, sha256_file, source_manifest)
from model_runtime_fixed import (load_wrapper, make_env, observation, renderer_metadata,
                                 reset_wait, close_environment, prepare_graphics_backend)
from entities import contact_objects, contact_reading
from objectives import GraspEvents, wrong_open_flags

KEY = ["suite","task_id","state_index","condition"]


def frozen_clean(cfg,suite):
    ev = cfg["evaluation"]
    path = ev["frozen_object_clean"] if suite=="libero_object" else ev["frozen_cross_suite_clean"]
    frame = pd.read_parquet(path)
    clean = frame.loc[frame.suite.eq(suite)&frame.condition.eq("clean")].copy()
    require(len(clean)==100 and int(clean.success.sum())==DENOMINATORS[suite]
            and not clean.duplicated(["task_id","state_index"]).any(),"Frozen clean eligibility changed")
    return clean


def load_patches(cfg):
    validation = load_json(ORIGIN_ROOT/"reports/AUTO_VALIDATION.json")
    require(validation["status"]=="COMPLETE" and validation["config_sha256"]==sha256_file(CONFIG)
            and validation["training_suite"]==cfg["training_suite"]
            and validation["evaluation_model_hooks"] is False,"Training not validated for normal evaluation")
    patches, hashes = {"clean":None}, {"clean":{"deployment_rgb":"none"}}
    for method in METHODS:
        folder = ORIGIN_ROOT/"training"/method
        pt, png = folder/"outer_0500.pt",folder/"outer_0500.png"
        checksums = {"pt":sha256_file(pt),"png":sha256_file(png)}
        require(checksums==validation["checkpoints"][method]["500"],"Validated patch bytes changed")
        with Image.open(png) as image:
            require(image.mode=="RGB" and image.size==(50,50),"Invalid deployment PNG")
            patches[method] = np.asarray(image).copy()
        hashes[method] = {**checksums,"deployment_rgb":hashlib.sha256(patches[method].tobytes()).hexdigest()}
    ref, ref_hash = reference_patch(cfg)
    require(ref_hash==validation["reference"],"Reference bytes changed")
    patches["reference_uada500"] = ref.permute(1,2,0).mul(255).round().byte().numpy()
    hashes["reference_uada500"] = {**ref_hash,"deployment_rgb":hashlib.sha256(patches["reference_uada500"].tobytes()).hexdigest()}
    require(set(patches)==set(CONDITIONS),"Deployment condition grid changed")
    return patches,hashes


def paste_png(obs,rgb,xy):
    if rgb is None:
        return obs
    require(rgb.shape==(50,50,3) and rgb.dtype==np.uint8,"Deployment patch must be real 50x50 uint8 RGB")
    out = dict(obs)
    out["wrist_image"] = np.asarray(obs["wrist_image"]).copy()
    x,y = map(int,xy)
    require(out["wrist_image"].shape==(224,224,3) and 0<=x<=174 and 0<=y<=174,"Placement/image grid changed")
    out["wrist_image"][y:y+50,x:x+50] = rgb
    require(np.array_equal(out["full_image"],obs["full_image"]),"Front changed at deployment")
    return out


def run_episode(env,wrapper,task,initial_state,seed,rgb,runtime,cfg):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw = reset_wait(env,initial_state,seed,cfg["evaluation"]["num_steps_wait"])
    measurable = contact_objects(env)
    trackers = {item["name"]:GraspEvents(cfg["event_metrics"]) for item in measurable}
    # Set height baseline before the first executed policy command.
    for item in measurable:
        grasp,height = contact_reading(env,item)
        trackers[item["name"]].update(grasp,height,False)
    trace, step, decision, success = [], 0, 0, False
    eligible_slots = wrong_slots = 0
    started = time.monotonic()
    limit = int(cfg["evaluation"]["max_steps_by_suite"][wrapper.gen_cfg.task_suite_name])
    while step<limit and not success:
        obs = observation(raw,runtime)
        with torch.no_grad():
            clean = np.asarray(wrapper.predict_action(obs,task.language),dtype=np.float32).reshape(8,7)
            attacked = clean.copy() if rgb is None else np.asarray(wrapper.predict_action(paste_png(obs,rgb,cfg["evaluation"]["patch_xy"]),
                                                                                        task.language),dtype=np.float32).reshape(8,7)
        require(np.isfinite(clean).all() and np.isfinite(attacked).all(),"Nonfinite deployment actions")
        close, wrong = wrong_open_flags(clean,attacked)
        for slot in range(min(8,limit-step)):
            action = process_libero_action(attacked[slot])
            raw, _, done, _ = env.step(action.tolist())
            step += 1
            eligible_slots += int(close[slot])
            wrong_slots += int(wrong[slot])
            readings = {}
            for item in measurable:
                grasp,height = contact_reading(env,item)
                readings[item["name"]] = trackers[item["name"]].update(grasp,height,bool(wrong[slot]))
            trace.append({"step":step,"decision_block":decision,"action_slot":slot,
                          "clean_gripper":float(clean[slot,6]),"attacked_gripper":float(attacked[slot,6]),
                          "executed_gripper":float(action[6]),"clean_close":int(close[slot]),"wrong_open":int(wrong[slot]),
                          "same_state_at_decision":True,"clean_action":clean[slot].tolist(),"attacked_action":attacked[slot].tolist(),
                          "object_proxy_readings":__import__("json").dumps(readings,sort_keys=True)})
            if done:
                success = True
                break
        decision += 1
    events = list(trackers.values())
    return {"success":int(success),"failure":int(not success),"executed_steps":step,"decision_blocks":decision,
            "eligible_close_slots":eligible_slots,"wrong_open_slots":wrong_slots,
            "wrong_open_rate":wrong_slots/eligible_slots if eligible_slots else None,
            "event_objects":len(events),"event_metrics_applicable":bool(events),
            "never_held_proxy":int(not any(e.ever_grasp for e in events)) if events else None,
            "stable_grasp_events":sum(e.grasps for e in events) if events else None,
            "contact_loss_events":sum(e.losses for e in events) if events else None,
            "suspected_drop_events":sum(e.drops for e in events) if events else None,
            "ever_lifted_proxy":int(any(e.ever_lift for e in events)) if events else None,
            "elapsed_s":time.monotonic()-started},trace


def validate_frame(rows,traces,cfg,task,suite,hashes,partial=False):
    conditions = set(CONDITIONS)
    require(rows.evaluation_repair_id.eq(REPAIR_ID).all()
            and traces.evaluation_repair_id.eq(REPAIR_ID).all(), "Mixed evaluation repair versions")
    require(not rows.empty and not rows.duplicated(KEY).any() and set(rows.condition)<=conditions,"Empty/duplicate/unknown rollout")
    require(rows.suite.eq(suite).all() and rows.task_id.eq(task).all() and rows.config_signature.eq(digest(cfg)).all(),"Shard metadata changed")
    require(rows.groupby("state_index").condition.nunique().eq(len(CONDITIONS)).all(),"Only complete paired states can be committed")
    states = set(rows.state_index.astype(int))
    require(states <= set(cfg["evaluation"]["state_indices"]) and (partial or states==set(cfg["evaluation"]["state_indices"])),"State grid changed")
    require(rows.success.isin([0,1]).all() and rows.failure.eq(1-rows.success).all()
            and rows.groupby("state_index").rollout_seed.nunique().eq(1).all()
            and rows.groupby("state_index").frozen_clean_success.nunique().eq(1).all(),"Unpaired/invalid success/eligibility")
    require(rows.checkpoint.eq(cfg["evaluation"]["checkpoints"][suite]).all()
            and rows.training_suite.eq(cfg["training_suite"]).all()
            and rows.renderer_vendor.str.lower().str.contains("nvidia").all(),"Model/training scope/renderer changed")
    require(not traces.empty and not traces.duplicated(KEY+["step"]).any() and traces.config_signature.eq(digest(cfg)).all()
            and traces.suite.eq(suite).all() and traces.task_id.eq(task).all(),"Trace incomplete/duplicated")
    critical = ["step","decision_block","action_slot","clean_gripper","attacked_gripper","executed_gripper","clean_close","wrong_open"]
    require(np.isfinite(traces[critical].to_numpy(float)).all(),"Nonfinite action traces")
    for row in rows.itertuples():
        expected = condition_metadata(row.condition,cfg)
        require(row.comparison_role==expected["comparison_role"]
                and (pd.isna(row.patch_training_suite) if expected["patch_training_suite"] is None
                     else row.patch_training_suite==expected["patch_training_suite"]), "Patch source/comparison role changed")
        require(row.deployment_rgb_sha256==hashes[row.condition]["deployment_rgb"],"Deployable pixels changed")
        part = traces.loc[traces.state_index.eq(row.state_index)&traces.condition.eq(row.condition)]
        require(len(part)==row.executed_steps and part.step.tolist()==list(range(1,row.executed_steps+1))
                and int(part.wrong_open.sum())==row.wrong_open_slots and int(part.clean_close.sum())==row.eligible_close_slots,
                "Wrong-open denominator must count executed slots only")
    require(len(traces)==int(rows.executed_steps.sum()),"Orphan traces")


def validate_source(identity):
    require(all(sha256_file(path) == checksum for path, checksum in identity.items()),
            "Recorded evaluation runtime source bytes changed")
    current = source_manifest()
    require(all(identity.get(path) == checksum for path, checksum in current.items()),
            "Evaluation repair/original source identity changed")


def completed_shard(task_id, suite, cfg, hashes):
    output = ROOT / "evaluation" / suite
    receipt_path = output / f"task_{task_id:02d}_STATUS.json"
    if not receipt_path.exists():
        return False
    receipt = load_json(receipt_path)
    rp, tp = output / f"task_{task_id:02d}_rollouts.parquet", output / f"task_{task_id:02d}_traces.parquet"
    require(receipt["status"] == "COMPLETE" and receipt["repair_id"] == REPAIR_ID
            and receipt["config_signature"] == digest(cfg) and receipt["patches"] == hashes
            and receipt["evaluation_training_hooks"] is False
            and receipt["preparation_signature"] == digest(preparation())
            and sha256_file(rp) == receipt["rollouts_sha256"]
            and sha256_file(tp) == receipt["traces_sha256"], "Completed evaluation changed")
    validate_source(receipt["source"])
    validate_frame(pd.read_parquet(rp), pd.read_parquet(tp), cfg, task_id, suite, hashes)
    return True


def require_smoke(cfg, prep, hashes):
    smoke = load_json(ROOT / "reports/GPU_SMOKE.json")
    require(smoke["status"] == "GPU_SMOKE_PASS" and smoke["repair_id"] == REPAIR_ID
            and smoke["config_signature"] == digest(cfg) and smoke["patches"] == hashes
            and smoke["preparation_signature"] == digest(prep)
            and smoke["suites"] == list(DENOMINATORS)
            and smoke["task_id"] == 0 and smoke["state_index"] == 40,
            "Actual four-suite renderer/policy smoke has not passed")


def run_suite(task_id, suite, smoke=False):
    cfg, prep = config(), preparation()
    patches, hashes = load_patches(cfg)
    require(task_id in range(10) and suite in DENOMINATORS, "Task/suite outside frozen grid")
    if smoke:
        require(task_id == 0, "Smoke only reuses original task0/init40")
    else:
        require_smoke(cfg, prep, hashes)
    output = ROOT / "evaluation" / suite
    output.mkdir(parents=True, exist_ok=True)
    if completed_shard(task_id, suite, cfg, hashes):
        print(f"EVALFIX_ALREADY_COMPLETE task={task_id} suite={suite}", flush=True)
        return
    progress_path = output / f"task_{task_id:02d}_PROGRESS.json"
    previous_progress = load_json(progress_path) if progress_path.exists() else {}
    progress = {"status": "RUNNING", "repair_id": REPAIR_ID, "task_id": task_id,
                "suite": suite, "smoke": smoke}
    def phase(name, **fields):
        progress.update(stage=name, updated_unix_s=time.time(), **fields)
        atomic_json(progress, progress_path)
        print(f"EVALFIX_STAGE task={task_id} suite={suite} stage={name} "
              f"state={progress.get('state_index')} condition={progress.get('condition')}", flush=True)

    env, wrapper = None, None
    started = time.monotonic()
    try:
        phase("gpu_guard")
        gpu = require_stage_gpu("evaluation")
        joint_path = output / f"task_{task_id:02d}_resume.pt"
        rows, traces, saved = [], [], None
        if joint_path.exists():
            saved = torch.load(joint_path, map_location="cpu", weights_only=False)
            require(saved["config_signature"] == digest(cfg) and saved["patches"] == hashes
                    and saved["repair_id"] == REPAIR_ID
                    and saved["preparation_signature"] == digest(prep), "Eval resume identity changed")
            validate_source(saved["source"])
            if saved["rows"]:
                validate_frame(pd.DataFrame(saved["rows"]), pd.DataFrame(saved["traces"]),
                               cfg, task_id, suite, hashes, partial=True)
            rows, traces = saved["rows"], saved["traces"]
        done_states = {int(row["state_index"]) for row in rows}
        selected = [40] if smoke else cfg["evaluation"]["state_indices"]
        remaining = [state for state in selected if state not in done_states]
        if not remaining and smoke and previous_progress.get("status") in {"SMOKE_STATE_COMPLETE", "COMPLETE"}:
            phase("smoke_state_reused", status="SMOKE_STATE_COMPLETE", paired_states=len(done_states))
            return
        runtime = runtime_config(cfg, suite)
        # Match earlier successful 3090 evaluation: model -> reset/render -> GL check.
        phase("graphics_import")
        prepare_graphics_backend()
        phase("model_load")
        wrapper = load_wrapper(runtime, suite)
        phase("environment_create")
        env, task, states = make_env(runtime, suite, task_id)
        graphics = {}
        renderer, vendor = renderer_metadata(env, diagnostics=graphics)
        phase("renderer_ready", renderer=renderer, renderer_vendor=vendor, graphics=graphics)
        require(not bool(wrapper.vla.get_action_stats(wrapper.gen_cfg.unnorm_key).get("mask", [True]*7)[6]),
                "Deployment gripper convention changed")
        identity = source_manifest(wrapper)
        if saved is not None:
            require(saved["source"] == identity, "Evaluation loaded source changed during resume")
        clean = frozen_clean(cfg, suite).loc[lambda frame: frame.task_id.astype(int).eq(task_id)]
        for state in remaining:
            frozen = clean.loc[clean.state_index.astype(int).eq(state)]
            require(len(frozen) == 1, "Missing frozen seed/qualification")
            seed = int(frozen.iloc[0].rollout_seed)
            order = list(CONDITIONS)
            random.Random(cfg["evaluation"]["order_seed"] + 10000 * list(DENOMINATORS).index(suite)
                          + task_id * 100 + state).shuffle(order)
            new_rows, new_traces = [], []
            for condition in order:
                phase("episode", state_index=state, condition=condition, rollout_seed=seed)
                row, trace = run_episode(env, wrapper, task, states[state], seed,
                                         patches[condition], runtime, cfg)
                metadata = {"experiment_id": cfg["experiment_id"], "config_signature": digest(cfg),
                            "evaluation_repair_id": REPAIR_ID, "suite": suite, "task_id": task_id,
                            "state_index": state, "condition": condition, "rollout_seed": seed,
                            "frozen_clean_success": int(frozen.iloc[0].success),
                            "training_suite": cfg["training_suite"], **condition_metadata(condition, cfg),
                            "evaluation_checkpoint_suite": suite,
                            "checkpoint": cfg["evaluation"]["checkpoints"][suite], "task_name": task.language,
                            "deployment_rgb_sha256": hashes[condition]["deployment_rgb"],
                            "runtime_source_signature": digest(identity)}
                new_rows.append({**row, **metadata, "condition_order": order.index(condition),
                                 "renderer": renderer, "renderer_vendor": vendor})
                new_traces.extend({**item, **metadata} for item in trace)
            rows.extend(new_rows)
            traces.extend(new_traces)
            validate_frame(pd.DataFrame(rows), pd.DataFrame(traces), cfg, task_id, suite, hashes, partial=True)
            atomic_torch({"config_signature": digest(cfg), "repair_id": REPAIR_ID, "patches": hashes,
                          "preparation_signature": digest(prep), "source": identity, "rows": rows,
                          "traces": traces}, joint_path)
            phase("paired_state_committed", paired_states=len({row["state_index"] for row in rows}))
        phase("environment_close")
        close_environment(env)
        env = None
        del wrapper
        wrapper = None
        gc.collect()
        torch.cuda.empty_cache()
        if smoke:
            phase("smoke_complete", status="SMOKE_STATE_COMPLETE", elapsed_s=time.monotonic()-started)
            return
        rf, tf = pd.DataFrame(rows), pd.DataFrame(traces)
        validate_frame(rf, tf, cfg, task_id, suite, hashes)
        rp, tp = output / f"task_{task_id:02d}_rollouts.parquet", output / f"task_{task_id:02d}_traces.parquet"
        atomic_parquet(rf, rp)
        atomic_parquet(tf, tp)
        atomic_json({"status": "COMPLETE", "repair_id": REPAIR_ID, "config_signature": digest(cfg),
                     "preparation_signature": digest(prep), "patches": hashes, "gpu": gpu,
                     "source": identity, "evaluation_training_hooks": False,
                     "rollouts_sha256": sha256_file(rp), "traces_sha256": sha256_file(tp),
                     "episodes": 50, "task_id": task_id, "suite": suite},
                    output / f"task_{task_id:02d}_STATUS.json")
        phase("complete", status="COMPLETE", elapsed_s=time.monotonic()-started)
    except BaseException as error:
        progress.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(),
                        elapsed_s=time.monotonic()-started, updated_unix_s=time.time())
        if hasattr(error, "details"):
            progress["graphics_failure"] = error.details
        atomic_json(progress, progress_path)
        raise
    finally:
        if env is not None:
            try:
                close_environment(env)
            except BaseException as cleanup_error:
                print(f"EVALFIX_CLEANUP_ERROR: {cleanup_error!r}", file=sys.stderr, flush=True)
        if wrapper is not None:
            del wrapper
            gc.collect()


def run_task(task_id, smoke=False):
    """One suite per fresh process, SERIAL on this job's single allocated GPU."""
    cfg, prep = config(), preparation()
    _, hashes = load_patches(cfg)
    if smoke:
        require(task_id == 0, "Smoke task must be0")
    else:
        require_smoke(cfg, prep, hashes)
    for suite in cfg["evaluation"]["suites"]:
        command = [sys.executable, "-X", "faulthandler", "-B", str(Path(__file__).resolve()),
                   "--task-id", str(task_id), "--suite", suite]
        if smoke:
            command.append("--smoke")
        result = subprocess.run(command, check=False)
        if result.returncode:
            path = ROOT / "evaluation" / suite / f"task_{task_id:02d}_PROGRESS.json"
            details = load_json(path) if path.exists() else {"task_id": task_id, "suite": suite}
            details.update(status="FAILED", worker_exit_code=result.returncode,
                           worker_terminated_abnormally=True, updated_unix_s=time.time())
            atomic_json(details, path)
            raise RuntimeError(f"Evaluation worker failed: task={task_id}, suite={suite}, "
                               f"returncode={result.returncode}; see {path} and faulthandler stderr")
    if smoke:
        # Evidence must contain the five original conditions at state40 in EVERY suite.
        for suite in DENOMINATORS:
            output = ROOT / "evaluation" / suite
            saved = torch.load(output / "task_00_resume.pt", map_location="cpu", weights_only=False)
            require(saved["repair_id"] == REPAIR_ID and saved["patches"] == hashes
                    and saved["preparation_signature"] == digest(prep), "Smoke provenance changed")
            validate_source(saved["source"])
            rf, tf = pd.DataFrame(saved["rows"]), pd.DataFrame(saved["traces"])
            validate_frame(rf, tf, cfg, 0, suite, hashes, partial=True)
            require(set(rf.loc[rf.state_index.eq(40), "condition"]) == set(CONDITIONS),
                    "GPU smoke paired state incomplete")
        atomic_json({"status": "GPU_SMOKE_PASS", "repair_id": REPAIR_ID,
                     "config_signature": digest(cfg), "preparation_signature": digest(prep),
                     "patches": hashes, "suites": list(DENOMINATORS), "task_id": 0,
                     "state_index": 40, "paired_episodes_reused": 20,
                     "actual_normal_rgb_rollouts": True, "training_restarted": False},
                    ROOT / "reports/GPU_SMOKE.json")
        print("GPU_SMOKE_PASS; 20 original-grid episodes committed, reused by full evaluation", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--suite", choices=list(DENOMINATORS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.suite:
        run_suite(args.task_id, args.suite, args.smoke)
    else:
        run_task(args.task_id, args.smoke)

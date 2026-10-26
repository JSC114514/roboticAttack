"""Normal OFT deployment ONLY: real Wrist PNG, no engine/teacher/replacement hooks."""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from eval_common import (ROOT, SOURCE_ROOT, SUITES, source_signature, atomic_json, atomic_parquet, config, deployment_conditions,
                    load_json, prepared, require, require_gpu, run_signature, runtime_config, sha256_file)
from collect import paste_png
from model_runtime_4090 import (load_wrapper, make_env, observation, reset_wait, close_environment,
                           renderer_metadata, prepare_graphics_backend)


def patches():
    validation = load_json(SOURCE_ROOT / "reports/AUTO_VALIDATION.json")
    require(validation["status"] == "COMPLETE" and validation["signature"] == source_signature(), "Training assets not validated")
    result, checksums = {"clean": None}, {"clean": "none"}
    for condition in deployment_conditions()[1:]:
        method, outer = condition.split("@")
        path = SOURCE_ROOT / "training" / method / f"outer_{int(outer):04d}.png"
        require(sha256_file(path) == validation["patches"][condition]["png"], "Validated deployment PNG changed")
        with Image.open(path) as im:
            require(im.mode == "RGB" and im.size == (50, 50), "PNG mode/shape changed")
            result[condition] = np.asarray(im).copy()
        checksums[condition] = sha256_file(path)
    return result, checksums


def episode(env, wrapper, task, initial, seed, rgb, runtime, cfg):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw = reset_wait(env, initial, seed, cfg["evaluation"]["num_steps_wait"])
    step, decision, success, traces = 0, 0, False, []
    limit = cfg["evaluation"]["max_steps_by_suite"][wrapper.gen_cfg.task_suite_name]
    while step < limit and not success:
        obs = observation(raw, runtime)
        obs = obs if rgb is None else paste_png(obs, rgb, tuple(cfg["evaluation"]["patch_xy"]))
        with torch.no_grad():
            actions = np.asarray(wrapper.predict_action(obs, task.language), dtype=np.float32).reshape(8, 7)
        require(np.isfinite(actions).all(), "Nonfinite normal deployment action")
        used = 0
        for slot in range(min(8, limit - step)):
            raw, _, done, _ = env.step(process_libero_action(actions[slot]).tolist())
            step += 1
            used += 1
            if done:
                success = True
                break
        traces.append({"decision_block": decision, "executed_slots": used, "step_end": step,
                       "predicted_environment_action_chunk": actions.tolist()})
        decision += 1
    return {"success": int(success), "failure": int(not success), "executed_steps": step,
            "decision_blocks": decision}, traces


def run_suite(task_id, suite):
    cfg, signature = config(), run_signature()
    parent_signature = source_signature()
    prepared()
    condition_patches, patch_hashes = patches()
    gpu_name = require_gpu("evaluation")
    folder = ROOT / "evaluation" / suite
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / f"task_{task_id:02d}_STATUS.json"
    rp, tp = folder / f"task_{task_id:02d}_rollouts.parquet", folder / f"task_{task_id:02d}_traces.parquet"
    if status_path.exists():
        old = load_json(status_path)
        if old["status"] == "COMPLETE":
            require(old["signature"] == signature and old["patches"] == patch_hashes
                    and sha256_file(rp) == old["rollouts_sha256"] and sha256_file(tp) == old["traces_sha256"],
                    "Completed evaluation assets changed")
            return
    resume_path = folder / f"task_{task_id:02d}_resume.pt"
    saved = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path.exists() else None
    if saved:
        require(saved["signature"] == signature and saved["patches"] == patch_hashes, "Eval resume changed")
    rows, traces = ([], []) if saved is None else (saved["rows"], saved["traces"])
    completed = {(r["state_index"], r["condition"]) for r in rows}
    require(len(completed) == len(rows), "Duplicate eval resume episodes")
    seeds = pd.read_parquet(cfg["evaluation"]["seed_sources"][suite])
    seeds = seeds.loc[seeds.suite.eq(suite) & seeds.condition.eq("clean") & seeds.task_id.eq(task_id)]
    require(set(seeds.state_index.astype(int)) == set(range(40, 50)) and len(seeds) == 10, "Frozen seed grid changed")
    runtime, env = runtime_config(suite), None
    started = time.monotonic()
    progress = {"status": "RUNNING", "signature": signature, "source_training_signature": parent_signature,
                "suite": suite, "task_id": task_id, "gpu_name": gpu_name, "episodes_complete": len(rows)}
    failure, completion = None, None
    def phase(name, **extra):
        progress.update(phase=name, elapsed_s=time.monotonic() - started, **extra)
        atomic_json(progress, status_path)
        print(f"FINAL8_EVAL_PHASE task={task_id} suite={suite} phase={name}", flush=True)
    try:
        phase("preload_graphics_before_model")
        prepare_graphics_backend()
        phase("load_model")
        wrapper = load_wrapper(runtime, suite)
        env, task, initial = make_env(runtime, suite, task_id, on_phase=phase)
        renderer, vendor = renderer_metadata(env)
        for state in range(40, 50):
            seed = int(seeds.loc[seeds.state_index.eq(state)].iloc[0].rollout_seed)
            order = deployment_conditions()
            random.Random(cfg["evaluation"]["order_seed"] + SUITES.index(suite) * 10000 + task_id * 100 + state).shuffle(order)
            for condition in order:
                if (state, condition) in completed:
                    continue
                progress = {"status": "RUNNING", "signature": signature, "suite": suite, "task_id": task_id,
                            "state_index": state, "condition": condition, "episodes_complete": len(rows),
                            "phase": "episode", "gpu_name": gpu_name,
                            "source_training_signature": parent_signature}
                atomic_json(progress, status_path)
                record, trace = episode(env, wrapper, task, initial[state], seed, condition_patches[condition], runtime, cfg)
                metadata = {"suite": suite, "task_id": task_id, "state_index": state, "condition": condition,
                            "rollout_seed": seed, "training_suite": "libero_10", "signature": signature,
                            "deployment_png_sha256": patch_hashes[condition], "renderer": renderer,
                            "renderer_vendor": vendor, "evaluation_training_hooks": False,
                            "gpu_name": gpu_name, "evaluation_gpu_type": "4090",
                            "source_training_signature": parent_signature}
                rows.append({**record, **metadata})
                traces.extend({**item, **metadata} for item in trace)
                atomic_torch_eval({"signature": signature, "patches": patch_hashes, "rows": rows, "traces": traces}, resume_path)
            print(f"FINAL8_EVAL task={task_id} suite={suite} state={state} episodes={len(rows)}/250", flush=True)
        frame = pd.DataFrame(rows)
        require(len(frame) == 250 and not frame.duplicated(["state_index", "condition"]).any()
                and frame.groupby("state_index").condition.nunique().eq(25).all(), "Evaluation condition grid incomplete")
        atomic_parquet(frame, rp)
        atomic_parquet(pd.DataFrame(traces), tp)
        completion = {"status": "COMPLETE", "signature": signature, "source_training_signature": parent_signature,
                      "patches": patch_hashes, "rollouts_sha256": sha256_file(rp),
                      "traces_sha256": sha256_file(tp), "episodes": 250, "gpu_name": gpu_name,
                      "evaluation_gpu_type": "4090"}
    except Exception as error:
        failure = error
        progress.update(status="FAILED", episodes_complete=len(rows), error=repr(error),
                        renderer_details=getattr(error, "details", {}))
        atomic_json(progress, status_path)
        raise
    finally:
        if env is not None:
            try:
                if failure is None:
                    phase("close_environment", episodes_complete=len(rows))
                close_environment(env)
            except Exception as cleanup_error:
                if failure is None:
                    progress.update(status="FAILED", error=repr(cleanup_error))
                    atomic_json(progress, status_path)
                    raise
                progress["cleanup_error"] = repr(cleanup_error)
                atomic_json(progress, status_path)
    require(completion is not None, "Missing complete episode grid")
    completion["elapsed_s"] = time.monotonic() - started
    atomic_json(completion, status_path)
    print(f"FINAL8_EVAL_COMPLETE task={task_id} suite={suite} episodes=250", flush=True)


def atomic_torch_eval(value, path):
    from common import atomic_torch
    atomic_torch(value, path)

def run_task(task_id):
    # Distinct process per suite releases CUDA memory and the EGL context.
    for suite in SUITES:
        command = [sys.executable, "-B", __file__, "--task-id", str(task_id), "--suite", suite]
        response = subprocess.run(command, check=False)
        if response.returncode != 0:
            path = ROOT / "evaluation" / suite / f"task_{task_id:02d}_STATUS.json"
            status = load_json(path) if path.exists() else {"suite": suite, "task_id": task_id}
            status.update(status="FAILED", child_returncode=response.returncode,
                          error=status.get("error", f"suite child exited with {response.returncode}"))
            atomic_json(status, path)
            raise subprocess.CalledProcessError(response.returncode, command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--suite", choices=SUITES)
    args = parser.parse_args()
    if args.suite:
        run_suite(args.task_id, args.suite)
    else:
        run_task(args.task_id)

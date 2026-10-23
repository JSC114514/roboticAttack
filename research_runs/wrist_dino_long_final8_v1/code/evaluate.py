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
from common import (ROOT, SUITES, atomic_json, atomic_parquet, config, deployment_conditions,
                    load_json, prepared, require, require_gpu, run_signature, runtime_config, sha256_file)
from collect import paste_png
from model_runtime import (load_wrapper, make_env, observation, reset_wait, close_environment,
                           renderer_metadata, prepare_graphics_backend)


def patches():
    validation = load_json(ROOT / "reports/AUTO_VALIDATION.json")
    require(validation["status"] == "COMPLETE" and validation["signature"] == run_signature(), "Training assets not validated")
    result, checksums = {"clean": None}, {"clean": "none"}
    for condition in deployment_conditions()[1:]:
        method, outer = condition.split("@")
        path = ROOT / "training" / method / f"outer_{int(outer):04d}.png"
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
    prepared()
    condition_patches, patch_hashes = patches()
    require_gpu("evaluation")
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
    try:
        prepare_graphics_backend()
        wrapper = load_wrapper(runtime, suite)
        env, task, initial = make_env(runtime, suite, task_id)
        renderer, vendor = renderer_metadata(env)
        for state in range(40, 50):
            seed = int(seeds.loc[seeds.state_index.eq(state)].iloc[0].rollout_seed)
            order = deployment_conditions()
            random.Random(cfg["evaluation"]["order_seed"] + SUITES.index(suite) * 10000 + task_id * 100 + state).shuffle(order)
            for condition in order:
                if (state, condition) in completed:
                    continue
                progress = {"status": "RUNNING", "signature": signature, "suite": suite, "task_id": task_id,
                            "state_index": state, "condition": condition, "episodes_complete": len(rows)}
                atomic_json(progress, status_path)
                record, trace = episode(env, wrapper, task, initial[state], seed, condition_patches[condition], runtime, cfg)
                metadata = {"suite": suite, "task_id": task_id, "state_index": state, "condition": condition,
                            "rollout_seed": seed, "training_suite": "libero_10", "signature": signature,
                            "deployment_png_sha256": patch_hashes[condition], "renderer": renderer,
                            "renderer_vendor": vendor, "evaluation_training_hooks": False}
                rows.append({**record, **metadata})
                traces.extend({**item, **metadata} for item in trace)
                atomic_torch_eval({"signature": signature, "patches": patch_hashes, "rows": rows, "traces": traces}, resume_path)
            print(f"FINAL8_EVAL task={task_id} suite={suite} state={state} episodes={len(rows)}/250", flush=True)
        frame = pd.DataFrame(rows)
        require(len(frame) == 250 and not frame.duplicated(["state_index", "condition"]).any()
                and frame.groupby("state_index").condition.nunique().eq(25).all(), "Evaluation condition grid incomplete")
        atomic_parquet(frame, rp)
        atomic_parquet(pd.DataFrame(traces), tp)
        atomic_json({"status": "COMPLETE", "signature": signature, "patches": patch_hashes,
                     "rollouts_sha256": sha256_file(rp), "traces_sha256": sha256_file(tp),
                     "episodes": 250, "elapsed_s": time.monotonic() - started}, status_path)
    except Exception as error:
        details = getattr(error, "details", {})
        atomic_json({"status": "FAILED", "signature": signature, "suite": suite, "task_id": task_id,
                     "episodes_complete": len(rows), "error": repr(error), "renderer_details": details}, status_path)
        raise
    finally:
        if env is not None:
            close_environment(env)


def atomic_torch_eval(value, path):
    from common import atomic_torch
    atomic_torch(value, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--suite", choices=SUITES)
    args = parser.parse_args()
    if args.suite:
        run_suite(args.task_id, args.suite)
    else:
        # Suite child exits release model AND EGL context before loading next suite.
        for suite in SUITES:
            subprocess.run([sys.executable, "-B", __file__, "--task-id", str(args.task_id), "--suite", suite], check=True)

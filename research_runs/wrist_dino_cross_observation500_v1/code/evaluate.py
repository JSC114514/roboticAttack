"""160 native episodes; immutable matched clean/UADA1000 outcomes are reused."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, SUITES, CONDITIONS, config, runtime_config, require, require_gpu,
                    prepared, run_signature, atomic_json, atomic_parquet, atomic_torch,
                    load_json, sha256_file)
from deployment import paste_png
from model_runtime import (load_wrapper, make_env, observation, reset_wait, close_environment,
                           renderer_metadata, prepare_graphics_backend)

@contextmanager
def observe_dino(wrapper, enabled):
    """A read-only hook: returning None leaves the consumed native output intact."""
    result = {}; calls = []
    handle = None
    if enabled:
        def hook(_module, _args, output):
            calls.append(1)
            if len(calls) == 2:
                require(output.shape[:2] == (1, 256), "Native Wrist DINO readout shape changed")
                result["value"] = output.detach()
        handle = wrapper.vla.vision_backbone.featurizer.register_forward_hook(hook)
    try:
        yield result
        if enabled:
            require(len(calls) == 2 and "value" in result, "Expected native Front then Wrist DINO")
    finally:
        if handle is not None:
            handle.remove()

def episode(env, wrapper, task, initial, seed, rgb, runtime, cfg, sample_path=None):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw = reset_wait(env, initial, seed, cfg["evaluation"]["num_steps_wait"])
    traces = []; step = 0; success = False; decision = 0
    limit = cfg["evaluation"]["max_steps_by_suite"][wrapper.gen_cfg.task_suite_name]
    while step < limit and not success:
        obs = paste_png(observation(raw, runtime), rgb, tuple(cfg["evaluation"]["patch_xy"]))
        position_before = np.asarray(raw["robot0_eef_pos"]).copy()
        monitor = decision in (0, 4, 8)
        with torch.no_grad(), observe_dino(wrapper, monitor) as features:
            actions = np.asarray(wrapper.predict_action(obs, task.language), dtype=np.float32).reshape(8, 7)
        require(np.isfinite(actions).all(), "Nonfinite native deployment actions")
        native = {}
        if monitor:
            value = features["value"].float()
            native = {"native_dino_rms": float(value.square().mean().sqrt()),
                      "native_dino_token_norms": value.norm(dim=-1)[0].cpu().tolist()}
            if decision == 0 and sample_path is not None:
                atomic_torch({"observation": obs, "wrist_dino": features["value"].cpu(),
                              "environment_actions": actions, "reset_seed": seed,
                              "read_only_hook": True}, sample_path)
        used_actions, eef, gripper = [], [], []
        for slot in range(min(8, limit - step)):
            applied = process_libero_action(actions[slot])
            raw, _, done, _ = env.step(applied.tolist())
            used_actions.append(np.asarray(applied).tolist())
            eef.append(np.asarray(raw["robot0_eef_pos"]).tolist())
            gripper.append(np.asarray(raw["robot0_gripper_qpos"]).tolist())
            step += 1
            if done:
                success = True; break
        traces.append({"decision_block": decision, "executed_slots": len(used_actions), "step_end": step,
                       "predicted_environment_action_chunk": actions.tolist(), "executed_environment_actions": used_actions,
                       "eef_position_before": position_before.tolist(), "eef_positions_after_slots": eef,
                       "gripper_qpos_after_slots": gripper, **native})
        decision += 1
    return {"success": int(success), "failure": int(not success), "executed_steps": step,
            "decision_blocks": decision}, traces

def task_complete(folder, task, signature, png_hash):
    path = folder / f"task_{task:02d}_STATUS.json"
    if not path.exists(): return False
    v = load_json(path)
    require(v["signature"] == signature, "Existing evaluation identity changed")
    if v["status"] != "COMPLETE": return False
    require(v["episodes"] == 2 and v["png_sha256"] == png_hash
            and sha256_file(folder / f"task_{task:02d}_rollouts.parquet") == v["rollouts_sha256"]
            and sha256_file(folder / f"task_{task:02d}_traces.parquet") == v["traces_sha256"], "Completed evaluation drift; refusing duplicate")
    return True

def run_suite(condition, suite):
    from common import METHOD, reset_seeds, lineage
    from resources import snapshot
    cfg, signature = config(), run_signature(); prepared(); gpu = require_gpu("evaluation")
    require(condition in CONDITIONS and suite in SUITES, "Invalid evaluation condition/suite")
    validation = load_json(ROOT / "reports/AUTO_VALIDATION.json")
    require(validation["status"] == "COMPLETE" and validation["signature"] == signature, "Training validation missing")
    patch_info = validation["patches"][condition]
    png = Path(patch_info["path"]); checksum = patch_info["png"]
    require(sha256_file(png) == checksum, "Validated deployment PNG changed")
    with Image.open(png) as im:
        require(im.mode == "RGB" and im.size == (50, 50), "Deployment PNG must be RGB50")
        rgb = np.asarray(im).copy()
    folder = ROOT / "evaluation" / condition / suite; folder.mkdir(parents=True, exist_ok=True)
    todo = [t for t in range(10) if not task_complete(folder, t, signature, checksum)]
    if not todo: print("EVALUATION_ALREADY_COMPLETE", condition, suite); return
    seeds = reset_seeds(suite); runtime = runtime_config(suite)
    prepare_graphics_backend(); wrapper = load_wrapper(runtime, suite)
    attempt = str(time.time_ns()); worker_start = time.monotonic()
    for task_id in todo:
        status_path = folder / f"task_{task_id:02d}_STATUS.json"
        resume = folder / f"task_{task_id:02d}_resume.pt"
        saved = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
        if saved:
            require(saved["signature"] == signature and saved["png_sha256"] == checksum, "Evaluation resume identity changed")
        rows, traces = ([], []) if saved is None else (saved["rows"], saved["traces"])
        completed = {int(r["state_index"]) for r in rows}
        require(len(completed) == len(rows) and completed <= {40, 41}, "Duplicated/invalid resumed episode")
        progress = {"status": "RUNNING", "signature": signature, "condition": condition, "suite": suite,
                    "task_id": task_id, "attempt": attempt, "episodes_complete": len(rows), "gpu_name": gpu}
        env, failure = None, None; start = time.monotonic()
        def phase(name, **extra):
            progress.update(phase=name, elapsed_s=time.monotonic()-start, **extra)
            atomic_json(progress, status_path)
        try:
            env, task, initial = make_env(runtime, suite, task_id, on_phase=phase)
            renderer, vendor = renderer_metadata(env)
            for state in (40, 41):
                if state in completed: continue
                seed = int(seeds.loc[seeds.task_id.eq(task_id) & seeds.state_index.eq(state)].iloc[0].rollout_seed)
                phase("episode", state_index=state, episodes_complete=len(rows))
                sample = folder / "attempts" / attempt / f"task_{task_id:02d}_first_inputs.pt" if state == 40 else None
                result, trace = episode(env, wrapper, task, initial[state], seed, rgb, runtime, cfg, sample)
                meta = {"suite": suite, "task_id": task_id, "state_index": state, "condition": condition,
                        "rollout_seed": seed, "signature": signature, "origin_signature": signature,
                        "origin_file": str(folder), "source_training_signature": signature if condition == METHOD else lineage()["parent"]["signature"],
                        "deployment_png_sha256": checksum, "renderer": renderer, "renderer_vendor": vendor,
                        "evaluation_training_hooks": False, "read_only_observation_hook": True,
                        "reused": False, "gpu_name": gpu, "attempt": attempt}
                rows.append({**result, **meta}); traces.extend({**item, **meta} for item in trace)
                atomic_torch({"signature": signature, "png_sha256": checksum, "rows": rows, "traces": traces}, resume)
                print(f"CROSS_OBS_EVAL {condition} {suite} task={task_id} state={state} success={result['success']}", flush=True)
            require(len(rows) == 2 and {r["state_index"] for r in rows} == {40, 41}, "Incomplete task episodes")
            atomic_parquet(pd.DataFrame(rows), folder / f"task_{task_id:02d}_rollouts.parquet")
            atomic_parquet(pd.DataFrame(traces), folder / f"task_{task_id:02d}_traces.parquet")
        except Exception as error:
            failure = error
            progress.update(status="FAILED", error=repr(error), episodes_complete=len(rows), **snapshot())
            atomic_json(progress, status_path); raise
        finally:
            if env is not None:
                try: close_environment(env)
                except Exception as error:
                    if failure is None:
                        progress.update(status="FAILED", phase="cleanup", error=repr(error), **snapshot())
                        atomic_json(progress, status_path); raise
        atomic_json({"status": "COMPLETE", "signature": signature, "episodes": 2, "png_sha256": checksum,
            "rollouts_sha256": sha256_file(folder / f"task_{task_id:02d}_rollouts.parquet"),
            "traces_sha256": sha256_file(folder / f"task_{task_id:02d}_traces.parquet"),
            "elapsed_s": time.monotonic()-start, "attempt": attempt, **snapshot()}, status_path)
    atomic_json({"status": "COMPLETE", "signature": signature, "condition": condition, "suite": suite,
                "worker_elapsed_s": time.monotonic()-worker_start, "new_tasks_this_attempt": len(todo), **snapshot()},
                folder / "attempts" / attempt / "WORKER.json")


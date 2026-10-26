"""Normal native RGB deployment; one model load per arm/suite, resumable tasks."""
from __future__ import annotations
import argparse
import json
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from PIL import Image
from run_core import (ARMS, ROOT, SUITES, finish, locked, native, prepared, read_json, require,
                      require_gpu, save_frame, save_torch, sha, signature, stage_done, write_json)


@contextmanager
def observe(wrapper, enabled):
    summaries, features, handles = {}, [], []
    if enabled:
        def dino_hook(_module, _args, output):
            features.append(output.detach().cpu())
        handles.append(wrapper.vla.vision_backbone.featurizer.register_forward_hook(dino_hook))
        def norm_hook(name):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                summaries[name] = {"shape": list(value.shape), "l2": float(value.detach().float().norm())}
            return hook
        handles.append(wrapper.vla.projector.register_forward_hook(norm_hook("projector")))
        layers = wrapper.vla.language_model.model.layers
        for i in sorted({0, len(layers) // 2, len(layers) - 1}):
            handles.append(layers[i].register_forward_hook(norm_hook(f"language_layer_{i}")))
    try:
        yield summaries, features
    finally:
        for handle in handles:
            handle.remove()


def episode(env, wrapper, task, initial, seed, rgb, runtime):
    from model_runtime import observation, reset_wait
    from deployment import paste_png
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    evaluation = native().config()["evaluation"]
    raw = reset_wait(env, initial, seed, evaluation["num_steps_wait"])
    limit = evaluation["max_steps_by_suite"][wrapper.gen_cfg.task_suite_name]
    step, decision, success, traces, captures = 0, 0, False, [], []
    while step < limit and not success:
        clean_obs = observation(raw, runtime)
        obs = paste_png(clean_obs, rgb, tuple(evaluation["patch_xy"]))
        selected = decision in [0, 1, 4, 8]
        with observe(wrapper, selected) as (internal, dino), torch.no_grad():
            actions = np.asarray(wrapper.predict_action(obs, task.language), dtype=np.float32).reshape(8, 7)
        require(np.isfinite(actions).all(), "Nonfinite formal native action")
        if selected:
            require(len(dino) == 2, "Read-only rollout DINO observer lost a view")
            captures.append({"decision": decision, "observation_before_overlay": clean_obs, "task_name": task.language,
                             "formal_actions": actions, "native_dino": torch.stack(dino, dim=1), "internal": internal})
        before_state = np.asarray(clean_obs["state"]).copy()
        used = 0
        for slot in range(min(8, limit - step)):
            raw, _, done, _ = env.step(process_libero_action(actions[slot]).tolist())
            step += 1; used += 1
            if done:
                success = True
                break
        traces.append({"decision": decision, "step_end": step, "executed_slots": used,
                       "actions": actions.tolist(), "proprio_before": before_state.tolist(),
                       "eef_after": np.asarray(raw["robot0_eef_pos"]).tolist(),
                       "gripper_after": np.asarray(raw["robot0_gripper_qpos"]).tolist(),
                       "internal_summary_json": json.dumps(internal)})
        decision += 1
    return {"success": int(success), "failure": int(not success), "executed_steps": step, "decision_blocks": decision}, traces, captures


def evaluate(arm, suite):
    prepared(); require_gpu("evaluation")
    training = ROOT / "training" / arm
    require(stage_done(training / "STATUS.json"), "Patch training incomplete")
    png = training / "outer_0500.png"
    patch_hash = sha(png)
    with Image.open(png) as im:
        require(im.mode == "RGB" and im.size == (50, 50), "Invalid deployment PNG")
        rgb = np.asarray(im).copy()
    folder = ROOT / "evaluation" / arm / suite
    with locked(folder / "worker.lock"):
        if all(stage_done(folder / f"task_{t:02d}_STATUS.json") for t in range(10)):
            print(arm, suite, "ALREADY_COMPLETE", flush=True)
            return
        from model_runtime import load_wrapper, make_env, close_environment, renderer_metadata, prepare_graphics_backend
        runtime = native().runtime_config(suite)
        prepare_graphics_backend()
        wrapper = load_wrapper(runtime, suite)
        seeds = pd.read_parquet(ROOT / "data/REUSED_BASELINE.parquet")
        seeds = seeds.loc[seeds.condition.eq("clean") & seeds.suite.eq(suite)]
        for task_id in range(10):
            status = folder / f"task_{task_id:02d}_STATUS.json"
            if stage_done(status):
                require(read_json(status)["png_sha256"] == patch_hash, "Completed PNG changed")
                continue
            resume = folder / f"task_{task_id:02d}_resume.pt"
            rows, traces, captures = [], [], []
            if resume.exists():
                saved = torch.load(resume, map_location="cpu", weights_only=False)
                require(saved["signature"] == signature() and saved["png_sha256"] == patch_hash, "Rollout resume changed")
                rows, traces, captures = saved["rows"], saved["traces"], saved["captures"]
            completed = {r["state_index"] for r in rows}
            require(len(completed) == len(rows) and completed <= {40, 41}, "Duplicate/out-of-grid rollout resume")
            env, started = None, time.monotonic()
            def phase(name, **extra):
                write_json({"status": "RUNNING", "signature": signature(), "arm": arm, "suite": suite,
                            "task_id": task_id, "phase": name, "episodes_complete": len(rows), **extra}, status)
            try:
                phase("create_environment")
                env, task, initial = make_env(runtime, suite, task_id, on_phase=phase)
                renderer, vendor = renderer_metadata(env)
                for state in [40, 41]:
                    if state in completed:
                        continue
                    seed_row = seeds.loc[seeds.task_id.eq(task_id) & seeds.state_index.eq(state)]
                    require(len(seed_row) == 1, "Missing paired rollout seed")
                    seed = int(seed_row.iloc[0].rollout_seed)
                    phase("episode", state_index=state)
                    result, trace, capture = episode(env, wrapper, task, initial[state], seed, rgb, runtime)
                    meta = {"suite": suite, "task_id": task_id, "state_index": state, "condition": arm,
                            "rollout_seed": seed, "signature": signature(), "png_sha256": patch_hash,
                            "renderer": renderer, "renderer_vendor": vendor, "readout_in_policy": False}
                    rows.append({**meta, **result})
                    traces.extend({**meta, **x} for x in trace)
                    captures.extend({**meta, **x} for x in capture)
                    save_torch({"signature": signature(), "png_sha256": patch_hash, "rows": rows,
                                "traces": traces, "captures": captures}, resume)
                require(len(rows) == 2 and {x["state_index"] for x in rows} == {40, 41}, "Task evaluation incomplete")
            finally:
                if env is not None:
                    close_environment(env)
            rp, tp, cp = [folder / f"task_{task_id:02d}_{suffix}" for suffix in ("rollouts.parquet", "traces.parquet", "CAPTURES.pt")]
            save_frame(pd.DataFrame(rows), rp); save_frame(pd.DataFrame(traces), tp); save_torch(captures, cp)
            finish(status, [rp, tp, cp], arm=arm, suite=suite, task_id=task_id,
                   png_sha256=patch_hash, elapsed_s=time.monotonic() - started, episodes=2)
            print(f"ROLLOUT {arm} {suite} task={task_id} complete=2/2", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("arm", choices=ARMS); p.add_argument("suite", choices=SUITES)
    args = p.parse_args(); evaluate(args.arm, args.suite)

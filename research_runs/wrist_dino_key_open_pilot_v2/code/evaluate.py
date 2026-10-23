"""Native PNG-only rollouts. One model load per suite; executed-slot and contact measurements only."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, SUITES, config, runtime_config, require, require_gpu, prepared,
                    run_signature, atomic_json, atomic_parquet, atomic_torch, load_json, sha256_file)
from deployment import paste_png
from model_runtime import (load_wrapper, make_env, observation, reset_wait, close_environment,
                           renderer_metadata, prepare_graphics_backend)
from entities import contact_objects, contact_reading
from resources import snapshot

@contextmanager
def read_action_head(wrapper):
    """Return None from every hook; do not replace Keys, actions, or any other activation."""
    head = getattr(wrapper.action_head, "module", wrapper.action_head); values = []
    h = head.model.register_forward_hook(lambda _m, _args, output: values.append(output.detach().float().cpu().numpy()))
    try:
        yield values
        require(len(values) == 1 and values[0].shape == (1, 8, 7), "Unexpected native L1 action-head invocation")
    finally: h.remove()

def native_prediction(wrapper, obs, language):
    with torch.no_grad(), read_action_head(wrapper) as values:
        actions = np.asarray(wrapper.predict_action(obs, language), dtype=np.float32).reshape(8, 7)
    require(np.isfinite(actions).all() and np.isfinite(values[0]).all(), "Nonfinite native actions")
    require(np.allclose(actions[:, 6], values[0][0, :, 6], atol=1e-5, rtol=1e-5), "Native gripper normalization changed")
    return actions, values[0][0]

def episode(env, wrapper, task, initial, seed, rgb, runtime, cfg):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw = reset_wait(env, initial, seed, cfg["evaluation"]["num_steps_wait"])
    objects = contact_objects(env)
    heights = {o["name"]: contact_reading(env, o)[1] for o in objects}
    previous = {o["name"]: {"contact": False, "lifted_streak": 0} for o in objects}
    traces = []; step = 0; success = False; decision = 0
    wrong_count = close_count = streak = max_streak = contact_losses = suspected_drops = 0
    limit = cfg["evaluation"]["max_steps_by_suite"][wrapper.gen_cfg.task_suite_name]
    while step < limit and not success:
        clean_obs = observation(raw, runtime)
        _, clean = native_prediction(wrapper, clean_obs, task.language)
        actions, attacked = native_prediction(wrapper, paste_png(clean_obs, rgb, tuple(cfg["evaluation"]["patch_xy"])), task.language)
        used, eef, qpos, events, flags = [], [], [], [], []
        for slot in range(min(8, limit-step)):
            applied = process_libero_action(actions[slot])
            raw, _, done, _ = env.step(applied.tolist())
            eligible = bool(clean[slot, 6] < .5)
            wrong = eligible and bool(attacked[slot, 6] > .5)
            close_count += int(eligible); wrong_count += int(wrong)
            streak = streak+1 if wrong else 0; max_streak = max(max_streak, streak)
            slot_events = []
            for obj in objects:
                grasp, height = contact_reading(env, obj); p = previous[obj["name"]]
                lifted = grasp and height > heights[obj["name"]]+.02
                contact_loss = p["contact"] and not grasp
                drop_proxy = p["lifted_streak"] >= 2 and not lifted
                contact_losses += int(contact_loss); suspected_drops += int(drop_proxy)
                slot_events.append({"name": obj["name"], "two_finger_contact": grasp, "height": height,
                                    "lifted_contact_proxy": lifted, "contact_loss_proxy": contact_loss,
                                    "suspected_drop_proxy": drop_proxy})
                p.update(contact=grasp, lifted_streak=p["lifted_streak"]+1 if lifted else 0)
            events.append(slot_events); flags.append({"slot": slot, "clean_close": eligible, "wrong_open": wrong})
            used.append(np.asarray(applied).tolist()); eef.append(np.asarray(raw["robot0_eef_pos"]).tolist())
            qpos.append(np.asarray(raw["robot0_gripper_qpos"]).tolist()); step += 1
            if done: success = True; break
        traces.append({"decision_block": decision, "executed_slots": len(used), "step_end": step,
                       "clean_same_state_raw_action": clean.tolist(), "attacked_raw_action": attacked.tolist(),
                       "executed_environment_actions": used, "eef_positions": eef, "gripper_qpos": qpos,
                       "executed_slot_flags": flags, "contact_proxy_events": events})
        decision += 1
    return {"success": int(success), "failure": int(not success), "executed_steps": step, "decision_blocks": decision,
            "eligible_executed_close_slots": close_count, "wrong_open_executed_slots": wrong_count,
            "max_consecutive_wrong_open_slots": max_streak, "contact_proxy_applicable": bool(objects),
            "contact_loss_proxy_events": contact_losses, "suspected_drop_proxy_events": suspected_drops}, traces

def task_complete(folder, task, signature, png_hash):
    p = folder / f"task_{task:02d}_STATUS.json"
    if not p.exists(): return False
    s = load_json(p); require(s["signature"] == signature and s["png_sha256"] == png_hash, "Evaluation identity changed")
    if s["status"] != "COMPLETE": return False
    require(s["episodes"] == 2 and sha256_file(folder / f"task_{task:02d}_rollouts.parquet") == s["rollouts_sha256"]
            and sha256_file(folder / f"task_{task:02d}_traces.parquet") == s["traces_sha256"], "Completed task artifact changed")
    return True

def run_suite(suite):
    cfg, signature = config(), run_signature(); prepared(); gpu = require_gpu("evaluation")
    validation = load_json(ROOT / "reports/AUTO_VALIDATION.json")
    require(validation["signature"] == signature and validation["evaluation_allowed"], "No validated learned PNG to evaluate")
    patches = validation["patches"]
    # Identical PNGs are evaluated only once; analysis explicitly expands aliases.
    unique = [m for m in METHODS if patches[m]["canonical"] == m]
    todo = [(m, t) for m in unique for t in range(10)
            if not task_complete(ROOT / "evaluation" / m / suite, t, signature, patches[m]["png"])]
    if not todo: print("SUITE_ALREADY_COMPLETE", suite); return
    history = pd.read_parquet(ROOT / "data/HISTORICAL_ROLLOUTS.parquet")
    seeds = history[history.suite.eq(suite) & history.condition.eq("clean")].set_index(["task_id", "state_index"]).rollout_seed
    runtime = runtime_config(suite); prepare_graphics_backend(); wrapper = load_wrapper(runtime, suite)
    attempt = str(time.time_ns())
    for task_id in range(10):
        conditions = [m for m in unique if (m, task_id) in todo]
        if not conditions: continue
        env = None; failure = None
        try:
            env, task, initial = make_env(runtime, suite, task_id)
            renderer, vendor = renderer_metadata(env)
            for method in conditions:
                folder = ROOT / "evaluation" / method / suite; folder.mkdir(parents=True, exist_ok=True)
                info = patches[method]; png = Path(info["path"])
                require(sha256_file(png) == info["png"], "Validated PNG changed")
                with Image.open(png) as im: rgb = np.asarray(im).copy()
                require(rgb.shape == (50, 50, 3) and rgb.dtype == np.uint8, "Invalid RGB50 PNG")
                resume = folder / f"task_{task_id:02d}_resume.pt"
                saved = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
                if saved: require(saved["signature"] == signature and saved["png"] == info["png"], "Evaluation resume changed")
                rows, traces = (saved["rows"], saved["traces"]) if saved else ([], [])
                completed = {r["state_index"] for r in rows}
                require(len(rows) == len(completed) and completed <= {40, 41}, "Duplicate resumed episode")
                for state in (40, 41):
                    if state in completed: continue
                    status = {"status": "RUNNING", "signature": signature, "png_sha256": info["png"],
                              "episodes_complete": len(rows), "state_index": state, "attempt": attempt}
                    atomic_json(status, folder / f"task_{task_id:02d}_STATUS.json")
                    seed = int(seeds.loc[(task_id, state)])
                    result, trace = episode(env, wrapper, task, initial[state], seed, rgb, runtime, cfg)
                    meta = {"suite": suite, "task_id": task_id, "state_index": state, "condition": method,
                            "rollout_seed": seed, "signature": signature, "deployment_png_sha256": info["png"],
                            "renderer": renderer, "renderer_vendor": vendor, "evaluation_gpu_type": "4090",
                            "evaluation_training_hooks": False, "read_only_action_hook": True,
                            "reused": False, "gpu_name": gpu, "attempt": attempt}
                    rows.append({**result, **meta}); traces.extend({**r, **meta} for r in trace)
                    atomic_torch({"signature": signature, "png": info["png"], "rows": rows, "traces": traces}, resume)
                    print("KEY_PILOT_EVAL", method, suite, task_id, state, result["success"], flush=True)
                atomic_parquet(pd.DataFrame(rows), folder / f"task_{task_id:02d}_rollouts.parquet")
                atomic_parquet(pd.DataFrame(traces), folder / f"task_{task_id:02d}_traces.parquet")
                # COMPLETE is written after successful environment cleanup below.
        except BaseException as error:
            failure = error; raise
        finally:
            if env is not None:
                try: close_environment(env)
                except BaseException:
                    if failure is None: raise
        for method in conditions:
            folder = ROOT / "evaluation" / method / suite
            atomic_json({"status": "COMPLETE", "signature": signature, "png_sha256": patches[method]["png"],
                         "episodes": 2, "attempt": attempt,
                         "rollouts_sha256": sha256_file(folder / f"task_{task_id:02d}_rollouts.parquet"),
                         "traces_sha256": sha256_file(folder / f"task_{task_id:02d}_traces.parquet"), **snapshot()},
                        folder / f"task_{task_id:02d}_STATUS.json")

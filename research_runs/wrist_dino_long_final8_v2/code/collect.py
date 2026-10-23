"""New Long replay from THIS batch's counted UADA250; no Object replay."""
from __future__ import annotations

import argparse
import hashlib
import time
import numpy as np
import torch
from common import (ROOT, atomic_json, atomic_torch, clean_records, config, observation_key,
                    prepared, require, require_gpu, run_signature, runtime_config, sha256_file)
from model_runtime import (load_wrapper, make_env, observation, reset_wait, close_environment,
                           renderer_metadata, prepare_graphics_backend, RendererInitializationError)


def paste_png(obs, rgb, xy):
    out = dict(obs)
    out["wrist_image"] = np.asarray(obs["wrist_image"]).copy()
    x, y = xy
    out["wrist_image"][y:y + 50, x:x + 50] = rgb
    require(np.array_equal(out["full_image"], obs["full_image"]), "Front changed")
    return out


def collect(task_id):
    from PIL import Image
    cfg, prep = config(), prepared()
    signature = prep["signature"]
    require_gpu("training")
    prefix = ROOT / "training/uada_original/outer_0250.png"
    with Image.open(prefix) as im:
        rgb = np.asarray(im.convert("RGB")).copy()
    require(rgb.shape == (50, 50, 3), "Wrong prefix patch")
    checksum = sha256_file(prefix)
    out = ROOT / "data" / f"replay_task_{task_id:02d}.pt"
    saved = torch.load(out, map_location="cpu", weights_only=False) if out.exists() else None
    if saved:
        require(saved["signature"] == signature and saved["prefix_sha256"] == checksum, "Replay resume changed")
        if saved["complete"]:
            return
    episodes = [] if saved is None else saved["episodes"]
    completed = {r["state_index"] for r in episodes}
    starts = sorted({int(r["state_index"]) for r in clean_records() if r["task_id"] == task_id})
    runtime = runtime_config()
    env = None
    started = time.monotonic()
    progress = {"status": "RUNNING", "signature": signature, "task_id": task_id,
                "prefix_sha256": checksum, "episodes_complete": len(episodes)}
    def phase(name, **extra):
        progress.update(phase=name, elapsed_s=time.monotonic() - started, **extra)
        atomic_json(progress, out.with_suffix(".json"))
        print(f"LONG_REPLAY_PHASE task={task_id} {name}", flush=True)
    failure = None
    try:
        phase("preload_graphics_before_model")
        prepare_graphics_backend()
        # This helper imports the model/TF stack; keep it AFTER EGL preload too.
        from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
        phase("load_model")
        wrapper = load_wrapper(runtime, "libero_10")
        env, task, initial = make_env(runtime, "libero_10", task_id, on_phase=phase)
        renderer, vendor = renderer_metadata(env)
        for state in starts:
            if state in completed:
                continue
            phase("episode", state_index=state)
            seed = cfg["training_source"]["seed"] + task_id * 100 + state
            raw = reset_wait(env, initial[state], seed, cfg["evaluation"]["num_steps_wait"])
            snapshots, step, success = [], 0, False
            while step < 520 and not success:
                obs = observation(raw, runtime)
                snapshots.append(obs)
                with torch.no_grad():
                    actions = np.asarray(wrapper.predict_action(paste_png(obs, rgb, (26, 137)), task.language)).reshape(8, 7)
                require(np.isfinite(actions).all(), "Nonfinite replay action")
                for slot in range(min(8, 520 - step)):
                    raw, _, done, _ = env.step(process_libero_action(actions[slot]).tolist())
                    step += 1
                    if done:
                        success = True
                        break
            count = len(snapshots)
            centers = sorted(set(round(j * (count - 1) / 5) for j in range(6)))
            records, clips = {}, []
            for center in centers:
                decisions = [max(0, center - 1), center, min(count - 1, center + 1)]
                uids = []
                for decision in decisions:
                    uid = "replay:" + observation_key(task_id, state, decision)
                    records[uid] = {"task_id": task_id, "task_name": task.language, "source_suite": "libero_10",
                                    "state_index": state, "decision_index": decision, "observation": snapshots[decision]}
                    uids.append(uid)
                clips.append({"task_id": task_id, "state_index": state, "decision_index": center, "uids": uids,
                              "phase_proxy": ("early", "middle", "late")[min(2, 3 * center // count)]})
            episodes.append({"state_index": state, "task_id": task_id, "success": bool(success), "steps": step,
                             "rollout_seed": seed, "records": records, "clips": clips})
            atomic_torch({"signature": signature, "prefix_sha256": checksum, "task_id": task_id,
                          "complete": len(episodes) == 5, "episodes": episodes}, out)
            phase("episode_saved", state_index=state, episodes_complete=len(episodes))
            print(f"LONG_REPLAY task={task_id} state={state} success={success} episodes={len(episodes)}/5", flush=True)
        atomic_json({"status": "COMPLETE", "signature": signature, "prefix_sha256": checksum,
                     "elapsed_s": time.monotonic() - started, "renderer": renderer, "vendor": vendor}, out.with_suffix(".json"))
    except Exception as error:
        failure = error
        progress.update(status="FAILED", error=repr(error))
        if isinstance(error, RendererInitializationError):
            progress["graphics_failure"] = error.details
        atomic_json(progress, out.with_suffix(".json"))
        raise
    finally:
        if env is not None:
            try:
                close_environment(env)
            except Exception as cleanup_error:
                if failure is None:
                    progress.update(status="FAILED", phase="close_environment", error=repr(cleanup_error))
                    atomic_json(progress, out.with_suffix(".json"))
                    raise
                progress["cleanup_error"] = repr(cleanup_error)
                atomic_json(progress, out.with_suffix(".json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    collect(parser.parse_args().task_id)

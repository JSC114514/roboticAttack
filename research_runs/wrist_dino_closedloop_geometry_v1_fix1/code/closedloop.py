"""Matched finite-candidate search and committed real-RGB episodes; no training."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from branch_state import BranchState, state_difference, state_vector
from geometry import observation_with_geometry, pixel_metrics
from loop_core import cfg, require, save_torch, write_json
from native_predictor import dino_difference
from task_progress import TaskProgress, score_candidates


def image_hash(raw):
    value = np.asarray(raw["agentview_image"]).tobytes() + np.asarray(raw["robot0_eye_in_hand_image"]).tobytes()
    return hashlib.sha256(value).hexdigest()


def window(env, raw, predictor, progress, runtime, candidate, remaining, *, observe=True, layers=False):
    from model_runtime import observation
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    record = {"actions": [], "normalized": [], "progress": [], "success": False, "steps": 0,
              "first_dino": None, "first_layers": {}, "pixel": {}, "action_parity_max": 0.}
    for block in range(cfg()["horizon_blocks"]):
        if record["steps"] >= remaining or record["success"]:
            break
        clean = observation(raw, runtime)
        attacked = observation_with_geometry(clean, candidate)
        prediction = predictor.predict(attacked, observe=observe and block == 0, layers=layers and block == 0)
        if block == 0:
            record["first_dino"] = prediction["dino"]
            record["first_layers"] = prediction["layers"]
            record["pixel"] = pixel_metrics(clean["wrist_image"], attacked["wrist_image"])
        record["actions"].append(prediction["actions"])
        record["normalized"].append(prediction["normalized"])
        record["action_parity_max"] = max(record["action_parity_max"], prediction["parity_error"])
        for action in prediction["actions"][:min(8, remaining - record["steps"])]:
            raw, _reward, done, _info = env.step(process_libero_action(action).tolist())
            record["steps"] += 1
            if done:
                record["success"] = True
                break
        record["progress"].append(progress.measure())
    require(record["actions"] and record["progress"], "Empty planning window")
    record["end_state"] = state_vector(env)
    record["end_image_hash"] = image_hash(raw)
    record["goal_fraction_mean"] = float(np.mean([x["goal_fraction"] for x in record["progress"]]))
    record["dense_progress_mean"] = float(np.mean([x["dense_progress"] for x in record["progress"]]))
    record["end_progress"] = record["progress"][-1]
    return raw, record


def action_difference(left, right):
    require(len(left["actions"]) == len(right["actions"]), "Committed policy chunk count differs from selected branch")
    return max(float(np.max(np.abs(a - b))) for a, b in zip(left["actions"], right["actions"]))


def check_commit(planned, committed):
    action_error = action_difference(planned, committed)
    state_error = state_difference(planned["end_state"], committed["end_state"])
    require(planned["steps"] == committed["steps"] and planned["success"] == committed["success"],
            "Selected branch and committed episode disagree on termination")
    require(action_error <= cfg()["replay_action_atol"] and state_error <= cfg()["replay_state_atol"],
            f"Selected branch did not replay in the real episode: actions={action_error}, physics={state_error}")
    return {"commit_action_max_error": action_error, "commit_state_max_error": state_error,
            "commit_images_equal": planned["end_image_hash"] == committed["end_image_hash"]}


def plan(env, raw, predictor, progress, runtime, remaining, condition, plan_index, capture_path=None):
    before = progress.measure()
    snapshot = BranchState(env, raw)
    outcomes, rows = [], []
    calls_before, started = predictor.calls, time.monotonic()
    candidates = cfg()["candidates"]
    rollback_max = 0.
    try:
        # Identity is first only to supply a same-origin clean native action and DINO reference.
        for index, candidate in enumerate(candidates):
            branch_raw, error = snapshot.restore(env)
            rollback_max = max(rollback_max, error)
            _, result = window(env, branch_raw, predictor, progress, runtime, candidate, remaining,
                               layers=plan_index in (0, 4, 8))
            if index == 0:
                reference_dino = result["first_dino"]
                target = np.where(result["normalized"][0][:, 0] >= 0, -1., 1.)
            result["uada_mse"] = float(np.mean([(a[:, 0] - target) ** 2 for a in result["normalized"]]))
            summary = {"plan_index": plan_index, "candidate_index": index, "candidate": candidate["name"],
                       "scale": candidate["scale"], "dx": candidate["dx"], "dy": candidate["dy"],
                       "success": bool(result["success"]), "steps": result["steps"], "policy_calls": len(result["actions"]),
                       "uada_mse": result["uada_mse"], "goal_fraction_mean": result["goal_fraction_mean"],
                       "dense_progress_mean": result["dense_progress_mean"],
                       "end_progress_json": json.dumps(result["end_progress"]),
                       "normalized_action_chunks": [a.tolist() for a in result["normalized"]],
                       "environment_action_chunks": [a.tolist() for a in result["actions"]],
                       "layer_summaries_json": json.dumps(result["first_layers"]),
                       "action_parity_max": result["action_parity_max"], **result["pixel"],
                       **dino_difference(reference_dino, result["first_dino"])}
            rows.append(summary); outcomes.append(result)
        chosen = score_candidates(rows, condition)
        for row in rows:
            row["selected"] = row["candidate_index"] == chosen
        if capture_path is not None:
            from model_runtime import observation
            clean_obs = observation(snapshot.raw, runtime)
            save_torch({"snapshot": snapshot.__dict__, "clean_observation": clean_obs,
                        "selected_observation": observation_with_geometry(clean_obs, candidates[chosen]),
                        "selected_candidate": candidates[chosen], "native_dino_clean": reference_dino.cpu().half(),
                        "native_dino_selected": outcomes[chosen]["first_dino"].cpu().half()}, capture_path)
        branch_raw, error = snapshot.restore(env)
        rollback_max = max(rollback_max, error)
        raw, committed = window(env, branch_raw, predictor, progress, runtime, candidates[chosen], remaining, observe=False)
        parity = check_commit(outcomes[chosen], committed)
        receipt = {"plan_index": plan_index, "snapshot_physics_sha256": snapshot.identity,
                   "selected_index": chosen, "selected_candidate": candidates[chosen]["name"],
                   "progress_before_json": json.dumps(before), "progress_after_json": json.dumps(committed["end_progress"]),
                   "simulated_candidate_steps": sum(r["steps"] for r in rows),
                   "candidate_count": len(rows), "policy_calls": predictor.calls - calls_before,
                   "rollback_state_max_error": rollback_max, "elapsed_s": time.monotonic() - started,
                   "goal_score_range": max(r["goal_fraction_mean"] for r in rows) - min(r["goal_fraction_mean"] for r in rows),
                   "dense_score_range": max(r["dense_progress_mean"] for r in rows) - min(r["dense_progress_mean"] for r in rows),
                   **parity}
        return raw, committed, receipt, rows
    except BaseException:
        # A failed hypothetical branch must not be mistaken for a committed episode.
        snapshot.restore(env)
        raise


def simple_episode(env, raw, predictor, runtime, condition, png, limit):
    from model_runtime import observation
    from deployment import paste_png
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    step, success, traces = 0, False, []
    progress = TaskProgress(env, cfg()["progress_distance_scale_m"])
    while step < limit and not success:
        obs = observation(raw, runtime)
        if condition == "uada_static":
            obs = paste_png(obs, png, (26, 137))
        result = predictor.predict(obs)
        used = 0
        for action in result["actions"][:min(8, limit - step)]:
            raw, _, done, _ = env.step(process_libero_action(action).tolist())
            step += 1; used += 1
            if done:
                success = True; break
        traces.append({"decision_block": len(traces), "step_end": step, "executed_slots": used,
                       "actions": result["actions"].tolist(), "normalized": result["normalized"].tolist(),
                       "task_progress_json": json.dumps(progress.measure())})
    return {"success": int(success), "failure": int(not success), "executed_steps": step,
            "decision_blocks": len(traces), "plans": 0}, traces, [], []


def adaptive_episode(env, raw, predictor, runtime, condition, limit, attempt):
    step, success, traces, plans, candidates = 0, False, [], [], []
    progress = TaskProgress(env, cfg()["progress_distance_scale_m"])
    journal = Path(attempt) / "LIVE_PLANS.jsonl"
    while step < limit and not success:
        index = len(plans)
        raw, committed, receipt, rows = plan(env, raw, predictor, progress, runtime, limit - step, condition, index,
                            capture_path=Path(attempt) / "FIRST_PLAN_INPUTS.pt" if index == 0 else None)
        previous = step
        step += committed["steps"]; success = committed["success"]
        receipt.update(step_start=previous, step_end=step, committed_success=bool(success))
        plans.append(receipt); candidates.extend(rows)
        for block, actions in enumerate(committed["actions"]):
            end = min(previous + (block + 1) * 8, step)
            traces.append({"decision_block": len(traces), "plan_index": index, "step_end": end,
                           "candidate": receipt["selected_candidate"], "actions": actions.tolist(),
                           "normalized": committed["normalized"][block].tolist(),
                           "task_progress_json": json.dumps(committed["progress"][block])})
        with journal.open("a") as stream:
            stream.write(json.dumps(receipt, allow_nan=False) + "\n"); stream.flush()
        if index % 5 == 0 or success or step >= limit:
            print(f"PLAN {condition} {runtime['task_suite'] if 'task_suite' in runtime else ''} block={index} step={step}/{limit} "
                  f"choice={receipt['selected_candidate']} calls={predictor.calls}", flush=True)
    return {"success": int(success), "failure": int(not success), "executed_steps": step,
            "decision_blocks": len(traces), "plans": len(plans)}, traces, plans, candidates

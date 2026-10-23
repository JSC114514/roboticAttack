#!/usr/bin/env python3
"""Fresh clean OFT-Long success trajectories, labels and anchors; A800 only."""
from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import time

import numpy as np
import torch

from gripper_common import (ROOT, atomic_json, atomic_torch, config, digest, load_json, preparation,
                            require, require_stage_gpu, runtime_config, sha256_file, source_manifest)
from model_runtime_copy import (add_paths, load_wrapper, observation, renderer_metadata, reset_wait,
                                unnormalize_action)
from engine import OFTEngine
from entities import pixel_masks, segmentation_mapping
from objectives import area_tokens, executed_gripper
from long_source import decision_indices, phase_bucket


def make_env(cfg, task_id):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = cfg["training_suite"]
    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
    require(bddl.resolve() == (Path(cfg["training_source"]["bddl_directory"])/task.bddl_file).resolve(),
            "Long environment BDDL differs from source")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                            camera_names=["agentview","robot0_eye_in_hand"], camera_heights=256, camera_widths=256,
                            camera_segmentations="element")
    env.reset()
    return env, task, bench.get_task_init_states(task_id)


def calibrate_gripper(env, state, seed):
    """Local simulator actuator sign check; reset before each command, then replay."""
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    values = {}
    for g in (0.,1.):
        raw = reset_wait(env,state,seed,10)
        start = float(np.abs(raw["robot0_gripper_qpos"]).sum())
        action = process_libero_action(np.array([0,0,0,0,0,0,g],dtype=np.float32))
        require(float(action[6]) == float(executed_gripper(g)), "Official gripper conversion differs")
        for _ in range(20):
            raw, _, _, _ = env.step(action.tolist())
        values[str(g)] = {"command":float(action[6]),"initial_abs_qpos_sum":start,
                          "final_abs_qpos_sum":float(np.abs(raw["robot0_gripper_qpos"]).sum())}
    require(values["1.0"]["final_abs_qpos_sum"] > values["0.0"]["final_abs_qpos_sum"] + 1e-5,
            "Actual simulator did not open for raw1 and close for raw0; stop before training")
    return values


def rollout_candidate(env, task, state, seed, wrapper, runtime, cfg, mapping):
    """Save current observations before executing each native action chunk."""
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw = reset_wait(env,state,seed,cfg["evaluation"]["num_steps_wait"])
    queue, snapshots, success, executed = deque(), [], False, 0
    for _ in range(cfg["training_source"]["max_steps"]):
        if not queue:
            obs = observation(raw,runtime)
            chunk = np.asarray(wrapper.predict_action(obs,task.language),dtype=np.float32).reshape(8,7)
            require(np.isfinite(chunk).all(), "Nonfinite Long native action")
            snapshots.append({"decision_index":len(snapshots),
                              "observation":{key:np.asarray(value).copy() for key,value in obs.items()},
                              "native_action":chunk.copy(),"sim_state":np.asarray(env.get_sim_state()).copy(),
                              "pixel_masks":pixel_masks(raw,mapping)})
            queue.extend(chunk.copy())
        raw, _, done, _ = env.step(process_libero_action(queue.popleft()).tolist())
        executed += 1
        if done:
            success = True
            break
    return {"success":success,"snapshots":snapshots,"executed_steps":executed}


def label_success(result, cfg, task_id, task_name, state_index, seed, rank, engine):
    chosen = decision_indices(len(result["snapshots"]),cfg["training_source"]["records_per_trajectory"])
    records, items = [], []
    for position, decision in enumerate(chosen):
        saved = result["snapshots"][decision]
        index = task_id*30+rank*6+position
        record = {"bank_index":index,"source_suite":cfg["training_suite"],"task_id":task_id,
                  "task_name":task_name,"trajectory_rank":rank,"state_index":state_index,
                  "rollout_seed":seed,"episode_success":True,"decision_index":decision,
                  "total_decisions":len(result["snapshots"]),"phase_bucket":phase_bucket(decision,len(result["snapshots"])),
                  "executed_episode_steps":result["executed_steps"],"observation":saved["observation"],
                  "sim_state":saved["sim_state"],"native_action":torch.as_tensor(saved["native_action"]).clone()}
        inputs, proprio = engine.base(record)
        with torch.no_grad():
            action, caps = engine.forward(inputs,proprio)
        delta = float(np.max(np.abs(unnormalize_action(engine.wrapper,action)-saved["native_action"])))
        require(delta <= .02, "Long native and differentiable action interfaces differ")
        record["clean_action"] = action.detach().cpu()
        masks = saved["pixel_masks"]
        task_tokens = area_tokens(masks["target"] | masks["gripper"]) > cfg["region_token_threshold"]
        background_tokens = area_tokens(masks["target"] | masks["robot"]) <= 0
        items.append({"bank_index":index,"source_suite":cfg["training_suite"],"task_id":task_id,"trajectory_rank":rank,
                      "decision_index":decision,"phase_bucket":record["phase_bucket"],"clean_action":action.detach().cpu(),
                      "clean_dino":caps["dino"].detach().cpu(),"clean_siglip":caps["siglip"].detach().cpu(),
                      "close_slots":(action[0,:,6]<cfg["close_threshold"]).cpu(),"task_tokens":task_tokens.cpu(),
                      "background_tokens":background_tokens.cpu(),"pixel_masks":masks,"native_alignment_max_abs":delta,
                      "background_permutation":np.random.default_rng(cfg["data_seed"]+index).permutation(256).tolist()})
        records.append(record)
    return records, items


def validate_resume(saved, cfg, task_id, prep, sources):
    require(saved["preparation_signature"]==digest(prep) and saved["source"]==sources
            and saved["task_id"]==task_id and saved["training_suite"]==cfg["training_suite"],
            "Long source resume identity changed")
    attempts = saved["attempts"]
    require([a["state_index"] for a in attempts] == cfg["training_source"]["state_indices"][:len(attempts)],
            "Long attempted states are not a contiguous frozen prefix")
    successes = [a for a in attempts if a["accepted"]]
    require(saved["successful_trajectories"]==len(successes)<=cfg["training_source"]["successful_trajectories_per_task"]
            and len(saved["records"])==len(saved["items"])==len(successes)*6, "Long resume records incomplete")
    for rank, attempt in enumerate(successes):
        group = saved["records"][rank*6:(rank+1)*6]
        require(all(r["trajectory_rank"]==rank and r["state_index"]==attempt["state_index"]
                    and r["source_suite"]==cfg["training_suite"] for r in group),
                "Long resumed records do not match accepted episodes")
        require([r["decision_index"] for r in group] == decision_indices(attempt["decisions"],6),
                "Long resumed frame selection changed")
    require([i["bank_index"] for i in saved["items"]] == [r["bank_index"] for r in saved["records"]],
            "Long resumed labels are not paired to source observations")


def collect(task_id):
    cfg, prep = config(), preparation()
    gpu = require_stage_gpu("labels")
    source_cfg, suite = cfg["training_source"], cfg["training_suite"]
    folder = ROOT/"data/labels"
    folder.mkdir(parents=True,exist_ok=True)
    final, receipt_path = folder/f"task_{task_id:02d}.pt", folder/f"task_{task_id:02d}.json"
    if final.exists() and receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt["status"]=="COMPLETE" and receipt["training_suite"]==suite
                and receipt["preparation_signature"]==digest(prep) and sha256_file(final)==receipt["sha256"],
                "Completed Long source labels changed; do not overwrite")
        print(f"LONG_SOURCE_ALREADY_COMPLETE task={task_id}",flush=True)
        return
    started = time.monotonic()
    runtime = runtime_config(cfg,suite)
    add_paths(runtime)
    wrapper = load_wrapper(runtime,suite)
    engine = OFTEngine(wrapper)
    env, task, states = make_env(cfg,task_id)
    sources = source_manifest(wrapper)
    expected_task = prep["assets"]["training_environment"]["tasks"][task_id]
    require(task.language==expected_task["task_name"], "Prepared Long task language changed")
    resume_path = folder/f"task_{task_id:02d}_resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path,map_location="cpu",weights_only=False)
        validate_resume(saved,cfg,task_id,prep,sources)
    else:
        saved = {"preparation_signature":digest(prep),"source":sources,"task_id":task_id,"training_suite":suite,
                 "records":[],"items":[],"attempts":[],"successful_trajectories":0,"elapsed_s":0.}
    previous_elapsed = float(saved["elapsed_s"])
    try:
        first_state = source_cfg["state_indices"][0]
        calibration = calibrate_gripper(env,states[first_state],source_cfg["seed"]+task_id*100+first_state)
        renderer, vendor = renderer_metadata()
        mapping = segmentation_mapping(env)
        for state_index in source_cfg["state_indices"][len(saved["attempts"]):]:
            if saved["successful_trajectories"]==source_cfg["successful_trajectories_per_task"]:
                break
            seed = source_cfg["seed"]+task_id*100+state_index
            attempt_started = time.monotonic()
            result = rollout_candidate(env,task,states[state_index],seed,wrapper,runtime,cfg,mapping)
            count = len(result["snapshots"])
            accepted = bool(result["success"] and count>=source_cfg["records_per_trajectory"])
            reason = "accepted" if accepted else ("too_few_distinct_decisions" if result["success"] else "clean_task_failed")
            if accepted:
                rank = saved["successful_trajectories"]
                records, items = label_success(result,cfg,task_id,task.language,state_index,seed,rank,engine)
                saved["records"].extend(records)
                saved["items"].extend(items)
                saved["successful_trajectories"] += 1
            saved["attempts"].append({"state_index":state_index,"seed":seed,"success":bool(result["success"]),
                                     "accepted":accepted,"reason":reason,"decisions":count,
                                     "executed_steps":result["executed_steps"],"elapsed_s":time.monotonic()-attempt_started})
            saved["elapsed_s"] = previous_elapsed+time.monotonic()-started
            validate_resume(saved,cfg,task_id,prep,sources)
            atomic_torch(saved,resume_path)
            print(f"LONG_SOURCE task={task_id} state={state_index} success={result['success']} "
                  f"accepted={saved['successful_trajectories']}/5 records={len(saved['records'])}/30",flush=True)
            del result
        require(saved["successful_trajectories"]==5 and len(saved["records"])==len(saved["items"])==30,
                f"Not enough Long successes in the frozen init0-39 grid for task{task_id}; no suite fallback")
        payload = {**saved,"calibration":calibration,"mapping":mapping,
                   "region_kind":"whole_goal_subject_objects_and_fixtures_plus_gripper"}
        atomic_torch(payload,final)
        atomic_json({"status":"COMPLETE","task_id":task_id,"training_suite":suite,"sha256":sha256_file(final),
                     "preparation_signature":digest(prep),"gpu":gpu,"renderer":renderer,"vendor":vendor,
                     "records":30,"successful_trajectories":5,"attempts":saved["attempts"],
                     "target_entities":mapping["target_names"],"region_kind":payload["region_kind"],
                     "elapsed_s":saved["elapsed_s"],"calibration":calibration},receipt_path)
    except BaseException as error:
        atomic_json({"status":"FAILED","task_id":task_id,"training_suite":suite,"error":repr(error),
                     "resume_preserved":resume_path.is_file()},folder/f"task_{task_id:02d}_FAILED.json")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id",type=int,choices=range(10),required=True)
    collect(parser.parse_args().task_id)

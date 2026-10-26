#!/usr/bin/env python3
"""User-submitted A800 shards: replay success trajectories and rebuild geom labels."""
from __future__ import annotations

import argparse
from collections import deque
import os
import time

import numpy as np
import torch

from gripper_common import (ROOT, atomic_json, atomic_torch, config, digest, load_json, preparation,
                            require, require_stage_gpu, runtime_config, sha256_file, source_manifest)
from model_runtime_copy import add_paths, load_wrapper, observation, renderer_metadata, reset_wait
from engine import OFTEngine
from entities import pixel_masks, segmentation_mapping
from objectives import area_tokens, executed_gripper


def make_env(cfg, task_id):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()["libero_object"]()
    task = bench.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file),
                            camera_names=["agentview","robot0_eye_in_hand"], camera_heights=256, camera_widths=256,
                            camera_segmentations="element")
    env.reset()
    return env, task, bench.get_task_init_states(task_id)


def check_observation(current, frozen):
    result = {}
    for key in ("full_image","wrist_image"):
        error = np.abs(np.asarray(current[key],dtype=np.int16)-np.asarray(frozen[key],dtype=np.int16))
        result[key+"_max_abs"], result[key+"_mae"] = float(error.max()), float(error.mean())
        require(error.max() <= 1 and error.mean() <= .01, f"Frozen source RGB replay drift: {key}")
    error = float(np.max(np.abs(np.asarray(current["state"])-np.asarray(frozen["state"]))))
    require(error <= 1e-5, "Frozen source proprio replay drift")
    result["proprio_max_abs"] = error
    return result


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


def collect(task_id):
    cfg = config()
    prep = preparation()
    gpu = require_stage_gpu("labels")
    folder = ROOT / "data/labels"
    folder.mkdir(parents=True,exist_ok=True)
    final, receipt_path = folder/f"task_{task_id:02d}.pt", folder/f"task_{task_id:02d}.json"
    if final.exists() and receipt_path.exists():
        receipt = __import__("json").loads(receipt_path.read_text())
        require(receipt["preparation_signature"] == digest(prep) and sha256_file(final) == receipt["sha256"],
                "Completed labels changed; do not overwrite")
        print(f"LABELS_ALREADY_COMPLETE task={task_id}",flush=True)
        return
    bank = torch.load(cfg["training_bank"],map_location="cpu",weights_only=False)
    runtime = runtime_config(cfg,"libero_object")
    add_paths(runtime)
    started = time.monotonic()
    wrapper = load_wrapper(runtime,"libero_object")
    engine = OFTEngine(wrapper)
    env, task, states = make_env(cfg,task_id)
    source_cfg = load_json(cfg["mask_runtime_config"])
    groups = [[(index,r) for index,r in enumerate(bank["records"])
               if int(r["task_id"])==task_id and int(r["trajectory_rank"])==rank] for rank in range(5)]
    require(all(len(group)==6 for group in groups), "Expected 5 trajectories x6 decisions")
    require(all(r["task_name"]==task.language for group in groups for _,r in group), "Task language mismatch")
    sources = source_manifest(wrapper)
    resume_path = folder/f"task_{task_id:02d}_resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path,map_location="cpu",weights_only=False)
        require(saved["preparation_signature"]==digest(prep) and saved["source"]==sources, "Label resume source changed")
        items, completed = saved["items"], saved["completed_ranks"]
    else:
        items, completed = [], []
    try:
        renderer, vendor = renderer_metadata()
        first_state = int(groups[0][0][1]["state_index"])
        calibration = calibrate_gripper(env,states[first_state],int(source_cfg["seed"])+task_id*100+first_state)
        for rank, group in enumerate(groups):
            if rank in completed:
                continue
            state_index = int(group[0][1]["state_index"])
            seed = int(source_cfg["seed"])+task_id*100+state_index
            raw = reset_wait(env,states[state_index],seed,cfg["evaluation"]["num_steps_wait"])
            mapping = segmentation_mapping(env)
            expected = {int(r["decision_index"]):(index,r) for index,r in group}
            queue, decision, captured, success = deque(), 0, [], False
            for _ in range(int(source_cfg["max_steps"])):
                if not queue:
                    obs = observation(raw,runtime)
                    if decision in expected:
                        index, record = expected[decision]
                        metrics = check_observation(obs,record["observation"])
                        inputs, proprio = engine.base(record)
                        with torch.no_grad():
                            action, caps = engine.forward(inputs,proprio)
                        metrics["old_action_max_abs"] = float((action.cpu()-record["clean_action"]).abs().max())
                        require(metrics["old_action_max_abs"] <= .02, "Frozen action replay changed")
                        masks = pixel_masks(raw,mapping)
                        task_tokens = area_tokens(masks["target"] | masks["gripper"]) > cfg["region_token_threshold"]
                        # Background must be a completely entity-free token, not just >0 background pixels.
                        background_tokens = area_tokens(masks["target"] | masks["robot"]) <= 0
                        captured.append({"bank_index":index,"task_id":task_id,"trajectory_rank":rank,
                                         "decision_index":decision,"clean_action":action.cpu(),
                                         "clean_dino":caps["dino"].detach().cpu(),
                                         "clean_siglip":caps["siglip"].detach().cpu(),
                                         "close_slots":(action[0,:,6]<cfg["close_threshold"]).cpu(),
                                         "task_tokens":task_tokens.cpu(),"background_tokens":background_tokens.cpu(),
                                         "pixel_masks":masks,"replay_metrics":metrics,
                                         "background_permutation":np.random.default_rng(cfg["data_seed"]+index).permutation(256).tolist()})
                    chunk = np.asarray(wrapper.predict_action(obs,task.language)).reshape(8,7)
                    queue.extend(chunk)
                    decision += 1
                from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
                raw, _, done, _ = env.step(process_libero_action(queue.popleft()).tolist())
                if done:
                    success = True
                    break
            require(success and len(captured)==6, f"Source success/decision replay failed task{task_id} rank{rank}")
            items.extend(captured)
            completed.append(rank)
            atomic_torch({"preparation_signature":digest(prep),"source":sources,"completed_ranks":completed,"items":items},resume_path)
            print(f"LABELS task={task_id} trajectories={len(completed)}/5 records={len(items)}/30",flush=True)
        require(len(items)==30 and len({i["bank_index"] for i in items})==30,"Duplicate/incomplete label shard")
        payload = {"items":items,"preparation_signature":digest(prep),"source":sources,"task_id":task_id,
                   "calibration":calibration,"region_kind":"actual_target_and_gripper_geom_ids"}
        atomic_torch(payload,final)
        atomic_json({"status":"COMPLETE","task_id":task_id,"sha256":sha256_file(final),
                     "preparation_signature":digest(prep),"gpu":gpu,"renderer":renderer,"vendor":vendor,
                     "records":30,"elapsed_s":time.monotonic()-started,"calibration":calibration},receipt_path)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id",type=int,choices=range(10),required=True)
    collect(parser.parse_args().task_id)

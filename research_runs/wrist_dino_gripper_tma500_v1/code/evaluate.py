#!/usr/bin/env python3
"""NORMAL OFT + uint8 Wrist RGB ONLY. This module does not import the training engine."""
from __future__ import annotations

import argparse
import gc
import hashlib
import random
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from gripper_common import (ROOT, CONFIG, METHODS, CONDITIONS, DENOMINATORS, atomic_json, atomic_parquet,
                            atomic_torch, config, digest, load_json, preparation, reference_patch,
                            require, require_stage_gpu, runtime_config, sha256_file, source_manifest)
from model_runtime_copy import load_wrapper, make_env, observation, renderer_metadata, reset_wait
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
    validation = load_json(ROOT/"reports/AUTO_VALIDATION.json")
    require(validation["status"]=="COMPLETE" and validation["config_sha256"]==sha256_file(CONFIG)
            and validation["evaluation_model_hooks"] is False,"Training not validated for normal evaluation")
    patches, hashes = {"clean":None}, {"clean":{"deployment_rgb":"none"}}
    for method in METHODS:
        folder = ROOT/"training"/method
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
    require(not rows.empty and not rows.duplicated(KEY).any() and set(rows.condition)<=conditions,"Empty/duplicate/unknown rollout")
    require(rows.suite.eq(suite).all() and rows.task_id.eq(task).all() and rows.config_signature.eq(digest(cfg)).all(),"Shard metadata changed")
    require(rows.groupby("state_index").condition.nunique().eq(len(CONDITIONS)).all(),"Only complete paired states can be committed")
    states = set(rows.state_index.astype(int))
    require(states <= set(cfg["evaluation"]["state_indices"]) and (partial or states==set(cfg["evaluation"]["state_indices"])),"State grid changed")
    require(rows.success.isin([0,1]).all() and rows.failure.eq(1-rows.success).all()
            and rows.groupby("state_index").rollout_seed.nunique().eq(1).all()
            and rows.groupby("state_index").frozen_clean_success.nunique().eq(1).all(),"Unpaired/invalid success/eligibility")
    require(rows.checkpoint.eq(cfg["evaluation"]["checkpoints"][suite]).all()
            and rows.training_suite.eq("libero_object").all()
            and rows.renderer_vendor.str.lower().str.contains("nvidia").all(),"Model/training scope/renderer changed")
    require(not traces.empty and not traces.duplicated(KEY+["step"]).any() and traces.config_signature.eq(digest(cfg)).all()
            and traces.suite.eq(suite).all() and traces.task_id.eq(task).all(),"Trace incomplete/duplicated")
    critical = ["step","decision_block","action_slot","clean_gripper","attacked_gripper","executed_gripper","clean_close","wrong_open"]
    require(np.isfinite(traces[critical].to_numpy(float)).all(),"Nonfinite action traces")
    for row in rows.itertuples():
        require(row.deployment_rgb_sha256==hashes[row.condition]["deployment_rgb"],"Deployable pixels changed")
        part = traces.loc[traces.state_index.eq(row.state_index)&traces.condition.eq(row.condition)]
        require(len(part)==row.executed_steps and part.step.tolist()==list(range(1,row.executed_steps+1))
                and int(part.wrong_open.sum())==row.wrong_open_slots and int(part.clean_close.sum())==row.eligible_close_slots,
                "Wrong-open denominator must count executed slots only")
    require(len(traces)==int(rows.executed_steps.sum()),"Orphan traces")


def run_task(task_id):
    cfg, prep = config(), preparation()
    gpu = require_stage_gpu("evaluation")
    patches, hashes = load_patches(cfg)
    folder = ROOT/"evaluation"
    folder.mkdir(parents=True,exist_ok=True)
    for suite in cfg["evaluation"]["suites"]:
        output = folder/suite
        output.mkdir(exist_ok=True)
        rp, tp = output/f"task_{task_id:02d}_rollouts.parquet",output/f"task_{task_id:02d}_traces.parquet"
        receipt_path = output/f"task_{task_id:02d}_STATUS.json"
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            require(receipt["status"]=="COMPLETE" and receipt["config_signature"]==digest(cfg) and receipt["patches"]==hashes
                    and sha256_file(rp)==receipt["rollouts_sha256"] and sha256_file(tp)==receipt["traces_sha256"],"Existing evaluation changed")
            validate_frame(pd.read_parquet(rp),pd.read_parquet(tp),cfg,task_id,suite,hashes)
            continue
        joint_path = output/f"task_{task_id:02d}_resume.pt"
        rows,traces = [],[]
        if joint_path.exists():
            saved = torch.load(joint_path,map_location="cpu",weights_only=False)
            require(saved["config_signature"]==digest(cfg) and saved["patches"]==hashes,"Eval resume identity changed")
            if saved["rows"]:
                validate_frame(pd.DataFrame(saved["rows"]),pd.DataFrame(saved["traces"]),cfg,task_id,suite,hashes,partial=True)
            rows,traces = saved["rows"],saved["traces"]
        done_states = {int(r["state_index"]) for r in rows}
        runtime = runtime_config(cfg,suite)
        wrapper = load_wrapper(runtime,suite)
        require(not bool(wrapper.vla.get_action_stats(wrapper.gen_cfg.unnorm_key).get("mask",[True]*7)[6]),"Deployment gripper convention changed")
        identity = source_manifest(wrapper)
        if joint_path.exists():
            require(saved["source"]==identity,"Evaluation loaded source changed during resume")
        env,task,states = make_env(runtime,suite,task_id)
        try:
            renderer,vendor = renderer_metadata()
            clean = frozen_clean(cfg,suite).loc[lambda f:f.task_id.astype(int).eq(task_id)]
            for state in cfg["evaluation"]["state_indices"]:
                if state in done_states:
                    continue
                frozen = clean.loc[clean.state_index.astype(int).eq(state)]
                require(len(frozen)==1,"Missing frozen seed/qualification")
                seed = int(frozen.iloc[0].rollout_seed)
                order = list(CONDITIONS)
                random.Random(cfg["evaluation"]["order_seed"]+10000*list(DENOMINATORS).index(suite)+task_id*100+state).shuffle(order)
                new_rows,new_traces = [],[]
                for condition in order:
                    row, trace = run_episode(env,wrapper,task,states[state],seed,patches[condition],runtime,cfg)
                    metadata = {"experiment_id":cfg["experiment_id"],"config_signature":digest(cfg),"suite":suite,
                                "task_id":task_id,"state_index":state,"condition":condition,"rollout_seed":seed,
                                "frozen_clean_success":int(frozen.iloc[0].success),"training_suite":"libero_object",
                                "checkpoint":cfg["evaluation"]["checkpoints"][suite],"task_name":task.language,
                                "deployment_rgb_sha256":hashes[condition]["deployment_rgb"],"runtime_source_signature":digest(identity)}
                    new_rows.append({**row,**metadata,"condition_order":order.index(condition),"renderer":renderer,"renderer_vendor":vendor})
                    new_traces.extend({**t,**metadata} for t in trace)
                rows.extend(new_rows); traces.extend(new_traces)
                validate_frame(pd.DataFrame(rows),pd.DataFrame(traces),cfg,task_id,suite,hashes,partial=True)
                atomic_torch({"config_signature":digest(cfg),"patches":hashes,"source":identity,"rows":rows,"traces":traces},joint_path)
                print(f"GRIPPER_EVAL task={task_id} suite={suite} states={len({r['state_index'] for r in rows})}/10",flush=True)
            rf,tf = pd.DataFrame(rows),pd.DataFrame(traces)
            validate_frame(rf,tf,cfg,task_id,suite,hashes)
            atomic_parquet(rf,rp); atomic_parquet(tf,tp)
            atomic_json({"status":"COMPLETE","config_signature":digest(cfg),"patches":hashes,"gpu":gpu,
                         "source":identity,"evaluation_training_hooks":False,"rollouts_sha256":sha256_file(rp),
                         "traces_sha256":sha256_file(tp),"episodes":50,"task_id":task_id,"suite":suite},receipt_path)
        finally:
            env.close()
            del env,wrapper
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id",type=int,choices=range(10),required=True)
    run_task(parser.parse_args().task_id)

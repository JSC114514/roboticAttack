"""Replay frozen clean action chunks and cache aligned renderer counterfactuals."""
from __future__ import annotations
import copy, time, hashlib
from functools import wraps
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import (ROOT, cfg, complete, finish, load_wrapper, prepared, progress, read_json,
                    require, require_gpu, resources, runtime_config, runtime_module, atomic_json, atomic_torch,
                    decode_action_chunk, validate_trace_actions)
from engine import Engine
from geometry import paired_deltas, render, task_candidates

def shard_path(split, suite, task):
    return ROOT/f"data/cache/{split}/{suite}/task_{task:02d}.pt"

def receipt_path(split, suite, task):
    return ROOT/f"data/cache/{split}/{suite}/task_{task:02d}.json"

def normalized_rms(left, right):
    return float(np.sqrt(np.mean(((np.asarray(left,dtype=float)-np.asarray(right,dtype=float))/255.0)**2)))

class NoVisibleTarget(RuntimeError):
    def __init__(self, attempts, failures):
        super().__init__("No task-relevant entity is visible in the wrist image")
        self.attempts,self.failures=attempts,failures

def select_entity(env, native_upright):
    entities, failures = task_candidates(env)
    attempts = []
    for entity in entities:
        zero, mask = render(env, entity, np.zeros(3))
        error = normalized_rms(native_upright, zero)
        require(error <= 1e-4, "Zero translation failed native RGB reproduction")
        attempts.append({"name":entity["name"], "visible_pixels":int(mask.sum()), "zero_rgb_nrmse":error})
        if mask.any(): return entity, zero, mask, attempts, failures
    raise NoVisibleTarget(attempts,failures)

def sample_fingerprint(env):
    """Read invariants only: never forward physics or refresh observables."""
    arrays={"sim_state":np.asarray(env.get_sim_state())}
    for name in ("body_xpos","body_xquat","geom_xpos","geom_xmat","site_xpos","site_xmat","cam_xpos","cam_xmat"):
        arrays[name]=np.asarray(getattr(env.sim.data,name))
    hashes={name:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for name,value in arrays.items()}
    clocks={name:{key:value.item() if isinstance(value,np.generic) else value
                  for key,value in vars(observable).items()
                  if key!="_sensor" and isinstance(value,(bool,int,float,str,np.generic))}
            for name,observable in env.env._observables.items()}
    return {"arrays":hashes,"time":float(env.sim.data.time),"observable_clocks":clocks}

def no_target_sample(native,error,reason):
    names=("translation_plus","translation_minus","bearing_plus","bearing_minus")
    return {"counterfactual_native":{name:native.copy() for name in names},
            "binding":{"entity":None,"candidate_attempts":getattr(error,"attempts",[]),
                       "resolution_failures":getattr(error,"failures",[])},
            "geometry":{"no_target":True,"reason":reason,"matched_displacement_m":0.0,
                        "visible_pixels":{name:0 for name in names}},
            "physics_modified":False,"counterfactual_scope":"no_visible_or_degenerate_target"}

def capture_scene(env, pixels):
    native = np.asarray(pixels)[::-1,::-1].copy()
    try:
        entity, zero, mask, attempts, failures = select_entity(env, native)
    except NoVisibleTarget as error:
        return no_target_sample(native,error,"no_visible_task_relevant_entity")
    eef = np.asarray(env.sim.data.site_xpos[env.env.robots[0].eef_site_id], dtype=float).copy()
    # Refresh a moving articulated entity's current point from the named task state.
    if entity["anchor"]=="joint_body":
        target=np.asarray(env.sim.data.body_xpos[entity["root_body_id"]],dtype=float).reshape(3)
    else:
        target=np.asarray(env.env.object_states_dict[entity["name"]].get_geom_state()["pos"],dtype=float).reshape(3)
    if float(np.linalg.norm((target-eef)[:2]))<=1e-9:
        return no_target_sample(native,NoVisibleTarget(attempts,failures),"degenerate_horizontal_relation")
    deltas, relation = paired_deltas(target, eef)
    images, visibility = {}, {}
    for name, delta in deltas.items():
        image, changed_mask = render(env, entity, delta)
        images[name] = image
        visibility[name] = int(changed_mask.sum())
    return {"counterfactual_native":images,
              "binding":{"entity":entity["name"], "parent":entity["parent"], "anchor":entity["anchor"],
                         "geom_count":len(entity["geom_ids"]),
                         "candidate_attempts":attempts, "resolution_failures":failures},
              "geometry":{**relation, "eef_xyz":eef.tolist(), "target_xyz":target.tolist(),
                          "deltas":{k:v.tolist() for k,v in deltas.items()}, "visible_pixels":visibility},
              "physics_modified":False, "counterfactual_scope":"renderer_scene_only"}

class CounterfactualCamera:
    """Capture at the native sensor timestamp, before the control step finishes."""
    def __init__(self,env):
        self.env=env; self.row=None; self.latest=None

    def __enter__(self):
        self.observable=self.env.env._observables["robot0_eye_in_hand_image"]
        self.original=self.observable._sensor
        @wraps(self.original)
        def sensor(cache):
            pixels=self.original(cache)
            if self.row is not None:
                before=sample_fingerprint(self.env)
                payload=capture_scene(self.env,pixels)
                require(sample_fingerprint(self.env)==before,
                        "Counterfactual rendering changed physics, poses, camera or sensor clocks")
                self.latest={**payload,"state_id":self.row["state_id"],
                             "native_pixels":np.array(pixels,copy=True),"sample_fingerprint":before,
                             "sample_time_s":float(self.env.sim.data.time)}
            return pixels
        self.observable._sensor=sensor
        return self

    def arm(self,row):
        self.row=row; self.latest=None

    def take(self,raw,base):
        require(self.latest is not None and self.latest["state_id"]==base["state_id"],
                "No matching native camera sample for selected decision")
        require(np.array_equal(self.latest["native_pixels"],raw["robot0_eye_in_hand_image"]),
                "Camera callback payload does not match the returned native observation")
        return self.latest

    def __exit__(self,*_):
        self.observable._sensor=self.original

def capture_frame(raw,env,runtime,task,base,engine,training,payload):
    observation=runtime.observation(raw,runtime_config())
    images={}
    for name,image in payload["counterfactual_native"].items():
        raw_cf={**raw,"robot0_eye_in_hand_image":image[::-1,::-1].copy()}
        images[name]=runtime.observation(raw_cf,runtime_config())["wrist_image"]
    record={**base, "language":task.language,"observation":observation,"counterfactual_wrist":images,
            **{k:payload[k] for k in ("binding","geometry","physics_modified","counterfactual_scope",
                                      "sample_fingerprint","sample_time_s")},
            "decision_time_s":float(env.sim.data.time),"native_camera_alignment_verified":True}
    if training:
        features = {}
        for name, wrist in images.items():
            features[name] = engine.spatial_observation({**observation, "wrist_image":wrist}).half()
        record["target_features"] = features
    return record

def replay_task(split, suite, task_id, index_rows, traces, engine, probe_only=False):
    output, receipt = shard_path(split,suite,task_id), receipt_path(split,suite,task_id)
    if complete(receipt): return
    runtime = runtime_module(); environment = None; records = []; audits=[]; started = time.monotonic()
    environment, task, initial = runtime.make_env(runtime_config(), suite, task_id)
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    try:
        episodes = sorted({r["episode_id"] for r in index_rows})
        if probe_only:episodes=episodes[:1]
        for episode_index, episode_id in enumerate(episodes):
            selected = {r["decision_block"]:r for r in index_rows if r["episode_id"] == episode_id}
            sample = next(iter(selected.values())); state = sample["state_index"]; seed = sample["rollout_seed"]
            raw = runtime.reset_wait(environment, initial[state], seed, runtime_config()["num_steps_wait"])
            current = traces.loc[(traces.suite == suite) & (traces.task_id == task_id) &
                                 (traces.state_index == state) & (traces.rollout_seed == seed)].sort_values("decision_block")
            require(len(current) == sample["decision_count"], "Trace decision count changed")
            require(min(selected)>0,"Selected decision has no preceding native sampling step")
            last_selected=max(selected)
            prefix_steps=int(current.loc[current.decision_block.lt(last_selected)].executed_slots.sum())
            progress("cache_probe" if probe_only else "cache",split=split,suite=suite,task_id=task_id,
                     episode_id=episode_id,phase="replay_prefix",required_prefix_steps=prefix_steps)
            executed = 0
            with CounterfactualCamera(environment) as sampler:
                for row in current.itertuples(index=False):
                    decision = int(row.decision_block)
                    if decision in selected:
                        record=capture_frame(raw,environment,runtime,task,selected[decision],engine,
                                             split=="train" and engine is not None,
                                             sampler.take(raw,selected[decision]))
                        if engine is not None: engine.verify_patch_path(record)
                        records.append(record)
                        if decision==last_selected:break
                    actions = decode_action_chunk(row.predicted_environment_action_chunk,
                                                  f"{episode_id}, decision_block={decision}")
                    for slot in range(int(row.executed_slots)):
                        sampler.arm(selected.get(decision+1) if slot==int(row.executed_slots)-1 else None)
                        raw, _reward, done, _info = environment.step(process_libero_action(actions[slot]).tolist())
                        executed += 1
                        if done and executed<prefix_steps:
                            failure={"episode_id":episode_id,"suite":suite,"task_id":task_id,"state_index":state,
                                     "rollout_seed":seed,"decision_block":decision,"slot":slot,
                                     "executed_steps":executed,"required_prefix_steps":prefix_steps,
                                     "source_end_step":int(current.step_end.max()),
                                     "captured_observations":len([r for r in records if r["episode_id"]==episode_id])}
                            atomic_json(failure,ROOT/f"reports/REPLAY_FAILURE_{time.time_ns()}.json")
                            raise RuntimeError("Replay ended before required observation: "+str(failure))
                    require(executed == int(row.step_end), "Saved trace step count mismatch")
            require(len([r for r in records if r["episode_id"]==episode_id]) == 5, "Incomplete episode frame capture")
            require(executed==prefix_steps,"Selected-prefix step count mismatch")
            audits.append({"episode_id":episode_id,"replayed_steps":executed,"last_selected_decision":last_selected,
                           "source_end_step":int(current.step_end.max()),"full_episode_replayed":False})
            progress("cache", split=split, suite=suite, task_id=task_id, trajectories_complete=episode_index+1,
                     trajectories_total=len(episodes), elapsed_s=time.monotonic()-started)
        if probe_only:
            finish(ROOT/"reports/REPLAY_PREFIX_PROBE.json",[],suite=suite,task_id=task_id,split=split,
                   frames=len(records),policy_loaded=False,replay_audits=audits,**resources())
            return
        expected = 40 if split == "train" else 20
        require(len(records) == expected, "Task cache frame count mismatch")
        atomic_torch({"signature":__import__("common").signature(), "split":split, "suite":suite,
                      "task_id":task_id, "records":records}, output)
        atomic_json({"elapsed_s":time.monotonic()-started, "frames":len(records),"replay_audits":audits,
                     **resources()}, output.with_suffix(".cost.json"))
        finish(receipt, [output, output.with_suffix(".cost.json")], split=split, suite=suite, task_id=task_id, frames=len(records))
    finally:
        if environment is not None: runtime.close_environment(environment)

def run_cache():
    prepared(); require_gpu("cache")
    c = cfg(); runtime = runtime_module(); runtime.prepare_graphics_backend()
    traces = pd.read_parquet(Path(c["source_run"])/"results/LONG4_TRACES.parquet" if Path(c["source_run"]).is_absolute()
                             else __import__("common").PROJECT/c["source_run"] / "results/LONG4_TRACES.parquet",
                             filters=[("condition","==","clean")])
    action_validation=validate_trace_actions(traces)
    index = read_json(ROOT/"data/INDEX.json")
    # Exercise the first unfinished task with real sensor callbacks before loading VLA.
    unfinished=[(split,suite,task) for suite in c["suites"] for split in ("train","validation")
                for task in c["task_splits"][suite][split] if not complete(receipt_path(split,suite,task))]
    if unfinished:
        split,suite,task=unfinished[0]
        rows=[r for r in index if r["suite"]==suite and r["split"]==split and r["task_id"]==task]
        replay_task(split,suite,task,rows,traces,None,probe_only=True)
    wrapper = load_wrapper(c["training_checkpoint_suite"]); engine = Engine(wrapper)
    artifacts = []
    for suite in c["suites"]:
        for split in ("train","validation"):
            for task in c["task_splits"][suite][split]:
                rows = [r for r in index if r["suite"]==suite and r["split"]==split and r["task_id"]==task]
                replay_task(split,suite,task,rows,traces,engine)
                artifacts.append(receipt_path(split,suite,task))
    require(read_json(ROOT/"reports/NATIVE_PATH_CHECK.json")["status"]=="PASS","Native patch-path check incomplete")
    artifacts.extend([ROOT/"reports/NATIVE_PATH_CHECK.json",ROOT/"reports/checks/initial.png"])
    if (ROOT/"reports/CACHE_REUSE.json").is_file():artifacts.append(ROOT/"reports/CACHE_REUSE.json")
    finish(ROOT/"reports/CACHE_COMPLETE.json", artifacts, frames=1120, heldout_frames_opened=False,
           train_features_checkpoint="libero_spatial",
           renderer_counterfactuals=4, source_action_validation=action_validation, **resources())

def cache_heldout_suite(suite):
    c=cfg(); require((ROOT/"results/WINNER.json").is_file(),"Held-out cache cannot open before winner sealing")
    traces=pd.read_parquet(__import__("common").PROJECT/c["source_run"]/"results/LONG4_TRACES.parquet",
                           filters=[("condition","==","clean")])
    validate_trace_actions(traces); index=read_json(ROOT/"data/INDEX.json")
    artifacts=[]
    for task in c["task_splits"][suite]["heldout"]:
        rows=[r for r in index if r["suite"]==suite and r["split"]=="heldout" and r["task_id"]==task]
        replay_task("heldout",suite,task,rows,traces,None); artifacts.append(receipt_path("heldout",suite,task))
    return artifacts

"""Regressions for runtime imports, sampling time, recovery and child costs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import common
import cache
import evaluate
import pipeline
import train
import worker


def test_actual_readonly_loader_and_four_checkpoint_interfaces():
    result=common.runtime_dependency_preflight()
    assert result["readonly_loader_import"]=="PASS"
    assert set(result["checkpoint_logic"])==set(common.cfg()["suites"])
    assert common.load_json is common.read_json and common.sha256_file is common.sha


def observable_module():
    root=Path(next(iter(importlib.util.find_spec("robosuite").submodule_search_locations)))
    spec=importlib.util.spec_from_file_location("relation_test_observables",root/"utils/observables.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sampling_environment():
    module=observable_module(); state={"step":0}
    @module.sensor(modality="image")
    def sensor(_):return np.full((4,4,3),state["step"]%256,dtype=np.uint8)
    observable=module.Observable("camera",sensor,sampling_rate=20)
    arrays={name:np.zeros((1,3)) for name in
            ("body_xpos","body_xquat","geom_xpos","geom_xmat","site_xpos","site_xmat","cam_xpos","cam_xmat")}
    data=SimpleNamespace(time=0.,**arrays)
    env=SimpleNamespace(env=SimpleNamespace(_observables={"robot0_eye_in_hand_image":observable}),
                        sim=SimpleNamespace(data=data),get_sim_state=lambda:np.array([state["step"]]))
    return env,observable,state,sensor


def test_camera_capture_uses_native_timestamp_not_step_end(monkeypatch):
    env,observable,state,original=sampling_environment()
    monkeypatch.setattr(cache,"capture_scene",lambda env,pixels:{"geometry":{"sampled_step":state["step"]}})
    with cache.CounterfactualCamera(env) as sampler:
        sampler.arm({"state_id":"selected"})
        for i in range(25):
            state["step"]=i+1; env.sim.data.time=.002*(i+1)
            observable.update(.002,{})
        payload=sampler.take({"robot0_eye_in_hand_image":observable.obs},{"state_id":"selected"})
        assert payload["geometry"]["sampled_step"]==int(observable.obs[0,0,0])
        assert payload["geometry"]["sampled_step"]<state["step"]
        assert payload["sample_time_s"]<env.sim.data.time
        with pytest.raises(RuntimeError,match="does not match"):
            sampler.take({"robot0_eye_in_hand_image":np.zeros_like(observable.obs)},{"state_id":"selected"})
    assert observable._sensor is original


@pytest.mark.parametrize("failure",["render_error","camera_mutation","clock_mutation"])
def test_camera_hook_restored_and_mutation_rejected(monkeypatch,failure):
    env,observable,state,original=sampling_environment()
    def broken(env,pixels):
        if failure=="render_error":raise ValueError("render failed")
        if failure=="camera_mutation":env.sim.data.cam_xpos[:]=1
        if failure=="clock_mutation":observable._time_since_last_sample+=1
        return {}
    monkeypatch.setattr(cache,"capture_scene",broken)
    with pytest.raises((ValueError,RuntimeError)):
        with cache.CounterfactualCamera(env) as sampler:
            sampler.arm({"state_id":"selected"}); observable.update(.002,{})
    assert observable._sensor is original


@pytest.mark.parametrize("done_at",[None,8,9,7])
@pytest.mark.parametrize("probe_only",[False,True])
def test_trace_replay_binds_five_frames_per_trajectory_after_each_reset(tmp_path,monkeypatch,done_at,probe_only):
    split="train" if probe_only else "validation"
    config=common.cfg(); (tmp_path/"config.json").write_text(json.dumps(config))
    monkeypatch.setattr(common,"ROOT",tmp_path); monkeypatch.setattr(cache,"ROOT",tmp_path)
    monkeypatch.setattr(common,"signature",lambda:"sampling_test")
    monkeypatch.setattr(cache,"resources",lambda:{})
    env,_,clock,_=sampling_environment(); closed=[]; sent_actions=[]
    def reset_wait(env,initial,seed,waits):
        fresh,observable,state,original=sampling_environment()
        clock.clear(); clock.update(state)
        # The installed sensor closes over its own mutable state dictionary.
        env.env=fresh.env; env.sim=fresh.sim; env.get_sim_state=fresh.get_sim_state
        env.clock=state; env.step_count=0
        return {"robot0_eye_in_hand_image":np.zeros((4,4,3),dtype=np.uint8)}
    def step(action):
        sent_actions.append(np.asarray(action,dtype=np.float32))
        observable=env.env._observables["robot0_eye_in_hand_image"]
        for _ in range(25):
            env.clock["step"]+=1; env.sim.data.time+=.002
            observable.update(.002,{})
        env.step_count+=1
        return {"robot0_eye_in_hand_image":observable.obs.copy()},0,env.step_count==done_at,{}
    env.step=step
    def observation(raw,cfg):
        return {"wrist_image":raw["robot0_eye_in_hand_image"][::-1,::-1].copy(),
                "full_image":np.zeros((4,4,3),dtype=np.uint8),"state":np.zeros(8)}
    runtime=SimpleNamespace(make_env=lambda *args:(env,SimpleNamespace(language="move"),list(range(50))),
                            reset_wait=reset_wait,observation=observation,
                            close_environment=lambda env:closed.append(True))
    monkeypatch.setattr(cache,"runtime_module",lambda:runtime)
    monkeypatch.setitem(sys.modules,"VLAAttacker.multiview.openvla_oft_experiments",
                        SimpleNamespace(process_libero_action=lambda action:action))
    def scene(env,pixels):
        native=pixels[::-1,::-1].copy()
        return {"counterfactual_native":{name:native.copy() for name in
                ("translation_plus","translation_minus","bearing_plus","bearing_minus")},
                "binding":{"entity":"test"},"geometry":{"no_target":False},
                "physics_modified":False,"counterfactual_scope":"test"}
    monkeypatch.setattr(cache,"capture_scene",scene)
    rows=[]; traces=[]
    for episode in range(4):
        for decision in range(9):
            traces.append({"suite":"libero_object","task_id":6,"state_index":40+episode,"rollout_seed":episode,
                           "decision_block":decision,
                           "predicted_environment_action_chunk":(np.arange(56).reshape(8,7)/100.+decision).tolist(),
                           "executed_slots":1,"step_end":decision+1})
        for decision in (1,3,4,6,8):
            rows.append({"episode_id":f"e{episode}","state_id":f"e{episode}_d{decision}","state_index":40+episode,
                         "rollout_seed":episode,"decision_block":decision,"decision_count":9})
    parquet=tmp_path/"replay_trace.parquet"
    pd.DataFrame(traces).to_parquet(parquet,index=False)
    restored=pd.read_parquet(parquet)
    first=restored.iloc[0].predicted_environment_action_chunk
    assert first.shape==(8,) and first.dtype==object
    assert common.validate_trace_actions(restored)["chunks_checked"]==36
    if done_at==7:
        with pytest.raises(RuntimeError,match="before required observation"):
            cache.replay_task(split,"libero_object",6,rows,restored,None,probe_only=probe_only)
        failure=common.read_json(next((tmp_path/"reports").glob("REPLAY_FAILURE_*.json")))
        assert failure["executed_steps"]==7 and failure["required_prefix_steps"]==8
        assert not cache.receipt_path(split,"libero_object",6).exists() and closed==[True]
        return
    cache.replay_task(split,"libero_object",6,rows,restored,None,probe_only=probe_only)
    if probe_only:
        result=common.read_json(tmp_path/"reports/REPLAY_PREFIX_PROBE.json")
        assert result["frames"]==5 and not result["policy_loaded"]
        assert result["replay_audits"][0]["replayed_steps"]==8 and len(sent_actions)==8
        assert not cache.receipt_path(split,"libero_object",6).exists()
        return
    shard=torch.load(cache.shard_path("validation","libero_object",6),weights_only=False)
    assert len(shard["records"])==20 and len({r["state_id"] for r in shard["records"]})==20
    assert all(r["native_camera_alignment_verified"] and r["sample_time_s"]<r["decision_time_s"] for r in shard["records"])
    assert common.complete(cache.receipt_path("validation","libero_object",6)) and closed==[True]
    assert np.array_equal(np.stack(sent_actions),np.asarray([r["predicted_environment_action_chunk"][0]
                                                           for r in traces if r["decision_block"]<8],dtype=np.float32))


def test_scene_geom_positions_restored_after_render_error(monkeypatch):
    import geometry
    geom=SimpleNamespace(objtype=5,objid=3,pos=np.array([1.,2.,3.]))
    scene=SimpleNamespace(ngeom=1,geoms=[geom])
    def fail(*args,**kwargs):
        assert np.array_equal(geom.pos,[1.1,2.,3.])
        raise ValueError("GPU render error")
    module=SimpleNamespace(mjr_render=fail)
    monkeypatch.setitem(sys.modules,"mujoco",module)
    with pytest.raises(ValueError,match="GPU render error"):
        with geometry.SceneTranslation(scene,[3],[.1,0,0]):module.mjr_render(None,scene,None)
    assert module.mjr_render is fail and np.array_equal(geom.pos,[1.,2.,3.])


def test_final_checkpoint_recovery_does_not_repeat_updates(tmp_path,monkeypatch):
    config=common.cfg(); (tmp_path/"config.json").write_text(json.dumps(config))
    monkeypatch.setattr(common,"ROOT",tmp_path); monkeypatch.setattr(train,"ROOT",tmp_path)
    monkeypatch.setattr(common,"signature",lambda:"recovery_test"); monkeypatch.setattr(train,"signature",lambda:"recovery_test")
    monkeypatch.setattr(train,"resources",lambda:{})
    folder=tmp_path/"training/translation"; folder.mkdir(parents=True)
    patch=torch.nn.Parameter(common.initial_patch())
    optimizer=torch.optim.Adam([patch],lr=.002)
    for epoch in config["checkpoint_epochs"]:common.save_png(patch,folder/f"epoch_{epoch:02d}.png")
    common.atomic_torch({"signature":"recovery_test","method":"translation","epoch":50,"step":48000,
                         "optimized":48000,"no_target":0,"patch":patch.detach(),"optimizer":optimizer.state_dict()},folder/"LAST.pt")
    before=common.sha(folder/"LAST.pt")
    train.train_method("translation",SimpleNamespace(wrapper=SimpleNamespace(device="cpu")))
    assert common.complete(folder/"COMPLETE.json")
    assert common.sha(folder/"LAST.pt")==before
    cost=common.read_json(next(folder.glob("COST_*.json")))
    assert cost["executed_candidate_updates_this_attempt"]==0
    assert cost["candidate_updates"]==48000


def test_suite_exception_writes_metrics_in_actual_child(tmp_path):
    path=tmp_path/"child_failure.json"
    code="""import evaluate
def fail(split,suite):
    raise RuntimeError('simulated suite model failure')
evaluate.suite_worker=fail
evaluate.main()
"""
    env={**os.environ,"PYTHONPATH":str(common.ROOT/"code"),"RELATION_SUITE_COST_PATH":str(path),
         "CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"}
    response=subprocess.run([sys.executable,"-B","-c",code,"--split","validation","--suite","libero_object"],
                            env=env,capture_output=True,text=True,timeout=60)
    assert response.returncode!=0
    cost=common.read_json(path)
    assert cost["status"]=="FAILED" and cost["suite"]=="libero_object"
    assert cost["pid"]!=os.getpid() and cost["measurement_scope"]=="this_process"
    assert cost["process_rss_hwm_bytes"]>0 and cost["elapsed_s"]>=0
    assert "simulated suite model failure" in cost["traceback"]


def test_killed_child_does_not_invent_parent_memory_as_child_peak(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluate,"ROOT",tmp_path)
    monkeypatch.setattr(evaluate,"cfg",lambda:{"suites":["libero_object"]})
    monkeypatch.setattr(evaluate,"complete",lambda path:False)
    def killed(command,check,env):return SimpleNamespace(returncode=-9)
    monkeypatch.setattr(evaluate.subprocess,"run",killed)
    with pytest.raises(RuntimeError,match="child failed"):
        evaluate.run_split("validation")
    cost=common.read_json(next((tmp_path/"reports/COSTS").glob("*.json")))
    assert cost["status"]=="PROCESS_EXITED_WITHOUT_FINAL_COST" and cost["exit_code"]==-9
    assert cost["process_rss_hwm_bytes"] is None and cost["cuda_peak_allocated_bytes"] is None


def test_old_revision_queue_blocks_duplicate_submission_before_preflight(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,"ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"active_queue",lambda:"123|PENDING|gres/gpu:4090:1|Priority|rel1_old_cache")
    def forbidden():raise AssertionError("Should not change receipts or submit")
    monkeypatch.setattr(pipeline,"preflight",forbidden)
    with pytest.raises(RuntimeError,match="older source revision"):
        pipeline.submit()
    assert not (tmp_path/"reports/SUBMISSION.json").exists()


def test_worker_rejects_changed_source_before_any_stage(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path)
    monkeypatch.setattr(common,"signature",lambda:"new_source")
    monkeypatch.setattr(worker,"signature",lambda:"new_source")
    monkeypatch.setattr(common,"resources",lambda:{})
    monkeypatch.setenv("SLURM_JOB_ID","123")
    monkeypatch.setenv("RELATION_EXPECTED_SIGNATURE","old_source")
    monkeypatch.setattr(sys,"argv",["worker.py","cache"])
    called=[]; monkeypatch.setattr(worker,"run",lambda stage:called.append(stage))
    with pytest.raises(RuntimeError,match="different source revision"):worker.main()
    assert called==[]
    cost=common.read_json(next((tmp_path/"reports/COSTS").glob("*.json")))
    assert cost["status"]=="FAILED"


def test_projection_does_not_hide_large_orthogonal_residual():
    negative=np.zeros((256,4)); positive=np.ones((256,4))
    perturbed=positive+np.tile([100.,-100.,100.,-100.],(256,1))
    result=evaluate.relation_metrics(perturbed,negative,positive)
    assert result["q"]==1 and result["orthogonal"]==10000
    assert result["target_distance_normalized"]==10000
    assert evaluate.relation_metrics(positive,positive,positive)["q"] is None

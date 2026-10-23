from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gripper_common as common
import collect_labels
import manage
from entities import goal_region_entities, segmentation_mapping
from long_source import decision_indices, phase_bucket, validate_records, records_digest


@pytest.mark.parametrize("count",[6,7,13,65])
def test_uniform_decisions_are_distinct_and_cover_full_episode(count):
    selected = decision_indices(count)
    assert len(selected)==len(set(selected))==6 and selected==sorted(selected)
    assert selected[0]==0 and selected[-1]==count-1
    assert {phase_bucket(i,count) for i in selected}=={0,1,2}


def test_short_episode_cannot_inflate_source_records():
    with pytest.raises(RuntimeError,match="too few"):
        decision_indices(5)


def test_multiobject_and_fixture_goals_map_to_actual_geometries():
    model = lambda name:SimpleNamespace(contact_geoms=[name],visual_geoms=[name])
    inner = SimpleNamespace(objects_dict={"mug_1":model("mug"),"pot_1":model("pot"),"basket_1":model("basket")},
                            fixtures_dict={"cabinet_1":model("cabinet")},
                            parsed_problem={"goal_state":[["And",["In","mug_1","basket_1_region"],
                                                          ["On","pot_1","stove_1_region"],
                                                          ["Close","cabinet_1_bottom_region"]]]})
    grip = model("gripper")
    inner.robots = [SimpleNamespace(robot_model=model("arm"),gripper=grip)]
    ids = {name:i for i,name in enumerate(["mug","pot","basket","cabinet","gripper","arm"])}
    env = SimpleNamespace(env=inner,sim=SimpleNamespace(model=SimpleNamespace(geom_name2id=lambda name:ids[name])))
    assert [name for name,_ in goal_region_entities(env)]==["cabinet_1","mug_1","pot_1"]
    mapping = segmentation_mapping(env)
    assert mapping["target_ids"]=={ids["mug"],ids["pot"],ids["cabinet"]}
    assert ids["basket"] not in mapping["target_ids"]
    assert mapping["gripper_ids"]=={ids["gripper"]} and mapping["robot_ids"]=={ids["gripper"],ids["arm"]}


def test_unknown_goal_subject_is_not_silently_background():
    env = SimpleNamespace(objects_dict={},fixtures_dict={},parsed_problem={"goal_state":[["Close","missing_region"]]})
    with pytest.raises(RuntimeError,match="no object/fixture"):
        goal_region_entities(env)


def source_records(cfg):
    records = []
    image = np.zeros((224,224,3),np.uint8)
    for task in range(10):
        for rank in range(5):
            for position,index in enumerate(decision_indices(10)):
                records.append({"bank_index":len(records),"source_suite":"libero_10","task_id":task,"task_name":"mock",
                                "trajectory_rank":rank,"state_index":rank,"episode_success":True,
                                "rollout_seed":cfg["training_source"]["seed"]+task*100+rank,"decision_index":index,
                                "total_decisions":10,"phase_bucket":phase_bucket(index,10),"executed_episode_steps":73,
                                "observation":{"full_image":image,"wrist_image":image,"state":np.zeros(8,np.float32)},
                                "sim_state":np.zeros(10),"native_action":torch.full((8,7),.2),
                                "clean_action":torch.full((1,8,7),.2)})
    return records


def test_long_records_accept_real_shape_contract_and_new_bank_path():
    cfg = common.config()
    records = source_records(cfg)
    assert validate_records(records,cfg) is records
    assert cfg["training_suite"]=="libero_10" and "LONG_TRAIN_BANK" in cfg["training_bank"]
    assert "training_bank_sha256" not in cfg and "mask_runtime_config" not in cfg


@pytest.mark.parametrize("change",["object_source","eval_state","duplicate_index","failed_episode","old_seed"])
def test_old_object_or_leaked_incomplete_sources_are_rejected(change):
    cfg = common.config()
    records = source_records(cfg)
    if change=="object_source": records[0]["source_suite"]="libero_object"
    if change=="eval_state": records[0]["state_index"]=40
    if change=="duplicate_index": records[0]["bank_index"]=1
    if change=="failed_episode": records[0]["episode_success"]=False
    if change=="old_seed": records[0]["rollout_seed"]=20260915
    with pytest.raises(RuntimeError):
        validate_records(records,cfg)


def test_record_identity_covers_pixels_and_actions_not_just_metadata():
    records = source_records(common.config())
    expected = records_digest(records)
    records[0]["observation"]["wrist_image"] = records[0]["observation"]["wrist_image"].copy()
    records[0]["observation"]["wrist_image"][0,0,0]=1
    assert records_digest(records)!=expected


def test_reference_is_tagged_as_object_not_a_long_matched_baseline():
    cfg = common.config()
    assert common.condition_metadata("reference_uada500",cfg)["patch_training_suite"]=="libero_object"
    assert all(common.condition_metadata(m,cfg)["patch_training_suite"]=="libero_10" for m in common.METHODS)
    assert common.condition_metadata("clean",cfg)["patch_training_suite"] is None


def test_pool_builds_new_long_bank_and_shares_it_across_methods(tmp_path,monkeypatch):
    cfg = deepcopy(common.config())
    cfg["training_bank"] = str(tmp_path/"data/LONG_TRAIN_BANK.pt")
    prep = {"status":"CPU_PREFLIGHT_PASS","mock":True}
    records = source_records(cfg)
    monkeypatch.setattr(manage,"ROOT",tmp_path)
    monkeypatch.setattr(manage,"config",lambda:cfg)
    monkeypatch.setattr(manage,"preparation",lambda:prep)
    for task in range(10):
        group = records[task*30:(task+1)*30]
        items = []
        for row in group:
            mask = torch.zeros(256,dtype=torch.bool); mask[:5]=True
            items.append({"bank_index":row["bank_index"],"source_suite":"libero_10","task_id":task,
                          "phase_bucket":row["phase_bucket"],"clean_action":row["clean_action"],
                          "clean_dino":torch.zeros(1,256,3),"clean_siglip":torch.zeros(1,256,3),
                          "close_slots":torch.ones(8,dtype=torch.bool),"task_tokens":mask,
                          "background_tokens":~mask,"background_permutation":list(range(256)),"pixel_masks":{}})
        path = tmp_path/"data/labels"/f"task_{task:02d}.pt"
        common.atomic_torch({"records":group,"items":items,"training_suite":"libero_10",
                             "preparation_signature":common.digest(prep),"source":{"mock":True}},path)
        common.atomic_json({"status":"COMPLETE","records":30,"successful_trajectories":5,"training_suite":"libero_10",
                            "preparation_signature":common.digest(prep),"sha256":common.sha256_file(path)},path.with_suffix(".json"))
    manage.build_pool()
    receipt = common.load_json(tmp_path/"data/POOL.json")
    assert receipt["training_suite"]=="libero_10" and receipt["accepted_records"]==300
    assert receipt["source_bank_sha256"]==common.sha256_file(cfg["training_bank"])
    bank = torch.load(cfg["training_bank"],weights_only=False)
    validate_records(bank["records"],cfg)
    expected_sha = common.sha256_file(cfg["training_bank"])
    manage.build_pool()
    assert common.sha256_file(cfg["training_bank"])==expected_sha
    monkeypatch.setattr(common,"ROOT",tmp_path)
    monkeypatch.setattr(common,"config",lambda:cfg)
    monkeypatch.setattr(common,"preparation",lambda:prep)
    pool,_ = common.read_training_pool()
    assert len(pool["training_order"])==500 and pool["source_bank_sha256"]==expected_sha


@pytest.mark.parametrize("interrupt",[False,True])
def test_fresh_long_collector_and_trajectory_resume_without_gpu(tmp_path,monkeypatch,interrupt):
    cfg = deepcopy(common.config())
    prep = {"mock":True,"assets":{"training_environment":{"tasks":[{"task_name":"mock"}]*10}}}
    image = np.zeros((224,224,3),np.uint8)
    obs = {"full_image":image,"wrist_image":image,"state":np.zeros(8,np.float32)}
    class Env:
        steps = state_index = 0
        interrupted = False
        closed = False
        def get_sim_state(self): return np.array([self.state_index,self.steps],np.float64)
        def step(self,action):
            if interrupt and self.state_index==2 and not self.interrupted:
                self.interrupted=True
                raise RuntimeError("intentional source interruption")
            self.steps+=1
            return {},0,self.steps>=49,{}
        def close(self): self.closed=True
    env = Env()
    wrapper = SimpleNamespace(cfg=SimpleNamespace(num_open_loop_steps=8),gen_cfg=SimpleNamespace(unnorm_key="libero_10_no_noops"),
                              vla=SimpleNamespace(_unnormalize_actions=lambda value,key:value),
                              predict_action=lambda observation,language:np.full((8,7),.2,np.float32))
    class Engine:
        def __init__(self,wrapper): self.wrapper=wrapper
        def base(self,record): return {},None
        def forward(self,inputs,proprio):
            return torch.full((1,8,7),.2),{"dino":torch.zeros(1,256,3),"siglip":torch.zeros(1,256,3)}
    def reset(env,state,seed,waits):
        env.steps,env.state_index=0,int(state[0]); return {}
    monkeypatch.setattr(collect_labels,"ROOT",tmp_path)
    monkeypatch.setattr(collect_labels,"config",lambda:cfg)
    monkeypatch.setattr(collect_labels,"preparation",lambda:prep)
    monkeypatch.setattr(collect_labels,"require_stage_gpu",lambda stage:{"gpu":"MOCK_A800"})
    monkeypatch.setattr(collect_labels,"add_paths",lambda runtime:None)
    monkeypatch.setattr(collect_labels,"load_wrapper",lambda runtime,suite:wrapper if suite=="libero_10" else pytest.fail("Object model loaded"))
    monkeypatch.setattr(collect_labels,"OFTEngine",Engine)
    monkeypatch.setattr(collect_labels,"make_env",lambda cfg,task:(env,SimpleNamespace(language="mock"),[np.array([s]) for s in range(50)]))
    monkeypatch.setattr(collect_labels,"source_manifest",lambda wrapper:{"mock":True})
    monkeypatch.setattr(collect_labels,"calibrate_gripper",lambda *args:{"mock":True})
    monkeypatch.setattr(collect_labels,"renderer_metadata",lambda:("MOCK","NVIDIA"))
    monkeypatch.setattr(collect_labels,"segmentation_mapping",lambda env:{"target_names":["mug_1","pot_1"]})
    monkeypatch.setattr(collect_labels,"reset_wait",reset)
    monkeypatch.setattr(collect_labels,"observation",lambda raw,runtime:obs)
    monkeypatch.setattr(collect_labels,"pixel_masks",lambda raw,mapping:{"target":np.zeros((256,256),np.uint8),
                        "gripper":np.zeros((256,256),np.uint8),"robot":np.zeros((256,256),np.uint8)})
    if interrupt:
        with pytest.raises(RuntimeError,match="intentional source"):
            collect_labels.collect(0)
        saved = torch.load(tmp_path/"data/labels/task_00_resume.pt",weights_only=False)
        assert saved["successful_trajectories"]==2 and len(saved["records"])==12
    collect_labels.collect(0)
    payload = torch.load(tmp_path/"data/labels/task_00.pt",weights_only=False)
    assert payload["training_suite"]=="libero_10" and len(payload["records"])==len(payload["items"])==30
    assert [a["state_index"] for a in payload["attempts"]]==list(range(5))
    assert {r["source_suite"] for r in payload["records"]}=={"libero_10"}
    assert len({r["bank_index"] for r in payload["records"]})==30 and env.closed
    collect_labels.collect(0)  # Completed-shard path does not repeat collection.

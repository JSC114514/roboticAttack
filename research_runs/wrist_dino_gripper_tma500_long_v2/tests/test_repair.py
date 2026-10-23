"""Real CPU processor integration plus isolated repair/recovery regressions.

Official image function bodies are loaded unchanged in an isolated namespace;
this avoids importing or instantiating a 7B VLA. No GPU or Slurm in these tests.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Union

import numpy as np
import pytest
from PIL import Image
import torch

import gripper_common as common
import collect_labels
import recovery
from alignment import action_alignment, NativeActionAlignmentError
from objectives import paired_masks
from preprocessing import aligned_pixels, deployment_uint8


@pytest.fixture(scope="module")
def actual_processor():
    import tensorflow as tf
    tf.config.set_visible_devices([],"GPU")
    cfg = common.config()
    source = Path(cfg["evaluation"]["paths"]["openvla_oft_root"])/"experiments/robot/openvla_utils.py"
    tree = ast.parse(source.read_text())
    names = {"prepare_images_for_vla","check_image_format","center_crop_image","crop_and_resize","resize_image_for_policy"}
    nodes = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==names
    namespace = dict(tf=tf,np=np,Image=Image,Union=Union,List=List,Any=Any,OPENVLA_IMAGE_SIZE=224)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),namespace)
    checkpoint = Path(cfg["evaluation"]["checkpoints"]["libero_10"])
    manifest=common.load_json(common.ROOT/"RECOVERY_SOURCE.json")
    payload=torch.load(manifest["tasks"]["5"]["path"],map_location="cpu",weights_only=False)
    processor_file=Path(next(p for p in payload["source"] if p.endswith("/processing_prismatic.py")
                             and "transformers_modules" in p))
    assert common.sha256_file(processor_file)==payload["source"][str(processor_file)]
    spec = importlib.util.spec_from_file_location("real_oft_processor_for_cpu_test",processor_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = module.PrismaticImageProcessor.from_pretrained(str(checkpoint),local_files_only=True)
    official = SimpleNamespace(prepare_images_for_vla=namespace["prepare_images_for_vla"])
    return processor,official,SimpleNamespace(center_crop=True)


@pytest.mark.parametrize("pattern",["random_uint8","subbyte_rgb","real_saved_observation"])
def test_native_forward_exact_and_rgb_gradient_connected(actual_processor,pattern):
    processor,official,cfg = actual_processor
    generator = torch.Generator().manual_seed(31)
    front = torch.rand(3,224,224,generator=generator)
    wrist = torch.rand(3,224,224,generator=generator)
    if pattern=="random_uint8":
        front=front.mul(255).round()/255; wrist=wrist.mul(255).round()/255
    elif pattern=="real_saved_observation":
        manifest = common.load_json(common.ROOT/"RECOVERY_SOURCE.json")
        saved = torch.load(manifest["tasks"]["5"]["path"],map_location="cpu",weights_only=False)
        obs = saved["records"][0]["observation"]
        front=torch.from_numpy(obs["full_image"].copy()).permute(2,0,1).float()/255
        wrist=torch.from_numpy(obs["wrist_image"].copy()).permute(2,0,1).float()/255
    wrist=wrist.requires_grad_(True)
    native_images=official.prepare_images_for_vla([deployment_uint8(front),deployment_uint8(wrist)],cfg)
    expected=torch.cat([processor(im,return_tensors="pt")["pixel_values"] for im in native_images],dim=1).bfloat16()
    actual=aligned_pixels(front,wrist,processor,official,cfg,torch.bfloat16)
    assert torch.equal(actual,expected), "STE must have zero forward gap, not merely a loose tolerance"
    grad=torch.autograd.grad(actual[:,6:9].float().sum(),wrist)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor
    proxy=_processor_normalize_tensor(wrist[None],processor,center_crop=True).bfloat16()
    expected_grad=torch.autograd.grad(proxy[:,:3].float().sum(),wrist)[0]
    assert torch.allclose(grad,expected_grad,atol=1e-6,rtol=1e-6)


def test_large_task_region_is_equal_subset_not_excluded():
    task=torch.zeros(256,dtype=torch.bool); task[:200]=True
    background=~task
    footprint=torch.zeros(256,dtype=torch.bool); footprint[190:205]=True
    pair=paired_masks(task,background,footprint,list(range(256)),list(reversed(range(256))))
    t,b=pair
    assert int(t.sum())==int(b.sum())==51
    assert not (t&~task).any() and not (b&~background).any() and not ((t|b)&footprint).any()
    again=paired_masks(task,background,footprint,list(range(256)),list(reversed(range(256))))
    assert all(torch.equal(a,b) for a,b in zip(pair,again))


def test_action_error_reports_actual_slot_and_dof_without_relaxing_tolerance():
    record={"native_action":torch.full((8,7),.2),"task_id":7,"state_index":2,"decision_index":11,"bank_index":216}
    action=torch.full((1,8,7),.2)
    ours=action.numpy().reshape(8,7).copy(); ours[3,1]+=.03
    report=action_alignment(record,action,ours)
    assert report["pass"] is False and report["max_error_slot"]==3 and report["max_error_dof"]==1
    assert report["tolerance"]==.02 and report["per_dof_max_abs"][1]==pytest.approx(.03)
    error=NativeActionAlignmentError(report,record,{})
    assert "task=7 state=2 decision=11 slot=3 dof=1" in str(error)


def test_close_sign_disagreement_is_detected_even_below_action_tolerance():
    record={"native_action":torch.full((8,7),.49),"task_id":0,"state_index":0,"decision_index":0,"bank_index":0}
    action=torch.full((1,8,7),.501)
    report=action_alignment(record,action,action.numpy().reshape(8,7))
    assert report["max_abs_error"]<.02 and report["close_slots_match"] is False and report["pass"] is False


def test_failed_label_exposes_the_exact_raw_observation_and_masks():
    record={"native_action":torch.full((8,7),.2),"task_id":6,"state_index":1,"decision_index":5,"bank_index":186,
            "observation":{"wrist_image":np.zeros((224,224,3),np.uint8)},"clean_action":torch.full((1,8,7),99.)}
    masks={"target":np.zeros((256,256),np.uint8)}
    action=torch.full((1,8,7),.2); action[0,4,2]+=.04
    wrapper=SimpleNamespace(cfg=SimpleNamespace(num_open_loop_steps=8),gen_cfg=SimpleNamespace(unnorm_key="long"),
                            vla=SimpleNamespace(_unnormalize_actions=lambda v,k:v))
    engine=SimpleNamespace(wrapper=wrapper,base=lambda r:({},None),forward=lambda *a:(action,{}))
    with pytest.raises(NativeActionAlignmentError) as captured:
        collect_labels.label_record(record,masks,common.config(),engine)
    error=captured.value
    assert error.details["max_error_slot"]==4 and error.details["max_error_dof"]==2
    assert error.record["observation"] is record["observation"] and error.masks is masks
    assert "clean_action" not in error.record


def test_actual_v1_recovery_exposes_raw_data_not_old_anchors():
    cfg=common.config()
    manifest=recovery.recovery_inventory(cfg)
    prep={"assets":{"recovery_source":manifest}}
    totals=0
    for task in range(10):
        old=recovery.load_recovery_task(cfg,prep,task)
        totals+=len(old["records"])
        assert "items" not in old and all("clean_action" not in r for r in old["records"])
        assert len(old["pixel_masks"])==len(old["records"])
    assert totals==228


@pytest.mark.parametrize("interrupted",[False,True])
def test_relabel_old_prefix_resumes_per_trajectory_without_replaying(tmp_path,monkeypatch,interrupted):
    cfg=common.config()
    prep={"assets":{"training_environment":{"tasks":[{"task_name":"mock"}]*10}}}
    source={"mock":True}
    records=[]
    from long_source import decision_indices,phase_bucket
    for rank in range(2):
        for position,decision in enumerate(decision_indices(10)):
            records.append({"bank_index":rank*6+position,"source_suite":"libero_10","task_id":0,"task_name":"mock",
                            "trajectory_rank":rank,"state_index":rank,"rollout_seed":cfg["training_source"]["seed"]+rank,
                            "episode_success":True,"decision_index":decision,"total_decisions":10,
                            "phase_bucket":phase_bucket(decision,10),"native_action":torch.full((8,7),.2),
                            "observation":{},"sim_state":np.zeros(3)})
    old={"sha256":"raw-sha","records":records,"pixel_masks":[{}]*12,
         "attempts":[{"state_index":r,"accepted":True,"decisions":10} for r in range(2)]}
    saved={"preparation_signature":common.digest(prep),"source":source,"task_id":0,"training_suite":"libero_10",
           "records":[],"items":[],"attempts":[],"successful_trajectories":0,"elapsed_s":0.,
           "recovery_source_sha256":"raw-sha","recovered_records":0}
    calls=0
    def fake_label(record,masks,cfg,engine):
        nonlocal calls
        calls+=1
        if interrupted and calls==8:
            raise RuntimeError("intentional relabel interruption")
        assert "clean_action" not in record
        return {**record,"clean_action":torch.full((1,8,7),.2)}, {
            "bank_index":record["bank_index"],"preprocessing":"native_uint8_crop_processor_v2",
            "native_close_slots_match":True,"native_alignment_max_abs":0.}
    monkeypatch.setattr(collect_labels,"label_record",fake_label)
    path=tmp_path/"resume.pt"
    import time
    if interrupted:
        with pytest.raises(RuntimeError,match="relabel interruption"):
            collect_labels.recover_raw_prefix(old,saved,cfg,0,prep,source,None,path,time.monotonic(),0.)
        saved=torch.load(path,weights_only=False)
        assert saved["recovered_records"]==6 and saved["successful_trajectories"]==1
        calls=8
    collect_labels.recover_raw_prefix(old,saved,cfg,0,prep,source,None,path,time.monotonic(),0.)
    final=torch.load(path,weights_only=False)
    assert final["recovered_records"]==12 and final["successful_trajectories"]==2
    assert [r["bank_index"] for r in final["records"]]==list(range(12))
    assert all(a["recovered_without_rerollout"] for a in final["attempts"])

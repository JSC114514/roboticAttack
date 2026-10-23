from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from objectives import (GraspEvents, area_tokens, executed_gripper, open_loss, paired_masks,
                        region_value, wrong_open_flags)
from entities import pixel_masks, contact_reading
from evaluate import paste_png, run_episode
from analyze import bootstrap_suite


@pytest.mark.parametrize("value,expected",[(0.,1.),(.49,1.),(.5,0.),(.51,-1.),(1.,-1.)])
def test_executed_gripper_sign(value,expected):
    assert float(executed_gripper(value))==expected


def test_only_close_slots_and_gripper_receive_target_gradient():
    a = torch.full((1,8,7),.2,requires_grad=True)
    slots = torch.tensor([True,False,True,False,False,False,False,False])
    loss = open_loss(a,slots)
    loss.backward()
    assert loss.item()==pytest.approx(.64)
    assert (a.grad[0,:,6]!=0).tolist()==slots.tolist()
    assert torch.equal(a.grad[:,:,:6],torch.zeros_like(a.grad[:,:,:6]))


def test_no_close_slots_raises():
    with pytest.raises(RuntimeError,match="No close"):
        open_loss(torch.zeros(1,8,7),torch.zeros(8,dtype=torch.bool))


def test_region_exact_restoration_and_gradient_support():
    clean = torch.rand(1,256,3,requires_grad=True)
    donor = torch.rand(1,256,3,requires_grad=True)
    mask = torch.arange(256)%4==0
    cf = region_value(clean,donor,mask)
    assert torch.equal(cf[:,~mask],clean[:,~mask]) and torch.equal(cf[:,mask],donor[:,mask])
    cf.sum().backward()
    assert clean.grad is None
    assert torch.equal(donor.grad[0,:,0]!=0,mask)


@pytest.mark.parametrize("full_scale,cf_scale",[(.1,.5),(.5,.1),(.1,.1)])
def test_detached_leaf_chain_matches_exact_max(full_scale,cf_scale):
    p = torch.tensor(.6,requires_grad=True)
    full = (.2+full_scale*p-1).square()
    gf = torch.autograd.grad(full,p)[0]
    d = 2*p
    leaf = d.detach().clone().requires_grad_(True)
    cf = (.2+cf_scale*leaf/2-1).square()
    if cf.item()>full.item():
        upstream = torch.autograd.grad(cf,leaf)[0]
        chosen = torch.autograd.grad(d,p,grad_outputs=upstream)[0]
    else:
        chosen = gf
    q = p.detach().clone().requires_grad_(True)
    f,c = (.2+full_scale*q-1).square(),(.2+cf_scale*q-1).square()
    expected = torch.autograd.grad(c if c.item()>f.item() else f,q)[0]
    assert torch.allclose(chosen,expected)


def test_paired_regions_exclude_actual_support_and_equal_count():
    task,bg,support = torch.zeros(256,dtype=torch.bool),torch.ones(256,dtype=torch.bool),torch.zeros(256,dtype=torch.bool)
    task[:5]=True; bg[:10]=False; support[0]=True; support[11]=True
    t,b = paired_masks(task,bg,support,list(reversed(range(256))))
    assert t.sum()==b.sum()==4 and not (t&b).any() and not ((t|b)&support).any()
    assert paired_masks(task,bg,torch.ones_like(task),list(range(256))) is None


@pytest.mark.parametrize("size",[224,256])
def test_dense_mask_mapping(size):
    assert torch.equal(area_tokens(torch.zeros(size,size)),torch.zeros(256))
    assert torch.allclose(area_tokens(torch.ones(size,size)),torch.ones(256),atol=1e-6)


def test_geom_ids_not_instance_labels_and_orientation():
    ids = np.zeros((256,256),np.int32); ids[0,0]=12; ids[0,1]=23; ids[0,2]=24
    raw = {"robot0_eye_in_hand_segmentation_element":ids[...,None]}
    masks = pixel_masks(raw,{"target_ids":{12},"gripper_ids":{23},"robot_ids":{23,24}})
    assert masks["target"][255,255] and masks["gripper"][255,254]
    assert not masks["background"][255,253]


def test_contact_requires_both_fingers_on_same_object():
    data = SimpleNamespace(ncon=2,contact=[SimpleNamespace(geom1=1,geom2=10),SimpleNamespace(geom1=2,geom2=10)],
                           body_xpos=np.array([[0,0,.5]]))
    env = SimpleNamespace(sim=SimpleNamespace(data=data))
    item = {"left_ids":{1},"right_ids":{2},"object_ids":{10},"body_id":0}
    assert contact_reading(env,item)==(True,.5)
    data.contact[1] = SimpleNamespace(geom1=2,geom2=11)
    assert contact_reading(env,item)==(False,.5)


def test_deployment_changes_only_fixed_wrist_support():
    obs = {"full_image":np.zeros((224,224,3),np.uint8),"wrist_image":np.zeros((224,224,3),np.uint8)}
    out = paste_png(obs,np.ones((50,50,3),np.uint8)*255,[26,137])
    assert np.array_equal(out["full_image"],obs["full_image"])
    assert not obs["wrist_image"].any()
    assert np.count_nonzero(out["wrist_image"]==255)==7500


def test_wrong_open_excludes_normal_release_and_ambiguous_threshold():
    clean,adv = np.zeros((8,7)),np.ones((8,7))
    clean[:2,6]=[1,.5]
    close,wrong = wrong_open_flags(clean,adv)
    assert close.sum()==wrong.sum()==6


@pytest.mark.parametrize("wrong_open,expected_drop",[(True,1),(False,0)])
def test_drop_proxy_requires_wrong_open_not_normal_release(wrong_open,expected_drop):
    tracker = GraspEvents({"stable_grasp_steps":3,"lift_height_m":.02,"suspected_drop_height_m":.02,"contact_loss_confirm_steps":3})
    tracker.update(False,.05,False)
    for height in [.05,.06,.07,.1]:
        tracker.update(True,height,False)
    for height in [.09,.08,.07,.06]:
        tracker.update(False,height,wrong_open)
    assert tracker.grasps==1 and tracker.losses==1 and tracker.drops==expected_drop


def test_missing_measurement_stays_missing():
    tracker = GraspEvents({"stable_grasp_steps":3,"lift_height_m":.02,"suspected_drop_height_m":.02,"contact_loss_confirm_steps":3})
    result = tracker.update(None,None,True)
    assert result["event_valid"] is False and result["object_height"] is None


def test_actual_executed_slots_only_and_no_training_hooks(monkeypatch):
    import evaluate
    obs = {"full_image":np.zeros((224,224,3),np.uint8),"wrist_image":np.zeros((224,224,3),np.uint8)}
    monkeypatch.setattr(evaluate,"reset_wait",lambda *a: {})
    monkeypatch.setattr(evaluate,"observation",lambda *a:obs)
    monkeypatch.setattr(evaluate,"contact_objects",lambda *a:[])
    class Wrapper:
        gen_cfg = SimpleNamespace(task_suite_name="libero_object")
        def predict_action(self,obs,language):
            a = np.zeros((8,7),np.float32)
            if obs["wrist_image"].any():
                a[:,6]=1
            return a
    class Env:
        def step(self,a):
            assert a[6]==-1
            return {},0,True,{}
    cfg = {"evaluation":{"num_steps_wait":10,"patch_xy":[26,137],"max_steps_by_suite":{"libero_object":280}},"event_metrics":{}}
    row,trace = run_episode(Env(),Wrapper(),SimpleNamespace(language="test"),None,42,np.ones((50,50,3),np.uint8),{},cfg)
    assert row["wrong_open_slots"]==row["eligible_close_slots"]==row["executed_steps"]==len(trace)==1
    assert row["never_held_proxy"] is None and row["event_metrics_applicable"] is False
    assert "engine" not in evaluate.__dict__


def test_task_episode_paired_bootstrap():
    from analyze import ATTACKS
    rows = [{"task_id":t,"state_index":s,"condition":m,"failure":t%2}
            for t in range(10) for s in range(2) for m in ATTACKS]
    point,samples = bootstrap_suite(pd.DataFrame(rows),200,42)
    assert np.allclose(point,.5)
    assert np.array_equal(samples[:,0],samples[:,1])

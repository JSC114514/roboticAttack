from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from engine import OFTEngine, seed_eot
from gripper_common import METHODS, config
from TMA_OFT import ExploringOFTGripperTMA, validate_resume


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x):
        local = F.adaptive_avg_pool2d(x,16).flatten(2).transpose(1,2)
        return self.scale*(local+x.mean(dim=(2,3))[:,None,:])


class Head(torch.nn.Module):
    def predict_action(self,hidden):
        return hidden


class VLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = torch.nn.Module()
        self.vision_backbone.featurizer = Encoder()
        self.vision_backbone.fused_featurizer = Encoder()
        self.norm_stats = {"libero_10_no_noops":{"proprio":{}}}
    def get_action_stats(self,key):
        return {"mask":[True]*6+[False]}
    def _unnormalize_actions(self,value,key):
        return value
    def predict_action(self,pixel_values,**kwargs):
        outputs = []
        for view in range(2):
            p = pixel_values[:,view*6:(view+1)*6]
            d = self.vision_backbone.featurizer(p[:,:3])
            s = self.vision_backbone.fused_featurizer(p[:,3:])
            outputs.append((d,s))
        grip = (.1+.2*outputs[1][0].mean()+.05*outputs[1][1].mean()).reshape(1,1,1).expand(1,8,1)
        hidden = torch.cat((torch.zeros(1,8,6),grip),dim=2)
        return hidden.detach().numpy()[0],hidden


class Wrapper:
    device = torch.device("cpu")
    dtype = torch.float32
    processor = SimpleNamespace(image_processor=None)
    gen_cfg = SimpleNamespace(unnorm_key="libero_10_no_noops",task_suite_name="libero_10")
    proprio_projector = noisy_action_projector = None
    def __init__(self):
        self.vla,self.action_head = VLA(),Head()
        self._official_openvla_utils = SimpleNamespace(normalize_proprio=lambda state,stats:state)
    def freeze_model(self):
        self.vla.eval()
        for p in self.vla.parameters():
            p.requires_grad_(False)
    def prepare_hf_inputs(self,obs,language):
        return {}
    def predict_action(self,obs,language):
        inputs,_ = self.engine.base({"observation":obs,"task_name":language})
        return self.vla.predict_action(**inputs)[0]


def fixture():
    wrapper = Wrapper()
    engine = OFTEngine(wrapper)
    wrapper.engine = engine
    engine.pixels = lambda front,wrist:torch.cat((front,front,wrist,wrist),dim=0)[None]
    records,items = [],[]
    for index in range(2):
        record = {"task_id":index,"task_name":"mock","observation":{
            "full_image":np.zeros((224,224,3),np.uint8),"wrist_image":np.full((224,224,3),index*10,np.uint8),"state":np.zeros(8)}}
        inputs,proprio = engine.base(record)
        with torch.no_grad():
            action,caps = engine.forward(inputs,proprio)
        task = torch.zeros(256,dtype=torch.bool); task[10:15]=True
        items.append({"bank_index":index,"close_slots":action[0,:,6]<.5,"clean_action":action,
                      "clean_dino":caps["dino"],"clean_siglip":caps["siglip"],
                      "task_tokens":task,"background_tokens":~task,"background_permutation":list(range(256))})
        records.append(record)
    cfg = config()
    cfg.update(outer_iterations=3,inner_loop=2,warmup_outer_iterations=0,checkpoint_outer_iterations=[0,1,3])
    return engine,records,{"items":items,"training_order":[0,1,0]},cfg


def test_source_rgb_paste_is_identical_and_gradient_connected():
    engine,records,pool,cfg = fixture()
    patch = torch.rand(3,50,50,requires_grad=True)
    image = records[0]["observation"]["wrist_image"]
    seed_eot(123)
    ours,support = engine.paste_with_support(image,patch,True)
    seed_eot(123)
    original = engine.transform._paste_patch_tensor(image,patch,True)
    assert torch.equal(ours,original)
    gradient = torch.autograd.grad(ours.sum(),patch)[0]
    assert gradient.abs().sum()>0
    seed_eot(123)
    _,support2 = engine.paste_with_support(image,torch.ones_like(patch),True)
    assert torch.equal(support,support2)


@pytest.mark.parametrize("method",METHODS)
def test_actual_training_loop_updates_only_rgb_and_matches_budget(tmp_path,method):
    engine,records,pool,cfg = fixture()
    ExploringOFTGripperTMA(engine).patchattack_unconstrained(records,pool,cfg,method,{"mock":True},folder=tmp_path/method)
    saved = torch.load(tmp_path/method/"resume.pt",weights_only=False)
    validate_resume(saved,saved["signature"],cfg,method)
    assert saved["outer"]==3 and int(next(iter(saved["optimizer"]["state"].values()))["step"])==6
    assert not torch.equal(saved["patch"],saved["initial"])
    assert all(not p.requires_grad and p.grad is None for p in engine.wrapper.vla.parameters())
    if method!=METHODS[0]:
        assert any(r["cf_gradient_selected"]>0 for r in saved["rows"])


def test_resume_reproduces_uninterrupted_training_exactly(tmp_path):
    engine,records,pool,cfg = fixture()
    trainer = ExploringOFTGripperTMA(engine)
    method = METHODS[1]
    trainer.patchattack_unconstrained(records,pool,cfg,method,{"mock":True},folder=tmp_path/"continuous")
    expected = torch.load(tmp_path/"continuous/resume.pt",weights_only=False)
    real_step = trainer.objective_step
    calls = 0
    def interrupted(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls==3:
            raise RuntimeError("intentional CPU mock interruption")
        return real_step(*args,**kwargs)
    trainer.objective_step = interrupted
    with pytest.raises(RuntimeError,match="intentional CPU mock"):
        trainer.patchattack_unconstrained(records,pool,cfg,method,{"mock":True},folder=tmp_path/"resumed")
    trainer.objective_step = real_step
    trainer.patchattack_unconstrained(records,pool,cfg,method,{"mock":True},folder=tmp_path/"resumed")
    actual = torch.load(tmp_path/"resumed/resume.pt",weights_only=False)
    assert torch.equal(actual["patch"],expected["patch"])
    for key in ("step","exp_avg","exp_avg_sq"):
        left = next(iter(actual["optimizer"]["state"].values()))[key]
        right = next(iter(expected["optimizer"]["state"].values()))[key]
        assert left==right if isinstance(left,int) else torch.equal(left,right)


def test_changed_resume_identity_rejected():
    with pytest.raises(RuntimeError,match="identity changed"):
        validate_resume({"signature":"other","method":METHODS[0],"config":{}},"expected",{},METHODS[0])

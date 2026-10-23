from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from pilot_engine import Engine,random_matched,replace_spatial,target_loss
from pilot_evaluate import feature_distance
from pilot_pixels import compose,initial_patch,load_png,save_png,verify_initial_png
from pilot_preprocessing import deployment_uint8


class Add(nn.Module):
    def forward(self,x): return x+.001


class Dino(nn.Module):
    num_prefix_tokens=5
    def __init__(self):
        super().__init__(); self.blocks=nn.ModuleList([Add() for _ in range(24)])
    def forward(self,x):
        value=x.mean().expand(1,261,1024)
        consumed=None
        for i,block in enumerate(self.blocks):
            value=block(value)
            if i==22: consumed=value[:,5:]
        return consumed


class Siglip(nn.Module):
    def forward(self,x): return (x.mean()*2).expand(1,256,1152)


class FakeWrapper:
    def __init__(self):
        backbone=nn.Module(); backbone.featurizer=Dino(); backbone.fused_featurizer=Siglip()
        self.vla=nn.Module(); self.vla.vision_backbone=backbone
        self.action_head=nn.Identity(); self.proprio_projector=nn.Identity(); self.noisy_action_projector=None
        self.device=torch.device('cpu'); self.fail=False
    def freeze_model(self):
        for module in (self.vla,self.action_head,self.proprio_projector):
            module.eval()
            for p in module.parameters(): p.requires_grad_(False)
    def predict_action(self,observation,language):
        values=[]
        for key in ('full_image','wrist_image'):
            x=torch.as_tensor(observation[key],dtype=torch.float32).permute(2,0,1)[None]/255
            d=self.vla.vision_backbone.featurizer(x)
            if self.fail: raise RuntimeError('native failure')
            s=self.vla.vision_backbone.fused_featurizer(x)
            values.append(d.mean()+s.mean())
        return torch.stack(values).sum().detach().expand(8,7).numpy().copy()


@pytest.fixture
def observation():
    return {'full_image':np.full((224,224,3),50,np.uint8),
            'wrist_image':np.full((224,224,3),100,np.uint8),'state':np.zeros(8,np.float32)}


def test_spatial_replacement_preserves_cls_register_values():
    value=torch.randn(1,261,1024); replacement=torch.randn(256,1024)
    result=replace_spatial(value,replacement)
    assert torch.equal(value[:,:5],result[:,:5]) and torch.equal(result[:,5:],replacement[None])
    assert torch.equal(value[:,5:],value[:,5:].clone())


def test_native_branch_identity_dino_isolation_and_full_cf_equivalence(observation):
    engine=Engine(FakeWrapper()); clean,features=engine.forward(observation,'pick')
    changed={**observation,'wrist_image':np.full((224,224,3),150,np.uint8)}
    cf,target=engine.forward(changed,'pick')
    for layer in (16,22):
        same,_=engine.forward(observation,'pick',{f'dino_b{layer}':features[f'dino_b{layer}']})
        assert np.array_equal(same,clean)
    dino,after=engine.forward(observation,'pick',{'dino_b22':target['dino_b22']})
    assert np.array_equal(after['siglip'],features['siglip']) and not np.array_equal(dino,clean)
    both,_=engine.forward(observation,'pick',{'dino_b22':target['dino_b22'],'siglip':target['siglip']})
    assert np.array_equal(both,cf)


def test_native_failure_cleans_all_hooks(observation):
    wrapper=FakeWrapper(); engine=Engine(wrapper); wrapper.fail=True
    with pytest.raises(RuntimeError,match='native failure'): engine.forward(observation,'pick')
    for module in [engine.dino,engine.siglip,engine.dino.blocks[16],engine.dino.blocks[22]]:
        assert not module._forward_hooks


@pytest.mark.parametrize('diverse',[False,True])
def test_png_quantization_before_eot_makes_train_forward_equal_to_png_deployment(tmp_path,diverse):
    patch=initial_patch(42).requires_grad_(True)
    wrist=np.random.default_rng(2).integers(0,256,(224,224,3),dtype=np.uint8)
    path=tmp_path/'patch.png'; save_png(patch,path)
    training,meta=compose(wrist,patch,20261019,diverse)
    deployed,other=compose(wrist,load_png(path),20261019,diverse)
    assert meta==other and np.array_equal(deployment_uint8(training),deployment_uint8(deployed))
    gradient=torch.autograd.grad(training.square().sum(),patch)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0


def test_eot_is_paired_across_candidates_and_does_not_mutate_rng():
    import random
    wrist=np.zeros((224,224,3),np.uint8)
    random.seed(3); np.random.seed(4)
    a=random.getstate(); b=np.random.get_state()
    _,first=compose(wrist,initial_patch(42),7,True)
    _,second=compose(wrist,initial_patch(43),7,True)
    assert first==second and random.getstate()==a
    assert np.array_equal(np.random.get_state()[1],b[1])


def test_initial_png_cache_cannot_silently_change_seed(tmp_path):
    path=tmp_path/'initial.png'; save_png(initial_patch(42),path)
    verify_initial_png(path,42)
    with pytest.raises(RuntimeError,match='declared random seed'): verify_initial_png(path,43)


def test_runtime_rgb_validation_checks_gradients_at_both_sites(observation,monkeypatch):
    import pilot_pixels
    wrapper=FakeWrapper(); engine=Engine(wrapper)
    def inputs(obs,language):
        values=[torch.as_tensor(np.array(obs[key],copy=True)).permute(2,0,1)[None].float()/255
                for key in ('full_image','wrist_image')]
        return {'pixel_values':torch.cat([values[0],values[0],values[1],values[1]],1)}
    wrapper.prepare_hf_inputs=inputs
    def forward(obs,language):
        values=inputs(obs,language)['pixel_values']; means=[]
        for start in (0,6):
            means.append(engine.dino(values[:,start:start+3]).mean()+engine.siglip(values[:,start+3:start+6]).mean())
        return torch.stack(means).sum().detach().expand(8,7).numpy().copy()
    wrapper.predict_action=forward
    def differentiable_pixels(wrapper,obs,patch,seed,diverse):
        rgb,meta=compose(obs['wrist_image'],patch,seed,diverse)
        front=torch.as_tensor(np.array(obs['full_image'],copy=True)).permute(2,0,1)[None].float()/255
        return torch.cat([front,front,rgb[None],rgb[None]],1),rgb,meta
    monkeypatch.setattr(pilot_pixels,'pixels',differentiable_pixels)
    result=engine.validate_rgb(observation,'pick')
    assert result['status']=='GPU_RGB_INTERFACE_PASS'
    assert set(result['rgb_gradient_l2_by_layer'])=={'16','22'}
    assert all(value>0 for value in result['rgb_gradient_l2_by_layer'].values())


def test_weighted_dino_target_loss_and_no_target_are_distinct():
    clean=torch.zeros(256,4); target=clean.clone(); target[:10]=2
    assert float(target_loss(clean,clean,target))==pytest.approx(1.)
    assert float(target_loss(target,clean,target))==0
    variable=clean.clone().requires_grad_(True)
    gradient=torch.autograd.grad(target_loss(variable,clean,target),variable)[0]
    assert gradient[:10].norm()>0 and gradient[10:].norm()==0
    assert target_loss(clean,clean,clean) is None
    assert feature_distance(clean.numpy(),clean.numpy(),target.numpy())==pytest.approx(1.)


def test_random_control_preserves_each_spatial_tokens_change_norm():
    rng=np.random.default_rng(1); clean=rng.normal(size=(256,32)).astype(np.float32)
    target=clean+rng.normal(size=clean.shape).astype(np.float32)*.1
    other=random_matched(clean,target,42)
    assert np.allclose(np.linalg.norm(other-clean,axis=-1),np.linalg.norm(target-clean,axis=-1),atol=2e-6)
    assert np.array_equal(other,random_matched(clean,target,42))
    assert not np.allclose(other-clean,target-clean)

"""Native nonlinear OFT actions and explicit wrist-only low-rank routing."""
from contextlib import contextmanager
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from wr_common import config, require, switch_suite


class WristLowRank(nn.Module):
    """Frozen base linear plus float32 adapter, enabled by an explicit view route."""
    def __init__(self, base, rank, alpha):
        super().__init__();self.base=base;self.enabled=False;self.scale=alpha/rank
        for p in base.parameters():p.requires_grad_(False)
        self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    def forward(self,x):
        y=self.base(x)
        if not self.enabled:return y
        return y+(F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(y.dtype)


@contextmanager
def route(adapters,enabled):
    previous=[m.enabled for m in adapters]
    try:
        for m in adapters:m.enabled=enabled
        yield
    finally:
        for m,v in zip(adapters,previous):m.enabled=v


def checkpoint_language(layers):
    for layer in layers:
        original=layer.forward
        def forward(*args,_original=original,**kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            return _original(*args,**kwargs)
        layer.forward=forward


class Engine:
    def __init__(self,wrapper,seed=42,gradient_checkpointing=True):
        self.wrapper=wrapper;self.backbone=wrapper.vla.vision_backbone
        self.dino=self.backbone.featurizer;self.siglip=self.backbone.fused_featurizer
        require(len(self.dino.blocks)==24 and self.dino.num_prefix_tokens==5,'Unexpected DINO layout')
        self.adapters={};self.replacement=None;self.captured=None;self.active=True
        self.front_features=None
        torch.manual_seed(seed)
        a=config()['adapter']
        for block in a['blocks']:
            attention=self.dino.blocks[block].attn
            for name in a['modules']:
                layer=WristLowRank(getattr(attention,name),a['rank'],a['alpha'])
                setattr(attention,name,layer);self.adapters[f'blocks.{block}.attn.{name}']=layer
        # Explicit image slicing, not a call counter. No checkpointed block consults
        # a mutable front/wrist counter during recomputation.
        self.original_forward=self.backbone.forward
        self.backbone.forward=self.visual_forward
        if gradient_checkpointing:checkpoint_language(wrapper.vla.language_model.model.layers)
        wrapper.vla.language_model.config.use_cache=False
        for module in (wrapper.vla,wrapper.action_head,wrapper.proprio_projector):
            module.eval()
        self.parameters=list(p for m in self.adapters.values() for p in (m.A,m.B))

    def visual_forward(self,pixels):
        require(pixels.ndim==4 and pixels.shape[0]==1 and pixels.shape[1]==12,'Require native two-view B=1')
        with torch.no_grad(),route(self.adapters.values(),False):
            front_dino=self.dino(pixels[:,:3]);front_sig=self.siglip(pixels[:,3:6])
            wrist_sig=self.siglip(pixels[:,9:12])
        self.front_features=front_dino.detach()
        if self.replacement is None:
            with route(self.adapters.values(),self.active):wrist=self.dino(pixels[:,6:9])
        else:wrist=self.replacement
        require(wrist.shape==(1,256,1024),'Native DINO interface changed')
        self.captured=wrist
        return torch.cat([torch.cat([front_dino,front_sig],dim=2),
                          torch.cat([wrist,wrist_sig],dim=2)],dim=1)

    def state_dict(self):
        return {name+'.'+key:p.detach().cpu().clone()
                for name,m in self.adapters.items() for key,p in (('A',m.A),('B',m.B))}
    def load_state_dict(self,state):
        require(set(state)==set(self.state_dict()),'Adapter keys differ')
        with torch.no_grad():
            for name,m in self.adapters.items():
                for key,p in (('A',m.A),('B',m.B)):p.copy_(state[name+'.'+key].to(p))

    def inputs(self,observation,language):
        return dict(self.wrapper.prepare_hf_inputs(copy.deepcopy(observation),language))

    def action(self,observation,language,pixels=None,replacement=None):
        inputs=self.inputs(observation,language)
        if pixels is not None:inputs['pixel_values']=pixels
        key=self.wrapper.gen_cfg.unnorm_key
        stats=self.wrapper.vla.norm_stats[key]['proprio']
        proprio=self.wrapper._official_openvla_utils.normalize_proprio(np.asarray(observation['state']),stats)
        self.replacement=replacement
        try:
            _,hidden=self.wrapper.vla.predict_action(**inputs,unnorm_key=key,do_sample=False,
                proprio=proprio,proprio_projector=self.wrapper.proprio_projector,
                noisy_action_projector=self.wrapper.noisy_action_projector,
                action_head=self.wrapper.action_head,use_film=False)
            head=getattr(self.wrapper.action_head,'module',self.wrapper.action_head)
            action=head.predict_action(hidden).reshape(8,7).float()
            require(torch.isfinite(action).all(),'Nonfinite native normalized action')
            return action
        finally:self.replacement=None;self.captured=None

    def value(self,observation,language):
        with torch.no_grad():return self.action(observation,language).detach()

    def vjp(self,observation,language,upstream):
        """Exact chain rule: release frozen LM graph before the vision backward."""
        pixels=self.inputs(observation,language)['pixel_values']
        with torch.no_grad(),route(self.adapters.values(),True):
            donor=self.dino(pixels[:,6:9]).detach()
        leaf=donor.clone().requires_grad_(True)
        action=self.action(observation,language,pixels,replacement=leaf)
        g=torch.autograd.grad(action,leaf,grad_outputs=upstream.to(action))[0]
        del action,leaf,donor
        require(torch.isfinite(g).all(),'Nonfinite LM feature gradient')
        with route(self.adapters.values(),True):feature=self.dino(pixels[:,6:9])
        torch.autograd.backward(feature,g)
        del feature,g

    def native(self,observation,language):
        with torch.no_grad():
            a=np.asarray(self.wrapper.predict_action(copy.deepcopy(observation),language),np.float32)
        require(a.shape==(8,7) and np.isfinite(a).all(),'Invalid deployed action')
        return a

    def assert_frozen(self):
        allowed={id(p) for p in self.parameters}
        for module in (self.wrapper.vla,self.wrapper.action_head,self.wrapper.proprio_projector):
            require(not module.training,'Frozen policy must remain eval')
            for p in module.parameters():
                require(p.requires_grad==(id(p) in allowed),'Unexpected trainable weight')
                if id(p) not in allowed:require(p.grad is None,'Gradient leaked into frozen base')

    def gate(self,observation,language,gradient=True):
        self.assert_frozen();tol=config()['action_tolerance']
        # Compare route-preserving forward against the unmodified native backbone,
        # with adapters disabled. The same original action head is used in both.
        self.active=False
        try:
            ours=self.native(observation,language)
            self.backbone.forward=self.original_forward
            original=self.native(observation,language)
        finally:self.backbone.forward=self.visual_forward;self.active=True
        identity=float(np.max(abs(ours-original)))
        require(identity<=tol,'Explicit view routing differs from original native policy')
        with torch.no_grad():
            a=self.action(observation,language)
            deployed=self.wrapper.vla._unnormalize_actions(a.cpu().numpy(),self.wrapper.gen_cfg.unnorm_key)
            native=self.native(observation,language)
        action_error=float(np.max(abs(deployed-native)))
        require(action_error<=tol,'Differentiable action interface differs from native deployment')
        result=dict(status='PASS',routing_max_abs=identity,action_max_abs=action_error,
                    trainable_parameters=sum(p.numel() for p in self.parameters),explicit_wrist_route=True)
        if gradient:
            # Check the nonlinear staged gradient against direct backprop at one
            # fixed native input. A scalar action coordinate avoids loss coupling.
            direct=self.action(observation,language)
            upstream=torch.zeros_like(direct);upstream[0,:3]=1/3
            grads=torch.autograd.grad(direct,self.parameters,grad_outputs=upstream,allow_unused=False)
            direct_values=[g.detach().clone() for g in grads];del grads,direct
            for p in self.parameters:p.grad=None
            self.vjp(observation,language,upstream)
            error=max(float((p.grad-g).abs().max()) for p,g in zip(self.parameters,direct_values))
            scale=max(float(g.abs().max()) for g in direct_values)
            require(error<=max(1e-5,scale*.02),'Staged/direct adapter gradients differ')
            require(all(torch.isfinite(p.grad).all() for p in self.parameters),'Nonfinite adapter gradient')
            require(any(float(p.grad.norm())>0 for p in self.parameters),'Zero adapter gradients')
            result.update(gradient_max_abs_error=error,gradient_reference_max_abs=scale)
            for p in self.parameters:p.grad=None
        self.assert_frozen();return result

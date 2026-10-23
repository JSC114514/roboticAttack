"""Frozen native VLA inference, scoped branch interventions, DINO-only gradients."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import torch
from pilot_common import atlas_path,require,runtime_config


def replace_spatial(value,replacement):
    require(tuple(value.shape)==(1,261,1024),'Native CLS/register/spatial layout changed')
    replacement=torch.as_tensor(replacement,device=value.device,dtype=value.dtype).reshape(1,256,1024)
    return torch.cat([value[:,:5],replacement],1)


class Engine:
    def __init__(self,wrapper=None):
        if wrapper is None:
            atlas_path()
            from atlas_runtime import load_wrapper
            wrapper=load_wrapper(runtime_config(),'libero_spatial')
        self.wrapper=wrapper; wrapper.freeze_model()
        self.dino=wrapper.vla.vision_backbone.featurizer
        self.siglip=wrapper.vla.vision_backbone.fused_featurizer
        require(len(self.dino.blocks)==24 and self.dino.num_prefix_tokens==5,'DINO architecture changed')
        for module in (wrapper.vla,wrapper.action_head,wrapper.proprio_projector,wrapper.noisy_action_projector):
            if module is not None:
                require(not module.training and all(not p.requires_grad for p in module.parameters()),'VLA must remain frozen/eval')

    @contextmanager
    def hooks(self,replacements=None,capture=True):
        replacements=replacements or {}; counts={}; values={}; handles=[]
        def install(module,key,spatial=False):
            counts[key]=0
            def hook(_module,_args,value):
                index=counts[key]; counts[key]+=1
                require(index<2,'Unexpected encoder calls; require front then wrist')
                if index!=1: return None
                changed=value
                if key in replacements:
                    if spatial: changed=replace_spatial(value,replacements[key])
                    else:
                        changed=torch.as_tensor(replacements[key],device=value.device,dtype=value.dtype).reshape_as(value)
                if capture: values[key]=changed[:,5:] if spatial else changed
                return changed if key in replacements else None
            handles.append(module.register_forward_hook(hook))
        for layer in (16,22): install(self.dino.blocks[layer],f'dino_b{layer}',spatial=True)
        install(self.dino,'consumed'); install(self.siglip,'siglip')
        try:
            yield values
            require(all(n==2 for n in counts.values()),'Native forward did not encode exactly front then wrist')
        finally:
            for handle in handles: handle.remove()

    @torch.no_grad()
    def forward(self,observation,language,replacements=None,capture=True):
        obs={k:np.array(v,copy=True) for k,v in observation.items()}
        with self.hooks(replacements,capture) as captured:
            action=np.asarray(self.wrapper.predict_action(obs,language),dtype=float).reshape(8,7)
        require(np.isfinite(action).all(),'Nonfinite native actions')
        arrays={k:v.detach().float().squeeze(0).cpu().numpy().copy() for k,v in captured.items()}
        if arrays:
            require(np.array_equal(arrays['dino_b22'],arrays['consumed']),'Policy no longer consumes raw block22 spatial outputs')
            require(all(np.isfinite(v).all() for v in arrays.values()),'Nonfinite visual feature')
            arrays.pop('consumed')
        return action,arrays

    def spatial(self,pixel_values,layer):
        require(layer in (16,22),'Only preregistered two sites are permitted')
        found=[]
        def hook(_module,_args,value): found.append(value[:,5:])
        handle=self.dino.blocks[layer].register_forward_hook(hook)
        try: self.dino(pixel_values[:,6:9])
        finally: handle.remove()
        require(len(found)==1,'DINO-only optimization must encode only wrist')
        require(found[0].shape==(1,256,1024),'Unexpected spatial feature size')
        return found[0]

    def validate_rgb(self,observation,language):
        from pilot_pixels import initial_patch,pixels,compose
        from pilot_preprocessing import deployment_uint8
        patch=initial_patch(42,self.wrapper.device).requires_grad_(True)
        values,rgb,_=pixels(self.wrapper,observation,patch,42,False)
        exact_obs={**observation,'wrist_image':deployment_uint8(rgb)}
        inputs=self.wrapper.prepare_hf_inputs({k:np.array(v,copy=True) for k,v in exact_obs.items()},language)
        require(torch.equal(values.detach(),inputs['pixel_values']),'PNG/native processor differs from optimization forward')
        clean_inputs=self.wrapper.prepare_hf_inputs({k:np.array(v,copy=True) for k,v in observation.items()},language)
        require(torch.equal(values[:,:6],clean_inputs['pixel_values'][:,:6]),'Patch changed front channels')
        _,features=self.forward(exact_obs,language)
        gradients={}
        for layer in (16,22):
            # Independent graphs share no already-consumed preprocessing backward.
            layer_values,_,_=pixels(self.wrapper,observation,patch,42,False)
            direct=self.spatial(layer_values,layer)
            if layer==22:
                require(np.array_equal(direct.detach().float().squeeze(0).cpu().numpy(),features['dino_b22']),
                        'Standalone DINO differs from native full-forward consumed features')
            gradient=torch.autograd.grad(direct.float().square().mean(),patch)[0]
            require(torch.isfinite(gradient).all() and gradient.norm()>0,'DINO-to-RGB gradient disconnected/nonfinite')
            gradients[str(layer)]=float(gradient.norm())
        require(all(p.grad is None for p in self.wrapper.vla.parameters()),'Model accumulated gradients')
        return {'status':'GPU_RGB_INTERFACE_PASS','native_pixels_exact':True,'front_unchanged':True,
                'consumed_readout_exact':True,'rgb_gradient_l2_by_layer':gradients,
                'forward':'native_uint8_PIL','backward':'explicit_STE','scientific_effect':'NOT_TESTED'}


def target_loss(patched,clean,target):
    """Energy-weighted spatial matching; baseline clean loss is 1 when defined."""
    patched=patched.float().reshape(256,-1)
    clean=torch.as_tensor(clean,device=patched.device,dtype=torch.float32).reshape_as(patched)
    target=torch.as_tensor(target,device=patched.device,dtype=torch.float32).reshape_as(patched)
    energy=(target-clean).square().mean(-1)
    total=energy.sum()
    if float(total.detach())==0: return None
    weights=energy/total
    denominator=(weights*energy).sum()
    return (weights*(patched-target).square().mean(-1)).sum()/denominator.clamp_min(torch.finfo(torch.float32).tiny)


def random_matched(clean,target,seed):
    """Independent random spatial directions, per-token L2 matched to the CF delta."""
    rng=np.random.default_rng(seed); delta=np.asarray(target,dtype=float)-clean
    noise=rng.normal(size=delta.shape)
    noise*=np.linalg.norm(delta,axis=-1,keepdims=True)/np.maximum(np.linalg.norm(noise,axis=-1,keepdims=True),1e-30)
    return (clean+noise).astype(np.float32)

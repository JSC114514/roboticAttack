"""Exact nonlinear action loss; staged VJP keeps one backward graph at a time."""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from at_common import ROOT, native, require, reference_engine_module, atomic_json, config
from at_math import objective

ref=reference_engine_module()

def install_checkpointing(module):
    for block in module:
        original=block.forward
        def forward(*args,_original=original,**kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            return _original(*args,**kwargs)
        block.forward=forward

class Engine(ref.Engine):
    def __init__(self,wrapper,checkpointing=False):
        super().__init__(wrapper)
        self.checkpointing=checkpointing
        if checkpointing:
            # Explicit wrappers preserve eval mode. HF checkpointing would require train().
            install_checkpointing(wrapper.vla.language_model.model.layers)
            install_checkpointing(self.dino.blocks)
            install_checkpointing(self.siglip.blocks)
        wrapper.vla.language_model.config.use_cache=False

    def action(self,observation,language,pixels=None,replacements=None):
        inputs=dict(self.wrapper.prepare_hf_inputs(copy.deepcopy(observation),language))
        if pixels is not None: inputs['pixel_values']=pixels
        stats=self.wrapper.vla.norm_stats[self.wrapper.gen_cfg.unnorm_key]['proprio']
        proprio=self.wrapper._official_openvla_utils.normalize_proprio(np.asarray(observation['state']),stats)
        with self.hooks(replacements) as captures:
            _,hidden=self.wrapper.vla.predict_action(**inputs,unnorm_key=self.wrapper.gen_cfg.unnorm_key,
                do_sample=False,proprio=proprio,proprio_projector=self.wrapper.proprio_projector,
                noisy_action_projector=self.wrapper.noisy_action_projector,
                action_head=self.wrapper.action_head,use_film=False)
        head=getattr(self.wrapper.action_head,'module',self.wrapper.action_head)
        action=head.predict_action(hidden).reshape(1,8,7).float()
        require(torch.isfinite(action).all(),'Nonfinite normalized actions')
        return action,captures

    def physical(self,action):
        return np.asarray(self.wrapper.vla._unnormalize_actions(
            action.detach().float().cpu().numpy().reshape(8,7),self.wrapper.gen_cfg.unnorm_key),np.float32)

    def staged_gradient(self,row,patch,method):
        pixels,_=ref.patch_pixels(row['observation'],patch,self.wrapper)
        with torch.no_grad(): _,caps=self.action(row['observation'],row['language'],pixels.detach())
        branches=('dino',) if method=='geometry_teacher' else ('dino','siglip')
        leaves={k:caps[k].detach().clone().requires_grad_(True) for k in branches}
        del caps
        action,_=self.action(row['observation'],row['language'],pixels.detach(),leaves)
        loss=objective(action,row['teacher'],method)
        require(torch.isfinite(loss),'Nonfinite action loss')
        metrics={'loss':float(loss.detach()),'target_mse':float((action[...,:3]-row['teacher']['target'].to(action)[...,:3]).square().mean().detach()),
                 'translation_change_l2':float((action[...,:3]-row['teacher']['clean'].to(action)[...,:3]).norm(dim=-1).mean().detach())}
        upstream=torch.autograd.grad(loss,tuple(leaves.values()))
        del action,loss,leaves
        # Language-model graph is released before vision recomputation.
        gradient=torch.zeros_like(patch)
        for name,weight in zip(branches,upstream):
            if name=='dino': donor=self.spatial(pixels)
            else:
                # Rebuild the STE graph after the DINO VJP consumed it.
                pixels,_=ref.patch_pixels(row['observation'],patch,self.wrapper)
                donor=self.siglip(pixels[:,9:12])
            require(torch.isfinite(weight).all(),'Nonfinite visual-leaf gradient')
            gradient+=torch.autograd.grad(donor,patch,grad_outputs=weight)[0]
            del donor
        require(torch.isfinite(gradient).all(),'Nonfinite RGB gradient')
        metrics['gradient_l2']=float(gradient.norm())
        return gradient,metrics

    def gate(self,row,patch,method,key):
        # Original PNG/preprocessing/native DINO equivalence and freezing gate.
        self.verify_patch_path(row)
        png=native.deployment_uint8(patch) if hasattr(native,'deployment_uint8') else patch.detach().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
        obs=native.paste_numpy(row['observation'],png)
        with torch.no_grad():
            action,caps=self.action(obs,row['language'])
            official=np.asarray(self.wrapper.predict_action(copy.deepcopy(obs),row['language'])).reshape(8,7)
            error=float(np.max(np.abs(self.physical(action)-official)))
            require(error<=config()['action_tolerance'],'Differentiable/native action mismatch')
            zero,zero_caps=self.action(obs,row['language'],replacements={'dino':caps['dino']})
            require(torch.equal(action,zero),'Zero DINO replacement changed action')
            require(torch.equal(caps['siglip'],zero_caps['siglip']),'Zero replacement changed SigLIP')
            clean,clean_caps=self.action(row['observation'],row['language'])
            positive=torch.as_tensor(row['target_features']['translation_plus'],device=patch.device)
            target,changed=self.action(row['observation'],row['language'],replacements={'dino':positive})
            require(torch.equal(clean_caps['siglip'],changed['siglip']),'Teacher changed SigLIP')
            require(torch.equal(clean.cpu(),row['teacher']['clean']),'Cached clean action changed')
            require(torch.equal(target.cpu(),row['teacher']['target']),'Cached teacher action changed')
        grad,metrics=self.staged_gradient(row,patch,method)
        require(float(grad.norm())>0,'Startup RGB gradient is zero')
        require(all(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters())
            for m in (self.wrapper.vla,self.wrapper.action_head,self.wrapper.proprio_projector) if m is not None),'Victim not frozen/eval')
        atomic_json({'status':'PASS','native_action_max_abs_error':error,'zero_replacement_exact':True,
            'teacher_only_wrist_dino':True,'cached_actions_exact':True,'model_frozen':True,
            'staged_vjp_cpu_equivalence_test':'tests/test_contracts.py',**metrics},ROOT/f'reports/GATES/{key}.json')

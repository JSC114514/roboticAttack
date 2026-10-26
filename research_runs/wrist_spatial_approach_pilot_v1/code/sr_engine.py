# 发布注释：冻结VLA，只优化RGB补丁；完整双分支前向与选择性梯度路径需分别解释。
# 运行来源与发布哈希分别记录在SOURCE_MANIFEST.json；历史运行目录不改动。
"""Exact nonlinear action loss; staged VJP keeps one backward graph at a time."""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from sr_common import ROOT, native, require, reference_engine_module, atomic_json, config
from sr_math import objective

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
        branches=('dino','siglip')
        leaves={k:caps[k].detach().clone().requires_grad_(True) for k in branches}
        del caps
        action,_=self.action(row['observation'],row['language'],pixels.detach(),leaves)
        command=self.command(action)
        loss=objective(command,row['teacher'],method)
        require(torch.isfinite(loss),'Nonfinite action loss')
        unit=torch.as_tensor(row['teacher']['unit'],device=command.device,dtype=command.dtype)
        metrics={'loss':float(loss.detach()),
            'first_slot_radial_goal_m':float((command[0,0]*unit).sum().detach()),
            'goal_saturated_fraction':float((command.detach().abs()>.05).float().mean())}
        upstream=torch.autograd.grad(loss,tuple(leaves.values()))
        del action,command,loss,leaves
        # Language-model graph is released before vision recomputation.
        gradient=torch.zeros_like(patch)
        for name,weight in zip(branches,upstream):
            if name=='dino': donor=self.spatial(pixels)
            else:
                # Rebuild the STE graph after the DINO VJP consumed it.
                pixels,_=ref.patch_pixels(row['observation'],patch,self.wrapper)
                donor=self.siglip(pixels[:,9:12])
            require(torch.isfinite(weight).all(),'Nonfinite visual-leaf gradient')
            branch_gradient=torch.autograd.grad(donor,patch,grad_outputs=weight)[0]
            metrics[name+'_gradient_l2']=float(branch_gradient.norm())
            gradient+=branch_gradient
            del donor
        require(torch.isfinite(gradient).all(),'Nonfinite RGB gradient')
        metrics['gradient_l2']=float(gradient.norm())
        return gradient,metrics

    def affine(self):
        if not hasattr(self,'_action_affine'):
            z=np.zeros((8,7),np.float32);o=np.ones((8,7),np.float32)
            f=self.wrapper.vla._unnormalize_actions;k=self.wrapper.gen_cfg.unnorm_key
            bias=np.asarray(f(z,k),np.float64)[0];scale=(np.asarray(f(o,k))-np.asarray(f(-o,k)))[0]*.5
            self._action_affine=(scale,bias)
        return self._action_affine

    def physical_tensor(self,action):
        scale,bias=self.affine()
        return action*torch.as_tensor(scale,device=action.device,dtype=action.dtype)+torch.as_tensor(bias,device=action.device,dtype=action.dtype)

    def command(self,action,clipped=False):
        values=self.physical_tensor(action)[...,:3]
        # Native deployment clips the controller INPUT. Train without this clip
        # so an initially saturated prediction does not lose its RGB gradient.
        if clipped:values=values.clamp(-1,1)
        return values*.05

    def gate(self,row,patch,method,key):
        self.verify_patch_path(row)
        png=patch.detach().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
        obs=native.paste_numpy(row['observation'],png)
        with torch.no_grad():
            pixels,_=ref.patch_pixels(row['observation'],patch,self.wrapper)
            expected=self.wrapper.prepare_hf_inputs(copy.deepcopy(obs),row['language'])['pixel_values']
            require(torch.equal(pixels.detach(),expected),'PNG pixels differ from train forward')
            action,caps=self.action(obs,row['language'])
            official=np.asarray(self.wrapper.predict_action(copy.deepcopy(obs),row['language'])).reshape(8,7)
            error=float(np.max(np.abs(self.physical(action)-official)))
            affine_error=float(np.max(np.abs(self.physical_tensor(action)[0].cpu().numpy()-official)))
            require(max(error,affine_error)<=config()['action_tolerance'],'Native action/affine mismatch')
            zero,_=self.action(obs,row['language'],replacements={'dino':caps['dino'],'siglip':caps['siglip']})
            require(torch.equal(action,zero),'Zero dual-branch replacement is not identity')
            clean,_=self.action(row['observation'],row['language'])
            require(torch.equal(clean.cpu(),row['teacher']['clean']),'Cached Goal clean actions changed')
        grad,metrics=self.staged_gradient(row,patch,method)
        require(float(grad.norm())>0,'Zero startup RGB gradient')
        modules=(self.wrapper.vla,self.wrapper.action_head,self.wrapper.proprio_projector,self.wrapper.noisy_action_projector)
        require(all(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters()) for m in modules if m is not None),'Victim not frozen/eval')
        atomic_json({'status':'PASS','native_action_max_abs_error':error,'native_affine_max_abs_error':affine_error,
            'png_pixels_exact':True,'dual_branch_identity':True,'both_wrist_branches_differentiated':True,
            'checkpoint':config()['checkpoint'],'model_frozen':True,**metrics},ROOT/f'reports/GATES/{key}.json')

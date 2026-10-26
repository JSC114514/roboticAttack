"""Restore only the wrist spatial outputs consumed by the native policy."""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from fr_common import require,config
from fr_math import branches

class Vision:
    def __init__(self,wrapper):
        self.wrapper=wrapper
        self.dino=wrapper.vla.vision_backbone.featurizer
        self.siglip=wrapper.vla.vision_backbone.fused_featurizer
        require(len(self.dino.blocks)==24,'DINO block layout differs')

    @contextmanager
    def hooks(self,mode='native',donor=None):
        selected=branches(mode)
        require(not selected or donor is not None,'Restoration requires same-observation donor')
        calls=dict(dino=0,siglip=0,b22=0);capture={};handles=[]
        def b22(_m,_args,value):
            i=calls['b22'];calls['b22']+=1;require(i<2,'Extra DINO block22 call')
            require(torch.is_tensor(value) and value.shape[1]==261,'DINO CLS/register/spatial layout differs')
            capture['b22_'+('front','wrist')[i]]=value[:,5:].detach().clone()
        try:
            handles.append(self.dino.blocks[22].register_forward_hook(b22))
            for branch,module in (('dino',self.dino),('siglip',self.siglip)):
                def hook(_m,_args,value,branch=branch):
                    i=calls[branch];calls[branch]+=1;require(i<2,'Extra visual branch call')
                    view=('front','wrist')[i];key=branch+'_'+view
                    require(torch.is_tensor(value) and value.ndim==3 and value.shape[:2]==(1,256),'Native spatial layout differs')
                    require(torch.isfinite(value).all(),'Nonfinite visual features')
                    capture[key]=value.detach().clone()
                    if donor is not None and view=='front':
                        require(torch.equal(value,donor[key]),'Front features changed during wrist-only patch/restore')
                    if view=='wrist' and branch in selected:
                        target=donor[key]
                        require(target.shape==value.shape and target.dtype==value.dtype and target.device==value.device,
                                'Donor shape/dtype/device differs')
                        return target
                handles.append(module.register_forward_hook(hook))
            yield capture
            require(calls==dict(dino=2,siglip=2,b22=2),'Policy must process front then wrist exactly once')
            for view in ('front','wrist'):
                require(torch.equal(capture['b22_'+view],capture['dino_'+view]),'Policy no longer consumes block22 spatial output')
        finally:
            for handle in handles:handle.remove()

    @torch.inference_mode()
    def action(self,observation,language,mode='native',donor=None):
        with self.hooks(mode,donor) as features:
            action=np.asarray(self.wrapper.predict_action(copy.deepcopy(observation),language),np.float32)
        require(action.shape==(8,7) and np.isfinite(action).all(),'Invalid native action block')
        return action,features

    def metrics(self,features,donor,mode):
        out={};selected=branches(mode)
        for name in ('dino','siglip'):
            delta=features[name+'_wrist'].float()-donor[name+'_wrist'].float()
            rms=float(delta.square().mean().sqrt())
            out[name+'_native_delta_rms']=rms
            out[name+'_used_delta_rms']=0. if name in selected else rms
        return out

    def gate(self,observation,deployed,language,expected_clean=None):
        from jr_rollout import predict,frozen_modules
        clean,donor=self.action(observation,language)
        direct=predict(self.wrapper,observation,language)
        zero,_=self.action(observation,language,'both',donor)
        raw,_=self.action(deployed,language,'native',donor)
        raw_direct=predict(self.wrapper,deployed,language)
        restored,_=self.action(deployed,language,'both',donor)
        values=dict(clean_capture_error=float(np.max(abs(clean-direct))),zero_restore_error=float(np.max(abs(clean-zero))),
            patch_capture_error=float(np.max(abs(raw-raw_direct))),both_restore_error=float(np.max(abs(restored-clean))))
        if expected_clean is not None:values['historical_first_action_error']=float(np.max(abs(clean-expected_clean)))
        require(max(values.values())<=config()['action_tolerance'],'Native/identity/both restore gate failed: '+str(values))
        require(all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in frozen_modules(self.wrapper)),
                'Model is not frozen/eval')
        return dict(status='PASS',block=22,wrist_spatial_only=True,front_unchanged=True,model_frozen=True,**values)

"""GPU-only native forward, passive spatial reads and explicit feature attenuation."""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from va_common import require,config,adapters
from va_math import blend,grid

class Vision:
    def __init__(self):
        _,native=adapters();self.wrapper=native.load_wrapper('libero_goal')
        self.dino=self.wrapper.vla.vision_backbone.featurizer
        self.siglip=self.wrapper.vla.vision_backbone.fused_featurizer
        require(len(self.dino.blocks)==24 and self.dino.num_prefix_tokens==5,'DINO architecture differs')
        devices=set();offload=[]
        for name in ('vla','action_head','proprio_projector','noisy_action_projector'):
            m=getattr(self.wrapper,name,None)
            if m is None:continue
            require(not m.training and all(not p.requires_grad for p in m.parameters()),'Model must be frozen/eval')
            devices.update(str(p.device) for p in m.parameters())
            for n,c in m.named_modules():
                if getattr(getattr(c,'_hf_hook',None),'offload',False):offload.append(n)
                if any(str(v) in ('cpu','disk','meta') for v in (getattr(c,'hf_device_map',{}) or {}).values()):offload.append(n)
        require(len(devices)==1 and next(iter(devices)).startswith('cuda:') and not offload,'CPU/disk offload forbidden')
        self.residency={'devices':sorted(devices),'offload':offload,'frozen':True}

    @contextmanager
    def hooks(self,gate_view=None,means=None,alpha=.5):
        captured={};calls={'dino':0,'siglip':0,'b22':0};handles=[]
        def b22(_m,_a,value):
            i=calls['b22'];calls['b22']+=1
            require(i<2,'Unexpected extra DINO call')
            captured['b22_'+('front','wrist')[i]]=value[:,5:].detach().float().cpu().numpy().squeeze(0).copy()
        handles.append(self.dino.blocks[22].register_forward_hook(b22))
        for branch,module in (('dino',self.dino),('siglip',self.siglip)):
            def hook(_m,_args,value,branch=branch):
                i=calls[branch];calls[branch]+=1;require(i<2,'Unexpected extra view')
                view=('front','wrist')[i];key=branch+'_'+view
                require(value.ndim==3 and value.shape[:2]==(1,256),'Spatial token layout changed')
                captured[key]=value.detach().float().cpu().numpy().squeeze(0).copy()
                require(np.isfinite(captured[key]).all(),'Nonfinite spatial features')
                if view==gate_view:
                    mean=torch.as_tensor(means[key],device=value.device,dtype=value.dtype)[None]
                    return blend(value,mean,alpha)
            handles.append(module.register_forward_hook(hook))
        try:
            yield captured
            require(all(n==2 for n in calls.values()),'Expected front then wrist exactly once per branch')
            for view in ('front','wrist'):
                require(np.array_equal(captured['b22_'+view],captured['dino_'+view]),'Consumed output differs from block22')
        finally:
            for h in handles:h.remove()

    @torch.no_grad()
    def features(self,observation,language):
        inputs=self.wrapper.prepare_hf_inputs(copy.deepcopy(observation),language)
        with self.hooks() as features:self.wrapper.vla.vision_backbone(inputs['pixel_values'])
        return features

    @torch.no_grad()
    def action(self,observation,language,gate_view=None,means=None,alpha=None):
        with self.hooks(gate_view,means,config()['gate_alpha'] if alpha is None else alpha) as features:
            action=np.asarray(self.wrapper.predict_action(copy.deepcopy(observation),language),dtype=np.float32).reshape(8,7)
        require(np.isfinite(action).all(),'Nonfinite action')
        return action,features

    def check(self,row):
        action,captured=self.action(row['observation'],row['language'])
        direct=self.features(row['observation'],row['language'])
        require(all(np.array_equal(captured[k],direct[k]) for k in captured),'Vision/full forward features differ')
        repeated,_=self.action(row['observation'],row['language'],gate_view='wrist',means=captured,alpha=1.)
        require(float(np.max(abs(action-repeated)))<=config()['action_tolerance'],'Identity feature gate changed action')
        a=row['teacher']['clean'].numpy().reshape(8,7)
        expected=self.wrapper.vla._unnormalize_actions(a,self.wrapper.gen_cfg.unnorm_key)
        error=float(np.max(abs(action-expected)))
        require(error<=config()['action_tolerance'],'Cached/native clean action differs')
        return {'status':'PASS','native_action_max_abs_error':error,'passive_features_identical':True,
                'identity_gate':True,'native_block':22,**self.residency}

def pooled(features):return np.stack([grid(features['dino_'+v]) for v in ('front','wrist')]).astype(np.float32)

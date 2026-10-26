"""Native OFT inference with an isolated wrist spatial-output substitution."""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from sd_common import require,native

class Engine:
    def __init__(self,wrapper):
        self.wrapper=wrapper;self.dino=wrapper.vla.vision_backbone.featurizer
        self.siglip=wrapper.vla.vision_backbone.fused_featurizer
        require(len(self.dino.blocks)==24 and self.dino.num_prefix_tokens==5,'DINO interface changed')
        self.frozen()
    def frozen(self):
        for module in (self.wrapper.vla,self.wrapper.action_head,self.wrapper.proprio_projector,self.wrapper.noisy_action_projector):
            if module is not None:
                require(not module.training and all(not p.requires_grad and p.grad is None for p in module.parameters()),'Model weights are not frozen/eval')
    @contextmanager
    def hooks(self,replacements):
        counts={'block':0,'dino':0,'siglip':0};out={};handles=[]
        def block(_m,_a,value):
            i=counts['block'];counts['block']+=1
            out[f'prefix_{i}']=value[:,:5];out[f'raw_b22_{i}']=value[:,5:]
        handles.append(self.dino.blocks[22].register_forward_hook(block))
        def branch(name):
            def hook(_m,_a,value):
                i=counts[name];counts[name]+=1;changed=value
                if i==1 and name in replacements:
                    changed=torch.as_tensor(replacements[name],device=value.device,dtype=value.dtype).reshape_as(value)
                out[f'{name}_{i}']=changed
                return changed if i==1 and name in replacements else None
            return hook
        handles.append(self.dino.register_forward_hook(branch('dino')))
        handles.append(self.siglip.register_forward_hook(branch('siglip')))
        try:
            yield out;require(all(n==2 for n in counts.values()),'Expected exactly front then wrist for both branches')
        finally:
            for h in handles:h.remove()
    @torch.no_grad()
    def forward(self,obs,language,replacements=None):
        replacements=replacements or {}
        with self.hooks(replacements) as captured:
            a=np.asarray(self.wrapper.predict_action(copy.deepcopy(obs),language),dtype=np.float32).reshape(8,7)
        require(np.isfinite(a).all(),'Nonfinite native action')
        f={k:v.detach().float().cpu().numpy().copy() for k,v in captured.items()}
        require(f['dino_1'].shape==(1,256,1024),'Wrong native spatial output shape')
        if 'dino' not in replacements:require(np.array_equal(f['raw_b22_1'],f['dino_1']),'Policy no longer consumes raw block22')
        return a,f

    @torch.no_grad()
    def encode_wrist(self,obs):
        inputs=self.wrapper.prepare_hf_inputs(copy.deepcopy(obs),'dummy');found=[]
        handle=self.dino.blocks[22].register_forward_hook(lambda _m,_a,v:found.append(v[:,5:]))
        try:self.dino(inputs['pixel_values'][:,6:9])
        finally:handle.remove()
        require(len(found)==1 and found[0].shape==(1,256,1024),'Standalone DINO interface changed')
        return found[0].float().cpu().numpy().copy()

def verify_isolation(clean,changed):
    for name in ('prefix_0','prefix_1','dino_0','siglip_0','siglip_1','raw_b22_0','raw_b22_1'):
        require(np.array_equal(clean[name],changed[name]),'DINO-only substitution contaminated '+name)

def load_engine():
    from sd_common import runtime_config
    require(native().runtime_config()==runtime_config(),'Read-only loader runtime does not match pinned settings')
    return Engine(native().load_wrapper('libero_spatial'))

"""Passive native hooks only. No activation replacement or VLA training."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from atlas_common import config,require
from atlas_math import pooled_grid


class Encoder:
    def __init__(self,wrapper):
        self.wrapper=wrapper; wrapper.freeze_model()
        self.dino=wrapper.vla.vision_backbone.featurizer
        require(len(self.dino.blocks)==24 and self.dino.num_prefix_tokens==5,'DINO architecture changed')
        require(all(not p.requires_grad for p in wrapper.vla.parameters()),'Model is not frozen')
        self.shapes={}

    @contextmanager
    def hooks(self):
        counts={}; captured={}; handles=[]; detail={}
        def install(module,name,fn):
            counts[name]=0
            def hook(_m,_args,value):
                index=counts[name]; counts[name]+=1
                require(index<2,'Unexpected extra view call')
                fn(config()['views'][index],value)
                # Intentionally no return: a read-only observation hook.
            handles.append(module.register_forward_hook(hook))
        for block,module in enumerate(self.dino.blocks):
            def record(view,x,block=block):
                require(tuple(x.shape)==(1,261,1024),'Block layout changed')
                for kind,value in [('spatial',pooled_grid(x[:,5:])),
                                   ('cls',x[:,:1].float().flatten().cpu().numpy().astype(np.float16)),
                                   ('register',x[:,1:5].float().flatten().cpu().numpy().astype(np.float16))]:
                    captured[f'{view}__b{block:02d}_{kind}']=value
                if block in config()['detail_blocks']:
                    detail[f'{view}__dense_b{block:02d}']=x[:,5:].float().squeeze(0).cpu().numpy().astype(np.float16)
            install(module,f'b{block:02d}',record)
        for block in config()['detail_blocks']:
            att=self.dino.blocks[block].attn
            def qkv(view,value,block=block,att=att):
                q,k,v=value.reshape(1,261,3,att.num_heads,-1).permute(2,0,3,1,4).unbind(0)
                q,k=att.q_norm(q),att.k_norm(k)
                for kind,x in [('q',q),('k',k),('v',v)]:
                    dense=x[:,:,5:].permute(0,2,1,3).reshape(1,256,1024)
                    captured[f'{view}__b{block:02d}_{kind}']=pooled_grid(dense)
                key=F.normalize(k[:,:,5:].float(),dim=-1)
                sim=(key@key.transpose(-2,-1)).mean(1).squeeze(0)
                captured[f'{view}__b{block:02d}_key_similarity']=sim.cpu().numpy().astype(np.float16).flatten()
            install(att.qkv,f'qkv{block:02d}',qkv)
        def consumed(view,x):
            require(tuple(x.shape)==(1,256,1024),'Consumed DINO layout changed')
            captured[f'{view}__consumed']=pooled_grid(x)
            detail[f'{view}__dense_consumed']=x.float().squeeze(0).cpu().numpy().astype(np.float16)
        install(self.dino,'consumed',consumed)
        try:
            yield captured,detail
            require(all(n==2 for n in counts.values()),f'Expected front then wrist: {counts}')
        finally:
            for handle in handles: handle.remove()

    def inputs(self,observation,instruction):
        # Official proprio normalization may be in-place; use independent arrays.
        obs={k:np.array(v,copy=True) for k,v in observation.items()}
        return dict(self.wrapper.prepare_hf_inputs(obs,instruction))

    @torch.no_grad()
    def extract(self,observation,instruction,validate=False):
        values=self.inputs(observation,instruction)
        with self.hooks() as (global_features,dense):
            vision=self.wrapper.vla.vision_backbone(values['pixel_values'])
        require(torch.isfinite(vision).all(),'Nonfinite native vision features')
        qa={'passive_only':True,'native_consumed_block':22,'block23_consumed_by_policy':False}
        for view in config()['views']:
            require(np.array_equal(dense[f'{view}__dense_b22'],dense[f'{view}__dense_consumed']),
                    'Native readout is not raw block22 spatial tokens')
        if validate:
            native=np.asarray(self.wrapper.predict_action({k:np.array(v,copy=True) for k,v in observation.items()},instruction))
            with self.hooks() as (check,_):
                hooked=np.asarray(self.wrapper.predict_action({k:np.array(v,copy=True) for k,v in observation.items()},instruction))
            require(np.array_equal(native,hooked),'Passive hooks changed native actions')
            require(all(np.array_equal(check[k],v) for k,v in global_features.items()),
                    'Direct vision and native full forward differ')
            qa.update(native_hook_action_max_abs=float(np.max(abs(native-hooked))),
                      full_forward_feature_identity=True)
        require(all(np.isfinite(v).all() for v in global_features.values()),'Nonfinite feature')
        return global_features,dense,qa

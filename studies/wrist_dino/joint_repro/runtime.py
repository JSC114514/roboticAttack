# 发布阅读说明：连接原生双视角、归一化、动作头和完整视觉梯度。RGB预处理与PNG部署保持同一算子；内部替换仅用于单列的机制诊断。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Native joint OFT runtime with unmodified two-view, two-branch visual routing.

Historical loader/render helpers are read-only dependencies, not historical
train/eval wrappers. Only language-layer recomputation changes memory use; it
preserves all forward operators and autograd paths and is checked in P03.
"""
import contextlib
import copy
import sys
import types
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,config,read,sha,require,module,setup_paths,atomic


def runtime_config():
    c=config();p=Path(c['project']);old=read(p/'research_runs/wrist_dino_spatial_b12_diag500_v1/config.json')['evaluation']
    return dict(old,checkpoints={s:c['model_dir'] for s in c['suites']},unnorm_keys={s:s+'_no_noops' for s in c['suites']},env_image_resolution=256,policy_image_resolution=224,num_steps_wait=10,num_open_loop_steps=8,max_steps_by_suite=c['max_steps'])


def native_runtime():
    p=setup_paths();d=p/'research_runs/wrist_dino_spatial_b12_diag500_v1/code';sys.path.insert(0,str(d))
    r=module('jr04_native_runtime',d/'model_runtime.py');r.add_paths(runtime_config());return r


def load_wrapper(monitor):
    """Load once per worker; forbid CPU/disk offload and every trainable model weight."""
    require(os.environ.get('SLURM_JOB_ID'),'GPU work requires Slurm allocation')
    c=config();device_name=torch.cuda.get_device_name(0)
    gpu_kind=next((k for k in ['4090','3090','A800'] if k.lower() in device_name.lower()),None)
    require(gpu_kind is not None,'Unsupported/forbidden GPU: '+device_name)
    # Historical renderer helper has a3090default; bind its guard to the actual
    # allocated device instead of inheriting a stale card-specific constant.
    os.environ['OFT_STAGE_GPU']=gpu_kind
    r=native_runtime();r.prepare_graphics_backend()
    # This shim supplies configuration only. It does not invoke old orchestrators
    # or let the historical loader synchronize/checkpoint Python files on disk.
    shim=types.ModuleType('jr_common')
    for n,v in dict(config=config,runtime_config=runtime_config,require=require,read=read,sha=sha).items():setattr(shim,n,v)
    sys.modules['jr_common']=shim
    loader=module('jr04_readonly_loading',Path(c['project'])/'research_runs/wrist_joint_checkpoint_patch_retest_v1/code/jr_loading.py')
    monitor.update(phase='loading_joint_model_once');w=loader.load_native_wrapper(r,c['suites'][0])
    for m in [w.vla,w.action_head,w.proprio_projector]:
        m.eval()
        for p in m.parameters():require(not p.requires_grad and p.device.type=='cuda','Trainable/offloaded model weight')
    require(not any(x in torch.cuda.get_device_name(0).lower() for x in c['forbidden_gpu_types']),'Forbidden GPU device')
    for s in c['suites']:require(s+'_no_noops' in w.vla.norm_stats,'Missing suite normalization')
    return w,r


import os

def switch_suite(w,suite):
    w.cfg.task_suite_name=suite;w.gen_cfg.task_suite_name=suite
    w.cfg.unnorm_key=suite+'_no_noops';w.gen_cfg.unnorm_key=suite+'_no_noops'


def policy_observation(obs):
    """Raw expert256px images use native resize, same as live simulation input.

    RLDS images are already upright; simulator raw frames are rotated in native
    observation(). Never rotate a decoded RLDS image a second time.
    """
    from VLAAttacker.multiview.openvla_oft_experiments import resize_oft_observation_images
    if all(np.asarray(obs[k]).shape==(224,224,3) for k in ['full_image','wrist_image']):return copy.deepcopy(obs)
    return resize_oft_observation_images(copy.deepcopy(obs),input_views=['front','wrist'],resize_size=224)


class Engine:
    def __init__(self,w,checkpoint=True):
        from white_patch.appply_random_transform import RandomPatchTransform
        self.w=w;self.transform=RandomPatchTransform(w.device,False);self.original_layers=[];self.checkpoint_enabled=checkpoint;self.cached_obs=None;self.cached_inputs=None;self.cached_key=None
        if checkpoint:
            from torch.utils.checkpoint import checkpoint as recompute
            for layer in w.vla.language_model.model.layers:
                original=layer.forward;self.original_layers.append((layer,original))
                def forward(*args,_fn=original,**kwargs):
                    if torch.is_grad_enabled():return recompute(_fn,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
                    return _fn(*args,**kwargs)
                layer.forward=forward
        w.vla.language_model.config.use_cache=False

    @contextlib.contextmanager
    def direct(self):
        """Temporarily use original language layers for a paired gradient witness."""
        wrapped=[(l,l.forward) for l,_ in self.original_layers]
        try:
            for l,fn in self.original_layers:l.forward=fn
            yield
        finally:
            for l,fn in wrapped:l.forward=fn

    def pixels(self,obs,patch=None,view='wrist',geometry=True):
        """Use the exact source EOT, then native differentiable crop/normalization."""
        from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor
        values=[]
        for name,key in [('front','full_image'),('wrist','wrist_image')]:
            if patch is not None and name==view:im=self.transform._paste_patch_tensor(obs[key],patch,geometry)
            else:im=torch.as_tensor(np.asarray(obs[key]),device=self.w.device,dtype=torch.float32).permute(2,0,1)/255
            values.append(_processor_normalize_tensor(im[None],self.w.processor.image_processor,center_crop=True))
        return torch.cat(values,dim=1).to(dtype=self.w.dtype)

    def patch_footprint(self,obs,view='wrist'):
        """Observe source EOT support by paired zero/onepatches, preserving RNG.

        No second transform is implemented. The original affine/paste function
        supplies both masks with identicalrandomdraws. The mask follows the
        same source center crop, then area-pools to native16x16spatialtokens.
        """
        import random
        from VLAAttacker.multiview.openvla_oft_experiments import _center_crop_resize_tensor
        state=(random.getstate(),np.random.get_state())
        key='wrist_image' if view=='wrist' else 'full_image'
        try:
            one=self.transform._paste_patch_tensor(obs[key],torch.ones(3,50,50,device=self.w.device),True)
            random.setstate(state[0]);np.random.set_state(state[1])
            zero=self.transform._paste_patch_tensor(obs[key],torch.zeros(3,50,50,device=self.w.device),True)
            support=((one-zero).abs().amax(0)>0).float()[None,None]
            support=_center_crop_resize_tensor(support,output_size=(224,224),crop_scale=.9)
            return torch.nn.functional.interpolate(support,size=(16,16),mode='area').reshape(-1)
        finally:
            random.setstate(state[0]);np.random.set_state(state[1])

    def action(self,obs,language,patch=None,view='wrist',geometry=True,capture=False,replacements=None):
        """Real nonlinear head forward; replacements accepted only for diagnostics.

        Normal training calls never supply replacements. Capture hooks return
        None and therefore do not modify activations or branch gradients.
        """
        w=self.w;key=(language,w.gen_cfg.unnorm_key)
        if obs is not self.cached_obs or key!=self.cached_key:
            self.cached_inputs=dict(w.prepare_hf_inputs(obs,language));self.cached_obs=obs;self.cached_key=key
        inputs=dict(self.cached_inputs)
        # Clean references must use the official native processor exactly.
        # Only RGB patch training uses the source differentiable EOT path.
        if patch is not None:inputs['pixel_values']=self.pixels(obs,patch,view,geometry)
        stats=w.vla.norm_stats[w.gen_cfg.unnorm_key]['proprio'];proprio=w._official_openvla_utils.normalize_proprio(np.asarray(obs['state']),stats)
        features={'dino':[],'siglip':[]};handles=[]
        def hook(key):
            def observe(module,args,output):
                index=len(features[key]);out=output
                if replacements is not None and (key,index) in replacements:out=replacements[key,index].to(output)
                features[key].append(out)
                if out.requires_grad and capture:out.retain_grad()
                return out if out is not output else None
            return observe
        try:
            for key,model in [('dino',w.vla.vision_backbone.featurizer),('siglip',w.vla.vision_backbone.fused_featurizer)]:handles.append(model.register_forward_hook(hook(key)))
            _,hidden=w.vla.predict_action(**inputs,unnorm_key=w.gen_cfg.unnorm_key,do_sample=False,proprio=proprio,proprio_projector=w.proprio_projector,noisy_action_projector=w.noisy_action_projector,action_head=w.action_head,use_film=False)
            head=getattr(w.action_head,'module',w.action_head);actions=head.predict_action(hidden).reshape(1,8,7).float()
            require(torch.isfinite(actions).all(),'Nonfinite native head actions')
            require(len(features['dino'])==len(features['siglip'])==2,'Native two-view visual interface changed')
            return actions,features
        finally:
            for h in handles:h.remove()

    def native(self,obs,language):
        """Deploy through original PNG/native helper; no training Tensor shortcut."""
        with torch.no_grad():return np.asarray(self.w.predict_action(copy.deepcopy(obs),language),np.float32).reshape(8,7)

    def assert_frozen(self):
        for m in [self.w.vla,self.w.action_head,self.w.proprio_projector]:
            require(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters()),'Model weight/gradient freeze violated')

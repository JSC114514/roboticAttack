"""Original RGB EOT and native-forward / explicit-STE-backward preprocessing."""
from __future__ import annotations
import random
import numpy as np
import torch
from PIL import Image
from pilot_common import config,require
from pilot_transform import RandomPatchTransform
from pilot_preprocessing import aligned_pixels,deployment_uint8


def initial_patch(seed,device='cpu'):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    return torch.rand(tuple(config()['patch_shape']),generator=generator,dtype=torch.float32).to(device)


def compose(wrist,patch,seed=0,diverse=False):
    require(tuple(patch.shape)==(3,50,50),'Every candidate must deploy one shared 50x50 RGB patch')
    # Preserve surrounding Python/NumPy streams; paired EOT does not affect optimizers.
    py_state=random.getstate(); np_state=np.random.get_state()
    try:
        random.seed(seed); np.random.seed(seed%(2**32-1))
        # Quantize the PATCH before EOT as a saved PNG would; preserve its STE.
        patch=patch.detach().clamp(0,1).mul(255).round().div(255)+(patch-patch.detach())
        image=torch.as_tensor(np.array(wrist,copy=True),device=patch.device).permute(2,0,1).float()/255
        if diverse: x,y=random.randint(0,174),random.randint(0,174)
        else: x,y=config()['patch_xy']
        canvas=torch.ones_like(image)*-100
        canvas[:,y:y+50,x:x+50]=patch
        matrix=torch.eye(3,device=patch.device)
        if diverse:
            transform=RandomPatchTransform(patch.device,resize_patch=False)
            matrix=transform.combined_transform_matrix().to(patch.device)
            canvas=transform.apply_affine_transform(canvas,matrix).squeeze(0)
        support=(canvas.detach()>=-20).any(0)
        rgb=torch.where(canvas < -20,image,canvas).clamp(0,1)
        return rgb,{'x':x,'y':y,'diverse':diverse,'seed':seed,
                    'matrix':matrix.detach().cpu().tolist(),'support_pixels':int(support.sum())}
    finally:
        random.setstate(py_state); np.random.set_state(np_state)


def pixels(wrapper,observation,patch,seed=0,diverse=False):
    rgb,metadata=compose(observation['wrist_image'],patch,seed,diverse)
    front=torch.as_tensor(np.array(observation['full_image'],copy=True),device=patch.device).permute(2,0,1).float()/255
    values=aligned_pixels(front,rgb,wrapper.processor.image_processor,
                          wrapper._official_openvla_utils,wrapper.gen_cfg,wrapper.dtype)
    return values,rgb,metadata


def save_png(patch,path):
    from pathlib import Path
    import os
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    value=patch.detach().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
    temp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    Image.fromarray(value).save(temp,format='PNG'); os.replace(temp,path)


def load_png(path,device='cpu'):
    with Image.open(path) as image:
        require(image.mode=='RGB' and image.size==(50,50),'PNG must be a native 50x50 RGB artifact')
        value=np.array(image,copy=True)
    return torch.as_tensor(value,device=device).permute(2,0,1).float()/255


def verify_initial_png(path,seed):
    expected=initial_patch(seed).mul(255).round().div(255)
    require(torch.equal(load_png(path),expected),'Initial PNG differs from declared random seed; preserve and inspect cache')

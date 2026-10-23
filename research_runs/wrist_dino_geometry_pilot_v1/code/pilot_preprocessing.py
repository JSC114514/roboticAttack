"""Native uint8/PIL forward with a differentiable RGB surrogate backward.

Never replace official rollout with the torch approximation. Quantization has
zero/undefined useful derivatives; the STE is explicit and only for optimization.
"""
from __future__ import annotations

import numpy as np
import torch

from pilot_common import require


def deployment_uint8(rgb):
    require(tuple(rgb.shape) == (3,224,224), "Expected RGB CHW224")
    return rgb.detach().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy().copy()


def native_pixels(front, wrist, processor, official, cfg, dtype):
    require(bool(cfg.center_crop), "OFT center-crop setting changed")
    images = official.prepare_images_for_vla([deployment_uint8(front),deployment_uint8(wrist)],cfg)
    values = [processor(image,return_tensors="pt")["pixel_values"] for image in images]
    require(len(values)==2 and all(tuple(v.shape)==(1,6,224,224) for v in values),
            "Expected actual Front/Wrist DINO+SigLIP processor layout")
    return torch.cat(values,dim=1).to(device=front.device,dtype=dtype).detach()


def aligned_pixels(front, wrist, processor, official, cfg, dtype):
    exact = native_pixels(front,wrist,processor,official,cfg,dtype)
    if not (front.requires_grad or wrist.requires_grad):
        return exact
    from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor
    surrogate = torch.cat([_processor_normalize_tensor(x[None],processor,center_crop=True)
                           for x in (front,wrist)],dim=1)
    # Subtract in FP32 BEFORE casting: the forward residual is exactly zero,
    # while its backward remains the torch crop/normalize Jacobian.
    return exact + (surrogate-surrogate.detach()).to(dtype=dtype)

"""OFT-specific differentiable adapter for the independent Exploring TMA copy."""
from __future__ import annotations

from contextlib import contextmanager
import random

import numpy as np
import torch
import torchvision

from gripper_common import require
from objectives import area_tokens, paired_masks
from source_transform import RandomPatchTransform


def seed_eot(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32-1))


class OFTEngine:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        wrapper.freeze_model()
        self.transform = RandomPatchTransform(wrapper.device, resize_patch=False)
        self.dino = wrapper.vla.vision_backbone.featurizer
        self.siglip = wrapper.vla.vision_backbone.fused_featurizer

    def base(self, record):
        obs = record["observation"]
        inputs = dict(self.wrapper.prepare_hf_inputs(obs, record["task_name"]))
        stats = self.wrapper.vla.norm_stats[self.wrapper.gen_cfg.unnorm_key]["proprio"]
        proprio = self.wrapper._official_openvla_utils.normalize_proprio(np.asarray(obs["state"]), stats)
        front = torch.as_tensor(np.asarray(obs["full_image"]), device=self.wrapper.device, dtype=torch.float32)
        wrist = torch.as_tensor(np.asarray(obs["wrist_image"]), device=self.wrapper.device, dtype=torch.float32)
        inputs["pixel_values"] = self.pixels(front.permute(2,0,1)/255, wrist.permute(2,0,1)/255)
        return inputs, proprio

    def pixels(self, front, wrist):
        from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor
        processor = self.wrapper.processor.image_processor
        return torch.cat([_processor_normalize_tensor(x[None], processor, center_crop=True)
                          for x in (front, wrist)], dim=1).to(self.wrapper.dtype)

    def paste_with_support(self, image, patch, geometry):
        # Independent source-transform copy; retain the exact Exploring RGB
        # sentinel, affine sampler and replacement, adding footprint bookkeeping.
        im = torchvision.transforms.ToTensor()(self.transform._to_rgb_pil(image)).to(self.wrapper.device)
        c, h, w = im.shape
        require(tuple(patch.shape) == (3, 50, 50) and (h,w) == (224,224), "RGB shape changed")
        canvas = torch.ones(c,h,w,device=self.wrapper.device) * -100
        support_canvas = torch.ones(c,h,w,device=self.wrapper.device) * -100
        x,y = random.randint(0,w-50), random.randint(0,h-50)
        canvas[:,y:y+50,x:x+50] = patch
        support_canvas[:,y:y+50,x:x+50] = 1
        if geometry:
            matrix = self.transform.combined_transform_matrix().to(self.wrapper.device)
            canvas = self.transform.apply_affine_transform(canvas,matrix).squeeze(0)
            support_canvas = self.transform.apply_affine_transform(support_canvas,matrix).squeeze(0)
        # Upper RGB envelope covers every actual replaced/interpolated pixel,
        # without making rejection/region selection depend on learned colors.
        support = (support_canvas >= -20).any(dim=0)
        require(not ((canvas.detach() >= -20).any(dim=0) & ~support).any(), "Conservative support missed RGB pixel")
        return torch.where(canvas < -20, im, canvas).clamp(0,1), support

    def eot(self, record, item, patch, seed, cfg):
        device = self.wrapper.device
        task, background = [item[key].to(device).bool() for key in ("task_tokens", "background_tokens")]
        for attempt in range(cfg["max_eot_attempts"]):
            used_seed = seed + attempt
            seed_eot(used_seed)
            wrist, support = self.paste_with_support(record["observation"]["wrist_image"], patch, cfg["geometry"])
            footprint = area_tokens(support) > cfg["patch_exclusion_threshold"]
            pair = paired_masks(task, background, footprint, item["background_permutation"])
            if pair is not None:
                front = torch.as_tensor(np.asarray(record["observation"]["full_image"]), device=device,
                                        dtype=torch.float32).permute(2,0,1)/255
                return self.pixels(front,wrist), pair, used_seed, attempt
        raise RuntimeError("No paired unoccluded regions in 32 predetermined EOT candidates; preserve state, stop")

    @contextmanager
    def hooks(self, replacement=None):
        captures, counts, handles = {}, {"dino":0,"siglip":0}, []
        for name, module in (("dino",self.dino),("siglip",self.siglip)):
            def hook(_module, args, output, *, name=name):
                call = counts[name]
                counts[name] += 1
                if call == 1:
                    require(isinstance(output,torch.Tensor) and output.shape[:2] == (1,256),
                            f"Actual consumed {name} is not the dense 16x16 grid")
                    captures[name] = output
                    if name == "dino":
                        captures["dino_input"] = args[0]
                    if name == "dino" and replacement is not None:
                        require(replacement.shape == output.shape, "CF output shape changed")
                        return replacement.to(output.dtype)
            handles.append(module.register_forward_hook(hook))
        try:
            yield captures
            require(counts == {"dino":2,"siglip":2}, "Expected Front then Wrist, one call per view")
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, inputs, proprio, pixels=None, replacement=None):
        values = {**inputs, "pixel_values": pixels if pixels is not None else inputs["pixel_values"]}
        with self.hooks(replacement) as captures:
            _detached, hidden = self.wrapper.vla.predict_action(
                **values, unnorm_key=self.wrapper.gen_cfg.unnorm_key, do_sample=False,
                proprio=proprio, proprio_projector=self.wrapper.proprio_projector,
                noisy_action_projector=self.wrapper.noisy_action_projector,
                action_head=self.wrapper.action_head, use_film=False)
        head = getattr(self.wrapper.action_head,"module",self.wrapper.action_head)
        action = head.predict_action(hidden).reshape(1,8,7).float()
        require(torch.isfinite(action).all(), "Nonfinite continuous action")
        return action, captures

    def direct_dino(self, pixels):
        value = self.dino(pixels[:,6:9])
        require(isinstance(value,torch.Tensor) and value.shape[:2] == (1,256), "Direct dense donor shape")
        return value

    def validate(self, record, item, cfg, patch):
        """GPU-only interface checks; not a scientific outcome gate."""
        from objectives import open_loss, region_value
        inputs, proprio = self.base(record)
        stats = self.wrapper.vla.get_action_stats(self.wrapper.gen_cfg.unnorm_key)
        require(not bool(stats.get("mask",[True]*7)[6]), "Gripper must retain raw [0,1] convention")
        with torch.no_grad():
            clean, captures = self.forward(inputs,proprio)
            official = np.asarray(self.wrapper.predict_action(record["observation"],record["task_name"])).reshape(8,7)
            ours = self.wrapper.vla._unnormalize_actions(clean.cpu().numpy()[0],self.wrapper.gen_cfg.unnorm_key)
            action_error = float(np.max(np.abs(official-ours)))
            require(action_error <= .02, "Normal differentiable action not aligned with official deployment")
            anchor_error = float((clean.cpu()-item["clean_action"]).abs().max())
            require(anchor_error <= .02 and torch.equal(clean[0,:,6].cpu()<.5,item["close_slots"]),
                    "Fresh clean action/close slots differ from label collection")
            zero_cf, zero_caps = self.forward(inputs,proprio,replacement=captures["dino"])
            require(torch.equal(clean,zero_cf) and torch.equal(captures["siglip"],zero_caps["siglip"]),
                    "Zero-delta CF must be exact identity")
        seed = cfg["data_seed"] * 100000 + 1000
        pixels, pair, used_seed, attempt = self.eot(record,item,patch,seed,cfg)
        normal, caps = self.forward(inputs,proprio,pixels)
        require(torch.equal(caps["dino_input"],pixels[:,6:9]), "Direct donor pixel layout differs")
        with torch.no_grad():
            direct = self.direct_dino(pixels)
            donor_error = float((caps["dino"].float()-direct.float()).abs().max())
            require(torch.allclose(caps["dino"],direct,rtol=1e-5,atol=1e-5), "Direct donor is not model-consumed DINO")
        full_grad = torch.autograd.grad(open_loss(normal,item["close_slots"].to(patch.device)),patch)[0]
        require(torch.isfinite(full_grad).all(), "Normal opening RGB gradient is nonfinite")
        del normal, caps, pixels
        pixels, pair2, seed2, attempt2 = self.eot(record,item,patch,seed,cfg)
        require(seed2==used_seed and attempt2==attempt and all(torch.equal(a,b) for a,b in zip(pair,pair2)),
                "EOT/region pairing changed during same-patch recomputation")
        donor = self.direct_dino(pixels)
        leaf = donor.detach().clone().requires_grad_(True)
        cf = region_value(captures["dino"],leaf,pair2[0])
        cf_action, cf_caps = self.forward(inputs,proprio,replacement=cf)
        require(torch.equal(cf_caps["siglip"],captures["siglip"]), "CF must keep Wrist SigLIP clean")
        upstream = torch.autograd.grad(open_loss(cf_action,item["close_slots"].to(patch.device)),leaf)[0]
        region_grad = torch.autograd.grad(donor,patch,grad_outputs=upstream)[0]
        require(torch.isfinite(region_grad).all(), "Region opening RGB gradient is nonfinite")
        return {"status":"GPU_INTERFACE_PASS","official_action_max_error":action_error,
                "fresh_anchor_max_error":anchor_error,"direct_dino_max_error":donor_error,
                "zero_cf_identity":True,"normal_rgb_gradient_l2":float(full_grad.norm()),
                "region_rgb_gradient_l2":float(region_grad.norm()),
                "normal_gradient_nonzero_at_probe":bool(full_grad.abs().sum()>0),
                "region_gradient_nonzero_at_probe":bool(region_grad.abs().sum()>0),
                "scientific_result":"NOT_TESTED"}

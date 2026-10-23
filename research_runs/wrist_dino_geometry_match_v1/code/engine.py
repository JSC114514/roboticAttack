"""Frozen OFT forwards and DINO-only differentiable wrist features."""
from __future__ import annotations
from contextlib import contextmanager
import copy,os
import numpy as np
import torch
from common import ROOT, cfg, require, initial_patch, save_png, load_png, paste_numpy, atomic_json, resources, signature

def deployment_uint8(rgb):
    require(tuple(rgb.shape) == (3, 224, 224), "Expected CHW224 RGB")
    return rgb.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy().copy()

def native_pixels(front, wrist, wrapper):
    official, processor = wrapper._official_openvla_utils, wrapper.processor.image_processor
    prepared = official.prepare_images_for_vla([deployment_uint8(front), deployment_uint8(wrist)], wrapper.gen_cfg)
    require(len(prepared) == 2, "Official image preparation changed")
    values = [processor(image, return_tensors="pt")["pixel_values"] for image in prepared]
    require(all(tuple(v.shape) == (1, 6, 224, 224) for v in values), "Expected DINO+SigLIP six-channel views")
    return torch.cat(values, dim=1).to(device=front.device, dtype=wrapper.dtype).detach()

def aligned_pixels(front, wrist, wrapper):
    exact = native_pixels(front, wrist, wrapper)
    if not (front.requires_grad or wrist.requires_grad): return exact
    from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor
    processor = wrapper.processor.image_processor
    surrogate = torch.cat([_processor_normalize_tensor(x[None], processor, center_crop=True) for x in (front, wrist)], dim=1)
    return exact + (surrogate-surrogate.detach()).to(dtype=wrapper.dtype)

def patch_pixels(observation, patch, wrapper):
    x, y = cfg()["patch_xy"]
    quantized = patch.detach().clamp(0, 1).mul(255).round().div(255) + (patch-patch.detach())
    front = torch.as_tensor(np.array(observation["full_image"], copy=True), device=patch.device).permute(2,0,1).float().div(255)
    wrist = torch.as_tensor(np.array(observation["wrist_image"], copy=True), device=patch.device).permute(2,0,1).float().div(255)
    mask = torch.zeros_like(wrist); mask[:, y:y+50, x:x+50] = 1
    canvas = torch.zeros_like(wrist); canvas[:, y:y+50, x:x+50] = quantized
    wrist = wrist*(1-mask) + canvas*mask
    return aligned_pixels(front, wrist, wrapper), wrist

class Engine:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.dino = wrapper.vla.vision_backbone.featurizer
        self.siglip = wrapper.vla.vision_backbone.fused_featurizer
        require(len(self.dino.blocks) == 24 and self.dino.num_prefix_tokens == 5, "DINO architecture changed")
        self._path_verified = False

    @contextmanager
    def hooks(self, replacements=None):
        replacements = replacements or {}; calls = {"b22":0, "dino":0, "siglip":0}; captured = {}; handles = []
        def block_hook(_module, _args, value):
            index = calls["b22"]; calls["b22"] += 1
            if index == 1: captured["b22"] = value[:,5:]
        handles.append(self.dino.blocks[22].register_forward_hook(block_hook))
        def branch_hook(name):
            def hook(_module, _args, value):
                index = calls[name]; calls[name] += 1
                if index != 1: return None
                changed = value
                if name in replacements:
                    changed = torch.as_tensor(replacements[name], device=value.device, dtype=value.dtype).reshape_as(value)
                captured[name] = changed
                return changed if name in replacements else None
            return hook
        handles.append(self.dino.register_forward_hook(branch_hook("dino")))
        handles.append(self.siglip.register_forward_hook(branch_hook("siglip")))
        try:
            yield captured
            require(all(v == 2 for v in calls.values()), "Native forward did not process front then wrist exactly once")
        finally:
            for handle in handles: handle.remove()

    @torch.no_grad()
    def forward(self, observation, language, replacements=None):
        with self.hooks(replacements) as captured:
            action = np.asarray(self.wrapper.predict_action(copy.deepcopy(observation), language), dtype=np.float32).reshape(8,7)
        require(np.isfinite(action).all(), "Non-finite native action")
        output = {k:v.detach().float().cpu().numpy().copy() for k,v in captured.items()}
        require(output["b22"].shape == (1,256,1024) and output["dino"].shape == (1,256,1024), "Visual feature shape changed")
        if not replacements or "dino" not in replacements:
            require(np.array_equal(output["b22"], output["dino"]), "Policy no longer consumes raw block-22 spatial output")
        return action, output

    def spatial(self, pixel_values):
        found = []
        def hook(_module, _args, value): found.append(value[:,5:])
        handle = self.dino.blocks[22].register_forward_hook(hook)
        try: self.dino(pixel_values[:,6:9])
        finally: handle.remove()
        require(len(found) == 1 and tuple(found[0].shape) == (1,256,1024), "DINO wrist feature capture failed")
        return found[0]

    @torch.no_grad()
    def spatial_observation(self, observation):
        inputs = self.wrapper.prepare_hf_inputs(copy.deepcopy(observation), "dummy")
        return self.spatial(inputs["pixel_values"]).detach().float().squeeze(0).cpu()

    def spatial_patch(self, observation, patch):
        values, wrist = patch_pixels(observation, patch, self.wrapper)
        return self.spatial(values), wrist

    def verify_patch_path(self, record):
        """One GPU integration check per cache worker, using actual PNG decoding."""
        if self._path_verified: return
        obs,language=record["observation"],record["language"]
        job='_'.join((os.environ.get('SLURM_JOB_ID','local'),os.environ.get('SLURM_ARRAY_TASK_ID','single'),
                      self.wrapper.gen_cfg.task_suite_name))
        path=ROOT/f"reports/checks/{job}/initial.png"
        patch=initial_patch(self.wrapper.device).requires_grad_(True)
        save_png(patch,path)
        decoded=load_png(path,self.wrapper.device)
        deployed=paste_numpy(obs,decoded.mul(255).round().byte().permute(1,2,0).cpu().numpy())
        values,_=patch_pixels(obs,patch,self.wrapper)
        native=self.wrapper.prepare_hf_inputs(copy.deepcopy(deployed),language)["pixel_values"]
        require(torch.equal(values.detach(),native),"Train preprocessing differs from actual PNG deployment")
        spatial=self.spatial(values)
        scalar=spatial.float().square().mean()
        gradient=torch.autograd.grad(scalar,patch)[0]
        require(torch.isfinite(gradient).all() and float(gradient.norm())>0,"PNG/DINO gradient integration failed")
        native_action,features=self.forward(deployed,language)
        require(np.array_equal(spatial.detach().float().cpu().numpy(),features["b22"]),
                "Training DINO features differ from native PNG forward")
        replaced,_=self.forward(obs,language,{"dino":features["dino"],"siglip":features["siglip"]})
        require(np.array_equal(native_action,replaced),"Dual branch replacement differs from native PNG action")
        modules=(self.wrapper.vla,self.wrapper.action_head,self.wrapper.proprio_projector,self.wrapper.noisy_action_projector)
        require(all(not p.requires_grad and p.grad is None for module in modules if module is not None
                    for p in module.parameters()),"Victim weights/gradients changed during integration check")
        atomic_json({"status":"PASS","signature":signature(),"state_id":record["state_id"],"png_pixels_match":True,
                     "dino_features_match":True,"dual_branch_action_match":True,"model_frozen":True,
                     "patch_gradient_l2":float(gradient.norm()),**resources()},ROOT/f"reports/GATES/{job}.json")
        self._path_verified=True

def signed_projection(patched, negative, positive):
    patched = patched.float().reshape(256,-1)
    negative = torch.as_tensor(negative, device=patched.device, dtype=torch.float32).reshape_as(patched)
    positive = torch.as_tensor(positive, device=patched.device, dtype=torch.float32).reshape_as(patched)
    direction = positive-negative
    energy = direction.square().mean(-1); total = energy.sum()
    if float(total.detach()) <= 1e-12: return None
    weights = energy/total
    denominator = (weights*energy).sum()
    numerator = (weights*((patched-negative)*direction).mean(-1)).sum()
    q = numerator/denominator.clamp_min(1e-12)
    loss = (q-1).square()
    residual = patched-(negative+q*direction)
    orthogonal = (weights*residual.square().mean(-1)).sum()/denominator.clamp_min(1e-12)
    return loss, q, orthogonal, float(total.detach())

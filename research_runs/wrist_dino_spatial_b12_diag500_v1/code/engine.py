"""Copy/extension of the proven Exploring-OFT native RGB engine."""
from __future__ import annotations
import random
from contextlib import contextmanager, ExitStack
import numpy as np
import torch
import torch.nn.functional as F
from common import require
from source_transform import RandomPatchTransform
from preprocessing import aligned_pixels
from objectives import action_loss, action_token_weights, geometry_reference, split_qkv, token_footprint

class OFTEngine:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        wrapper.freeze_model()
        self.dino = wrapper.vla.vision_backbone.featurizer
        self.siglip = wrapper.vla.vision_backbone.fused_featurizer
        self.transform = RandomPatchTransform(wrapper.device, resize_patch=False)
        require((self.transform.angle, self.transform.shx, self.transform.shy) == (30, .2, .2),
                "Exploring original rotation/shear changed")
        self.attentions = {layer: self.dino.blocks[layer].attn for layer in (8, 16)}
        require(getattr(self.dino, "num_prefix_tokens", None) == 5, "Actual DINO must have CLS+four registers")
        require(all(not p.requires_grad for module in (wrapper.vla, wrapper.action_head, wrapper.proprio_projector)
                    if module is not None for p in module.parameters()), "Model parameters must be frozen")

    def base(self, record):
        obs = record["observation"]
        inputs = dict(self.wrapper.prepare_hf_inputs(obs, record["task_name"]))
        stats = self.wrapper.vla.norm_stats[self.wrapper.gen_cfg.unnorm_key]["proprio"]
        proprio = self.wrapper._official_openvla_utils.normalize_proprio(np.asarray(obs["state"]), stats)
        return inputs, proprio

    def pixels(self, record, patch, seed, diverse=False, return_rgb=False):
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        obs = record["observation"]
        front = torch.as_tensor(np.asarray(obs["full_image"]), device=patch.device).permute(2, 0, 1).float() / 255
        wrist = torch.as_tensor(np.asarray(obs["wrist_image"]), device=patch.device).permute(2, 0, 1).float() / 255
        if diverse:
            x, y = random.randint(0, 174), random.randint(0, 174)
        else:
            x, y = 26, 137
        canvas = torch.ones_like(wrist) * -100
        canvas[:, y:y + 50, x:x + 50] = patch
        if diverse:
            matrix = self.transform.combined_transform_matrix().to(patch.device)
            canvas = self.transform.apply_affine_transform(canvas, matrix).squeeze(0)
        support = (canvas.detach() >= -20).any(dim=0)
        rgb = torch.where(canvas < -20, wrist, canvas).clamp(0, 1)
        pixels = aligned_pixels(front, rgb, self.wrapper.processor.image_processor,
                                self.wrapper._official_openvla_utils, self.wrapper.gen_cfg, self.wrapper.dtype)
        if return_rgb:
            quantized = rgb.detach().mul(255).round().div(255)
            return pixels, token_footprint(support), (quantized + (rgb - rgb.detach())).unsqueeze(0)
        return pixels, token_footprint(support)

    @contextmanager
    def hooks(self, replacement=None, leaf=False):
        captures, counts, handles = {}, {"dino": 0, "siglip": 0}, []
        for name, module in (("dino", self.dino), ("siglip", self.siglip)):
            def hook(_module, _args, output, *, name=name):
                index = counts[name]
                counts[name] += 1
                if index == 1:
                    require(output.shape[:2] == (1, 256), "Actual consumed dense visual tokens changed")
                    value = output
                    if name == "dino" and replacement is not None:
                        require(replacement.shape == output.shape, "DINO replacement shape changed")
                        value = replacement.to(device=output.device, dtype=output.dtype)
                    elif name == "dino" and leaf:
                        value = output.detach().requires_grad_(True)
                    captures[name] = value
                    return value
            handles.append(module.register_forward_hook(hook))
        try:
            yield captures
            require(counts == {"dino": 2, "siglip": 2}, "Expected Front then Wrist encoder calls")
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def qkv_hooks(self, enabled):
        captures, counts, handles = {}, {layer: 0 for layer in self.attentions}, []
        if enabled:
            for layer, attn in self.attentions.items():
                def hook(_module, _args, output, *, layer=layer):
                    index = counts[layer]
                    counts[layer] += 1
                    if index == 1:
                        captures[layer] = output
                handles.append(attn.qkv.register_forward_hook(hook))
        try:
            yield captures
            if enabled:
                require(all(n == 2 for n in counts.values()), "Native QKV did not encode Front then Wrist")
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, inputs, proprio, pixels=None, replacement=None, leaf=False, attention=False):
        values = {**inputs, "pixel_values": inputs["pixel_values"] if pixels is None else pixels}
        with self.hooks(replacement, leaf) as captures, self.qkv_hooks(attention) as qkv:
            _unused, hidden = self.wrapper.vla.predict_action(
                **values, unnorm_key=self.wrapper.gen_cfg.unnorm_key, do_sample=False,
                proprio=proprio, proprio_projector=self.wrapper.proprio_projector,
                noisy_action_projector=self.wrapper.noisy_action_projector,
                action_head=self.wrapper.action_head, use_film=False)
        head = getattr(self.wrapper.action_head, "module", self.wrapper.action_head)
        action = head.predict_action(hidden).reshape(1, 8, 7).float()
        require(torch.isfinite(action).all(), "Nonfinite native continuous action")
        observer = getattr(self, "_diagnostic_observer", None)
        if observer is not None:
            observer["action_hidden"] = hidden
        return action, captures, qkv

    def direct_dino(self, pixels):
        value = self.dino(pixels[:, 6:9])
        require(value.shape[:2] == (1, 256) and torch.isfinite(value).all(), "Invalid Wrist dense DINO")
        return value

    def geometric_pixels(self, record, scale=1., lateral=0.):
        obs = record["observation"]
        front = torch.as_tensor(np.asarray(obs["full_image"]), device=self.wrapper.device).permute(2, 0, 1).float() / 255
        wrist = torch.as_tensor(np.asarray(obs["wrist_image"]), device=self.wrapper.device).permute(2, 0, 1).float() / 255
        theta = wrist.new_tensor([[1 / scale, 0, -2 * lateral / 224], [0, 1 / scale, 0]])[None]
        grid = F.affine_grid(theta, wrist[None].shape, align_corners=False)
        transformed = F.grid_sample(wrist[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        return aligned_pixels(front, transformed, self.wrapper.processor.image_processor,
                              self.wrapper._official_openvla_utils, self.wrapper.gen_cfg, self.wrapper.dtype)

    def mechanism_reference(self, record, inputs, proprio, target, cfg, geometry=False, attention=False):
        action, captures, qkv = self.forward(inputs, proprio, leaf=True, attention=attention)
        loss = action_loss(action, target, "tma", cfg)
        gradient = torch.autograd.grad(loss, captures["dino"])[0].detach().float()
        clean = captures["dino"].detach()
        result = {"clean": clean, "weights": action_token_weights(gradient), "direction": None,
                  "geometry_scores": [0.] * 4, "geometry_selected": -1, "attention_scales": {}}
        if attention:
            for layer, value in qkv.items():
                q, k = split_qkv(value.detach(), self.attentions[layer])
                result["attention_scales"][layer] = float((q.float().square().mean().sqrt() * k.float().square().mean().sqrt()
                                                         * self.attentions[layer].scale).clamp_min(1e-6))
        del action, captures, qkv, loss
        if geometry:
            with torch.no_grad():
                references = [self.direct_dino(self.geometric_pixels(record, scale=s)).detach() for s in cfg["geometry_scales"]]
                references += [self.direct_dino(self.geometric_pixels(record, lateral=x)).detach() for x in cfg["geometry_lateral_pixels"]]
            direction, scores, selected = geometry_reference(clean, references, gradient)
            result.update(direction=direction, geometry_scores=scores, geometry_selected=selected)
        return result

    def validate(self, record, patch):
        inputs, proprio = self.base(record)
        with torch.no_grad():
            action, clean, _ = self.forward(inputs, proprio)
            official = np.asarray(self.wrapper.predict_action(record["observation"], record["task_name"])).reshape(8, 7)
            ours = self.wrapper.vla._unnormalize_actions(action.cpu().numpy()[0], self.wrapper.gen_cfg.unnorm_key)
            error = float(np.abs(official - ours).max())
            require(error <= .02, "Native continuous head interface mismatch")
        pixels, footprint = self.pixels(record, patch, 20261011, diverse=True)
        require(torch.equal(pixels[:, :6], inputs["pixel_values"][:, :6]), "Front changed during patch training")
        normal, captures, _ = self.forward(inputs, proprio, pixels)
        with torch.no_grad():
            direct = self.direct_dino(pixels)
            require(torch.allclose(direct, captures["dino"], rtol=1e-5, atol=1e-5), "DINO objective uses a different readout")
            _restored, restored, _ = self.forward(inputs, proprio, pixels, replacement=clean["dino"])
            require(torch.equal(restored["siglip"], captures["siglip"]), "DINO restoration altered SigLIP")
            same, _, _ = self.forward(inputs, proprio, replacement=clean["dino"])
            require(torch.allclose(same, action, rtol=1e-4, atol=1e-4), "Clean DINO identity restoration failed")
        gradient = torch.autograd.grad(normal[..., :3].square().mean(), patch)[0]
        require(torch.isfinite(gradient).all() and gradient.norm() > 0, "RGB backward is nonfinite or disconnected")
        return {"status": "GPU_INTERFACE_PASS", "native_action_max_error": error, "rgb_gradient_l2": float(gradient.norm()),
                "dino_prefix_tokens": 5, "footprint_tokens": int((footprint > 0).sum()),
                "forward": "native uint8/PIL", "backward": "explicit STE", "scientific_result": "NOT_TESTED"}

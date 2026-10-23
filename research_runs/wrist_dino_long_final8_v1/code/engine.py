"""Copy/adaptation of Long-v2 OFTEngine; no task/background feature regions."""
from __future__ import annotations

import random
from contextlib import contextmanager
import numpy as np
import torch
import torchvision
from common import require
from source_transform import RandomPatchTransform
from preprocessing import aligned_pixels


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
        return inputs, proprio

    def pixels(self, record, patch, seed, geometry=False, diverse=False, fixed=False):
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        obs = record["observation"]
        front = torch.as_tensor(np.asarray(obs["full_image"]), device=patch.device).permute(2, 0, 1).float() / 255
        image = torchvision.transforms.ToTensor()(self.transform._to_rgb_pil(obs["wrist_image"])).to(patch.device)
        if diverse:
            # Preserve legacy diverse draw order (site before brightness/contrast).
            random.randint(0, 174)
            random.randint(0, 174)
            brightness, contrast = random.uniform(.8, 1.2), random.uniform(.8, 1.2)
            patch = (((patch - .5) * contrast + .5) * brightness).clamp(0, 1)
        x, y = (26, 137) if fixed else (random.randint(0, 174), random.randint(0, 174))
        canvas = torch.ones_like(image) * -100
        canvas[:, y:y + 50, x:x + 50] = patch
        if geometry:
            matrix = self.transform.combined_transform_matrix().to(patch.device)
            canvas = self.transform.apply_affine_transform(canvas, matrix).squeeze(0)
        wrist = torch.where(canvas < -20, image, canvas).clamp(0, 1)
        return aligned_pixels(front, wrist, self.wrapper.processor.image_processor,
                              self.wrapper._official_openvla_utils, self.wrapper.gen_cfg, self.wrapper.dtype)

    @contextmanager
    def hooks(self, replacement=None):
        captures, counts, handles = {}, {"dino": 0, "siglip": 0}, []
        for name, module in (("dino", self.dino), ("siglip", self.siglip)):
            def hook(_module, _args, output, *, name=name):
                index = counts[name]
                counts[name] += 1
                if index == 1:
                    require(isinstance(output, torch.Tensor) and output.shape[:2] == (1, 256), "DINO dense grid changed")
                    captures[name] = output
                    if name == "dino" and replacement is not None:
                        require(replacement.shape == output.shape, "Restore shape changed")
                        return replacement.to(output.dtype)
            handles.append(module.register_forward_hook(hook))
        try:
            yield captures
            require(counts == {"dino": 2, "siglip": 2}, "Expected Front then Wrist calls")
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, inputs, proprio, pixels=None, replacement=None):
        values = {**inputs, "pixel_values": inputs["pixel_values"] if pixels is None else pixels}
        with self.hooks(replacement) as captures:
            _discard, hidden = self.wrapper.vla.predict_action(
                **values, unnorm_key=self.wrapper.gen_cfg.unnorm_key, do_sample=False,
                proprio=proprio, proprio_projector=self.wrapper.proprio_projector,
                noisy_action_projector=self.wrapper.noisy_action_projector,
                action_head=self.wrapper.action_head, use_film=False)
        head = getattr(self.wrapper.action_head, "module", self.wrapper.action_head)
        action = head.predict_action(hidden).reshape(1, 8, 7).float()
        require(torch.isfinite(action).all(), "Nonfinite continuous action")
        return action, captures

    def direct_dino(self, pixels):
        value = self.dino(pixels[:, 6:9])
        require(value.shape[:2] == (1, 256) and torch.isfinite(value).all(), "Invalid dense DINO target")
        return value

    def validate(self, record, patch):
        inputs, proprio = self.base(record)
        with torch.no_grad():
            action, _ = self.forward(inputs, proprio)
            official = np.asarray(self.wrapper.predict_action(record["observation"], record["task_name"])).reshape(8, 7)
            ours = self.wrapper.vla._unnormalize_actions(action.cpu().numpy()[0], self.wrapper.gen_cfg.unnorm_key)
            error = float(np.abs(official - ours).max())
            require(error <= .02, "Continuous head differs from native deployment")
        pixels = self.pixels(record, patch, 20261009, geometry=True)
        require(torch.equal(pixels[:, :6], inputs["pixel_values"][:, :6]), "Front changed during training")
        normal, captures = self.forward(inputs, proprio, pixels)
        with torch.no_grad():
            direct = self.direct_dino(pixels)
            require(torch.allclose(direct, captures["dino"], rtol=1e-5, atol=1e-5),
                    "Teacher feature is not actual consumed dense DINO")
        gradient = torch.autograd.grad(normal[..., 0].square().mean(), patch)[0]
        require(torch.isfinite(gradient).all(), "Nonfinite native RGB gradient")
        return {"status": "GPU_INTERFACE_PASS", "official_action_max_error": error,
                "native_forward": "official_uint8/PIL", "backward": "explicit_STE", "gradient_l2": float(gradient.norm()),
                "scientific_result": "NOT_TESTED"}

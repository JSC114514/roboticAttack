"""Native OFT action objective; DINO is only the virtual RGB proposal path."""
from __future__ import annotations
import numpy as np
import torch
from common import require
from engine_native import OFTEngine
from objectives import action_loss, target_action
from meta import split_rgb_gradient, cosine

class MetaEngine:
    def __init__(self, wrapper, cfg):
        self.native = OFTEngine(wrapper)
        self.wrapper, self.cfg = wrapper, cfg
        self._bases = {}

    def base(self, record):
        if record["uid"] not in self._bases:
            self._bases[record["uid"]] = self.native.base(record)
        return self._bases[record["uid"]]

    def clear_bases(self):
        self._bases.clear()

    def gradients(self, record, patch, seed, split):
        inputs, proprio = self.base(record)
        target = target_action(record["expert_actions"].to(patch.device), "uada", self.cfg)
        pixels, _ = self.native.pixels(record, patch, seed, diverse=True)
        action, captures, _ = self.native.forward(inputs, proprio, pixels)
        loss = action_loss(action, target, "uada", self.cfg)
        require(torch.isfinite(loss), "Nonfinite native UADA loss")
        loss_value = float(loss.detach())
        if split:
            full, gd = split_rgb_gradient(loss, captures["dino"], captures["siglip"], patch)
        else:
            full, gd = torch.autograd.grad(loss, patch)[0].detach(), None
        return {"loss": loss_value, "full": full, "dino": gd}

    @torch.no_grad()
    def inspect(self, record, patch, seed, fixed=False):
        inputs, proprio = self.base(record)
        pixels = None
        if patch is not None:
            pixels, _ = self.native.pixels(record, patch, seed, diverse=not fixed)
        action, caps, _ = self.native.forward(inputs, proprio, pixels)
        target = target_action(record["expert_actions"].to(action.device), "uada", self.cfg)
        return {"loss": float(action_loss(action, target, "uada", self.cfg)),
                "action": action.cpu(), "dino": caps["dino"].cpu(), "siglip": caps["siglip"].cpu()}

    def validate(self, record, patch):
        inputs, proprio = self.base(record)
        with torch.no_grad():
            action, _, _ = self.native.forward(inputs, proprio)
            reference = np.asarray(self.wrapper.predict_action(record["observation"], record["task_name"])).reshape(8, 7)
            ours = self.wrapper.vla._unnormalize_actions(action.cpu().numpy()[0], self.wrapper.gen_cfg.unnorm_key)
            error = float(np.abs(reference-ours).max())
            require(error <= .02, "Native action-head interface mismatch")
        pixels, _ = self.native.pixels(record, patch, 20261011, diverse=False)
        require(torch.equal(pixels[:, :6], inputs["pixel_values"][:, :6]), "Training changed Front input")
        actions, caps, _ = self.native.forward(inputs, proprio, pixels)
        target = target_action(record["expert_actions"].to(patch.device), "uada", self.cfg)
        loss = action_loss(actions, target, "uada", self.cfg)
        direct = torch.autograd.grad(loss, patch, retain_graph=True)[0].detach()
        combined, gd = split_rgb_gradient(loss, caps["dino"], caps["siglip"], patch)
        relative = float((combined-direct).norm()/direct.norm().clamp_min(1e-20))
        alignment = cosine(combined, direct)
        require(torch.isfinite(combined).all() and float(direct.norm()) > 0 and float(gd.norm()) > 0,
                "Native DINO/RGB backward disconnected")
        # BF16 accumulation order differs for decomposed encoder VJPs. These
        # are numerical interface tolerances, never an efficacy acceptance gate.
        require(relative <= .02 and alignment >= .999, "Native VJP decomposition differs from direct RGB autograd")
        return {"status": "GPU_INTERFACE_PASS", "native_action_max_error": error,
                "decomposition_relative_error": relative, "decomposition_cosine": alignment,
                "dino_gradient_l2": float(gd.norm()), "rgb_gradient_l2": float(direct.norm()),
                "model_parameters_updated": False, "inference_branch_replaced": False,
                "forward": "native uint8/PIL", "backward": "explicit STE", "first_order": True}

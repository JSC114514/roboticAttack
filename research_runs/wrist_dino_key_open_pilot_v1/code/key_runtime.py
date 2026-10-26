"""Post-k_norm spatial-Key intervention. Q/V, prefix tokens and Front are never written."""
from __future__ import annotations
from contextlib import contextmanager
import torch
from common import require
from engine_native import OFTEngine
from key_math import open_loss, action_metrics, project_rms

@contextmanager
def spatial_key(module, expected_calls, replacement=None, prefix=5):
    captured = {}; calls = 0
    def hook(_module, _args, output):
        nonlocal calls
        calls += 1
        if calls != expected_calls: return None
        require(output.ndim == 4 and output.shape[0] == 1 and output.shape[2] == prefix+256,
                "Unexpected native post-k_norm Key layout")
        value = output
        if replacement is not None:
            require(replacement.shape == output[:, :, prefix:].shape, "Key replacement shape mismatch")
            value = torch.cat([output[:, :, :prefix], replacement.to(output)], dim=2)
            require(torch.equal(value[:, :, :prefix], output[:, :, :prefix]), "CLS/register Key changed")
        captured["key"] = value[:, :, prefix:]
        return value if replacement is not None else None
    handle = module.register_forward_hook(hook)
    try:
        yield captured
        require(calls == expected_calls and "key" in captured, "Wrong Front/Wrist Key invocation count")
    finally:
        handle.remove()

class KeyEngine(OFTEngine):
    def __init__(self, wrapper):
        super().__init__(wrapper)
        self.key_norm = self.dino.blocks[-2].attn.k_norm
        self.cached_uid = None

    def inputs(self, record):
        if self.cached_uid != record["uid"]:
            self.cached_inputs, self.cached_proprio = self.base(record)
            self.cached_uid = record["uid"]
        return self.cached_inputs, self.cached_proprio

    def native(self, record, patch=None, seed=42, fixed=False, replacement=None):
        inputs, proprio = self.inputs(record)
        pixels = inputs["pixel_values"] if patch is None else self.pixels(record, patch, seed, diverse=not fixed)[0]
        with spatial_key(self.key_norm, 2, replacement) as value:
            action, _, _ = self.forward(inputs, proprio, pixels)
        return action, value["key"]

    def rgb_key(self, record, patch, seed, fixed):
        pixels = self.pixels(record, patch, seed, diverse=not fixed)[0]
        with spatial_key(self.key_norm, 1) as value:
            self.direct_dino(pixels)
        return value["key"]

    def make_target(self, record, clean_action, cfg):
        mask = record["close_slots"].to(self.wrapper.device)
        with torch.no_grad():
            a0, k = self.native(record)
        require(torch.equal(a0[:, :, 6] < .5, clean_action.to(a0)[:, :, 6] < .5), "Clean-close slots changed")
        k0 = k.detach().float(); scale = k0.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        z = torch.zeros_like(k0, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=cfg["key_lr"])
        initial_loss = float(open_loss(a0, mask)); best_loss = initial_loss; best_rms = 0.
        best_key, best_action = k0.clone(), a0.detach().clone(); rows = []
        for step in range(cfg["key_steps"]):
            optimizer.zero_grad(set_to_none=True)
            action, _ = self.native(record, replacement=k0+scale*z)
            opening = open_loss(action, mask)
            require(opening.requires_grad, "Injected Key is disconnected from the consumed native action")
            loss = opening + cfg["key_regularizer"]*z.square().mean()
            loss.backward()
            require(z.grad is not None and torch.isfinite(z.grad).all(), "Key target gradient disconnected/nonfinite")
            gradient = float(z.grad.norm())
            action_gradient = float((z.grad-2*cfg["key_regularizer"]*z.detach()/z.numel()).norm())
            optimizer.step(); project_rms(z, cfg["key_rms_limit"])
            with torch.no_grad():
                actual, consumed = self.native(record, replacement=k0+scale*z)
                normalized = (consumed.float()-k0)/scale
                maximum_rms = float(normalized.square().mean((-2, -1)).sqrt().max())
                actual_loss = float(open_loss(actual, mask)); norm = float(normalized.square().mean())
                feasible = maximum_rms <= cfg["key_rms_limit"] + 1e-6
                if feasible and (actual_loss < best_loss or actual_loss == best_loss and norm < best_rms):
                    best_key, best_action = consumed.detach().float().clone(), actual.detach().clone()
                    best_loss, best_rms = actual_loss, norm
                rows.append({"step": step+1, "objective": float(loss.detach()), "open_loss": actual_loss,
                             "gradient_l2": gradient, "opening_gradient_l2": action_gradient, "maximum_actual_head_rms": maximum_rms,
                             "within_rms_limit": feasible, **action_metrics(actual, mask)})
            if best_loss == 0.: break
        nonzero = float(((best_key-k0)/scale).square().mean(-1).sum()) > 1e-12
        outcome = "MARGIN_REACHED" if best_loss == 0 else "PARTIAL" if best_loss < initial_loss else "NO_IMPROVEMENT"
        return {"uid": record["uid"], "mask": mask.cpu(), "clean_key": k0.cpu(), "scale": scale.cpu(),
                "target_key": best_key.cpu(), "target_action": best_action.cpu(), "nonzero": nonzero,
                "outcome": outcome, "initial_open_loss": initial_loss, "best_open_loss": best_loss,
                "steps": rows, "metrics": action_metrics(best_action, mask)}

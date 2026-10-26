"""Read-only native observers; never substitute a branch or alter policy weights."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import torch
from run_core import native, require
native()
from engine import OFTEngine
from model_runtime import load_wrapper


class ReadoutEngine(OFTEngine):
    @contextmanager
    def hooks(self, replacement=None, leaf=False):
        require(replacement is None and not leaf, "Feature substitution is outside this experiment")
        captures, values, handles = {}, {"dino": [], "siglip": []}, []
        for name, module in (("dino", self.dino), ("siglip", self.siglip)):
            def hook(_module, _args, output, name=name):
                require(output.shape[:2] == (1, 256), "Native dense token shape changed")
                values[name].append(output)
                # A read-only forward hook returns None.
            handles.append(module.register_forward_hook(hook))
        try:
            yield captures
            require(all(len(x) == 2 for x in values.values()), "Expected front then wrist encoder calls")
            captures.update(dino=values["dino"][1], siglip=values["siglip"][1],
                            dino_all=torch.stack(values["dino"], dim=1))
        finally:
            for handle in handles:
                handle.remove()

    def context(self, record):
        inputs, proprio = self.base(record)
        with torch.no_grad():
            embedding = self.wrapper.vla.get_input_embeddings()(inputs["input_ids"])
            mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
            language = (embedding.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return {"inputs": inputs, "proprio": proprio, "language": language.detach(),
                "state": torch.as_tensor(np.asarray(proprio), device=self.wrapper.device).reshape(1, 8).float()}

    def run(self, record, context, patch=None, seed=None):
        pixels = None if patch is None else self.pixels(record, patch, seed, diverse=True)[0]
        action, captures, _ = self.forward(context["inputs"], context["proprio"], pixels)
        return action[..., 0], captures["dino_all"]


def load_engine():
    engine = ReadoutEngine(load_wrapper(native().runtime_config(), "libero_spatial"))
    require(not engine.dino.training, "Frozen native DINO must use eval mode")
    return engine


def wrong_target(record, device):
    expert = record["expert_actions"].to(device)[..., 0]
    return torch.where(expert >= 0, -torch.ones_like(expert), torch.ones_like(expert)).detach()

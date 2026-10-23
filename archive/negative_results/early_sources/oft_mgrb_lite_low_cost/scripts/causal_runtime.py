from __future__ import annotations

import copy
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import require


@dataclass
class Runtime:
    adapter: Any
    registry: dict[str, Any]

    @property
    def roots(self):
        return self.adapter.roots

    def release(self):
        self.adapter.release()


def load_runtime(cfg: dict[str, Any], suite: str) -> Runtime:
    attribution = Path(cfg["paths"]["attribution_root"])
    for path in (attribution, Path(cfg["paths"]["openvla_oft_root"]), Path(cfg["paths"]["libero_root"])):
        value = str(path)
        sys.path[:] = [entry for entry in sys.path if entry != value]
        sys.path.insert(0, value)
    from e_rescue_clean_internal_v6.src.models.runtime import load_adapter
    import yaml

    base = attribution / "e_rescue_clean_internal_v6" / "configs" / "base.yaml"
    config = yaml.safe_load(base.read_text(encoding="utf-8"))
    config["environment"]["local_openvla_root"] = cfg["paths"]["openvla_oft_root"]
    config["data"]["libero_root"] = cfg["paths"]["libero_root"]
    config["data"]["benchmark"] = suite
    config["model"]["checkpoint"] = cfg["checkpoints"][suite]
    config["model"]["unnorm_key"] = cfg["unnorm_keys"][suite]
    adapter, registry = load_adapter(config)
    return Runtime(adapter=adapter, registry=registry)


def prepare(runtime: Runtime, observation: dict[str, Any], language: str) -> dict[str, Any]:
    prepared = runtime.adapter.prepare_observation(copy.deepcopy(observation), language)
    import torch
    for key, value in prepared.items():
        if torch.is_tensor(value):
            require(value.is_cuda, f"Prepared tensor {key} left CUDA")
    return prepared


def visual_spec(site: str, view: str):
    from src.hooks.manager import HookSpec

    require(view in {"front", "wrist"}, "Unknown view")
    call = 0 if view == "front" else 1
    if site == "dino":
        return HookSpec(path="vla.vision_backbone.featurizer", hook="output", call_index=call)
    if site == "siglip":
        return HookSpec(path="vla.vision_backbone.fused_featurizer", hook="output", call_index=call)
    raise ValueError(site)


def run(runtime: Runtime, prepared: dict[str, Any], observe: list[Any], *, replace_specs=None, replace_values=None):
    import torch
    from src.hooks.manager import TensorHooks
    from src.hooks.registry import resolve_module

    replace_specs, replace_values = list(replace_specs or []), list(replace_values or [])
    require(len(replace_specs) == len(replace_values), "Replacement count mismatch")
    specs = [*replace_specs, *observe]
    modules = {id(resolve_module(runtime.roots, spec.path)): resolve_module(runtime.roots, spec.path) for spec in specs}
    before = {key: (len(module._forward_hooks), len(module._forward_pre_hooks)) for key, module in modules.items()}
    with torch.inference_mode(), ExitStack() as stack:
        if replace_specs:
            stack.enter_context(TensorHooks(runtime.roots, replace_specs, mode="replace", values=replace_values))
        captured = stack.enter_context(TensorHooks(runtime.roots, observe, mode="capture"))
        action = runtime.adapter.predict_prepared(prepared)
    after = {key: (len(module._forward_hooks), len(module._forward_pre_hooks)) for key, module in modules.items()}
    require(before == after, "Hook cleanup failed")
    values = [value.detach().clone() for value in captured.captured]
    require(all(value.is_cuda and bool(torch.isfinite(value).all()) for value in values), "Invalid CUDA activation")
    return action, values

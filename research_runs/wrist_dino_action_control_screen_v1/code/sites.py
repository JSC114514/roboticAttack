"""Site registry and exact native-tensor replacement hooks."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import re
import torch
from common import require


@dataclass(frozen=True)
class Site:
    name: str
    kind: str
    layer: int | None
    headwise: bool


def site_registry(names):
    sites = {}
    patterns = {
        r"([qkv])_spatial_b(\d\d)": lambda m: Site(m.group(0), m.group(1), int(m.group(2)), True),
        r"attn_spatial_b(\d\d)": lambda m: Site(m.group(0), "attn", int(m.group(1)), False),
        r"mlp_spatial_b(\d\d)": lambda m: Site(m.group(0), "mlp", int(m.group(1)), False),
        r"block_(spatial|cls|register)_b(\d\d)": lambda m: Site(m.group(0), "block_" + m.group(1), int(m.group(2)), False),
    }
    fixed = {
        "dino_wrist_output": Site("dino_wrist_output", "dino", None, False),
        "siglip_wrist_output": Site("siglip_wrist_output", "siglip", None, False),
        "projector_wrist_output": Site("projector_wrist_output", "projector", None, False),
    }
    for name in names:
        value = fixed.get(name)
        if value is None:
            for pattern, builder in patterns.items():
                match = re.fullmatch(pattern, name)
                if match:
                    value = builder(match)
                    break
        require(value is not None, f"Unknown candidate site: {name}")
        sites[name] = value
    require(len(sites) == len(names), "Duplicate candidate site")
    return sites


def selected(value: torch.Tensor, site: Site, prefix: int = 5, dense: int = 256):
    require(isinstance(value, torch.Tensor) and value.shape[0] == 1, f"Non-tensor/batch at {site.name}")
    if site.kind in ("q", "k"):
        require(value.ndim == 4 and value.shape[2] == prefix + dense, f"Unexpected Q/K layout at {site.name}")
        return value[:, :, prefix:, :]
    if site.kind == "v":
        require(value.ndim == 3 and value.shape[1] == prefix + dense and value.shape[2] % 3 == 0,
                f"Unexpected QKV layout at {site.name}")
        # The caller supplies num_heads during extraction through SiteHook.
        raise RuntimeError("V extraction requires attention metadata")
    if site.kind in ("attn", "mlp", "block_spatial"):
        require(value.ndim == 3 and value.shape[1] == prefix + dense, f"Unexpected spatial layout at {site.name}")
        return value[:, prefix:, :]
    if site.kind == "block_cls":
        require(value.ndim == 3 and value.shape[1] == prefix + dense, f"Unexpected CLS layout at {site.name}")
        return value[:, :1, :]
    if site.kind == "block_register":
        require(value.ndim == 3 and value.shape[1] == prefix + dense, f"Unexpected register layout at {site.name}")
        return value[:, 1:prefix, :]
    if site.kind in ("dino", "siglip"):
        require(value.ndim == 3 and value.shape[1] == dense, f"Unexpected consumed vision layout at {site.name}")
        return value
    if site.kind == "projector":
        require(value.ndim == 3 and value.shape[1] == 2 * dense, f"Unexpected two-view projector layout at {site.name}")
        return value[:, dense:, :]
    raise RuntimeError(f"No extraction for {site.name}")


def replace_selected(value: torch.Tensor, replacement: torch.Tensor, site: Site,
                     prefix: int = 5, dense: int = 256, heads: int | None = None):
    replacement = replacement.to(device=value.device, dtype=value.dtype)
    if site.kind in ("q", "k"):
        expected = value[:, :, prefix:, :]
        require(replacement.shape == expected.shape, f"Replacement shape changed at {site.name}")
        return torch.cat((value[:, :, :prefix, :], replacement), dim=2)
    if site.kind == "v":
        require(heads is not None and value.shape[2] % (3 * heads) == 0, "Invalid V attention metadata")
        head_dim = value.shape[2] // (3 * heads)
        shaped = value.reshape(value.shape[0], value.shape[1], 3, heads, head_dim)
        q, k, v = shaped.unbind(2)
        expected = v[:, prefix:, :, :].permute(0, 2, 1, 3)
        require(replacement.shape == expected.shape, f"Replacement shape changed at {site.name}")
        new_v = torch.cat((v[:, :prefix, :, :], replacement.permute(0, 2, 1, 3)), dim=1)
        return torch.stack((q, k, new_v), dim=2).reshape_as(value)
    if site.kind in ("attn", "mlp", "block_spatial"):
        expected = value[:, prefix:, :]
        require(replacement.shape == expected.shape, f"Replacement shape changed at {site.name}")
        return torch.cat((value[:, :prefix, :], replacement), dim=1)
    if site.kind == "block_cls":
        require(replacement.shape == value[:, :1, :].shape, f"Replacement shape changed at {site.name}")
        return torch.cat((replacement, value[:, 1:, :]), dim=1)
    if site.kind == "block_register":
        require(replacement.shape == value[:, 1:prefix, :].shape, f"Replacement shape changed at {site.name}")
        return torch.cat((value[:, :1, :], replacement, value[:, prefix:, :]), dim=1)
    if site.kind in ("dino", "siglip"):
        require(replacement.shape == value.shape, f"Replacement shape changed at {site.name}")
        return replacement
    if site.kind == "projector":
        require(replacement.shape == value[:, dense:, :].shape, f"Replacement shape changed at {site.name}")
        return torch.cat((value[:, :dense, :], replacement), dim=1)
    raise RuntimeError(f"No replacement for {site.name}")


class SiteHook:
    def __init__(self, engine, site: Site, replacement=None):
        self.engine = engine
        self.site = site
        self.replacement = replacement
        self.calls = 0
        self.captured = None
        self.module, self.call_index, self.expected_calls, self.heads = self._module()

    def _module(self):
        site, dino = self.site, self.engine.dino
        if site.kind in ("q", "k"):
            attention = dino.blocks[site.layer].attn
            return (attention.q_norm if site.kind == "q" else attention.k_norm), 1, 2, attention.num_heads
        if site.kind == "v":
            attention = dino.blocks[site.layer].attn
            return attention.qkv, 1, 2, attention.num_heads
        if site.kind == "attn":
            return dino.blocks[site.layer].attn.proj, 1, 2, None
        if site.kind == "mlp":
            return dino.blocks[site.layer].mlp, 1, 2, None
        if site.kind.startswith("block_"):
            return dino.blocks[site.layer], 1, 2, None
        if site.kind == "dino":
            return dino, 1, 2, None
        if site.kind == "siglip":
            return self.engine.siglip, 1, 2, None
        if site.kind == "projector":
            return self.engine.wrapper.vla.projector, 0, 1, None
        raise RuntimeError(f"No native module for {site.name}")

    def _extract(self, value):
        if self.site.kind == "v":
            head_dim = value.shape[2] // (3 * self.heads)
            shaped = value.reshape(value.shape[0], value.shape[1], 3, self.heads, head_dim)
            return shaped[:, 5:, 2, :, :].permute(0, 2, 1, 3)
        return selected(value, self.site)

    def _hook(self, _module, _arguments, output):
        index = self.calls
        self.calls += 1
        if index != self.call_index:
            return None
        clean = self._extract(output)
        if self.replacement is None:
            self.captured = clean
            return None
        changed = replace_selected(output, self.replacement, self.site, heads=self.heads)
        self.captured = self._extract(changed)
        return changed

    @contextmanager
    def installed(self):
        handle = self.module.register_forward_hook(self._hook)
        try:
            yield self
        finally:
            handle.remove()
        require(self.calls == self.expected_calls and self.captured is not None,
                f"Native invocation count changed at {self.site.name}: {self.calls}")


def nonselected_equal(before: torch.Tensor, after: torch.Tensor, site: Site, prefix=5, dense=256, heads=None):
    """CPU/unit-test helper: verify replacement changes only the declared slice."""
    if site.kind in ("q", "k", "attn", "mlp", "block_spatial"):
        axis = 2 if site.kind in ("q", "k") else 1
        return bool(torch.equal(before.narrow(axis, 0, prefix), after.narrow(axis, 0, prefix)))
    if site.kind == "v":
        head_dim = before.shape[2] // (3 * heads)
        a = before.reshape(before.shape[0], before.shape[1], 3, heads, head_dim)
        b = after.reshape_as(a)
        return bool(torch.equal(a[:, :, :2], b[:, :, :2]) and torch.equal(a[:, :prefix, 2], b[:, :prefix, 2]))
    if site.kind == "block_cls":
        return bool(torch.equal(before[:, 1:], after[:, 1:]))
    if site.kind == "block_register":
        return bool(torch.equal(before[:, :1], after[:, :1]) and torch.equal(before[:, prefix:], after[:, prefix:]))
    if site.kind == "projector":
        return bool(torch.equal(before[:, :dense], after[:, :dense]))
    return True

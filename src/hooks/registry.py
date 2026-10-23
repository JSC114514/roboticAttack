"""从真实模块树发现视觉、Transformer、SwiGLU 与动作头位置。"""

from __future__ import annotations

from typing import Any


def resolve_module(roots: dict[str, Any], path: str) -> Any:
    pieces = path.split(".")
    module = roots[pieces[0]]
    for piece in pieces[1:]:
        module = module[int(piece)] if piece.isdigit() and hasattr(module, "__getitem__") else getattr(module, piece)
    return module


def module_tree(roots: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_name, root in roots.items():
        for name, module in root.named_modules():
            rows.append({
                "path": root_name if not name else f"{root_name}.{name}",
                "class": f"{module.__class__.__module__}.{module.__class__.__name__}",
                "direct_parameters": int(sum(p.numel() for p in module.parameters(recurse=False))),
            })
    return rows


def _largest_layer_list(language_model: Any) -> tuple[str, Any]:
    import torch

    candidates = [(len(module), name, module) for name, module in language_model.named_modules() if isinstance(module, torch.nn.ModuleList) and len(module) >= 3]
    if not candidates:
        raise RuntimeError("无法从实际 language_model 中找到 Transformer ModuleList")
    _size, name, layers = max(candidates, key=lambda item: item[0])
    return name, layers


def discover_registry(adapter: Any) -> dict[str, Any]:
    vla = adapter.vla
    layer_list_name, layers = _largest_layer_list(vla.language_model)
    prefix = "vla.language_model" + (f".{layer_list_name}" if layer_list_name else "")
    layer_records = []
    for index, layer in enumerate(layers):
        layer_path = f"{prefix}.{index}"
        if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
            raise RuntimeError(f"Transformer 层缺少 self_attn/mlp: {layer_path}")
        mlp = layer.mlp
        for attribute in ("gate_proj", "up_proj", "down_proj"):
            if not hasattr(mlp, attribute):
                raise RuntimeError(f"SwiGLU 层缺少 {attribute}: {layer_path}.mlp")
        attention = layer.self_attn
        heads = int(getattr(attention, "num_heads", getattr(vla.config, "num_attention_heads", 0)))
        head_dim = int(getattr(attention, "head_dim", 0))
        if heads <= 0 or head_dim <= 0:
            raise RuntimeError(f"无法读取 Attention 头布局: {layer_path}")
        down_weight = mlp.down_proj.weight
        intermediate = int(mlp.gate_proj.out_features)
        if down_weight.shape[1] != intermediate:
            raise RuntimeError(
                f"W_down 布局审计失败: {layer_path}, weight={tuple(down_weight.shape)}, intermediate={intermediate}"
            )
        layer_records.append({
            "index": index,
            "layer": layer_path,
            "residual": {"path": layer_path, "hook": "output"},
            "attention_output": {"path": f"{layer_path}.self_attn", "hook": "output"},
            "attention_head_output": {"path": f"{layer_path}.self_attn.o_proj", "hook": "input", "heads": heads, "head_dim": head_dim},
            "swiglu_post_gate": {"path": f"{layer_path}.mlp.down_proj", "hook": "input", "intermediate": intermediate},
            "ffn_writeback": {"path": f"{layer_path}.mlp.down_proj", "hook": "output"},
            "down_weight_layout": "[hidden, intermediate]",
        })
    primary_blocks = vla.vision_backbone.featurizer.blocks
    fractions = adapter.config["hooks"]["vision_block_fractions"]
    indices = sorted({min(len(primary_blocks) - 1, max(0, round(float(f) * (len(primary_blocks) - 1)))) for f in fractions})
    vision = []
    for index in indices:
        vision.append({"encoder": "siglip_or_primary", "index": index, "path": f"vla.vision_backbone.featurizer.blocks.{index}", "hook": "output", "per_view_call_order": ["front", "wrist"]})
        if getattr(vla.vision_backbone, "fused_featurizer", None) is not None:
            vision.append({"encoder": "dinov2_or_fused", "index": index, "path": f"vla.vision_backbone.fused_featurizer.blocks.{index}", "hook": "output", "per_view_call_order": ["front", "wrist"]})
    action_path = "action_head.model" if hasattr(adapter.action_head, "model") else "action_head"
    return {
        "model_class": f"{vla.__class__.__module__}.{vla.__class__.__name__}",
        "layers": layer_records,
        "vision": vision,
        "projector": {"path": "vla.projector", "hook": "output"},
        "action_head_input": {"path": action_path, "hook": "input"},
        "action_head_output": {"path": action_path, "hook": "output"},
        "token_layout": {
            "visual_tokens_per_view": adapter.num_patches_per_view,
            "projector": {"front": [0, adapter.num_patches_per_view], "wrist": [adapter.num_patches_per_view, 2 * adapter.num_patches_per_view]},
            "transformer": {"bos": [0, 1], "front": [1, 1 + adapter.num_patches_per_view], "wrist": [1 + adapter.num_patches_per_view, 1 + 2 * adapter.num_patches_per_view]},
            "language_start": 2 + adapter.num_visual_tokens,
            "action_prefix_tokens": adapter.num_prefix_tokens,
        },
    }


def token_slice(registry: dict[str, Any], site: str, sequence_length: int, space: str = "transformer") -> slice:
    layout = registry["token_layout"]
    if site in ("front", "wrist", "bos"):
        start, stop = layout[space][site]
        return slice(int(start), int(stop))
    if site == "language":
        return slice(int(layout["language_start"]), sequence_length)
    if site == "action":
        return slice(int(layout["action_prefix_tokens"]), sequence_length)
    raise KeyError(site)


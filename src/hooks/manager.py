"""捕获、替换、加法与 Attention edge knockout 的短生命周期 Hook。"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

from src.hooks.registry import resolve_module


def primary_tensor(value: Any) -> tuple[Any, Callable[[Any], Any]]:
    import torch

    if torch.is_tensor(value):
        return value, lambda tensor: tensor
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            if torch.is_tensor(item):
                def rebuild(tensor: Any, container=value, position=index) -> Any:
                    items = list(container)
                    items[position] = tensor
                    return tuple(items) if isinstance(container, tuple) else items
                return item, rebuild
    raise TypeError(f"Hook 输出没有 Tensor: {type(value)!r}")


@dataclass
class HookSpec:
    path: str
    hook: str = "output"
    token_slice: slice | None = None
    feature_indices: Any | None = None
    head_index: int | None = None
    num_heads: int | None = None
    head_dim: int | None = None
    call_index: int | None = None


class TensorHooks(AbstractContextManager["TensorHooks"]):
    """同一 context 内完成一组节点捕获/干预，并在退出时强制卸载。"""

    def __init__(self, roots: dict[str, Any], specs: list[HookSpec], mode: str = "capture", values: list[Any] | None = None, scale: float = 1.0, retain_grad: bool = False) -> None:
        if mode not in {"capture", "replace", "add", "identity"}:
            raise ValueError(mode)
        self.roots, self.specs, self.mode = roots, specs, mode
        self.values = values or []
        self.scale, self.retain_grad = float(scale), bool(retain_grad)
        self.captured: list[Any | None] = [None] * len(specs)
        self.handles: list[Any] = []
        self.calls = [0] * len(specs)

    @staticmethod
    def _selection(tensor: Any, spec: HookSpec) -> tuple[Any, tuple[Any, Any]]:
        token = spec.token_slice if spec.token_slice is not None else slice(None)
        if spec.head_index is not None:
            shaped = tensor.reshape(*tensor.shape[:-1], int(spec.num_heads), int(spec.head_dim))
            return shaped[:, token, int(spec.head_index), :], ("head", shaped)
        feature = spec.feature_indices if spec.feature_indices is not None else slice(None)
        return tensor[:, token, feature], ("feature", tensor)

    @staticmethod
    def _write(tensor: Any, spec: HookSpec, value: Any) -> Any:
        result = tensor.clone()
        token = spec.token_slice if spec.token_slice is not None else slice(None)
        if spec.head_index is not None:
            shaped = result.reshape(*result.shape[:-1], int(spec.num_heads), int(spec.head_dim))
            shaped[:, token, int(spec.head_index), :] = value.to(device=tensor.device, dtype=tensor.dtype)
            return shaped.reshape_as(result)
        feature = spec.feature_indices if spec.feature_indices is not None else slice(None)
        result[:, token, feature] = value.to(device=tensor.device, dtype=tensor.dtype)
        return result

    def _process(self, index: int, value: Any) -> Any:
        tensor, rebuild = primary_tensor(value)
        spec = self.specs[index]
        selected, _ = self._selection(tensor, spec)
        if self.mode == "capture":
            if self.retain_grad and tensor.requires_grad:
                # 保留实际参与下游计算的完整 Tensor 的梯度；仅创建 sibling view 不会收到 .grad。
                tensor.retain_grad()
                self.captured[index] = tensor
            else:
                self.captured[index] = selected.detach().clone()
            return value
        if self.mode == "identity":
            replacement = selected
        elif self.mode == "add":
            replacement = selected + self.scale * self.values[index].to(selected)
        else:
            replacement = selected + self.scale * (self.values[index].to(selected) - selected) if self.scale != 1.0 else self.values[index].to(selected)
        return rebuild(self._write(tensor, spec, replacement))

    def _matches(self, index: int) -> bool:
        current = self.calls[index]
        self.calls[index] += 1
        return self.specs[index].call_index is None or current == self.specs[index].call_index

    def __enter__(self) -> "TensorHooks":
        for index, spec in enumerate(self.specs):
            module = resolve_module(self.roots, spec.path)
            if spec.hook == "output":
                def output_hook(_module: Any, _inputs: Any, output: Any, slot=index) -> Any:
                    return self._process(slot, output) if self._matches(slot) else output
                self.handles.append(module.register_forward_hook(output_hook))
            elif spec.hook == "input":
                def input_hook(_module: Any, inputs: Any, slot=index) -> Any:
                    return self._process(slot, inputs) if self._matches(slot) else inputs
                self.handles.append(module.register_forward_pre_hook(input_hook))
            else:
                raise ValueError(spec.hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if exc is None and self.mode == "capture" and any(value is None for value in self.captured):
            raise RuntimeError("至少一个 Hook 未触发；拒绝无声跳过")


class AttentionEdgeKnockout(AbstractContextManager["AttentionEdgeKnockout"]):
    """通过 additive attention mask 切断指定 source→target 边（所有 Head）。"""

    def __init__(self, module: Any, source: slice, target: slice, magnitude: float = -1.0e4) -> None:
        self.module, self.source, self.target, self.magnitude = module, source, target, magnitude
        self.handle: Any | None = None

    def __enter__(self) -> "AttentionEdgeKnockout":
        import torch

        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                raise RuntimeError("Attention Hook 无法读取 hidden_states")
            batch, length = hidden.shape[:2]
            mask_in_kwargs = "attention_mask" in kwargs
            mask_in_args = not mask_in_kwargs and len(args) > 1
            mask = kwargs.get("attention_mask", args[1] if mask_in_args else None)
            if mask is None:
                mask = torch.zeros((batch, 1, length, length), device=hidden.device, dtype=hidden.dtype)
            else:
                mask = mask.clone()
            if mask.dtype == torch.bool:
                mask[..., self.target, self.source] = False
            else:
                mask[..., self.target, self.source] = self.magnitude
            if mask_in_args:
                mutable = list(args)
                mutable[1] = mask
                args = tuple(mutable)
            else:
                kwargs["attention_mask"] = mask
            return args, kwargs
        self.handle = self.module.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

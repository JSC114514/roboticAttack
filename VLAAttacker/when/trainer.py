"""Small trainer-side utilities shared by WHEN entry scripts."""

from __future__ import annotations

from typing import Any


def assert_cuda_available() -> None:
    """Fail fast when a GPU job is required but CUDA is unavailable."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WHEN training/evaluation; refusing CPU fallback.")


def freeze_module(module: Any) -> None:
    """Freeze a torch module in eval mode."""
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def assert_optimizer_patch_only(optimizer: Any, allowed_params: set[int]) -> None:
    """Ensure an optimizer only owns explicitly allowed trainable tensors."""
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            seen.add(id(param))
    extra = seen - set(allowed_params)
    if extra:
        raise RuntimeError("WHEN optimizer contains model parameters; only patch/inner perturbation are allowed.")

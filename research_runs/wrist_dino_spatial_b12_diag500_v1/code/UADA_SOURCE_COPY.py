"""Isolated copy of Exploring-UADA patch-optimization primitives.

The upstream ``UADA.py`` stays byte-for-byte at the repository version and is
read-only. OFT experiments import this copy so adaptations never edit source.
"""

from __future__ import annotations

import torch
import transformers


def init_uada_patch(patch_size, device):
    patch = torch.rand(patch_size).to(device)
    patch.requires_grad_(True)
    patch.retain_grad()
    return patch


def build_uada_adamw(patch, lr, warmup, num_training_steps, weight_decay=0.0):
    optimizer = transformers.AdamW([patch], lr=lr, weight_decay=weight_decay)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
        last_epoch=-1,
    )
    return optimizer, scheduler


def step_uada_patch(patch, optimizer, vla, empty_cuda_cache=False):
    optimizer.step()
    patch.data = patch.data.clamp(0, 1)
    optimizer.zero_grad()
    vla.zero_grad()
    if empty_cuda_cache:
        torch.cuda.empty_cache()

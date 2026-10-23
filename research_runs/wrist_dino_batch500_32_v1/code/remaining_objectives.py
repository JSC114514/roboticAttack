"""Read-only native Value target and exact full-batch generator gradients."""
from __future__ import annotations
from contextlib import ExitStack
import numpy as np
import torch
import torch.nn.functional as F
from common import require
from feature_objectives import capture_qkv, flattened_cosine, high_component


def value_forward(engine, pixels, layers):
    with ExitStack() as stack:
        captures = {i: stack.enter_context(capture_qkv(engine.dino.blocks[i].attn)) for i in layers}
        _dense = engine.direct_dino(pixels)
    result = {}
    for i, outputs in captures.items():
        qkv = outputs[0]
        b, tokens, channels = qkv.shape
        heads = int(engine.dino.blocks[i].attn.num_heads)
        require(b == 1 and tokens == 261 and channels % (3 * heads) == 0, "Native Value shape changed")
        _q, _k, v = qkv.reshape(b, tokens, 3, heads, channels // (3 * heads)).permute(2, 0, 3, 1, 4).unbind(0)
        result[i] = v[:, :, int(engine.dino.num_prefix_tokens):]
    return result


def value_loss(clean, current):
    require(set(clean) == set(current) and bool(clean), "Value layer correspondence changed")
    per_layer = {i: flattened_cosine(clean[i].detach(), current[i]) for i in clean}
    loss = torch.stack(list(per_layer.values())).mean()
    return loss, {f"value_layer{i}_cosine": float(value.detach()) for i, value in per_layer.items()}


def batch_feature_loss(clean, attacked, criterion, temperature):
    require(clean.shape == attacked.shape and clean.ndim >= 2, "Cross-image feature shape changed")
    query = F.normalize(clean.detach().float().reshape(clean.shape[0], -1), dim=1)
    positive = F.normalize(attacked.float().reshape(attacked.shape[0], -1), dim=1)
    similarity = query @ positive.T
    matched = similarity.diagonal().mean()
    if criterion == "cos":
        return matched, {"feature_cosine": float(matched.detach()), "feature_negative_nce": 0.}
    # Approved 9-B1 is an explicit degeneracy control, NOT a cosine fallback:
    # one-class cross entropy and its feature gradient are exactly zero.
    require(criterion == "nce" and len(query) >= 1 and temperature > 0, "Invalid InfoNCE batch/temperature")
    loss = -F.cross_entropy(similarity / temperature, torch.arange(len(query), device=query.device))
    return loss, {"feature_cosine": float(matched.detach()), "feature_negative_nce": float(loss.detach()),
                  "nce_negative_count": float(len(query) - 1), "nce_degenerate": float(len(query) == 1)}


def rgb_terms(clean, attacked, settings):
    quality = F.mse_loss(attacked.float(), clean.float())
    high = F.mse_loss(high_component(attacked, settings["high_frequency_radius"]),
                      high_component(clean, settings["high_frequency_radius"]))
    loss = settings["quality_weight"] * quality - settings["high_frequency_weight"] * high
    return loss, quality, high


def cached_generator_backward(engine, records, clean_features, clean_rgb, rgb, seeds, criterion, cfg):
    """Exactly backpropagate coupled B8 loss while keeping one DINO graph live.

    First pass caches the forward. The full BxB loss supplies a cotangent for
    each image. Recompute each image once, sum RGB cotangents, then perform a
    single backward through the existing generator graph. No B1-NCE shortcut.
    """
    require(len(records) == len(seeds) == len(clean_rgb) == clean_features.shape[0], "Incomplete generator batch")
    settings, values, qualities, highs = cfg["generator"], [], [], []
    # Match grad-enabled native attention dispatch in both passes. The first
    # pass uses an independent RGB leaf and discards each graph immediately;
    # it does NOT retain eight encoder graphs or touch the generator graph.
    with torch.enable_grad():
        for record, seed, original in zip(records, seeds, clean_rgb):
            scratch = rgb.detach().clone().requires_grad_(True)
            pixels, _footprint, rendered_rgb = engine.pixels(record, scratch, seed, diverse=True, return_rgb=True)
            dense = engine.direct_dino(pixels)
            values.append(dense.detach())
            with torch.no_grad():
                _aux, quality, high = rgb_terms(original, rendered_rgb, settings)
                qualities.append(float(quality)); highs.append(float(high))
            del pixels, rendered_rgb, dense, scratch
    leaf = torch.cat(values, dim=0).detach().requires_grad_(True)
    feature, detail = batch_feature_loss(clean_features, leaf, criterion, cfg["nce_temperature"])
    cotangents = torch.autograd.grad(settings["feature_weight"] * feature, leaf)[0].detach()
    rgb_gradient, maximum_error = torch.zeros_like(rgb), 0.
    for index, (record, seed, original) in enumerate(zip(records, seeds, clean_rgb)):
        pixels, _footprint, rendered_rgb = engine.pixels(record, rgb, seed, diverse=True, return_rgb=True)
        dense = engine.direct_dino(pixels)
        scale = values[index].detach().float().abs().max().clamp_min(1.)
        error = float((dense.detach().float() - values[index].float()).abs().max() / scale)
        maximum_error = max(maximum_error, error)
        require(error <= cfg["recompute_tolerance"], "Frozen eval-mode DINO recompute changed; refusing inexact cached gradient")
        auxiliary, _quality, _high = rgb_terms(original, rendered_rgb, settings)
        proxy = (dense.float() * cotangents[index:index + 1]).sum() + auxiliary / len(records)
        rgb_gradient += torch.autograd.grad(proxy, rgb)[0].detach()
        del pixels, dense, proxy, auxiliary, rendered_rgb
    require(torch.isfinite(rgb_gradient).all(), "Nonfinite cached RGB gradient")
    rgb.backward(rgb_gradient)
    quality, high = float(np.mean(qualities)), float(np.mean(highs))
    value = float(settings["feature_weight"] * feature.detach()) + settings["quality_weight"] * quality - settings["high_frequency_weight"] * high
    detail.update(rgb_quality_mse=quality, rgb_high_frequency_mse=high,
                  recompute_relative_max_error=maximum_error, training_image_batch_size=len(records),
                  attacked_dino_forwards_per_update=2 * len(records))
    return value, detail

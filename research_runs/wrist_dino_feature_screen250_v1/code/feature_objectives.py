"""Paper-based DINO targets and an author-architecture universal generator.

Training-only readout hooks never replace activations or alter inference.
"""
from __future__ import annotations
import importlib.util
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from common import ROOT, require

_spec = importlib.util.spec_from_file_location("advencoder_author_generator", ROOT / "vendor/adv_gan.py")
_author = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_author)
Generator = _author.Generator


def flattened_cosine(clean, attacked):
    require(clean.shape == attacked.shape and clean.ndim >= 2, "Feature pair shape changed")
    return F.cosine_similarity(clean.float().reshape(clean.shape[0], -1),
                               attacked.float().reshape(attacked.shape[0], -1), dim=1, eps=1e-8).mean()


def centered_loss(clean, attacked, mean):
    require(mean.shape == clean.shape and torch.isfinite(mean).all(), "Invalid fixed feature mean")
    return flattened_cosine(clean.float() - mean, attacked.float() - mean)


def weighted_key_loss(clean, attacked, saliency):
    require(clean.shape == attacked.shape and saliency.shape == (clean.shape[0], clean.shape[2]), "Key/saliency shape changed")
    weight = saliency.detach().float()[:, None, :, None]
    return flattened_cosine(clean.float() * weight, attacked.float() * weight)


def split_native_qk(qkv, attention):
    b, n, total = qkv.shape
    h = attention.num_heads
    require(total % (3 * h) == 0, "Native QKV dimensions changed")
    q, k, _ = qkv.reshape(b, n, 3, h, total // (3 * h)).permute(2, 0, 3, 1, 4).unbind(0)
    return attention.q_norm(q), attention.k_norm(k)


def key_and_saliency(qkv, attention, prefix=5):
    q, k = split_native_qk(qkv, attention)
    require(q.shape[2] == prefix + 256, "DINO CLS/register/dense count changed")
    # Normalize with ALL native keys before selecting spatial keys. Registers
    # are not incorrectly treated as spatial tokens or omitted from softmax.
    logits = (q[:, :, :1].float() * float(attention.scale)) @ k.float().transpose(-2, -1)
    saliency = logits.softmax(dim=-1)[:, :, 0, prefix:].mean(dim=1)
    saliency = saliency / saliency.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return k[:, :, prefix:], saliency.detach()


@contextmanager
def capture_qkv(attention):
    captured = []
    handle = attention.qkv.register_forward_hook(lambda _m, _args, output: captured.append(output))
    try:
        yield captured
        require(len(captured) == 1, "Expected one native Wrist QKV call")
    finally:
        handle.remove()


def key_forward(engine, pixels):
    layer = len(engine.dino.blocks) - 2
    require(layer >= 0, "DINO depth invalid")
    attention = engine.dino.blocks[layer].attn
    with capture_qkv(attention) as captured:
        dense = engine.direct_dino(pixels)
    key, saliency = key_and_saliency(captured[0], attention, engine.dino.num_prefix_tokens)
    return dense, key, saliency, layer


def high_component(images, radius=8):
    """Device-safe version of author generate_high's radial FFT high-pass."""
    require(images.ndim == 4, "Expected BCHW raw RGB")
    h, w = images.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=images.device), torch.arange(w, device=images.device), indexing="ij")
    low = ((yy - h / 2).square() + (xx - w / 2).square()).sqrt() < radius
    spectrum = torch.fft.fftshift(torch.fft.fftn(images.float(), dim=(-2, -1)), dim=(-2, -1))
    spectrum = spectrum * (~low)
    return torch.fft.ifftn(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1)).real


def generator_loss(clean_features, attacked_features, clean_rgb, attacked_rgb, settings):
    cosine = flattened_cosine(clean_features, attacked_features)
    quality = F.mse_loss(attacked_rgb.float(), clean_rgb.float())
    high = F.mse_loss(high_component(attacked_rgb, settings["high_frequency_radius"]),
                      high_component(clean_rgb, settings["high_frequency_radius"]))
    loss = settings["feature_weight"] * cosine + settings["quality_weight"] * quality - settings["high_frequency_weight"] * high
    return loss, {"feature_cosine": float(cosine.detach()), "rgb_quality_mse": float(quality.detach()),
                  "rgb_high_frequency_mse": float(high.detach())}


def new_generator(settings, seed, device, filters=(1024, 512, 256, 128)):
    # Independent initialization stream, never change the frozen policy RNG.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        generator = Generator(settings["latent_dimension"], list(filters), 3, 1)
        noise = torch.randn(1, settings["latent_dimension"], 1, 1)
    generator = generator.to(device).train()
    noise = noise.to(device).detach()
    require(not noise.requires_grad, "Fixed generator noise must not be optimized")
    return generator, noise


def generated_rgb(generator, noise):
    output = generator(noise)
    require(output.shape == (1, 3, 64, 64) and torch.isfinite(output).all(), "Author generator output changed")
    return F.interpolate(output, size=(50, 50), mode="bilinear", align_corners=False)[0].clamp(0, 1)

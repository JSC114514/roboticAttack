"""Same published/adapted objectives; B gradients averaged before one step."""
from __future__ import annotations
import numpy as np
import torch
from common import require
from objectives import action_loss, target_action
from feature_objectives import (key_forward, weighted_key_loss, centered_loss, flattened_cosine,
                                generator_loss)
from new_objectives import features_forward, multilayer_reference, objective
from remaining_objectives import cached_generator_backward


def prepare_references(engine, records, family, cfg):
    references = []
    with torch.no_grad():
        for record in records:
            inputs, proprio = engine.base(record)
            ref = {"inputs": inputs, "proprio": proprio}
            if family in {"uada_original", "tma_xyz"}:
                action_family = "uada" if family == "uada_original" else "tma"
                ref["target"] = target_action(record["expert_actions"].to(engine.wrapper.device), action_family, cfg)
            elif family == "dino_key":
                _dense, key, saliency, layer = key_forward(engine, inputs["pixel_values"])
                ref.update(key=key.detach(), saliency=saliency.detach(), layer=layer)
            elif family in {"dino_multilayer", "dino_affinity"}:
                dense, captures = features_forward(engine, inputs["pixel_values"], family, cfg)
                ref["reference"] = multilayer_reference(captures, cfg["objective_epsilon"]) if family == "dino_multilayer" else dense.detach()
            else:
                ref["clean"] = engine.direct_dino(inputs["pixel_values"]).detach()
                if family.startswith("advencoder_"):
                    ref["clean_rgb"] = torch.as_tensor(np.asarray(record["observation"]["wrist_image"]),
                         device=engine.wrapper.device).permute(2, 0, 1)[None].float() / 255
            references.append(ref)
    return references


def pixel_batch_backward(engine, records, references, rgb, seeds, family, cfg, center=None):
    require(len(records) == len(references) == len(seeds), "Incomplete pixel batch")
    statistics = []
    for record, ref, seed in zip(records, references, seeds):
        pixels, footprint = engine.pixels(record, rgb, seed, diverse=True)
        detail = {}
        if family in {"uada_original", "tma_xyz"}:
            action, captures, qkv = engine.forward(ref["inputs"], ref["proprio"], pixels)
            action_family = "uada" if family == "uada_original" else "tma"
            loss = action_loss(action, ref["target"], action_family, cfg)
            del action, captures, qkv
        elif family == "dino_key":
            dense, key, _saliency, _layer = key_forward(engine, pixels)
            loss = weighted_key_loss(ref["key"], key, ref["saliency"])
            detail = {"feature_cosine": float(loss.detach()), "key_layer": float(ref["layer"])}
            del dense, key, _saliency
        elif family == "dino_center":
            dense = engine.direct_dino(pixels)
            loss = centered_loss(ref["clean"], dense, center)
            detail = {"feature_cosine": float(loss.detach()), "uncentered_cosine": float(flattened_cosine(ref["clean"], dense).detach())}
            del dense
        else:
            require(family in {"dino_multilayer", "dino_affinity"}, "Unknown pixel objective")
            dense, captures = features_forward(engine, pixels, family, cfg)
            loss, detail = objective(family, dense, captures, ref["reference"], engine, footprint, cfg)
            del dense, captures
        require(torch.isfinite(loss), "Nonfinite actual training objective")
        value = float(loss.detach())
        (loss / len(records)).backward()
        statistics.append({"main_loss": value, **detail})
        del pixels, footprint, loss
    means = {key: float(np.mean([s[key] for s in statistics])) for key in statistics[0]}
    return means.pop("main_loss"), means


def generator_batch_backward(engine, records, references, rgb, seeds, family, cfg):
    clean = torch.cat([ref["clean"] for ref in references], dim=0)
    clean_rgb = [ref["clean_rgb"] for ref in references]
    if family == "advencoder_cos" and len(records) == 1:
        # Retain the original efficient B1 branch and its exact scalar objective.
        pixels, _footprint, rendered = engine.pixels(records[0], rgb, seeds[0], diverse=True, return_rgb=True)
        dense = engine.direct_dino(pixels)
        loss, detail = generator_loss(clean, dense, clean_rgb[0], rendered, cfg["generator"])
        require(torch.isfinite(loss), "Nonfinite generator objective")
        loss.backward()
        return float(loss.detach()), detail
    criterion = "nce" if family == "advencoder_nce" else "cos"
    require(family in {"advencoder_cos", "advencoder_nce"}, "Unknown generator objective")
    return cached_generator_backward(engine, records, clean, clean_rgb, rgb, seeds, criterion, cfg)

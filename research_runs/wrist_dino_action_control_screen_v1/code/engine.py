"""Native OFT forward with one bounded internal site intervention."""
from __future__ import annotations
import numpy as np
import torch
from common import require
from screen_math import clean_scale, primary_loss
from sites import SiteHook


class ControlEngine:
    def __init__(self, wrapper, cfg):
        self.wrapper = wrapper
        self.cfg = cfg
        wrapper.freeze_model()
        self.dino = wrapper.vla.vision_backbone.featurizer
        self.siglip = wrapper.vla.vision_backbone.fused_featurizer
        require(len(self.dino.blocks) == 24 and self.dino.num_prefix_tokens == 5,
                "Expected 24-block DINO with CLS+four registers")
        require(all(not parameter.requires_grad for module in
                    (wrapper.vla, wrapper.action_head, wrapper.proprio_projector)
                    if module is not None for parameter in module.parameters()), "Native model is not frozen")
        self.cached_uid = None

    def context(self, record):
        if self.cached_uid != record["uid"]:
            observation = record["observation"]
            self.cached_inputs = dict(self.wrapper.prepare_hf_inputs(observation, record["task_name"]))
            stats = self.wrapper.vla.norm_stats[self.wrapper.gen_cfg.unnorm_key]["proprio"]
            self.cached_proprio = self.wrapper._official_openvla_utils.normalize_proprio(
                np.asarray(observation["state"]), stats)
            self.cached_uid = record["uid"]
        return self.cached_inputs, self.cached_proprio

    def forward(self, record, site, replacement=None):
        inputs, proprio = self.context(record)
        hook = SiteHook(self, site, replacement)
        with hook.installed():
            _unused, hidden = self.wrapper.vla.predict_action(
                **inputs, unnorm_key=self.wrapper.gen_cfg.unnorm_key, do_sample=False,
                proprio=proprio, proprio_projector=self.wrapper.proprio_projector,
                noisy_action_projector=self.wrapper.noisy_action_projector,
                action_head=self.wrapper.action_head, use_film=False)
        head = getattr(self.wrapper.action_head, "module", self.wrapper.action_head)
        action = head.predict_action(hidden).reshape(1, 8, 7).float()
        require(torch.isfinite(action).all() and torch.isfinite(hook.captured).all(), f"Nonfinite native value at {site.name}")
        return action, hook.captured

    def interface_check(self, record, sites, margin):
        rows = []
        for site in sites.values():
            with torch.no_grad():
                clean_action, clean_site = self.forward(record, site)
            clean_site = clean_site.detach().float()
            scale = clean_scale(clean_site, site.headwise)
            z = torch.zeros_like(clean_site, requires_grad=True)
            action, actual = self.forward(record, site, clean_site + scale * z)
            identity = float((action.detach() - clean_action).abs().max())
            require(identity <= 1e-5 and torch.equal(actual.detach(), clean_site.to(actual)),
                    f"Clean identity replacement failed at {site.name}: {identity}")
            loss = primary_loss(action, record, "wrong_open", margin)
            gradient = torch.autograd.grad(loss, z, allow_unused=False)[0]
            require(torch.isfinite(gradient).all(), f"Disconnected/nonfinite intervention at {site.name}")
            rows.append({"site": site.name, "shape": list(clean_site.shape), "identity_action_max_abs": identity,
                         "wrong_open_gradient_l2": float(gradient.norm()), "gradient_zero": bool(gradient.norm() == 0)})
        return rows

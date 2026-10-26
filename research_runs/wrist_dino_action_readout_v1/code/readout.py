"""DINO-conditioned action readout and continuous-action LAFEAT adaptation."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class ActionReadout(nn.Module):
    def __init__(self, feature_dim=1024, language_dim=4096, width=128, tokens=256):
        super().__init__()
        self.tokens = tokens
        self.feature = nn.Linear(feature_dim, width)
        self.language = nn.Linear(language_dim, width)
        self.proprio = nn.Linear(8, width)
        side = int(tokens ** .5)
        if side * side != tokens:
            raise ValueError("DINO grid must be square")
        self.row = nn.Parameter(torch.randn(side, width) * .02)
        self.col = nn.Parameter(torch.randn(side, width) * .02)
        self.view = nn.Parameter(torch.randn(2, width) * .02)
        self.query = nn.Parameter(torch.randn(8, width) * .02)
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0., batch_first=True, norm_first=True)
        self.blocks = nn.TransformerDecoder(layer, 2)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, features, language, proprio):
        # [batch, front/wrist, dense tokens, channels], no CLS/register tokens.
        batch, views, tokens, _ = features.shape
        if views != 2 or tokens != self.tokens:
            raise ValueError("Native two-view dense token layout changed")
        pos = (self.row[:, None] + self.col[None, :]).reshape(tokens, -1)
        memory = self.feature(features.float()) + pos[None, None] + self.view[None, :, None]
        context = self.language(language.float()) + self.proprio(proprio.float())
        query = self.query[None].expand(batch, -1, -1) + context[:, None]
        return self.output(self.blocks(query, memory.flatten(1, 2))).squeeze(-1)


def energy_logits(action, clean, wrong):
    scale = (clean - wrong).square().mean(-1, keepdim=True).detach().clamp_min(1e-6)
    return -torch.stack(((action - clean).square().mean(-1), (action - wrong).square().mean(-1)), -1) / scale


def patch_objective(native_action, helper_action, clean, wrong, arm, outer, beta=.5):
    native_action = native_action.reshape(-1, 8)
    clean, wrong = clean.reshape(-1, 8).detach(), wrong.reshape(-1, 8).detach()
    uada = F.mse_loss(native_action, wrong)
    if outer > 400:
        return uada, {"beta": 0., "uada_mse": float(uada.detach()), "phase": "uada_final100"}
    logits = energy_logits(native_action, clean, wrong)
    active = torch.zeros_like(logits[:, :1])
    if arm == "C_dino_readout":
        h = helper_action.reshape(-1, 8)
        helper_logits = energy_logits(h, clean, wrong)
        active = (helper_logits[:, :1] >= helper_logits[:, 1:]).detach().to(logits.dtype) * beta
        logits = (1 - active) * logits + active * helper_logits
    loss = F.cross_entropy(logits, torch.ones(logits.shape[0], device=logits.device, dtype=torch.long))
    return loss, {"beta": float(active.mean()), "uada_mse": float(uada.detach()), "phase": "energy_first400"}


def response_loss(prediction, teacher, scales, eta):
    # variants: clean, p, p+, p-. Denominators use TRAIN trajectories only.
    delta_p, delta_t = prediction[:, 1:] - prediction[:, :1], teacher[:, 1:] - teacher[:, :1]
    response_p = (prediction[:, 2] - prediction[:, 3]) / (2 * eta)
    response_t = (teacher[:, 2] - teacher[:, 3]) / (2 * eta)
    action = ((prediction - teacher).square() / scales["action"]).mean()
    delta = ((delta_p - delta_t).square() / scales["delta"]).mean()
    response = ((response_p - response_t).square() / scales["response"]).mean()
    return action + delta + response, {"action": action, "delta": delta, "response": response}

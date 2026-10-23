"""Pure gradient combination and small CPU checks for the delayed register run."""
from __future__ import annotations

from types import SimpleNamespace

import torch


def auxiliary_active(outer: int, cfg: dict) -> bool:
    return int(outer) >= int(cfg["auxiliary_start_outer"])


def combine_gradients(main: torch.Tensor, auxiliary: torch.Tensor | None,
                      ratio: float, project_conflict: bool, eps: float = 1e-12):
    """Preserve main; cap auxiliary using its RAW norm before optional projection.

    This is a constraint on raw gradients, not a guarantee about Adam's
    preconditioned update or closed-loop task success.
    """
    if not 0 <= ratio <= 1 or not bool(torch.isfinite(main).all()):
        raise ValueError("Invalid main gradient or auxiliary ratio")
    mn = main.norm()
    empty = {
        "aux_grad_norm": 0., "aux_grad_scale": 0.,
        "raw_gradient_cosine": 0., "projected_gradient_cosine": 0.,
        "gradient_conflict_fraction": 0., "projection_applied_fraction": 0.,
        "effective_aux_gradient_ratio": 0., "aux_zero_gradient_steps": 0.,
    }
    if auxiliary is None:
        return main, empty
    if auxiliary.shape != main.shape or auxiliary.device != main.device or not bool(torch.isfinite(auxiliary).all()):
        raise ValueError("Invalid auxiliary shape/device/numerics")
    an = auxiliary.norm()
    if float(mn) <= eps or float(an) <= eps:
        return main, {**empty, "aux_grad_norm": float(an),
                      "aux_zero_gradient_steps": float(float(an) <= eps)}
    dot = (main * auxiliary).sum()
    conflict = bool(dot < 0)
    applied = bool(project_conflict and conflict)
    used = auxiliary - dot / mn.square() * main if applied else auxiliary
    scale = ratio * mn / an
    contribution = scale * used
    used_norm = used.norm()
    used_cosine = (main * used).sum() / (mn * used_norm).clamp_min(eps * eps) if float(used_norm) > eps else mn.new_zeros(())
    return main + contribution, {
        "aux_grad_norm": float(an), "aux_grad_scale": float(scale),
        "raw_gradient_cosine": float(dot / (mn * an)),
        "projected_gradient_cosine": float(used_cosine),
        "gradient_conflict_fraction": float(conflict),
        "projection_applied_fraction": float(applied),
        "effective_aux_gradient_ratio": float(contribution.norm() / mn),
        "aux_zero_gradient_steps": 0.,
    }


def cpu_preflight(cfg: dict) -> dict:
    """Exercise the actual objective and hooks, without an OFT model or CUDA."""
    from timm.models.vision_transformer import Attention
    from white_patch.UADA_OFT_DINO_REGISTER_DELAYED import ExploringOFTDINORegisterDelayedAttacker

    def check(condition, message):
        if not condition:
            raise RuntimeError("Delayed objective CPU check: " + message)

    check(not auxiliary_active(250, cfg) and auxiliary_active(251, cfg), "auxiliary onset")
    main = torch.tensor([1., 0.])
    for auxiliary in (torch.tensor([1., 1.]), torch.tensor([-1., 1.]), torch.tensor([-1., 0.]), torch.zeros(2)):
        for projected in (False, True):
            combined, stats = combine_gradients(main, auxiliary, .1, projected)
            contribution = combined - main
            check(bool(combined.isfinite().all()) and stats["effective_aux_gradient_ratio"] <= .100001,
                  "finite auxiliary cap")
            if projected:
                check(float((main * contribution).sum()) >= -1e-7, "main gradient opposition")
    combined, stats = combine_gradients(main, torch.tensor([-1., 1.]), .1, True)
    check(abs(stats["effective_aux_gradient_ratio"] - .1 / 2 ** .5) < 1e-6,
          "post-projection contribution was renormalized")
    zero, _ = combine_gradients(torch.zeros(2), main, .1, True)
    check(bool(zero.eq(0).all()), "zero main norm")
    unchanged, _ = combine_gradients(main, None, .1, True)
    check(torch.equal(unchanged, main), "action-only gradient changed")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        blocks = [SimpleNamespace(attn=Attention(32, num_heads=4, qk_norm=True))
                  for _ in range(max(cfg["register_attention_layers"]) + 1)]
        for block in blocks:
            block.attn.eval()
            for parameter in block.attn.parameters():
                parameter.requires_grad_(False)
        attacker = ExploringOFTDINORegisterDelayedAttacker.__new__(ExploringOFTDINORegisterDelayedAttacker)
        attacker.wrapper = SimpleNamespace(
            vla=SimpleNamespace(vision_backbone=SimpleNamespace(featurizer=SimpleNamespace(blocks=blocks))),
            device=torch.device("cpu"))
        attacker.patch_tokens = torch.zeros(256)
        attacker.patch_tokens[80:96] = 1
        front, wrist = torch.randn(1, 261, 32), torch.randn(1, 261, 32, requires_grad=True)
        with attacker._capture_dino_qkv(cfg) as captures:
            for pixels in (front, wrist):
                for layer in cfg["register_attention_layers"]:
                    blocks[layer].attn.qkv(pixels)
        check(all(len(values) == 2 and not values[0].requires_grad and values[1].requires_grad
                  for values in captures.values()), "front/wrist call order")
        for target in ("register", "spatial"):
            loss, metrics = attacker._attention_masses(captures, cfg, target)
            expected = []
            for layer in cfg["register_attention_layers"]:
                attn = blocks[layer].attn
                q, k, _ = captures[layer][1].reshape(1, 261, 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
                q, k = attn.q_norm(q), attn.k_norm(k)
                probability = ((q.float() * attn.scale) @ k.float().transpose(-2, -1)).softmax(-1)
                queries = torch.arange(1, 5) if target == "register" else torch.cat((torch.arange(5, 85), torch.arange(101, 261)))
                expected.append(-probability[:, :, queries, :][:, :, :, 85:101].sum(-1).log().mean())
            check(bool(torch.allclose(loss, torch.stack(expected).mean(), atol=1e-6, rtol=1e-6)), "attention formula")
            grad = torch.autograd.grad(loss, wrist, retain_graph=True)[0]
            check(bool(grad.isfinite().all()) and float(grad.norm()) > 1e-12, "auxiliary backward")
            check(metrics["target_query_count"] == (4 if target == "register" else 240), "query selection")
        check(all(not blocks[layer].attn.qkv._forward_hooks for layer in cfg["register_attention_layers"]),
              "QKV hooks not removed")
    return {"status": "PASS", "attention_formula": True, "wrist_hook_order": True,
            "backward_finite": True, "gradient_cap": .1, "onset_outer": 251,
            "no_post_projection_renormalization": True, "full_oft_model_loaded": False,
            "cuda_initialized": torch.cuda.is_initialized()}

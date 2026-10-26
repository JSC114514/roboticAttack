"""Exact official RGB policy call with read-only DINO / action / layer observers."""
from __future__ import annotations
import copy
import time
import numpy as np
from loop_core import require


class NativePredictor:
    def __init__(self, wrapper, instruction):
        self.wrapper, self.instruction = wrapper, instruction
        self.calls, self.seconds = 0, 0.
        self.handles = []
        self.captures = {}
        head = getattr(wrapper.action_head, "module", wrapper.action_head)
        self.head = head
        self.had_instance_method = "predict_action" in head.__dict__
        self.old_instance_method = head.__dict__.get("predict_action")
        original = head.predict_action
        def observe_action(*args, **kwargs):
            result = original(*args, **kwargs)
            self.captures["normalized"] = result.detach().float().cpu().numpy().reshape(8, 7).copy()
            return result
        head.predict_action = observe_action
        def dino_hook(_module, _args, result):
            self.dino_calls += 1
            if self.dino_calls == 2 and self.observe:
                require(tuple(result.shape) == (1, 256, 1024), "Native consumed wrist DINO shape changed")
                self.captures["dino"] = result.detach().float()
        self.handles.append(wrapper.vla.vision_backbone.featurizer.register_forward_hook(dino_hook))
        def layer_hook(name):
            def hook(_module, _args, result):
                if self.layers:
                    value = result[0] if isinstance(result, tuple) else result
                    self.captures[name] = {"shape": list(value.shape), "l2": float(value.detach().float().norm())}
            return hook
        self.handles.append(wrapper.vla.projector.register_forward_hook(layer_hook("projector")))
        layers = wrapper.vla.language_model.model.layers
        for index in sorted({0, len(layers) // 2, len(layers) - 1}):
            self.handles.append(layers[index].register_forward_hook(layer_hook(f"llm_{index}")))

    def close(self):
        for handle in self.handles:
            handle.remove()
        if self.had_instance_method:
            self.head.predict_action = self.old_instance_method
        else:
            delattr(self.head, "predict_action")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def predict(self, obs, *, observe=False, layers=False):
        import torch
        self.captures, self.dino_calls = {}, 0
        self.observe, self.layers = observe, layers
        started = time.monotonic()
        with torch.no_grad():
            actions = np.asarray(self.wrapper.predict_action(copy.deepcopy(obs), self.instruction), dtype=np.float32).reshape(8, 7)
        self.seconds += time.monotonic() - started
        self.calls += 1
        require(self.dino_calls == 2, "Official RGB policy no longer processes front then wrist")
        require("normalized" in self.captures, "Action observer did not see the native head")
        normalized = self.captures["normalized"]
        expected = np.asarray(self.wrapper.vla._unnormalize_actions(normalized, self.wrapper.gen_cfg.unnorm_key)).reshape(8, 7)
        parity = float(np.abs(actions[:, :6] - expected[:, :6]).max())
        require(parity <= 2e-5 and np.isfinite(actions).all() and np.isfinite(normalized).all(),
                f"Official / normalized native action mismatch: {parity}")
        return {"actions": actions.copy(), "normalized": normalized, "parity_error": parity,
                "dino": self.captures.get("dino"),
                "layers": {k: v for k, v in self.captures.items() if k.startswith(("llm_", "projector"))}}


def dino_difference(clean, current):
    import torch
    require(clean is not None and current is not None, "Missing native DINO observation")
    with torch.no_grad():
        delta = current - clean
        norms = delta.norm(dim=-1)[0]
        relative = float(delta.norm() / clean.norm().clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(clean.flatten(), current.flatten(), dim=0))
        return {"dino_relative_l2": relative, "dino_cosine": cosine,
                "dino_token_delta_l2": norms.cpu().tolist(), "dino_delta_norm": float(delta.norm())}

"""Observe actual native steps; probe passes preserve RNG and optimizer leaves."""
from __future__ import annotations
import json
import random
import time
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT, atomic_json, atomic_torch, experiment, require, sha256_file
from objectives import action_loss

@contextmanager
def preserve_rng():
    state = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1])
        torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])

def quantized(value):
    return value.detach().clamp(0, 1).mul(255).round().div(255)

def cosine(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0, eps=1e-20))

def step_metrics(before, after, gradient):
    change = after.detach() - before
    return {"gradient_l2": float(gradient.norm()), "rgb_update_l2": float(change.norm()),
            "rgb_update_linf": float(change.abs().max()),
            "native_gradient_dot_update": float((gradient * change).sum()),
            "update_vs_native_descent_cosine": cosine(-gradient, change),
            "quantized_channel_change_fraction": float((quantized(before) != quantized(after)).float().mean()),
            "rgb_min": float(after.min()), "rgb_max": float(after.max()),
            "saturated_fraction": float(((after <= 0) | (after >= 1)).float().mean())}

class Monitor:
    def __init__(self, folder, signature, cfg):
        self.cfg, self.signature = cfg, signature
        self.folder = folder / "attempts" / str(time.time_ns())
        self.folder.mkdir(parents=True, exist_ok=False)
        self.path = self.folder / "INNER_STEPS.jsonl"
        self.stream = self.path.open("a", encoding="utf-8", buffering=65536)
        self.rows = 0; self.last = None

    def step(self, outer, inner, uid, seed, loss, lr, before, after, gradient):
        row = {"signature": self.signature, "arm": self.cfg["arm"], "outer": outer, "inner": inner,
               "uid": uid, "eot_seed": seed, "native_uada_loss": float(loss.detach()), "learning_rate": lr,
               **step_metrics(before, after, gradient)}
        require(all(np.isfinite(v) for v in row.values() if isinstance(v, float)), "Nonfinite step monitor")
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows += 1; self.last = [outer, inner]
        return row

    def should_probe(self, outer, inner):
        settings = experiment()
        return outer in settings["diagnostic_outers"] and inner in settings["diagnostic_inners"]

    def probe(self, engine, record, inputs, proprio, target, seed, diverse,
              outer, inner, before, after, gradient, action_before, dino_before):
        with preserve_rng(), torch.no_grad():
            clean, clean_features, _ = engine.forward(inputs, proprio)
            pixels, _ = engine.pixels(record, after, seed, diverse=diverse)
            changed, changed_features, _ = engine.forward(inputs, proprio, pixels)
            pixels_png, _ = engine.pixels(record, quantized(after), seed, diverse=diverse)
            changed_png, _, _ = engine.forward(inputs, proprio, pixels_png)
            fixed_before, _ = engine.pixels(record, quantized(before), seed, diverse=False)
            fixed_after, _ = engine.pixels(record, quantized(after), seed, diverse=False)
            deployment_before, _, _ = engine.forward(inputs, proprio, fixed_before)
            deployment_after, _, _ = engine.forward(inputs, proprio, fixed_after)
            l_before = float(action_loss(action_before, target, "uada", self.cfg))
            l_after = float(action_loss(changed, target, "uada", self.cfg))
            report = {"signature": self.signature, "arm": self.cfg["arm"], "outer": outer, "inner": inner,
                      "uid": record["uid"], "eot_seed": seed, "diverse": diverse,
                      "native_uada_before": l_before, "native_uada_after": l_after,
                      "native_uada_change": l_after - l_before,
                      "png_uada_after": float(action_loss(changed_png, target, "uada", self.cfg)),
                      "png_vs_training_action_l2": float((changed_png - changed).norm()),
                      "fixed_png_uada_before": float(action_loss(deployment_before, target, "uada", self.cfg)),
                      "fixed_png_uada_after": float(action_loss(deployment_after, target, "uada", self.cfg)),
                      "dino_before_relative_l2": float((dino_before - clean_features["dino"]).norm() / clean_features["dino"].norm().clamp_min(1e-12)),
                      "dino_after_relative_l2": float((changed_features["dino"] - clean_features["dino"]).norm() / clean_features["dino"].norm().clamp_min(1e-12)),
                      "action_axis_rms_before": (action_before - clean).square().mean((0, 1)).sqrt().cpu().tolist(),
                      "action_axis_rms_after": (changed - clean).square().mean((0, 1)).sqrt().cpu().tolist()}
            tensor = {"rgb_before": before.cpu(), "rgb_after": after.detach().cpu(), "native_gradient": gradient.cpu(),
                      "action_before": action_before.cpu(), "action_after": changed.cpu(), "clean_action": clean.cpu(),
                      "wrong_target": target.cpu(), "dino_before": dino_before.cpu(),
                      "dino_after": changed_features["dino"].cpu(), "dino_clean": clean_features["dino"].cpu(),
                      "deployment_action_before": deployment_before.cpu(), "deployment_action_after": deployment_after.cpu(),
                      "source_observation": record["observation"], "report": report}
            stem = self.folder / f"probe_{outer:04d}_{inner:02d}"
            atomic_torch(tensor, stem.with_suffix(".pt")); atomic_json(report, stem.with_suffix(".json"))

    def flush(self):
        self.stream.flush()

    def close(self):
        if self.stream.closed:
            return
        self.stream.close()
        atomic_json({"signature": self.signature, "rows": self.rows, "last_step": self.last,
                     "log_sha256": sha256_file(self.path), "rng_preserving_probes": True}, self.folder / "MONITOR.json")

def collect_steps(folder):
    """Newest attempt replaces only replayed step keys; old logs remain intact."""
    import pandas as pd
    rows = {}
    for path in sorted((folder / "attempts").glob("*/INNER_STEPS.jsonl")):
        text = path.read_text()
        lines = text.splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A killed worker can leave a partial final line. No completed
                # final run can omit that step because coverage is checked.
                require(index == len(lines) - 1 and not text.endswith("\n"), "Corrupted completed monitoring line")
                continue
            row["monitor_file"] = str(path.relative_to(ROOT))
            rows[(row["outer"], row["inner"])] = row
    return pd.DataFrame([rows[key] for key in sorted(rows)])

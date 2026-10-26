"""Bounded native probes with RNG preservation; no optimizer or policy mutation."""
from __future__ import annotations
from contextlib import contextmanager
import json
import random
import time
import numpy as np
import torch
from common import ROOT, atomic_json, atomic_torch, require
from meta import cosine

def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])

@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)

def comparison(value, clean):
    d = value["dino"].float() - clean["dino"].float()
    a = value["action"].float() - clean["action"].float()
    return {"uada_loss": value["loss"], "dino_relative_l2": float(d.norm()/clean["dino"].float().norm().clamp_min(1e-20)),
            "action_l2": float(a.norm()), "action_axis_rms": a.square().mean((0, 1)).sqrt().tolist(),
            "gripper_sign_flip_fraction": float((value["action"][..., 6].ge(0) != clean["action"][..., 6].ge(0)).float().mean())}

class Monitor:
    def __init__(self, folder, signature, cfg):
        self.signature, self.cfg = signature, cfg
        self.folder = folder / "attempts" / str(time.time_ns())
        self.folder.mkdir(parents=True, exist_ok=False)
        self.stream = (self.folder / "INNER_STEPS.jsonl").open("a", encoding="utf-8")

    def write(self, row):
        self.stream.write(json.dumps({"signature": self.signature, **row}, allow_nan=False) + "\n")
        self.stream.flush()

    def capture(self, engine, current, successor, seed, tensors, outer, inner):
        records = {"current": current}
        if successor is not None:
            records["successor"] = successor
        result, payload = {}, {}
        with preserve_rng():
            for name, record in records.items():
                clean = engine.inspect(record, None, seed)
                before = engine.inspect(record, tensors["before"], seed)
                after = engine.inspect(record, tensors["after"], seed)
                rgb = {k: tensors[k].mul(255).round().div(255) for k in ("before", "after")}
                png_before = engine.inspect(record, rgb["before"], seed)
                png_after = engine.inspect(record, rgb["after"], seed)
                fixed_before = engine.inspect(record, rgb["before"], seed, fixed=True)
                fixed_after = engine.inspect(record, rgb["after"], seed, fixed=True)
                result[name] = {"uid": record["uid"], "before": comparison(before, clean), "after": comparison(after, clean),
                    "native_loss_change": after["loss"]-before["loss"], "png_loss_change": png_after["loss"]-png_before["loss"],
                    "fixed_png_loss_change": fixed_after["loss"]-fixed_before["loss"],
                    "png_vs_training_action_l2": float((png_after["action"]-after["action"]).float().norm()),
                    "fixed_png_before": comparison(fixed_before, clean), "fixed_png_after": comparison(fixed_after, clean)}
                payload[name] = {"clean": clean, "before": before, "after": after,
                                 "png_before": png_before, "png_after": png_after,
                                 "fixed_png_before": fixed_before, "fixed_png_after": fixed_after}
            if successor is not None and tensors["query_gradient"] is not None:
                at_p = engine.gradients(successor, tensors["before"].detach().requires_grad_(True), seed, split=False)
                result["query_at_P_vs_Pprime"] = {"loss_at_P": at_p["loss"],
                    "gradient_cosine": cosine(at_p["full"], tensors["query_gradient"]),
                    "gradient_difference_l2": float((at_p["full"]-tensors["query_gradient"]).norm())}
        stem = self.folder / f"probe_{outer:04d}_{inner:02d}"
        payload["rgb_and_gradients"] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
        atomic_torch(payload, stem.with_suffix(".pt"))
        atomic_json({"signature": self.signature, "outer": outer, "inner": inner, "seed": seed,
                     "rng_preserved": True, "evidence": result}, stem.with_suffix(".json"))

    def close(self):
        self.stream.close()

def collect_steps(folder):
    import pandas as pd
    records = {}
    for path in sorted((folder / "attempts").glob("*/INNER_STEPS.jsonl")):
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Only an unterminated final line can be interrupted safely.
                    require(not line.endswith("\n") and not stream.read(), "Corrupt completed monitoring record")
                    break
                records[(row["outer"], row["inner"])] = row
    return pd.DataFrame([records[k] for k in sorted(records)])

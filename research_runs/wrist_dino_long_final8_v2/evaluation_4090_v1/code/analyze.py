"""CPU validation and paired task/state statistics. Fresh clean qualifies CASR."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from eval_common import (ROOT, METHODS, SUITES, prepared, source_signature, atomic_json, atomic_parquet, config, deployment_conditions,
                    load_json, require, run_signature, sha256_file)


def task_casr(frame, condition):
    clean = frame.loc[frame.condition.eq("clean"), ["task_id", "state_index", "success"]].rename(columns={"success": "clean_success"})
    attacked = frame.loc[frame.condition.eq(condition), ["task_id", "state_index", "failure"]]
    pairs = attacked.merge(clean, on=["task_id", "state_index"], validate="one_to_one")
    means = pairs.loc[pairs.clean_success.eq(1)].groupby("task_id").failure.mean()
    return float(means.mean()), pairs


def paired_bootstrap(pairs, draws=10000, seed=20261009):
    rng = np.random.default_rng(seed)
    tasks = sorted(pairs.task_id.unique())
    values = [pairs.loc[pairs.task_id.eq(t), ["clean_success", "value"]].to_numpy(float) for t in tasks]
    estimates = []
    for _ in range(draws):
        results = []
        for index in rng.integers(0, len(values), len(values)):
            group = values[index]
            sample = group[rng.integers(0, len(group), len(group))]
            valid = sample[:, 0] == 1
            if valid.any():
                results.append(sample[valid, 1].mean())
        if results:
            estimates.append(np.mean(results))
    return np.quantile(estimates, [.025, .975]).tolist()


def analyze():
    prepared()
    signature, frames, traces = run_signature(), [], []
    for suite in SUITES:
        for task in range(10):
            folder = ROOT / "evaluation" / suite
            status = load_json(folder / f"task_{task:02d}_STATUS.json")
            rp, tp = folder / f"task_{task:02d}_rollouts.parquet", folder / f"task_{task:02d}_traces.parquet"
            require(status["status"] == "COMPLETE" and status["signature"] == signature
                    and status["rollouts_sha256"] == sha256_file(rp) and status["traces_sha256"] == sha256_file(tp),
                    "Evaluation shard incomplete or changed")
            frames.append(pd.read_parquet(rp)); traces.append(pd.read_parquet(tp))
    raw = pd.concat(frames, ignore_index=True)
    require(len(raw) == 10000 and not raw.duplicated(["suite", "task_id", "state_index", "condition"]).any(), "10000-episode grid incomplete")
    require(raw.success.isin([0, 1]).all() and raw.failure.eq(1 - raw.success).all()
            and raw.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(), "Unpaired/invalid outcomes")
    summaries, contrasts = [], []
    comparisons = [("uada_dino", "uada_original"), ("uada_dino_freq", "uada_dino"),
                   ("tma_xyz_dino", "tma_xyz"), ("tma_xyz_dino_freq", "tma_xyz_dino"),
                   ("tma_xyz_dino_teacher", "tma_xyz_dino"), ("uada_dino_teacher", "uada_dino")]
    for suite in SUITES:
        subset = raw.loc[raw.suite.eq(suite)]
        clean_success = int(subset.loc[subset.condition.eq("clean")].success.sum())
        for condition in deployment_conditions()[1:]:
            point, pairs = task_casr(subset, condition)
            pairs = pairs.assign(value=pairs.failure)
            ci = paired_bootstrap(pairs)
            method, outer = condition.split("@")
            summaries.append({"suite": suite, "method": method, "outer": int(outer), "task_equal_casr": point,
                              "ci_low": ci[0], "ci_high": ci[1], "fresh_clean_success": clean_success,
                              "attack_failure_rate": float(subset.loc[subset.condition.eq(condition)].failure.mean())})
        for outer in config()["evaluation_checkpoints"]:
            for left, right in comparisons:
                lp, a = task_casr(subset, f"{left}@{outer}")
                rp, b = task_casr(subset, f"{right}@{outer}")
                joint = a.merge(b[["task_id", "state_index", "failure"]], on=["task_id", "state_index"], suffixes=("_a", "_b"))
                joint["value"] = joint.failure_a - joint.failure_b
                ci = paired_bootstrap(joint)
                contrasts.append({"suite": suite, "outer": outer, "left": left, "right": right,
                                  "casr_difference": lp - rp, "ci_low": ci[0], "ci_high": ci[1]})
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    atomic_parquet(raw, output / "FINAL8_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(traces, ignore_index=True), output / "FINAL8_TRACES.parquet")
    pd.DataFrame(summaries).to_csv(output / "FINAL8_SUMMARY.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "FINAL8_CONTRASTS.csv", index=False)
    atomic_json({"status": "COMPLETE", "signature": signature, "episodes": 10000, "methods": 8,
                 "source_suite": "libero_10", "independent_patch_training_seeds": 1,
                 "source_training_signature": source_signature(), "evaluation_gpu_type": "4090",
                 "note": "CASR fresh-clean qualified; Object/Spatial/Goal transfer, Long source; viewed development grid"},
                output / "STATUS.json")
    print(pd.DataFrame(summaries).pivot_table(index=["method", "outer"], columns="suite", values="task_equal_casr").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stats",), required=True)
    parser.parse_args()
    analyze()

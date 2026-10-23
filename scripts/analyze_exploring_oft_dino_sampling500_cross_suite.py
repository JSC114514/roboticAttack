#!/usr/bin/env python3
"""Task-aware, suite-separated paired CASR analysis for DINO sampling-500."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "oft_ramp_recovery_aware/scripts"))
from common import atomic_json, atomic_parquet, require, sha256_file  # noqa: E402
from statistics import paired_contrast_bootstrap, task_episode_bootstrap  # noqa: E402
from run_exploring_oft_dino_sampling500_cross_suite import (  # noqa: E402
    BASE, CONTROL, EXPERIMENT_ID, GRAY, GUIDED, CONDITIONS,
    frozen_clean, load_config, validate_result, validate_sources,
)

PREFIX = "DINO_SAMPLING500_XSUITE"
CONTRASTS = (("GUIDED_MINUS_SITE_UADA", GUIDED, CONTROL),
             ("SITE_UADA_MINUS_OLD_UADA", CONTROL, BASE),
             ("GUIDED_MINUS_OLD_UADA", GUIDED, BASE),
             ("GUIDED_MINUS_GRAY", GUIDED, GRAY),
             ("SITE_UADA_MINUS_GRAY", CONTROL, GRAY))


def analyze(config: Path) -> dict:
    cfg = load_config(config)
    sources = validate_sources(cfg)
    output = Path(cfg["output_root"])
    shard_rollouts, shard_traces = [], []
    for suite in cfg["suites"]:
        for task_id in range(10):
            rpath = output / suite / f"task_{task_id:02d}_rollouts.parquet"
            tpath = output / suite / f"task_{task_id:02d}_traces.parquet"
            require(rpath.is_file() and tpath.is_file(), f"Missing shard: {suite} {task_id}")
            rollouts, traces = pd.read_parquet(rpath), pd.read_parquet(tpath)
            validate_result(rollouts, traces, suite, task_id, cfg, sources)
            shard_rollouts.append(rollouts)
            shard_traces.append(traces)
    all_rollouts = pd.concat(shard_rollouts, ignore_index=True)
    all_traces = pd.concat(shard_traces, ignore_index=True)
    require(len(all_rollouts) == 1600 and not all_rollouts.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(),
        "Frozen four-suite episode grid drift")
    summary_frames, task_frames, contrast_rows, eligible_frames = [], [], [], []
    denominators = {}
    for suite_index, suite in enumerate(cfg["suites"]):
        clean = frozen_clean(cfg, suite)
        subset = all_rollouts.loc[all_rollouts.suite == suite]
        linked = subset.merge(clean[["task_id", "state_index", "success", "rollout_seed"]],
                              on=["task_id", "state_index"], suffixes=("", "_clean"),
                              validate="many_to_one")
        require(len(linked) == len(subset)
                and (linked.frozen_clean_success == linked.success_clean).all()
                and (linked.rollout_seed == linked.rollout_seed_clean).all(),
                f"Frozen clean linkage drift: {suite}")
        eligible = subset.loc[subset.frozen_clean_success == 1].copy()
        eligible["casr"] = eligible.failure.astype(float)
        denominator = int(clean.success.sum())
        require(len(eligible) == denominator * len(CONDITIONS),
                f"Clean-success denominator drift: {suite}")
        denominators[suite] = denominator
        eligible_frames.append(eligible)
        summary = task_episode_bootstrap(
            eligible, "casr", "condition", 10_000, int(cfg["bootstrap_seed"]) + suite_index)
        summary.insert(0, "suite", suite)
        summary_frames.append(summary)
        per_task = (eligible.groupby(["task_id", "condition"], as_index=False)
                    .agg(clean_success_denominator=("state_index", "nunique"),
                         casr=("casr", "mean"), attacked_success=("success", "mean"),
                         mean_episode_steps=("episode_steps", "mean")))
        per_task.insert(0, "suite", suite)
        task_frames.append(per_task)
        for contrast_index, (role, left, right) in enumerate(CONTRASTS):
            result = paired_contrast_bootstrap(
                eligible, "casr", "condition", left, right, 10_000,
                int(cfg["bootstrap_seed"]) + 10 + 10 * suite_index + contrast_index)
            require(result["paired_episodes"] == denominator,
                    f"Paired contrast grid drift: {suite} {role}")
            contrast_rows.append({"suite": suite, "contrast_role": role, **result})
    atomic_parquet(all_rollouts, output / f"{PREFIX}_ROLLOUTS.parquet")
    atomic_parquet(all_traces, output / f"{PREFIX}_TRACES.parquet")
    atomic_parquet(pd.concat(eligible_frames, ignore_index=True), output / f"{PREFIX}_ELIGIBLE.parquet")
    summary_path = output / f"{PREFIX}_PER_SUITE.csv"
    contrast_path = output / f"{PREFIX}_CONTRASTS.csv"
    pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)
    pd.concat(task_frames, ignore_index=True).to_csv(output / f"{PREFIX}_PER_TASK.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(contrast_path, index=False)
    status = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
              "new_attack_episodes": len(all_rollouts),
              "eligible_attack_rows": sum(len(frame) for frame in eligible_frames),
              "clean_success_denominators": denominators,
              "conditions": CONDITIONS,
              "patch_sha256": {method: digest for method, (_, digest) in sources.items()},
              "per_suite_sha256": sha256_file(summary_path),
              "contrasts_sha256": sha256_file(contrast_path),
              "claim_boundary": (
                  "The primary DINO claim is GUIDED_MINUS_SITE_UADA at identical placement. "
                  "GRAY is an occlusion control. Object is the training suite; Spatial and Goal "
                  "are zero-shot transfer; Long is secondary. No pooled four-suite headline.")}
    atomic_json(status, output / "STATUS.json")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

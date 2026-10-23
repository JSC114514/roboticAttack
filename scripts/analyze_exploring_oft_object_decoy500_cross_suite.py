#!/usr/bin/env python3
"""Suite-separated paired CASR analysis for the Object-Decoy-500 evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "oft_ramp_recovery_aware" / "scripts"))

from common import atomic_json, atomic_parquet, require, sha256_file  # noqa: E402
from statistics import paired_contrast_bootstrap, task_episode_bootstrap  # noqa: E402
from run_exploring_oft_object_decoy500_cross_suite import (  # noqa: E402
    BASELINE, EXPERIMENT_ID, R03, R10, frozen_clean, load_config, validate_result,
    validate_sources,
)

PREFIX = "OBJECT_DECOY500_XSUITE"
CONTRASTS = (
    ("R03_MINUS_UADA500", R03, BASELINE),
    ("R10_MINUS_UADA500", R10, BASELINE),
    ("R10_MINUS_R03", R10, R03),
)


def analyze(config: Path) -> None:
    cfg = load_config(config)
    validate_sources(cfg)
    output = Path(cfg["output_root"])
    shard_rollouts, shard_traces = [], []
    for suite in cfg["suites"]:
        for task_id in range(int(cfg["task_count"])):
            rpath = output / suite / f"task_{task_id:02d}_rollouts.parquet"
            tpath = output / suite / f"task_{task_id:02d}_traces.parquet"
            require(rpath.is_file() and tpath.is_file(), f"Missing shard: {suite} {task_id}")
            rollouts, traces = pd.read_parquet(rpath), pd.read_parquet(tpath)
            validate_result(rollouts, traces, suite, task_id, cfg)
            shard_rollouts.append(rollouts)
            shard_traces.append(traces)
    all_rollouts = pd.concat(shard_rollouts, ignore_index=True)
    all_traces = pd.concat(shard_traces, ignore_index=True)
    require(len(all_rollouts) == 1200 and not all_rollouts.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(),
        "Frozen episode grid drift")

    summary_frames, contrast_rows, task_frames, eligible_frames = [], [], [], []
    denominators = {}
    for suite_index, suite in enumerate(cfg["suites"]):
        clean = frozen_clean(cfg, suite)
        subset = all_rollouts.loc[all_rollouts.suite == suite].copy()
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
        require(len(eligible) == denominator * len(cfg["conditions"]),
                f"Clean-success denominator drift: {suite}")
        denominators[suite] = denominator
        eligible_frames.append(eligible)
        summary = task_episode_bootstrap(
            eligible, "casr", "condition", 10_000, int(cfg["bootstrap_seed"]) + suite_index)
        summary.insert(0, "suite", suite)
        summary_frames.append(summary)
        tasks = (eligible.groupby(["task_id", "condition"], as_index=False)
                 .agg(clean_success_denominator=("state_index", "nunique"),
                      casr=("casr", "mean"), attacked_success=("success", "mean"),
                      mean_episode_steps=("episode_steps", "mean")))
        tasks.insert(0, "suite", suite)
        task_frames.append(tasks)
        for contrast_index, (role, left, right) in enumerate(CONTRASTS):
            contrast = paired_contrast_bootstrap(
                eligible, "casr", "condition", left, right, 10_000,
                int(cfg["bootstrap_seed"]) + 10 + 10 * suite_index + contrast_index)
            require(contrast["paired_episodes"] == denominator,
                    f"Contrast pairing drift: {suite} {role}")
            contrast_rows.append({"suite": suite, "contrast_role": role, **contrast})

    eligible_all = pd.concat(eligible_frames, ignore_index=True)
    atomic_parquet(all_rollouts, output / f"{PREFIX}_ROLLOUTS.parquet")
    atomic_parquet(all_traces, output / f"{PREFIX}_TRACES.parquet")
    atomic_parquet(eligible_all, output / f"{PREFIX}_ELIGIBLE.parquet")
    summary_path = output / f"{PREFIX}_PER_SUITE.csv"
    contrast_path = output / f"{PREFIX}_CONTRASTS.csv"
    pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)
    pd.concat(task_frames, ignore_index=True).to_csv(output / f"{PREFIX}_PER_TASK.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(contrast_path, index=False)
    status = {
        "status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
        "new_attack_episodes": len(all_rollouts), "eligible_attack_rows": len(eligible_all),
        "frozen_clean_success_denominators": denominators,
        "conditions": cfg["conditions"],
        "patch_sha256": {method: cfg["patch_sources"][method]["sha256"]
                         for method in cfg["conditions"]},
        "per_suite_sha256": sha256_file(summary_path),
        "contrasts_sha256": sha256_file(contrast_path),
        "claim_boundary": (
            "Object is the training suite; Spatial and Goal are zero-shot transfer; "
            "Long is reported separately. CASR is conditioned on frozen clean success; "
            "no pooled cross-suite headline."
        ),
    }
    atomic_json(status, output / "STATUS.json")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    analyze(Path(args.config).resolve())


if __name__ == "__main__":
    main()

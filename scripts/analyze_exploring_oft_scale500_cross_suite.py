#!/usr/bin/env python3
"""Suite-separated paired CASR analysis; no pooled cross-suite headline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "oft_ramp_recovery_aware" / "scripts"))

from common import atomic_json, atomic_parquet, load_json, require, sha256_file  # noqa: E402
from statistics import paired_contrast_bootstrap, task_episode_bootstrap  # noqa: E402

from run_exploring_oft_scale500_cross_suite import frozen_clean, validate  # noqa: E402


CONTRASTS = (
    ("DINO_CF_MINUS_UADA", "exploring_oft_dino_cf", "exploring_oft_uada"),
    ("WORST_CHUNK_MINUS_DINO_CF", "exploring_oft_dino_cf_worst_chunk", "exploring_oft_dino_cf"),
    ("DINO_RESTORE_MINUS_UADA", "exploring_oft_dino_restore", "exploring_oft_uada"),
    ("DINO_CF_MINUS_SIGLIP_CF", "exploring_oft_dino_cf", "exploring_oft_siglip_cf"),
)


def analyze(config: Path) -> None:
    cfg = load_json(config)
    output = Path(cfg["output_root"])
    shard_rollouts, shard_traces = [], []
    for suite in cfg["suites"]:
        for task_id in range(int(cfg["task_count"])):
            rpath = output / suite / f"task_{task_id:02d}_rollouts.parquet"
            tpath = output / suite / f"task_{task_id:02d}_traces.parquet"
            require(rpath.is_file() and tpath.is_file(), f"Missing cross-suite shard: {suite} {task_id}")
            rollouts, traces = pd.read_parquet(rpath), pd.read_parquet(tpath)
            validate(rollouts, traces, suite, task_id, cfg)
            shard_rollouts.append(rollouts)
            shard_traces.append(traces)
    all_rollouts = pd.concat(shard_rollouts, ignore_index=True)
    all_traces = pd.concat(shard_traces, ignore_index=True)
    expected = len(cfg["suites"]) * int(cfg["task_count"]) * len(cfg["state_indices"]) * len(cfg["conditions"])
    require(len(all_rollouts) == expected == 2000, "Frozen 2000-episode grid drift")
    require(not all_rollouts.duplicated(["suite", "task_id", "state_index", "condition"]).any(),
            "Duplicate combined cross-suite rollout")
    summary_frames, contrast_rows, task_frames = [], [], []
    eligible_frames = []
    denominators = {}
    for suite_index, suite in enumerate(cfg["suites"]):
        clean = frozen_clean(cfg, suite)
        subset = all_rollouts.loc[all_rollouts.suite == suite].copy()
        check = subset.merge(clean[["task_id", "state_index", "success", "rollout_seed"]],
                             on=["task_id", "state_index"], suffixes=("", "_clean"), validate="many_to_one")
        require(len(check) == len(subset)
                and (check.frozen_clean_success == check.success_clean).all()
                and (check.rollout_seed == check.rollout_seed_clean).all(),
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
    atomic_parquet(all_rollouts, output / "SCALE500_XSUITE_ROLLOUTS.parquet")
    atomic_parquet(all_traces, output / "SCALE500_XSUITE_TRACES.parquet")
    atomic_parquet(eligible_all, output / "SCALE500_XSUITE_ELIGIBLE.parquet")
    summary_path = output / "SCALE500_XSUITE_PER_SUITE.csv"
    contrast_path = output / "SCALE500_XSUITE_CONTRASTS.csv"
    pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)
    pd.concat(task_frames, ignore_index=True).to_csv(output / "SCALE500_XSUITE_PER_TASK.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(contrast_path, index=False)
    status = {
        "status": "COMPLETE", "experiment_id": cfg["experiment_id"],
        "new_attack_episodes": len(all_rollouts), "eligible_attack_rows": len(eligible_all),
        "frozen_clean_success_denominators": denominators,
        "conditions": cfg["conditions"], "manual_visual_review": cfg["manual_visual_review"],
        "per_suite_sha256": sha256_file(summary_path),
        "contrasts_sha256": sha256_file(contrast_path),
        "claim_boundary": "Object source; Spatial/Goal zero-shot; Long mandatory secondary; no pooled headline",
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

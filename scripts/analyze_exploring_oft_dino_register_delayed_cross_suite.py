#!/usr/bin/env python3
"""Task/state paired bootstrap for OFT-DINO-REGISTER-DELAYED2000-V2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dino_register_delayed_common import (
    CHECKPOINTS, FINAL_OUTER, CONDITIONS, CONFIG, DENOMINATORS, METHODS, REFERENCES, config,
    digest_json, reference_patches, require_preparation,
)
from run_exploring_oft_dino_register_delayed_cross_suite import (
    EXPERIMENT_ID, KEY, load_config, patches, validate_new_shard, validate_runtime,
)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet, require, sha256_file

PAIRS = [
    (f"{method}_minus_{reference}", method, reference)
    for method in METHODS for reference in REFERENCES
] + [
    ("register_minus_spatial", "register_late_g10", "spatial_late_g10"),
    ("projected_minus_register", "register_late_projected_g10", "register_late_g10"),
    ("projected_minus_spatial", "register_late_projected_g10", "spatial_late_g10"),
    ("uniform_axis0_minus_original_uada", "uniform_axis0", "original_uada"),
]


def contrast_vectors():
    vectors = {}
    for name, left, right in PAIRS:
        value = np.zeros(len(CONDITIONS))
        value[CONDITIONS.index(left)] = 1
        value[CONDITIONS.index(right)] = -1
        vectors[name] = (value, left, right)
    return vectors


def hierarchical_samples(eligible, draws, rng):
    require(draws >= 1 and not eligible.duplicated(["task_id", "state_index", "condition"]).any(),
            "Duplicate paired bootstrap key")
    wide = eligible.pivot(index=["task_id", "state_index"], columns="condition", values="casr")
    wide = wide.reindex(columns=list(CONDITIONS))
    require(not wide.isna().any().any()
            and np.isfinite(wide.to_numpy(float)).all()
            and wide.isin([0, 1]).all().all(),
            "Unpaired/nonfinite/binary CASR data")
    tasks = sorted(wide.index.get_level_values("task_id").unique())
    require(tasks == list(range(10)), "All ten tasks require clean-success states")
    arrays = [wide.xs(task, level="task_id").to_numpy(float) for task in tasks]
    selected = rng.integers(0, 10, size=(draws, 10))
    samples = np.zeros((draws, len(CONDITIONS)))
    for position in range(10):
        for task, values in enumerate(arrays):
            ids = np.flatnonzero(selected[:, position] == task)
            states = rng.integers(0, len(values), size=(len(ids), len(values)))
            samples[ids] += values[states].mean(axis=1) / 10
    per_task = np.stack([values.mean(axis=0) for values in arrays])
    return per_task.mean(axis=0), samples, per_task


def paired_statistics(eligible, draws=10000, seed=20260907, checkpoint_outer=2000):
    require(checkpoint_outer in CHECKPOINTS and eligible.checkpoint_outer.eq(checkpoint_outer).all(),
            "Bootstrap must use one matched budget at a time")
    rng = np.random.default_rng(seed)
    points, samples, tasks = {}, {}, {}
    for suite in DENOMINATORS:
        point, sample, per_task = hierarchical_samples(eligible.loc[eligible.suite.eq(suite)], draws, rng)
        points[suite], samples[suite], tasks[suite] = point, sample, per_task
    groups = {suite: [suite] for suite in DENOMINATORS}
    groups["object_spatial_goal_macro"] = ["libero_object", "libero_spatial", "libero_goal"]
    groups["spatial_goal_macro"] = ["libero_spatial", "libero_goal"]
    groups["four_suite_macro"] = list(DENOMINATORS)
    summaries, comparisons = [], []
    for group, suites in groups.items():
        point = np.mean([points[s] for s in suites], axis=0)
        sample = np.mean([samples[s] for s in suites], axis=0)
        task_values = np.concatenate([tasks[s] for s in suites], axis=0)
        for index, condition in enumerate(CONDITIONS):
            low, high = np.percentile(sample[:, index], [2.5, 97.5])
            summaries.append({
                "checkpoint_outer": checkpoint_outer,
                "suite_or_endpoint": group,
                "condition": condition,
                "task_equal_casr": float(point[index]),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "clean_success_states": sum(DENOMINATORS[s] for s in suites),
                "tasks": 10 * len(suites),
                "suite_equal": len(suites) > 1,
            })
        for name, (vector, left, right) in contrast_vectors().items():
            delta, distribution = task_values @ vector, sample @ vector
            low, high = np.percentile(distribution, [2.5, 97.5])
            comparisons.append({
                "checkpoint_outer": checkpoint_outer,
                "suite_or_endpoint": group,
                "contrast": name,
                "left": left,
                "right": right,
                "difference_left_minus_right": float(point @ vector),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "positive_tasks": int((delta > 0).sum()),
                "negative_tasks": int((delta < 0).sum()),
                "tied_tasks": int((delta == 0).sum()),
                "tasks": len(delta),
                "paired_clean_success_states": sum(DENOMINATORS[s] for s in suites),
            })
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def validate_raw(frame, cfg):
    expected_rows = len(CHECKPOINTS) * len(CONDITIONS) * 400
    require(len(frame) == expected_rows
            and set(frame.condition) == set(CONDITIONS)
            and set(frame.suite) == set(DENOMINATORS)
            and set(frame.checkpoint_outer) == set(CHECKPOINTS)
            and not frame.duplicated(KEY).any(),
            "Incomplete/duplicate raw register grid")
    require(frame.groupby(["checkpoint_outer", "suite", "condition"]).size().eq(100).all()
            and frame.groupby(["checkpoint_outer", "suite", "task_id", "condition"]).size().eq(10).all()
            and frame.success.isin([0, 1]).all()
            and frame.failure.eq(1 - frame.success).all()
            and frame.frozen_clean_success.isin([0, 1]).all()
            and np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(),
            "Raw coverage/success/finite data failed")
    require(frame.groupby(["suite", "task_id", "state_index"]).frozen_clean_success.nunique().eq(1).all()
            and frame.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(),
            "Unpaired clean qualification or seeds")
    for suite, expected in DENOMINATORS.items():
        eligible = frame.loc[frame.suite.eq(suite) & frame.frozen_clean_success.eq(1)]
        require(eligible.groupby(["checkpoint_outer", "condition"]).size().eq(expected).all(),
                f"Frozen eligible count changed: {suite}")
    return frame.loc[frame.frozen_clean_success.eq(1)].assign(casr=lambda f: f.failure.astype(float)).copy()


def write_closeout(output, train, summary, contrasts, training, costs):
    """Readable result/method record; never declare success from attention alone."""
    lines = [
        "# OFT-DINO-REGISTER-DELAYED2000-V2 实验闭环", "",
        "状态：评测完成，待最终方法选择。以下是实际闭环结果；CASR 越高表示攻击越强。", "",
        "## 方法", "",
        "三组均从 seed42 随机 3×50×50 RGB 开始；Exploring 的只读副本提供初始化、",
        "AdamW、cosine 和 RGB 更新，保留 OFT dim0 opposite-boundary MSE 主目标。",
        "仅 Object clean300/replay258 训练，单腕部补丁；Front 不变。前250轮仅动作目标，",
        "251轮起加入10%辅助梯度。register 目标是 block8/16 的四个 register query",
        "读取实际补丁 footprint；spatial 组是同权重非补丁 dense query 控制。",
        "projected 组仅去除辅助梯度中反对动作梯度的分量，缩放仍用投影前范数。",
        "测试使用正常 OFT 前向，没有辅助 hook 或特征替换。", "",
        "## 同预算结果", "",
        "| 轮次 | 方法 | Object | Spatial | Goal | Long（辅助） | 三suite主指标 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    endpoints = [*DENOMINATORS, "object_spatial_goal_macro"]
    for outer in CHECKPOINTS:
        for condition in CONDITIONS:
            values = []
            for endpoint in endpoints:
                row = summary.loc[summary.checkpoint_outer.eq(outer)
                                  & summary.condition.eq(condition)
                                  & summary.suite_or_endpoint.eq(endpoint)]
                require(len(row) == 1, "Closeout summary grid incomplete")
                values.append(f"{float(row.iloc[0].task_equal_casr) * 100:.2f}%")
            lines.append(f"| {outer} | {condition} | " + " | ".join(values) + " |")
    lines.extend(["", "主判断固定使用2000轮，不挑更有利的1000轮替代。Long完整报告但不作为主机制结论。",
                  "", "## 2000轮三suite配对差值", "",
                  "| 比较 | 差值（pp） | 95%配对区间（pp） | 正/负/平任务数 |",
                  "|---|---:|---|---|"])
    for row in contrasts.loc[contrasts.checkpoint_outer.eq(FINAL_OUTER)
                             & contrasts.suite_or_endpoint.eq("object_spatial_goal_macro")].itertuples():
        lines.append(f"| {row.contrast} | {row.difference_left_minus_right * 100:+.2f} | "
                     f"[{row.ci95_low * 100:+.2f}, {row.ci95_high * 100:+.2f}] | "
                     f"{row.positive_tasks}/{row.negative_tasks}/{row.tied_tasks} |")
    lines.extend(["", "## 训练是否变化", "",
                  "| 轮次 | 方法 | 主辅梯度平均cos | 冲突比例 | 实际辅助比例 | RGB对500轮MAE |",
                  "|---:|---|---:|---:|---:|---:|"])
    for row in training.itertuples():
        lines.append(f"| {row.checkpoint_outer} | {row.condition} | {row.mean_raw_gradient_cosine:.4f} | "
                     f"{row.conflict_fraction:.4f} | {row.mean_effective_aux_ratio:.4f} | "
                     f"{row.rgb_mae_from_outer500:.6f} |")
    lines.extend(["", "## 耗时", "",
                  "| 轮次 | 方法 | 训练到该轮（小时） | 测试GPU时间之和（小时） |",
                  "|---:|---|---:|---:|"])
    for row in costs:
        elapsed = row.get("cumulative_training_elapsed_s", row.get("recorded_training_elapsed_s", 0.))
        lines.append(f"| {row['checkpoint_outer']} | {row['condition']} | {elapsed / 3600:.2f} | "
                     f"{row['evaluation_elapsed_s'] / 3600:.2f} |")
    lines.extend(["", "旧基线只复用同预算权重并重测，未重新训练。并行墙钟时间不等于GPU时间之和。", "",
                  "## 解释边界", "",
                  "仅一个训练seed；任务→初态配对bootstrap不代表训练seed间方差。目标suite此前已用于开发。",
                  "优于原始UADA不能单独归因DINO，因为已有replay/EOT也不同；DINO增益应看相对uniform。",
                  "注意力上升、梯度方向保护与闭环攻击增强必须分别报告。完整分suite区间、",
                  "每task结果和两预算对比见同目录SUMMARY/CONTRASTS/PER_TASK文件。", "",
                  "本轮按计划作为OFT最终方法筛选；不自动扩展诊断、跨模型或实机作业。", ""])
    destination = Path(output) / "OFT_CLOSEOUT.md"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines))
    os.replace(temporary, destination)
    return destination


def analyze(draws=10000):
    train, cfg = config(), load_config()
    require(draws == train["bootstrap_draws"] == 10000, "Formal bootstrap is frozen at 10000 draws")
    preparation = require_preparation(train)
    _, hashes = patches(cfg, "cpu")
    output = Path(cfg["output_root"])
    require(json.loads((output / "EVAL_CONFIG.json").read_text()) == cfg,
            "Final evaluation config changed")
    parts, trace_parts, artifact_hashes = [], [], {}
    for task in range(10):
        receipt = json.loads((output / f"task_{task:02d}_STATUS.json").read_text())
        require(receipt["status"] == "COMPLETE"
                and receipt["task_id"] == task
                and receipt["config_signature"] == digest_json(cfg)
                and receipt["patches"] == hashes,
                "Task receipt incomplete or patch identity changed")
        for suite in cfg["suites"]:
            rp, tp = [output / suite / f"task_{task:02d}_{kind}.parquet" for kind in ("rollouts", "traces")]
            require(sha256_file(rp) == receipt["suites"][suite]["rollouts_sha256"]
                    and sha256_file(tp) == receipt["suites"][suite]["traces_sha256"],
                    "Committed evaluation shard bytes changed")
            rows, traces = pd.read_parquet(rp), pd.read_parquet(tp)
            validate_new_shard(rows, traces, suite, task, cfg, hashes)
            validate_runtime(rows, traces, cfg, suite, task)
            for frame, path in ((rows, rp), (traces, tp)):
                frame["source_path"], frame["source_sha256"] = str(path), sha256_file(path)
                artifact_hashes[str(path)] = sha256_file(path)
            parts.append(rows)
            trace_parts.append(traces)
    frame = pd.concat(parts, ignore_index=True)
    frame["eligibility_reason"] = np.where(frame.frozen_clean_success.eq(1),
                                           "eligible_clean_success", "frozen_clean_failure")
    eligible = validate_raw(frame, cfg)
    require(len(eligible) == len(CHECKPOINTS) * len(CONDITIONS) * sum(DENOMINATORS.values()),
            "Expected all eligible repeated-measure rows")
    results = [paired_statistics(eligible.loc[eligible.checkpoint_outer.eq(outer)],
                                 draws, train["bootstrap_seed"], outer) for outer in CHECKPOINTS]
    summary = pd.concat([pair[0] for pair in results], ignore_index=True)
    contrasts = pd.concat([pair[1] for pair in results], ignore_index=True)
    suite_summary = summary.loc[summary.suite_or_endpoint.isin(DENOMINATORS)].copy()
    endpoints = summary.loc[~summary.suite_or_endpoint.isin(DENOMINATORS)].copy()
    per_task = eligible.groupby(["checkpoint_outer", "suite", "task_id", "condition"], as_index=False).agg(
        clean_success_states=("state_index", "nunique"),
        casr=("casr", "mean"),
        attacked_success=("success", "mean"),
    )
    prefix = "DINO_REGISTER_DELAYED2000_XSUITE"
    atomic_parquet(frame, output / f"{prefix}_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(trace_parts, ignore_index=True), output / f"{prefix}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{prefix}_ELIGIBLE.parquet")
    for suffix, data in (("SUMMARY", suite_summary), ("ENDPOINTS", endpoints),
                         ("CONTRASTS", contrasts), ("PER_TASK", per_task)):
        data.to_csv(output / f"{prefix}_{suffix}.csv", index=False)
    costs = []
    training = pd.read_csv(Path(train["output_root"]) / "REGISTER_TRAINING_METRICS.csv")
    training.to_csv(output / f"{prefix}_TRAINING_METRICS.csv", index=False)
    refs = reference_patches(train)
    for outer in CHECKPOINTS:
        for method in METHODS:
            folder = Path(train["output_root"]) / "training" / method
            status = json.loads((folder / "status.json").read_text())
            history = pd.read_parquet(folder / "training.parquet")
            attack = frame.loc[frame.checkpoint_outer.eq(outer) & frame.condition.eq(method)]
            row = history.loc[history.outer_iteration.eq(outer)].iloc[0]
            costs.append({
                "checkpoint_outer": outer, "condition": method, "fresh_start_outer": 0,
                "optimizer_updates": outer * train["inner_loop"],
                "cumulative_training_elapsed_s": float(row.elapsed_s),
                "whole_training_elapsed_s": float(status["elapsed_s"]),
                "whole_training_costs": status["costs"],
                "evaluation_elapsed_s": float(attack.elapsed_s.sum()),
                "evaluation_episodes": len(attack), "training_seed_count": 1,
            })
        for condition, item in refs[outer].items():
            attack = frame.loc[frame.checkpoint_outer.eq(outer) & frame.condition.eq(condition)]
            costs.append({
                "checkpoint_outer": outer, "condition": condition,
                "new_training_elapsed_s": 0., "optimizer_updates": outer * train["inner_loop"],
                "recorded_training_elapsed_s": item["recorded_training_elapsed_s"],
                "evaluation_elapsed_s": float(attack.elapsed_s.sum()),
                "evaluation_episodes": len(attack), "pt_sha256": item["pt"],
                "note": f"Existing matched {outer} checkpoint; reevaluated without retraining.",
            })
    atomic_json(costs, output / f"{prefix}_COSTS.json")
    report = {
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "preparation_signature": preparation["signature"],
        "evaluation_config_signature": digest_json(cfg),
        "checkpoints": hashes,
        "evaluation_checkpoint_outer_iterations": list(CHECKPOINTS),
        "attack_episodes": len(CHECKPOINTS) * len(CONDITIONS) * 400,
        "new_patch_attack_episodes": len(CHECKPOINTS) * len(METHODS) * 400,
        "reference_attack_episodes": len(CHECKPOINTS) * len(REFERENCES) * 400,
        "eligible_rows": len(eligible),
        "artifact_hashes": artifact_hashes,
        "prefix": prefix,
    }
    primary = summary.loc[summary.checkpoint_outer.eq(FINAL_OUTER)
                          & summary.suite_or_endpoint.eq(train["primary_endpoint"])]
    closeout = {
        "status": "READY_FOR_FINAL_METHOD_REVIEW", "experiment_id": EXPERIMENT_ID,
        "primary_checkpoint_outer": FINAL_OUTER, "primary_endpoint": train["primary_endpoint"],
        "method_specs": train["method_specs"], "auxiliary_start_outer": train["auxiliary_start_outer"],
        "training_suite": "libero_object", "front_camera_attacked": False,
        "patch_shape": train["patch_shape"], "patch_xy": train["patch_xy"],
        "fresh_start_outer": 0, "training_seed_count": 1,
        "selection_basis": "Fixed 2000-budget three-suite macro; report individual suites and paired deltas.",
        "primary_results": primary.to_dict("records"),
        "primary_contrasts": contrasts.loc[
            contrasts.checkpoint_outer.eq(FINAL_OUTER)
            & contrasts.suite_or_endpoint.eq(train["primary_endpoint"])].to_dict("records"),
        "descriptive_method_order": primary.loc[primary.condition.isin(METHODS)]
            .sort_values("task_equal_casr", ascending=False).condition.tolist(),
        "limitations": [
            "Single training seed; state/task bootstrap does not quantify between-training-seed variance.",
            "Hypothesis and design were informed by previous target-suite results.",
            "Higher attention mass alone does not establish closed-loop attack effectiveness.",
            "Long is fully reported but is not the primary mechanism endpoint.",
            "Projection constrains raw patch gradients, not the Adam update or closed-loop outcome.",
        ],
        "closure_policy": train["closure_policy"],
    }
    atomic_json(closeout, output / "OFT_CLOSEOUT.json")
    report["closeout_report"] = str(write_closeout(output, train, summary, contrasts, training, costs))
    report["closeout_json"] = str(output / "OFT_CLOSEOUT.json")
    atomic_json(report, output / "STATUS.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.draws), indent=2))

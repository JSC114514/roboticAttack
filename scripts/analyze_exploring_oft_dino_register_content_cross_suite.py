#!/usr/bin/env python3
"""Task/state paired bootstrap for OFT-DINO-REGISTER-CONTENT500-V1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dino_register_content_common import (
    CHECKPOINTS, FINAL_OUTER, CONDITIONS, CONFIG, DENOMINATORS, METHODS, REFERENCES, config,
    digest_json, reference_patches, require_preparation,
)
from run_exploring_oft_dino_register_content_cross_suite import (
    EXPERIMENT_ID, KEY, load_config, patches, validate_new_shard, validate_runtime,
)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet, require, sha256_file

PAIRS = [
    (f"{method}_minus_{reference}", method, reference)
    for method in METHODS for reference in REFERENCES
] + [
    ("content_minus_action", "register_content_g10", "uniform_action"),
    ("content_minus_attention", "register_content_g10", "register_attention_g10"),
    ("content_minus_dense", "register_content_g10", "dense_content_g10"),
    ("attention_minus_action", "register_attention_g10", "uniform_action"),
    ("dense_minus_action", "dense_content_g10", "uniform_action"),
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


def paired_statistics(eligible, draws=10000, seed=20260907, checkpoint_outer=500):
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
    """Describe actual RGB-only results, not proxy-only claims."""
    lines = [
        "# OFT-DINO-REGISTER-CONTENT500-V1 实验闭环", "",
        "评测完成，待方法选择；CASR越高表示任务失败率越高。", "",
        "四组从seed42随机7500参数RGB开始，500轮；前250轮动作目标完全一致。",
        "A仅动作；B注意力；C第16层四register K/V内容；D四普通dense K/V内容。",
        "C/D额外干净分支只用于训练；正常贴图主损失和评测均不替换特征。",
        "辅助raw RGB梯度范数固定10%，均不投影；不是对Adam更新方向的保证。",
        "Object clean300/replay258，Front不变；cosine总时间轴2000，500停止。", "",
        "## 同预算正常闭环结果", "",
        "| 方法 | Object | Spatial | Goal | Long（辅助） | Spatial/Goal主指标 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = []
        for endpoint in [*DENOMINATORS, "spatial_goal_macro"]:
            row = summary.loc[summary.checkpoint_outer.eq(500)
                              & summary.condition.eq(condition)
                              & summary.suite_or_endpoint.eq(endpoint)]
            require(len(row) == 1, "Content closeout grid incomplete")
            values.append(f"{float(row.iloc[0].task_equal_casr) * 100:.2f}%")
        lines.append("| " + condition + " | " + " | ".join(values) + " |")
    lines.extend(["", "## 500轮Spatial/Goal主要差值", "",
                  "| 比较 | 差值pp | 配对95%区间pp | 正/负/平任务数 |",
                  "|---|---:|---|---|"])
    for row in contrasts.loc[contrasts.suite_or_endpoint.eq("spatial_goal_macro")
                             & contrasts.contrast.isin(["content_minus_action",
                                                       "content_minus_attention",
                                                       "content_minus_dense"])].itertuples():
        lines.append(f"| {row.contrast} | {row.difference_left_minus_right * 100:+.2f} | "
                     f"[{row.ci95_low * 100:+.2f}, {row.ci95_high * 100:+.2f}] | "
                     f"{row.positive_tasks}/{row.negative_tasks}/{row.tied_tasks} |")
    lines.extend(["", "## 补丁与代理变化", "",
                  "| 方法 | RGB对250轮MAE | 辅助比例 | CF对clean动作MAE | donor K/V变化 |",
                  "|---|---:|---:|---:|---:|"])
    for row in training.itertuples():
        lines.append(f"| {row.condition} | {row.rgb_mae_from_outer250:.6f} | "
                     f"{row.mean_effective_aux_ratio:.4f} | {row.mean_cf_clean_action_mae:.6f} | "
                     f"{row.mean_donor_kv_change:.6f} |")
    lines.extend(["", "## 耗时", "",
                  "| 方法 | 训练小时 | 评测GPU小时之和 |", "|---|---:|---:|"])
    for row in costs:
        elapsed = row.get("cumulative_training_elapsed_s", row.get("recorded_training_elapsed_s", 0.))
        lines.append(f"| {row['condition']} | {elapsed / 3600:.2f} | "
                     f"{row['evaluation_elapsed_s'] / 3600:.2f} |")
    lines.extend(["", "原始UADA500复用已有权重，不重新训练；以上GPU时间和不等于并行墙钟。",
                  "单训练seed；任务/初态bootstrap不是训练seed方差。目标suite已参与开发。",
                  "C>A/B/D才支持本次register内容目标的独立增量；C/D相近不能称register独有。",
                  "CF代理提升而正常RGB闭环无提升时，不宣称攻击增强。",
                  "均匀采样是共同数据分布，不能包装为DINO方法。Long完整报告但非主要结论。",
                  "不自动扩2000轮、搜索层/权重、跨模型或实机。", ""])
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
    prefix = "DINO_REGISTER_CONTENT500_XSUITE"
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
        "selection_basis": "Fixed 500-budget Spatial/Goal macro; compare content-action, content-attention, content-dense.",
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
            "The 10% ratio constrains raw RGB gradients, not the Adam update or closed-loop outcome.",
            "Content-branch action loss is a training proxy; improvement requires normal RGB-only evaluation.",
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

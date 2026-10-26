"""Validate actual assets; paired per-suite results with no suite averaging."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, ALL_METHODS, REFERENCE_METHODS, BASE_SIGNATURE, SUITES, atomic_json,
                    atomic_parquet, config, deployment_conditions, analysis_conditions, reference_frames,
                    prepared, load_bank, load_json, require, run_signature, sha256_file)
from train import validate_resume

COMPARISONS = tuple((m, "uada_original") for m in REFERENCE_METHODS if m != "uada_original") + tuple(
    (m, reference) for m in METHODS for reference in REFERENCE_METHODS)

def validate_training():
    cfg, signature, patches, changes = config(), run_signature(), {}, []
    prepared()
    bank = load_bank()
    expected_uids = [row[0] for row in bank["candidates"]]
    for method in METHODS:
        folder = ROOT / "training" / method
        status = load_json(folder / "status.json")
        require(status["status"] == "COMPLETE" and status["outer"] == cfg["outer_iterations"]
                and status["signature"] == signature, "Method not complete")
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        validate_resume(saved, method, cfg, signature)
        frame = pd.read_parquet(folder / "training.parquet")
        require(frame.uid.tolist() == expected_uids, "Training no longer uses shared frozen Long order")
        require(saved["outer"] == cfg["outer_iterations"], "Complete resume clock incomplete")
        require(frame.outer_iteration.tolist() == list(range(1, cfg["outer_iterations"] + 1))
                and frame.optimizer_updates.tolist() == list(range(cfg["inner_loop"],
                    cfg["outer_iterations"] * cfg["inner_loop"] + 1, cfg["inner_loop"])), "Training budget/history incomplete")
        require(np.isfinite(frame[["main_loss", "gradient_l2", "rgb_outer_mae", "learning_rate", "tv", "high_frequency_fraction"]].to_numpy()).all(),
                "Nonfinite training history")
        require(frame.effective_rgb_updates.sum() > 0 and not torch.equal(saved["patch"], saved["initial"]),
                "No actual RGB learning occurred")
        expected_initial = saved["initial"]
        manifest = load_json(folder / "RUN_MANIFEST.json")
        require(manifest["signature"] == signature and not manifest["old_patch_loaded"]
                and not manifest["model_parameters_updated"], "Fresh/frozen training contract changed")
        if method != "advencoder_cos":
            require(torch.equal(expected_initial, torch.rand(cfg["patch_shape"], generator=torch.Generator().manual_seed(cfg["seed"]))),
                    "Pixel methods must share fresh random initialization")
        else:
            require(manifest["generator_parameterization"] and manifest["trainable_parameter_count"] > 7500,
                    "Generator optimization was replaced by pixel optimization")
        checkpoint_changes, previous_value = [], None
        for outer in cfg["checkpoint_outer_iterations"]:
            pt, png = folder / f"outer_{outer:04d}.pt", folder / f"outer_{outer:04d}.png"
            asset = torch.load(pt, map_location="cpu", weights_only=False)
            value = asset["patch"]
            require(asset["signature"] == signature and asset["method"] == method and asset["outer_iterations"] == outer
                    and asset["config"] == cfg and torch.equal(asset["initial_patch"], expected_initial)
                    and asset["optimizer_updates"] == outer * cfg["inner_loop"] and value.shape == (3, 50, 50)
                    and torch.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(), "Invalid RGB snapshot")
            if outer == 0:
                require(torch.equal(value, expected_initial), "Snapshot0 differs from fresh initial RGB")
            with Image.open(png) as im:
                require(im.mode == "RGB" and im.size == (50, 50), "PNG shape/mode changed")
                expected = value.permute(1, 2, 0).mul(255).round().byte().numpy()
                require(np.array_equal(np.asarray(im), expected), "PNG does not match trained RGB")
            if previous_value is not None:
                checkpoint_changes.append({"outer": outer, "rgb_mae_since_previous_snapshot": float((value - previous_value).abs().mean()),
                                           "png_values_changed_since_previous_snapshot": int((value.mul(255).round() != previous_value.mul(255).round()).sum())})
            previous_value = value
            if outer == cfg["outer_iterations"]:
                require(torch.equal(value, saved["patch"]), "Final RGB snapshot differs from complete optimizer state")
            if outer in cfg["evaluation_checkpoints"]:
                patches[f"{method}@{outer}"] = {"pt": sha256_file(pt), "png": sha256_file(png)}
        changes.append({"method": method, "real_rgb_updates": int(frame.effective_rgb_updates.sum()),
                        "initial_rgb_mae": float(frame.iloc[-1].initial_rgb_mae),
                        "elapsed_h": float(frame.iloc[-1].elapsed_s / 3600), "unique_selected_windows": int(frame.uid.nunique()),
                        "unique_selected_episodes": int(frame.episode_id.nunique()), "checkpoint_changes": checkpoint_changes})
    atomic_json({"status": "COMPLETE", "signature": signature, "patches": patches,
                 "training_changes": changes, "model_parameters_updated": False, "manual_appearance_gate": False,
                 "scientific_result": "PENDING_CLOSED_LOOP"}, ROOT / "reports/AUTO_VALIDATION.json")
    print(f"AUTO_VALIDATION_COMPLETE: three methods,{cfg['outer_iterations']*cfg['inner_loop']} updates each; three final deployment PNGs", flush=True)

def task_casr(frame, condition):
    clean = frame.loc[frame.condition.eq("clean"), ["task_id", "state_index", "success"]].rename(columns={"success": "clean_success"})
    attack = frame.loc[frame.condition.eq(condition), ["task_id", "state_index", "failure"]]
    pairs = attack.merge(clean, on=["task_id", "state_index"], validate="one_to_one")
    means = pairs.loc[pairs.clean_success.eq(1)].groupby("task_id").failure.mean()
    require(len(means) > 0, "Suite has no clean-success qualified tasks")
    return float(means.mean()), pairs

def paired_bootstrap(pairs, draws=10000, seed=20260907):
    rng = np.random.default_rng(seed)
    # Resample qualified states within tasks; no artificial zero-denominator tasks.
    eligible = pairs.loc[pairs.clean_success.eq(1)]
    groups = [g.value.to_numpy(float) for _, g in eligible.groupby("task_id")]
    require(groups, "No clean-success states for paired bootstrap")
    estimates = np.empty(draws)
    for draw in range(draws):
        values = []
        for index in rng.integers(0, len(groups), len(groups)):
            group = groups[index]
            values.append(group[rng.integers(0, len(group), len(group))].mean())
        estimates[draw] = np.mean(values)
    return np.quantile(estimates, [.025, .975]).tolist()

def summarize(raw, draws=10000):
    conditions = analysis_conditions()
    checkpoints = config()["evaluation_checkpoints"]
    states = config()["evaluation"]["state_indices"]
    expected = len(SUITES) * 10 * len(states) * len(conditions)
    require(len(raw) == expected and not raw.duplicated(["suite", "task_id", "state_index", "condition"]).any(), "Screening episode grid incomplete/duplicated")
    require(set(raw.suite) == set(SUITES) and set(raw.condition) == set(conditions), "Suite/condition coverage changed")
    require(raw.success.isin([0, 1]).all() and raw.failure.eq(1 - raw.success).all(), "Invalid success/failure labels")
    require(raw.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(), "Unpaired reset seeds")
    require(raw.groupby(["suite", "task_id", "state_index"]).condition.nunique().eq(len(conditions)).all(), "Missing condition pairs")
    summaries, contrasts, tasks = [], [], []
    for suite in SUITES:
        sub = raw.loc[raw.suite.eq(suite)]
        clean = sub.loc[sub.condition.eq("clean")]
        require(len(clean) == 10 * len(states)
                and set(zip(clean.task_id, clean.state_index)) == {(t, s) for t in range(10) for s in states}, "Invalid suite grid")
        clean_success = int(clean.success.sum())
        clean_sr = float(clean.success.mean())
        summaries.append({"suite": suite, "method": "clean", "outer": 0, "task_success_rate": clean_sr,
                          "attack_failure_rate": 1 - clean_sr, "task_equal_casr": 0., "ci_low": 0., "ci_high": 0.,
                          "fresh_clean_success": clean_success, "qualified_tasks": int(clean.groupby("task_id").success.sum().gt(0).sum())})
        for outer in checkpoints:
            for method in ALL_METHODS:
                condition = f"{method}@{outer}"
                point, pairs = task_casr(sub, condition)
                ci = paired_bootstrap(pairs.assign(value=pairs.failure), draws=draws)
                attack = sub.loc[sub.condition.eq(condition)]
                summaries.append({"suite": suite, "method": method, "outer": outer, "task_success_rate": float(attack.success.mean()),
                                  "attack_failure_rate": float(attack.failure.mean()), "task_equal_casr": point,
                                  "ci_low": ci[0], "ci_high": ci[1], "fresh_clean_success": clean_success,
                                  "qualified_tasks": int(pairs.loc[pairs.clean_success.eq(1)].task_id.nunique())})
                for task, group in pairs.groupby("task_id"):
                    valid = group.clean_success.eq(1)
                    tasks.append({"suite": suite, "method": method, "outer": outer, "task_id": int(task),
                                  "clean_success_states": int(valid.sum()),
                                  "casr": float(group.loc[valid].failure.mean()) if valid.any() else None})
        comparisons = [(left, outer, right, outer, "enhancement_vs_baseline")
                       for outer in checkpoints for left, right in COMPARISONS]
        if len(checkpoints) > 1:
            comparisons += [(method, checkpoints[-1], method, checkpoints[0], "training_rounds") for method in METHODS]
        for left, left_outer, right, right_outer, kind in comparisons:
            lp, a = task_casr(sub, f"{left}@{left_outer}")
            rp, b = task_casr(sub, f"{right}@{right_outer}")
            joint = a.merge(b[["task_id", "state_index", "failure"]], on=["task_id", "state_index"], suffixes=("_a", "_b"), validate="one_to_one")
            joint["value"] = joint.failure_a - joint.failure_b
            ci = paired_bootstrap(joint, draws=draws)
            qualified = joint.loc[joint.clean_success.eq(1)]
            task_deltas = qualified.groupby("task_id").value.mean()
            contrasts.append({"suite": suite, "outer": left_outer, "left": left, "right": right,
                              "left_outer": left_outer, "right_outer": right_outer, "contrast_type": kind,
                              "casr_difference": lp - rp, "ci_low": ci[0], "ci_high": ci[1],
                              "positive_tasks": int(task_deltas.gt(0).sum()),
                              "negative_tasks": int(task_deltas.lt(0).sum()),
                              "tied_tasks": int(task_deltas.eq(0).sum())})
    return pd.DataFrame(summaries), pd.DataFrame(contrasts), pd.DataFrame(tasks)

def analyze():
    signature, frames, traces = run_signature(), [], []
    prepared()
    for suite in SUITES:
        for task in range(10):
            folder = ROOT / "evaluation" / suite
            status = load_json(folder / f"task_{task:02d}_STATUS.json")
            rp, tp = folder / f"task_{task:02d}_rollouts.parquet", folder / f"task_{task:02d}_traces.parquet"
            require(status["status"] == "COMPLETE" and status["signature"] == signature
                    and status["rollouts_sha256"] == sha256_file(rp) and status["traces_sha256"] == sha256_file(tp), "Evaluation shard incomplete/changed")
            frames.append(pd.read_parquet(rp)); traces.append(pd.read_parquet(tp))
    new_raw, new_trace = pd.concat(frames, ignore_index=True), pd.concat(traces, ignore_index=True)
    require(len(new_raw) == 240 and new_raw.signature.eq(signature).all()
            and set(new_raw.condition) == set(deployment_conditions()), "New rollout coverage/identity changed")
    validation = load_json(ROOT / "reports/AUTO_VALIDATION.json")
    require(validation["status"] == "COMPLETE" and validation["signature"] == signature, "Training validation identity changed")
    for method in METHODS:
        condition = method + "@250"
        png = ROOT / "training" / method / "outer_0250.png"
        expected = validation["patches"][condition]["png"]
        require(sha256_file(png) == expected
                and new_raw.loc[new_raw.condition.eq(condition)].deployment_png_sha256.eq(expected).all(),
                "Evaluation is not using validated final PNG")
    for frame in (new_raw, new_trace):
        frame["source_run"] = ROOT.name
        frame["reused"] = False
    old_raw, old_trace = reference_frames()
    raw, trace = pd.concat([old_raw, new_raw], ignore_index=True), pd.concat([old_trace, new_trace], ignore_index=True)
    expected_signatures = np.where(raw.reused, BASE_SIGNATURE, signature)
    require(raw.signature.eq(expected_signatures).all() and not raw.evaluation_training_hooks.any(), "Unexpected runtime/hooks")
    keys = ["suite", "task_id", "state_index", "condition"]
    require(not trace.duplicated(keys + ["decision_block"]).any(), "Duplicate trace blocks")
    counts = trace.groupby(keys).size().rename("trace_rows")
    check = raw.set_index(keys).join(counts)
    require(check.trace_rows.eq(check.decision_blocks).all() and len(trace) == int(raw.decision_blocks.sum()), "Missing/extra trace records")
    require(raw.renderer_vendor.str.lower().str.contains("nvidia").all() and raw.renderer.str.contains("4090").all(), "Non-4090 hardware renderer")
    summary, contrasts, tasks = summarize(raw)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    atomic_parquet(raw, output / "OBJECTIVES250_ROLLOUTS.parquet")
    atomic_parquet(trace, output / "OBJECTIVES250_TRACES.parquet")
    summary.to_csv(output / "OBJECTIVES250_SUMMARY.csv", index=False)
    contrasts.to_csv(output / "OBJECTIVES250_CONTRASTS.csv", index=False)
    tasks.to_csv(output / "OBJECTIVES250_PER_TASK.csv", index=False)
    lines = ["# DINO三新目标与UADA/Key/生成器：250轮闭环筛选", "", "主表逐suite CASR；表中为百分比。不跨suite平均。",
             "clean、UADA、Key、生成器为匹配v1原始记录只读复用；三新目标为本批新增240回合。",
             "", "## 无补丁清洁对照（v1复用）", "", "| Suite | 成功/回合 | SR% | clean-success任务数 |", "|---|---:|---:|---:|"]
    for row in summary.loc[summary.method.eq("clean")].itertuples():
        count = int(raw.loc[raw.suite.eq(row.suite) & raw.condition.eq("clean")].shape[0])
        lines.append(f"| {row.suite} | {row.fresh_clean_success}/{count} | {100*row.task_success_rate:.2f} | {row.qualified_tasks} |")
    for outer in config()["evaluation_checkpoints"]:
        table = summary.loc[summary.outer.eq(outer)].pivot(index="method", columns="suite", values="task_equal_casr")
        print(f"outer={outer}\n" + (100 * table).round(2).to_string(), flush=True)
        lines += ["", f"## {outer}轮", "", "| 方法 | Object | Spatial | Goal | Long |", "|---|---:|---:|---:|---:|"]
        for method in ALL_METHODS:
            values = [f"{table.loc[method, suite]*100:.2f}" for suite in SUITES]
            lines.append("| " + method + " | " + " | ".join(values) + " |")
        lines += ["", "候选相对同预算UADA的配对差（百分点）/正向任务数：", "",
                  "| 方法 | Suite | 差值pp | 正向task | 负向task | 持平task |", "|---|---|---:|---:|---:|---:|"]
        for row in contrasts.loc[contrasts.outer.eq(outer) & contrasts.right.eq("uada_original")].itertuples():
            lines.append(f"| {row.left} | {row.suite} | {100*row.casr_difference:+.2f} | {row.positive_tasks} | {row.negative_tasks} | {row.tied_tasks} |")
    lines += ["", "数据来自官方Long train专家演示；Long来源内，其他三套件补丁迁移。",
              "测试使用正常OFT和RGB PNG，无训练hook/生成器推理。单训练seed；40/41为已查看开发初态。",
              "每task仅两个初态，是低成本探索筛选，不等于最终验证；Long来源内。",
              "任务成功率、配对基线差值/区间、每任务结果分别见SUMMARY、CONTRASTS、PER_TASK；不作跨suite平均。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json({"status": "COMPLETE", "signature": signature, "base_signature": BASE_SIGNATURE,
                 "episodes": len(raw), "new_episodes": 240, "reused_episodes": 320, "methods": len(ALL_METHODS),
                 "evaluation_checkpoints": config()["evaluation_checkpoints"], "source_suite": "libero_10",
                 "training_data": "official expert demonstrations", "independent_patch_training_seeds": 1,
                 "summary_sha256": sha256_file(output / "OBJECTIVES250_SUMMARY.csv"),
                 "raw_sha256": sha256_file(output / "OBJECTIVES250_ROLLOUTS.parquet"),
                 "traces_sha256": sha256_file(output / "OBJECTIVES250_TRACES.parquet")}, output / "STATUS.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "stats"), required=True)
    {"validate": validate_training, "stats": analyze}[parser.parse_args().stage]()

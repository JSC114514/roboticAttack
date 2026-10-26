"""Validate actual assets; paired per-suite results with no suite averaging."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, SUITES, atomic_json, atomic_parquet, config, deployment_conditions,
                    load_json, require, run_signature, sha256_file)
from train import validate_resume

COMPARISONS = (("uada_dino_freq", "uada_original"), ("tma_xyz_dino", "tma_xyz"))

def validate_training():
    cfg, signature, patches, changes = config(), run_signature(), {}, []
    for method in METHODS:
        folder = ROOT / "training" / method
        status = load_json(folder / "status.json")
        require(status["status"] == "COMPLETE" and status["outer"] == cfg["outer_iterations"]
                and status["signature"] == signature, "Method not complete")
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        validate_resume(saved, method, cfg, signature)
        frame = pd.read_parquet(folder / "training.parquet")
        require(saved["outer"] == cfg["outer_iterations"], "Complete resume clock incomplete")
        require(frame.outer_iteration.tolist() == list(range(1, cfg["outer_iterations"] + 1))
                and frame.optimizer_updates.tolist() == list(range(cfg["inner_loop"],
                    cfg["outer_iterations"] * cfg["inner_loop"] + 1, cfg["inner_loop"])), "Training budget/history incomplete")
        require(np.isfinite(frame[["main_loss", "gradient_l2", "rgb_outer_mae", "learning_rate", "tv", "high_frequency_fraction"]].to_numpy()).all(),
                "Nonfinite training history")
        require(frame.effective_rgb_updates.sum() > 0 and not torch.equal(saved["patch"], saved["initial"]),
                "No actual RGB learning occurred")
        expected_initial = torch.rand(cfg["patch_shape"], generator=torch.Generator().manual_seed(cfg["seed"]))
        require(torch.equal(saved["initial"], expected_initial), "Not the shared fresh random initialization")
        checkpoint_changes, previous_value = [], None
        for outer in cfg["checkpoint_outer_iterations"]:
            pt, png = folder / f"outer_{outer:04d}.pt", folder / f"outer_{outer:04d}.png"
            asset = torch.load(pt, map_location="cpu", weights_only=False)
            value = asset["patch"]
            require(asset["signature"] == signature and asset["method"] == method and asset["outer_iterations"] == outer
                    and asset["config"] == cfg and torch.equal(asset["initial_patch"], expected_initial)
                    and asset["optimizer_updates"] == outer * cfg["inner_loop"] and value.shape == (3, 50, 50)
                    and torch.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(), "Invalid RGB snapshot")
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
    print("AUTO_VALIDATION_COMPLETE: four methods,100000 optimizer updates each; eight deployment PNGs at1000/2000", flush=True)

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
    conditions = deployment_conditions()
    checkpoints = config()["evaluation_checkpoints"]
    require(len(raw) == 400 * len(conditions) and not raw.duplicated(["suite", "task_id", "state_index", "condition"]).any(), "Expected3600 unique episodes")
    require(set(raw.suite) == set(SUITES) and set(raw.condition) == set(conditions), "Suite/condition coverage changed")
    require(raw.success.isin([0, 1]).all() and raw.failure.eq(1 - raw.success).all(), "Invalid success/failure labels")
    require(raw.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(), "Unpaired reset seeds")
    require(raw.groupby(["suite", "task_id", "state_index"]).condition.nunique().eq(len(conditions)).all(), "Missing condition pairs")
    summaries, contrasts, tasks = [], [], []
    for suite in SUITES:
        sub = raw.loc[raw.suite.eq(suite)]
        clean = sub.loc[sub.condition.eq("clean")]
        require(len(clean) == 100 and set(zip(clean.task_id, clean.state_index)) == {(t, s) for t in range(10) for s in range(40, 50)}, "Invalid suite grid")
        clean_success = int(clean.success.sum())
        clean_sr = float(clean.success.mean())
        summaries.append({"suite": suite, "method": "clean", "outer": 0, "task_success_rate": clean_sr,
                          "attack_failure_rate": 1 - clean_sr, "task_equal_casr": 0., "ci_low": 0., "ci_high": 0.,
                          "fresh_clean_success": clean_success, "qualified_tasks": int(clean.groupby("task_id").success.sum().gt(0).sum())})
        for outer in checkpoints:
            for method in METHODS:
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
        comparisons += [(method, checkpoints[-1], method, checkpoints[0], "training_rounds") for method in METHODS]
        for left, left_outer, right, right_outer, kind in comparisons:
            lp, a = task_casr(sub, f"{left}@{left_outer}")
            rp, b = task_casr(sub, f"{right}@{right_outer}")
            joint = a.merge(b[["task_id", "state_index", "failure"]], on=["task_id", "state_index"], suffixes=("_a", "_b"), validate="one_to_one")
            joint["value"] = joint.failure_a - joint.failure_b
            ci = paired_bootstrap(joint, draws=draws)
            contrasts.append({"suite": suite, "outer": left_outer, "left": left, "right": right,
                              "left_outer": left_outer, "right_outer": right_outer, "contrast_type": kind,
                              "casr_difference": lp - rp, "ci_low": ci[0], "ci_high": ci[1]})
    return pd.DataFrame(summaries), pd.DataFrame(contrasts), pd.DataFrame(tasks)

def analyze():
    signature, frames, traces = run_signature(), [], []
    for suite in SUITES:
        for task in range(10):
            folder = ROOT / "evaluation" / suite
            status = load_json(folder / f"task_{task:02d}_STATUS.json")
            rp, tp = folder / f"task_{task:02d}_rollouts.parquet", folder / f"task_{task:02d}_traces.parquet"
            require(status["status"] == "COMPLETE" and status["signature"] == signature
                    and status["rollouts_sha256"] == sha256_file(rp) and status["traces_sha256"] == sha256_file(tp), "Evaluation shard incomplete/changed")
            frames.append(pd.read_parquet(rp)); traces.append(pd.read_parquet(tp))
    raw, trace = pd.concat(frames, ignore_index=True), pd.concat(traces, ignore_index=True)
    require(raw.signature.eq(signature).all() and not raw.evaluation_training_hooks.any(), "Unexpected runtime/hooks")
    require(raw.renderer_vendor.str.lower().str.contains("nvidia").all() and raw.renderer.str.contains("4090").all(), "Non-4090 hardware renderer")
    summary, contrasts, tasks = summarize(raw)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    atomic_parquet(raw, output / "LONG4_ROLLOUTS.parquet")
    atomic_parquet(trace, output / "LONG4_TRACES.parquet")
    summary.to_csv(output / "LONG4_SUMMARY.csv", index=False)
    contrasts.to_csv(output / "LONG4_CONTRASTS.csv", index=False)
    tasks.to_csv(output / "LONG4_PER_TASK.csv", index=False)
    lines = ["# 官方Long四路从零2000轮结果", "", "只报告逐suite CASR；表中为百分比。"]
    for outer in config()["evaluation_checkpoints"]:
        table = summary.loc[summary.outer.eq(outer)].pivot(index="method", columns="suite", values="task_equal_casr")
        print(f"outer={outer}\n" + (100 * table).round(2).to_string(), flush=True)
        lines += ["", f"## {outer}轮", "", "| 方法 | Object | Spatial | Goal | Long |", "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            values = [f"{table.loc[method, suite]*100:.2f}" for suite in SUITES]
            lines.append("| " + method + " | " + " | ".join(values) + " |")
    lines += ["", "数据来自官方Long train专家演示；Long来源内，其他三套件补丁迁移。",
              "测试使用正常OFT和RGB PNG，无DINO训练hook。单训练seed；40–49为已查看开发网格。",
              "任务成功率、配对基线差值/区间、每任务结果分别见SUMMARY、CONTRASTS、PER_TASK；不作跨suite平均。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json({"status": "COMPLETE", "signature": signature, "episodes": 3600, "methods": len(METHODS),
                 "evaluation_checkpoints": config()["evaluation_checkpoints"], "source_suite": "libero_10",
                 "training_data": "official expert demonstrations", "independent_patch_training_seeds": 1,
                 "summary_sha256": sha256_file(output / "LONG4_SUMMARY.csv"), "raw_sha256": sha256_file(output / "LONG4_ROLLOUTS.parquet")}, output / "STATUS.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "stats"), required=True)
    {"validate": validate_training, "stats": analyze}[parser.parse_args().stage]()

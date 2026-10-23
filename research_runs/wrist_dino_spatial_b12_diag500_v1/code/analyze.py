"""32 RGB assets and fully paired per-suite statistics; no across-suite mean."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, FAMILIES, BATCHES, SUITES, arm_info, atomic_json, atomic_parquet,
                    condition, config, deployment_conditions, load_json, prepared, require, run_signature, sha256_file)
from train import validate_resume, verify_final_assets


def validate_training():
    prepared()
    cfg, signature, patches, changes = config(), run_signature(), {}, []
    for method in METHODS:
        info, folder = arm_info(method), ROOT / "training" / method
        status = load_json(folder / "status.json")
        require(status["status"] == "COMPLETE" and status["outer"] == 500 and status["signature"] == signature, "Arm not complete")
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        validate_resume(saved, method, cfg, signature)
        verify_final_assets(folder, saved, method, cfg, signature)
        frame = pd.read_parquet(folder / "training.parquet")
        inner = pd.read_parquet(folder / "inner_steps.parquet")
        require(len(inner) == 25000 and not inner.duplicated(["outer_iteration", "inner_iteration"]).any()
                and inner.groupby("outer_iteration").inner_iteration.nunique().eq(50).all(), "Inner-step monitor incomplete")
        diagnostics = pd.read_parquet(folder / "diagnostics.parquet")
        expected_points = {(o, i) for o in cfg["monitor"]["diagnostic_outer_iterations"]
                           for i in cfg["monitor"]["diagnostic_inner_iterations"]}
        require(set(zip(diagnostics.outer_iteration, diagnostics.inner_iteration)) == expected_points
                and len(diagnostics) == len(expected_points) * info["batch"], "Full-model monitoring incomplete")
        require(np.isfinite(diagnostics[["uada_loss_before", "uada_loss_after", "tma_loss_before", "tma_loss_after",
                                        "method_same_input_loss_change"]].to_numpy()).all(), "Nonfinite internal diagnostic")
        require(frame.outer_iteration.tolist() == list(range(1, 501))
                and frame.optimizer_updates.tolist() == list(range(50, 25001, 50))
                and frame.sample_updates.tolist() == list(range(50*info["batch"], 25001*info["batch"], 50*info["batch"])),
                "Incomplete optimizer/sample clock")
        require(np.isfinite(frame[["main_loss", "gradient_l2", "rgb_outer_mae", "learning_rate", "tv", "high_frequency_fraction"]].to_numpy()).all(),
                "Nonfinite training history")
        require(frame.effective_rgb_updates.sum() > 0 and not torch.equal(saved["patch"], saved["initial"]), "No actual RGB learning")
        manifest = load_json(folder / "RUN_MANIFEST.json")
        require(manifest["signature"] == signature and manifest["candidate_id"] == info["id"]
                and manifest["training_image_batch_size"] == info["batch"] and not manifest["old_patch_loaded"]
                and not manifest["model_parameters_updated"], "Fresh/frozen/batch contract changed")
        if not info["objective"].startswith("advencoder_"):
            expected = torch.rand(cfg["patch_shape"], generator=torch.Generator().manual_seed(cfg["seed"]))
            require(torch.equal(saved["initial"], expected), "Pixel seed/initialization changed")
        if info["id"] == 9:
            require(frame.nce_negative_count.eq(info["batch"] - 1).all(), "NCE negative count changed")
            if info["batch"] == 1:
                require(frame.nce_degenerate.eq(1).all() and frame.feature_negative_nce.eq(0).all(), "B1 NCE silently replaced")
        snapshots, previous = [], None
        for outer in cfg["checkpoint_outer_iterations"]:
            pt = folder / f"outer_{outer:04d}.pt"
            asset = torch.load(pt, map_location="cpu", weights_only=False)
            value = asset["patch"]
            require(asset["signature"] == signature and asset["method"] == method and asset["outer_iterations"] == outer
                    and asset["config"] == cfg and torch.equal(asset["initial_patch"], saved["initial"])
                    and value.shape == (3, 50, 50) and torch.isfinite(value).all()
                    and ((value >= 0) & (value <= 1)).all(), "Invalid RGB snapshot")
            if outer == 0:
                require(torch.equal(value, saved["initial"]), "Initial snapshot changed")
            with Image.open(pt.with_suffix(".png")) as im:
                require(im.mode == "RGB" and im.size == (50, 50)
                        and np.array_equal(np.asarray(im), value.permute(1, 2, 0).mul(255).round().byte().numpy()), "PNG differs from RGB")
            if previous is not None:
                snapshots.append({"outer": outer, "rgb_mae": float((value - previous).abs().mean()),
                                  "changed_png_values": int((value.mul(255).round() != previous.mul(255).round()).sum())})
            previous = value
        pt = folder / "outer_0500.pt"
        patches[condition(method)] = {"pt": sha256_file(pt), "png": sha256_file(pt.with_suffix(".png"))}
        changes.append({"method": method, "candidate_id": info["id"], "batch_size": info["batch"],
                        "elapsed_h": float(saved["elapsed_s"] / 3600), "rgb_changes": snapshots,
                        "real_rgb_updates": int(frame.effective_rgb_updates.sum()),
                        "nce_degenerate_control": info["id"] == 9 and info["batch"] == 1})
        del saved
    atomic_json({"status": "COMPLETE", "signature": signature, "patches": patches, "training_changes": changes,
                 "model_parameters_updated": False, "manual_appearance_gate": False,
                 "scientific_result": "PENDING_CLOSED_LOOP"}, ROOT / "reports/AUTO_VALIDATION.json")
    print("BATCH500_VALIDATED: 16 fresh PNGs; 25000 updates per arm; appearance is not a gate", flush=True)


def task_casr(frame, name):
    clean = frame.loc[frame.condition.eq("clean"), ["task_id", "state_index", "success"]].rename(columns={"success": "clean_success"})
    attack = frame.loc[frame.condition.eq(name), ["task_id", "state_index", "failure"]]
    pairs = attack.merge(clean, on=["task_id", "state_index"], validate="one_to_one")
    means = pairs.loc[pairs.clean_success.eq(1)].groupby("task_id").failure.mean()
    require(len(means) > 0, "Suite has no clean-success qualified tasks")
    return float(means.mean()), pairs


def paired_bootstrap(pairs, draws=10000, seed=20260907):
    """Same task->qualified-state bootstrap, vectorized to keep 32-arm stats cheap."""
    rng = np.random.default_rng(seed)
    groups = [g.value.to_numpy(float) for _, g in pairs.loc[pairs.clean_success.eq(1)].groupby("task_id")]
    require(bool(groups), "No clean-success groups")
    selected = rng.integers(0, len(groups), size=(draws, len(groups)))
    values = np.zeros((draws, len(groups)))
    for index, group in enumerate(groups):
        rows, slots = np.nonzero(selected == index)
        samples = rng.integers(0, len(group), size=(len(rows), len(group)))
        values[rows, slots] = group[samples].mean(axis=1)
    return np.quantile(values.mean(axis=1), [.025, .975]).tolist()


def comparisons():
    result = []
    for left in METHODS:
        info = arm_info(left)
        for right in METHODS:
            other = arm_info(right)
            if info["id"] not in {1, 12} and other["id"] in {1, 12} and info["batch"] == other["batch"]:
                result.append((left, right, "candidate_vs_matched_baseline"))
            elif info["id"] == 12 and other["id"] == 1 and info["batch"] == other["batch"]:
                result.append((left, right, "baseline_objectives"))
            elif info["id"] == other["id"] and info["batch"] > 1 and other["batch"] == 1:
                kind = "nce_batch_plus_negatives_vs_degenerate_b1" if info["id"] == 9 else "batch_vs_b1"
                result.append((left, right, kind))
            elif info["id"] == 9 and other["id"] == 4 and info["batch"] == other["batch"]:
                kind = "nce_degenerate_vs_cos_b1" if info["batch"] == 1 else "nce_vs_cos_same_batch"
                result.append((left, right, kind))
    require(len(result) == len(set(result)) == 36, "Unexpected comparison grid")
    return result


def summarize(raw, draws=10000):
    names, states = deployment_conditions(), config()["evaluation"]["state_indices"]
    require(len(raw) == 40 * len(states) * len(names) and not raw.duplicated(["suite", "task_id", "state_index", "condition"]).any(), "Incomplete/duplicate grid")
    require(set(raw.suite) == set(SUITES) and set(raw.condition) == set(names), "Suite/condition coverage changed")
    require(raw.success.isin([0, 1]).all() and raw.failure.eq(1 - raw.success).all(), "Invalid outcomes")
    require(raw.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(), "Unpaired seeds")
    summaries, contrasts, tasks = [], [], []
    for suite in SUITES:
        sub = raw.loc[raw.suite.eq(suite)]
        clean = sub.loc[sub.condition.eq("clean")]
        require(set(zip(clean.task_id, clean.state_index)) == {(t, s) for t in range(10) for s in states}
                and sub.groupby(["task_id", "state_index"]).condition.nunique().eq(len(names)).all(), "Suite grid changed")
        qualified = int(clean.groupby("task_id").success.sum().gt(0).sum())
        summaries.append({"suite": suite, "method": "clean", "candidate_id": 0, "batch_size": 0,
                          "task_success_rate": float(clean.success.mean()), "attack_failure_rate": float(clean.failure.mean()),
                          "task_equal_casr": 0., "ci_low": 0., "ci_high": 0., "fresh_clean_success": int(clean.success.sum()),
                          "qualified_tasks": qualified, "nce_degenerate_control": False})
        pairs, estimates = {}, {}
        for method in METHODS:
            info = arm_info(method)
            point, paired = task_casr(sub, condition(method))
            paired["value"] = paired.failure
            ci = paired_bootstrap(paired, draws)
            subset = sub.loc[sub.condition.eq(condition(method))]
            summaries.append({"suite": suite, "method": method, "candidate_id": info["id"], "batch_size": info["batch"],
                              "task_success_rate": float(subset.success.mean()), "attack_failure_rate": float(subset.failure.mean()),
                              "task_equal_casr": point, "ci_low": ci[0], "ci_high": ci[1], "fresh_clean_success": int(clean.success.sum()),
                              "qualified_tasks": qualified, "nce_degenerate_control": info["id"] == 9 and info["batch"] == 1})
            pairs[method], estimates[method] = paired, point
            for task, group in paired.groupby("task_id"):
                eligible = group.clean_success.eq(1)
                tasks.append({"suite": suite, "method": method, "candidate_id": info["id"], "batch_size": info["batch"],
                              "task_id": int(task), "clean_success_states": int(eligible.sum()),
                              "casr": float(group.loc[eligible].failure.mean()) if eligible.any() else None})
        for left, right, kind in comparisons():
            joint = pairs[left].merge(pairs[right][["task_id", "state_index", "failure"]], on=["task_id", "state_index"],
                                     suffixes=("_a", "_b"), validate="one_to_one")
            joint["value"] = joint.failure_a - joint.failure_b
            ci = paired_bootstrap(joint, draws)
            delta = joint.loc[joint.clean_success.eq(1)].groupby("task_id").value.mean()
            contrasts.append({"suite": suite, "left": left, "right": right, "contrast_type": kind,
                              "casr_difference": estimates[left] - estimates[right], "ci_low": ci[0], "ci_high": ci[1],
                              "positive_tasks": int(delta.gt(0).sum()), "negative_tasks": int(delta.lt(0).sum()),
                              "tied_tasks": int(delta.eq(0).sum())})
    return pd.DataFrame(summaries), pd.DataFrame(contrasts), pd.DataFrame(tasks)


def analyze():
    prepared()
    signature, frames, traces = run_signature(), [], []
    for suite in SUITES:
        for task in range(10):
            folder = ROOT / "evaluation" / suite
            status = load_json(folder / f"task_{task:02d}_STATUS.json")
            rp, tp = folder / f"task_{task:02d}_rollouts.parquet", folder / f"task_{task:02d}_traces.parquet"
            require(status["status"] == "COMPLETE" and status["signature"] == signature
                    and status["rollouts_sha256"] == sha256_file(rp) and status["traces_sha256"] == sha256_file(tp), "Incomplete/changed shard")
            frames.append(pd.read_parquet(rp))
            traces.append(pd.read_parquet(tp))
    raw, trace = pd.concat(frames, ignore_index=True), pd.concat(traces, ignore_index=True)
    require(raw.signature.eq(signature).all() and not raw.evaluation_training_hooks.any(), "Unexpected runtime/hooks")
    require(raw.renderer_vendor.str.lower().str.contains("nvidia").all() and raw.renderer.str.contains("4090").all(), "Non-4090 renderer")
    summaries, contrasts, tasks = summarize(raw)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    atomic_parquet(raw, output / "BATCH500_ROLLOUTS.parquet")
    atomic_parquet(trace, output / "BATCH500_TRACES.parquet")
    summaries.to_csv(output / "BATCH500_SUMMARY.csv", index=False)
    contrasts.to_csv(output / "BATCH500_CONTRASTS.csv", index=False)
    tasks.to_csv(output / "BATCH500_PER_TASK.csv", index=False)
    validation = load_json(ROOT / "reports/AUTO_VALIDATION.json")
    changes = {row["method"]: row for row in validation["training_changes"]}
    timing = pd.DataFrame(validation["training_changes"])
    timing.drop(columns=["rgb_changes"]).to_csv(output / "BATCH500_TRAINING_COST.csv", index=False)
    lines = ["# 16组500轮：逐套件CASR与同batch基线", "", "数据来自本轮配对1360回合；不跨套件平均。",
             "Spatial是来源内，其余三个套件是补丁迁移；40/41为预先固定的开发初态；仅作初步比较。9-B1是退化对照。", "",
             "|Suite|Clean成功/20|", "|---|---:|"]
    for row in summaries.loc[summaries.method.eq("clean")].itertuples():
        lines.append(f"|{row.suite}|{row.fresh_clean_success}| ")
    for batch in BATCHES:
        lines += ["", f"## Batch size={batch}", "", "|ID|方法|Object%|Spatial%|Goal%|Long%|训练小时|", "|---|---|---:|---:|---:|---:|---:|"]
        for i, family in FAMILIES.items():
            method = f"m{i:02d}_{family}_b{batch}"
            values = [f"{100*float(summaries.loc[summaries.method.eq(method) & summaries.suite.eq(s),'task_equal_casr'].iloc[0]):.2f}" for s in SUITES]
            label = family + ("（退化对照）" if i == 9 and batch == 1 else "")
            lines.append(f"|{i}|{label}|" + "|".join(values) + f"|{changes[method]['elapsed_h']:.2f}|")
        lines += ["", "同batch相对两种基线的差值（百分点）：", "",
                  "|ID|Suite|对UADA|对TMA|", "|---|---|---:|---:|"]
        for method in METHODS:
            info = arm_info(method)
            if info["batch"] != batch or info["id"] in {1, 12}:
                continue
            for suite in SUITES:
                d = contrasts.loc[contrasts.left.eq(method) & contrasts.suite.eq(suite) & contrasts.contrast_type.eq("candidate_vs_matched_baseline")]
                values = [100*float(d.loc[d.right.eq(f"m{i:02d}_{FAMILIES[i]}_b{batch}"),"casr_difference"].iloc[0]) for i in (1, 12)]
                lines.append(f"|{info['id']}|{suite}|{values[0]:+.2f}|{values[1]:+.2f}|")
    lines += ["", "所有组同25000次更新，但B2处理样本次数更多。NCE的B变化也改变负样本数。",
              "每任务结果、同方法B对比、NCE对cos、区间和成本见同目录CSV；不自动扩轮或跨模型。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json({"status": "COMPLETE", "signature": signature, "episodes": len(raw), "methods": len(METHODS),
                 "summary_sha256": sha256_file(output / "BATCH500_SUMMARY.csv"),
                 "raw_sha256": sha256_file(output / "BATCH500_ROLLOUTS.parquet"),
                 "traces_sha256": sha256_file(output / "BATCH500_TRACES.parquet"),
                 "independent_patch_training_seeds": 1}, output / "STATUS.json")
    from diagnostic_report import report
    report()
    print("BATCH500_COMPLETE: 16 arms / 4 suites / 1360 paired episodes; see results/RESULTS.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "stats"), required=True)
    {"validate": validate_training, "stats": analyze}[parser.parse_args().stage]()

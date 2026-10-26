"""Complete factorial cells, paired task/state intervals and explicit provenance."""
from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, SUITES, CELLS, arm_info, config, experiment, prepared,
                    run_signature, atomic_json, atomic_parquet, load_json, require, sha256_file)
from monitor import collect_steps

CONTRASTS = {
    "dino_without_frequency": [-1, 1, 0, 0],
    "dino_with_frequency": [0, 0, -1, 1],
    "frequency_without_dino": [-1, 0, 1, 0],
    "frequency_with_dino": [0, -1, 0, 1],
    "combined_vs_uada": [-1, 0, 0, 1],
    "interaction": [1, -1, -1, 1],
}

def validate_training():
    prepared(); signature = run_signature(); patches = {}; evidence = []
    for arm in METHODS:
        os.environ["OFT_FACTORIAL_ARM"] = arm
        from train import validate_resume
        cfg = config(arm); folder = ROOT / "training" / arm
        status = load_json(folder / "status.json")
        require(status["status"] == "COMPLETE" and status["signature"] == signature and status["outer"] == 1000,
                f"Incomplete training: {arm}")
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        validate_resume(saved, arm, cfg, signature)
        expected = torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(42))
        require(torch.equal(expected, saved["initial"]) and saved["outer"] == 1000, "Initialization/budget mismatch")
        rows = pd.read_parquet(folder / "training.parquet")
        require(rows.outer_iteration.tolist() == list(range(1, 1001))
                and rows.optimizer_updates.tolist() == list(range(50, 50001, 50)), "Outer/update sequence incomplete")
        require(rows.effective_rgb_updates.sum() > 0 and not torch.equal(expected, saved["patch"]), "No actual RGB learning")
        # The saved scheduler is already advanced after outer1000. Check the
        # exact horizon explicitly rather than mistaking epoch1000 for its end.
        expected_lr = .002 * max(0., .5 * (1 + math.cos(math.pi * (1000 - 20) / (cfg["schedule_outer_iterations"] - 20))))
        require(abs(saved["optimizer"]["param_groups"][0]["lr"] - expected_lr) < 1e-12, "Wrong cosine horizon")
        for outer in cfg["checkpoint_outer_iterations"]:
            path = folder / f"outer_{outer:04d}.pt"
            value = torch.load(path, map_location="cpu", weights_only=False)
            require(value["signature"] == signature and value["method"] == arm and value["config"] == cfg
                    and value["outer_iterations"] == outer and value["optimizer_updates"] == outer * 50
                    and torch.equal(value["initial_patch"], expected), "Snapshot provenance changed")
            patch = value["patch"]
            require(patch.shape == (3, 50, 50) and torch.isfinite(patch).all()
                    and ((patch >= 0) & (patch <= 1)).all(), "Invalid RGB snapshot")
            with Image.open(path.with_suffix(".png")) as image:
                require(image.mode == "RGB" and image.size == (50, 50)
                        and np.array_equal(np.asarray(image), patch.permute(1, 2, 0).mul(255).round().byte().numpy()),
                        "PNG differs from actual trained RGB")
            if outer == 1000:
                require(torch.equal(patch, saved["patch"]), "Final checkpoint/optimizer RGB mismatch")
        steps = collect_steps(folder)
        require(len(steps) == 50000 and list(zip(steps.outer, steps.inner)) == [(o, i) for o in range(1, 1001) for i in range(50)]
                and steps.signature.eq(signature).all() and steps.arm.eq(arm).all(), "Actual inner-step monitoring incomplete")
        numeric = steps.select_dtypes(include=["number"])
        require(np.isfinite(numeric.to_numpy()).all(), "Nonfinite monitored step")
        atomic_parquet(steps, folder / "MONITORING_STEPS.parquet")
        probes = {}
        for path in sorted((folder / "attempts").glob("*/probe_*.json")):
            report = load_json(path)
            require(report["signature"] == signature and report["arm"] == arm, "Probe provenance changed")
            require(path.with_suffix(".pt").exists(), "Missing full probe tensors")
            probes[(report["outer"], report["inner"])] = report
        require(set(probes) == {(o, i) for o in experiment()["diagnostic_outers"] for i in experiment()["diagnostic_inners"]},
                "Fixed native/PNG probes incomplete")
        active = steps.rgb_update_l2.gt(0)
        evidence.append({"arm": arm, "protocol": cfg["protocol"], "schedule_outer_iterations": cfg["schedule_outer_iterations"],
                         "updates": 50000, "probe_count": len(probes), "training_elapsed_h": status["elapsed_s"] / 3600,
                         "active_step_fraction": float(active.mean()),
                         "active_step_adverse_native_direction_fraction": float(steps.loc[active, "native_gradient_dot_update"].gt(0).mean()),
                         "probe_uada_increase_fraction": float(np.mean([v["native_uada_change"] > 0 for v in probes.values()])),
                         "max_png_action_gap": max(v["png_vs_training_action_l2"] for v in probes.values()),
                         "steps_sha256": sha256_file(folder / "MONITORING_STEPS.parquet")})
        patches[arm] = {"pt": sha256_file(folder / "outer_1000.pt"), "png": sha256_file(folder / "outer_1000.png")}
    atomic_json({"status": "COMPLETE", "signature": signature, "patches": patches, "training": evidence,
                 "source_goal": "native expert-opposite X MSE, eight slots", "model_weights_updated": False,
                 "scientific_result": "PENDING_CLOSED_LOOP"}, ROOT / "reports/AUTO_VALIDATION.json")
    print("FACTORIAL_TRAINING_VALIDATED: 3x50000 updates, exact native PNGs and full step logs", flush=True)

def validate_grid(raw):
    require(len(raw) == 4000 and not raw.duplicated(["protocol", "suite", "task_id", "state_index", "condition"]).any(),
            "Expected4000 unique protocol/suite/task/state/condition rows")
    require(raw.success.isin([0, 1]).all() and raw.failure.eq(1 - raw.success).all(), "Invalid episode outcomes")
    require(int((~raw.reused).sum()) == 1200 and int(raw.reused.sum()) == 2800, "New/reused episodes mixed")
    for protocol in ("legacy", "original"):
        for suite in SUITES:
            sub = raw.loc[raw.protocol.eq(protocol) & raw.suite.eq(suite)]
            expected = {(t, s, c) for t in range(10) for s in range(40, 50) for c in ("clean", *CELLS)}
            require(set(zip(sub.task_id, sub.state_index, sub.condition)) == expected, "Missing factorial pairs")
            require(sub.groupby(["task_id", "state_index"]).rollout_seed.nunique().eq(1).all(), "Unpaired reset seeds")

def bootstrap_cells(groups, draws, rng):
    # Every draw resamples tasks, then states within each selected task. All four
    # treatments share sampled states, preserving paired outcomes and denominators.
    task_indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    sampled = np.empty((draws, len(groups), len(CELLS)))
    for task, group in enumerate(groups):
        occurrences = task_indices == task
        count = int(occurrences.sum())
        if count:
            # Repeated occurrences of the same task get independent state draws.
            indices = rng.integers(0, len(group), size=(count, len(group)))
            sampled[occurrences] = group[indices].mean(axis=1)
    return sampled.mean(axis=1)

def summarize(raw, draws=10000, seed=20261017):
    validate_grid(raw)
    summaries = []; contrasts = []; tasks = []
    for protocol in ("legacy", "original"):
        suite_draws = []; suite_points = []; suite_eligible = []
        for suite_index, suite in enumerate(SUITES):
            sub = raw.loc[raw.protocol.eq(protocol) & raw.suite.eq(suite)]
            pivot = sub.pivot(index=["task_id", "state_index"], columns="condition", values="failure")
            eligible = pivot.loc[pivot.clean.eq(0), list(CELLS)]
            groups = [g.to_numpy(float) for _, g in eligible.groupby(level="task_id")]
            require(groups, f"No clean-success tasks: {protocol}/{suite}")
            points = np.mean([g.mean(axis=0) for g in groups], axis=0)
            rng = np.random.default_rng(seed + 100 * (protocol == "original") + suite_index)
            boot = bootstrap_cells(groups, draws, rng)
            suite_points.append(points); suite_draws.append(boot); suite_eligible.append(eligible.to_numpy())
            for j, cell in enumerate(CELLS):
                cond = sub.loc[sub.condition.eq(cell)]
                ci = np.quantile(boot[:, j], [.025, .975])
                summaries.append({"protocol": protocol, "suite": suite, "method": cell, "casr": float(points[j]),
                                  "ci_low": float(ci[0]), "ci_high": float(ci[1]), "task_success_rate": float(cond.success.mean()),
                                  "attack_failure_rate": float(cond.failure.mean()), "clean_success_states": len(eligible),
                                  "qualified_tasks": len(groups), "new_episodes": int((~cond.reused).sum())})
                for task in range(10):
                    valid = eligible.loc[eligible.index.get_level_values("task_id") == task, cell]
                    tasks.append({"protocol": protocol, "suite": suite, "method": cell, "task_id": task,
                                  "eligible_states": len(valid), "casr": float(valid.mean()) if len(valid) else None})
            contrasts.extend(contrast_rows(protocol, suite, points, boot, eligible.to_numpy()))
        points = np.mean(suite_points, axis=0); boot = np.mean(suite_draws, axis=0)
        for j, cell in enumerate(CELLS):
            ci = np.quantile(boot[:, j], [.025, .975])
            summaries.append({"protocol": protocol, "suite": "four_suite_equal_mean", "method": cell,
                              "casr": float(points[j]), "ci_low": float(ci[0]), "ci_high": float(ci[1])})
        contrasts.extend(contrast_rows(protocol, "four_suite_equal_mean", points, boot, np.concatenate(suite_eligible)))
    return pd.DataFrame(summaries), pd.DataFrame(contrasts), pd.DataFrame(tasks)

def contrast_rows(protocol, suite, points, boot, eligible):
    result = []
    for name, coefficients in CONTRASTS.items():
        coef = np.asarray(coefficients); values = boot @ coef; ci = np.quantile(values, [.025, .975])
        row = {"protocol": protocol, "suite": suite, "contrast": name, "difference": float(points @ coef),
               "ci_low": float(ci[0]), "ci_high": float(ci[1])}
        if np.abs(coef).sum() == 2:
            difference = eligible @ coef
            row.update(new_failures=int((difference > 0).sum()), lost_failures=int((difference < 0).sum()))
        result.append(row)
    return result

def recommendation(summary):
    series = summary.loc[summary.protocol.eq("original") & summary.suite.eq("four_suite_equal_mean")].set_index("method").casr
    if series.uada_dino_freq > max(series.uada_original, series.uada_freq):
        return "SECOND_SEED_COMPARISON_WORTHWHILE"
    if series.uada_freq > series.uada_original and series.uada_freq >= max(series.uada_dino, series.uada_dino_freq):
        return "KEEP_FREQUENCY_ONLY_STOP_DINO_ENHANCEMENT"
    if series.uada_dino > series.uada_original and series.uada_dino > max(series.uada_freq, series.uada_dino_freq):
        return "DINO_ONLY_GAIN_REQUIRES_MANUAL_SECOND_SEED_PLAN"
    return "STOP_THIS_ENHANCEMENT_ROUTE"

def stats():
    from evaluate import task_complete
    prepared(); signature = run_signature(); output = ROOT / "results"; output.mkdir(exist_ok=True)
    old_status = output / "STATUS.json"
    if old_status.exists() and load_json(old_status).get("status") == "COMPLETE":
        old = load_json(old_status)
        require(old["signature"] == signature and all(sha256_file(output / p) == h for p, h in old["artifacts"].items()),
                "Completed factorial results changed")
        print("FACTORIAL_RESULTS_ALREADY_COMPLETE", flush=True); return
    validation = load_json(ROOT / "reports/AUTO_VALIDATION.json")
    require(validation["signature"] == signature and validation["status"] == "COMPLETE", "Missing training validation")
    frames = [pd.read_parquet(ROOT / "data/HISTORICAL_ROLLOUTS.parquet")]; trace_frames = []
    for arm in METHODS:
        for suite in SUITES:
            folder = ROOT / "evaluation" / arm / suite
            for task in range(10):
                require(task_complete(folder, task, signature, validation["patches"][arm]["png"]), "Incomplete evaluation shard")
                frames.append(pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet"))
                trace_frames.append(pd.read_parquet(folder / f"task_{task:02d}_traces.parquet"))
    raw = pd.concat(frames, ignore_index=True)
    raw["comparison_signature"] = signature
    new = raw.loc[~raw.reused]
    require(new.signature.eq(signature).all() and not raw.evaluation_training_hooks.any(), "Unexpected inference identity/hooks")
    require(new.renderer_vendor.str.lower().str.contains("nvidia").all() and new.renderer.str.contains("4090").all(), "Wrong GPU renderer")
    summary, contrasts, tasks = summarize(raw, experiment()["bootstrap_draws"], experiment()["bootstrap_seed"])
    atomic_parquet(raw, output / "FACTORIAL_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(trace_frames, ignore_index=True), output / "NEW_TRACES.parquet")
    for name, frame in (("SUMMARY", summary), ("CONTRASTS", contrasts), ("PER_TASK", tasks)):
        frame.to_csv(output / f"FACTORIAL_{name}.csv", index=False)
    decision = {"recommendation": recommendation(summary), "future_jobs_submitted": False,
                "independent_training_seeds": 1, "development_states": True,
                "new_episodes": 1200, "reused_episodes": 2800}
    atomic_json(decision, output / "DECISION.json")
    costs = [load_json(p) for p in (ROOT / "reports/WORKER_COSTS").glob("*.json")]
    atomic_json({"training": validation["training"], "gpu_worker_hours_including_failed_attempts":
                 sum(c["elapsed_s"] for c in costs) / 3600, "workers": costs}, output / "INTERNAL_EVIDENCE.json")
    lines = ["# UADA factorial missing-cell results", "", "CASR percentages; higher is stronger. New1200 / reused2800 episodes."]
    for protocol in ("legacy", "original"):
        lines += ["", f"## {protocol}", "", "| Method | Object | Spatial | Goal | Long | Equal mean |",
                  "|---|---:|---:|---:|---:|---:|"]
        table = summary.loc[summary.protocol.eq(protocol)].pivot(index="method", columns="suite", values="casr")
        for cell in CELLS:
            lines.append("| " + cell + " | " + " | ".join(f"{100*table.loc[cell, s]:.2f}" for s in (*SUITES, "four_suite_equal_mean")) + " |")
    lines += ["", "Recommendation: **" + decision["recommendation"] + "**.",
              "", "Single patch-training seed42. Task/state intervals do not represent training-seed stability.",
              "Protocol differences are bundled; no isolated EOT attribution. Historical origin signatures/files are retained.",
              "Native UADA objective is unchanged. Monitoring explains optimization, not a substitute success endpoint.",
              "No follow-up job is submitted automatically. See CONTRASTS, PER_TASK and INTERNAL_EVIDENCE."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    artifacts = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "STATUS.json"}
    atomic_json({"status": "COMPLETE", "signature": signature, "artifacts": artifacts, **decision}, old_status)
    print("FACTORIAL_COMPLETE " + decision["recommendation"], flush=True)

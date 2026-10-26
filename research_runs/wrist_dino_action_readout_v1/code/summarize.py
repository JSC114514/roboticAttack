"""Paired clean-conditional, task-equal metrics and measured internal evidence."""
from __future__ import annotations
import numpy as np
import pandas as pd
from run_core import (ARMS, BASE, BASE_CONDITION, ROOT, SUITES, finish, prepared, read_json,
                      require, save_frame, signature, stage_done, write_json)


def compare(frame):
    keys = ["suite", "task_id", "state_index", "rollout_seed"]
    require(not frame.duplicated(keys + ["condition"]).any(), "Duplicate paired outcomes")
    wide = frame.pivot(index=keys, columns="condition", values="success").reset_index()
    require(all(c in wide for c in ["clean", BASE_CONDITION, *ARMS]), "Missing comparison condition")
    require(not wide[["clean", BASE_CONDITION, *ARMS]].isna().any().any(), "Unpaired outcomes")
    eligible = wide.loc[wide.clean.eq(1)].copy()
    rows, tasks = [], []
    for suite in SUITES:
        group = eligible.loc[eligible.suite.eq(suite)]
        for condition in [BASE_CONDITION, *ARMS]:
            task_rates = (1 - group.groupby("task_id")[condition].mean())
            rows.append({"suite": suite, "condition": condition, "casr": float(task_rates.mean()),
                         "clean_success_states": len(group), "tasks_with_clean_success": len(task_rates)})
            tasks.extend({"suite": suite, "task_id": int(t), "condition": condition, "casr": float(v)} for t, v in task_rates.items())
    metrics = pd.DataFrame(rows)
    scores = metrics.groupby("condition").casr.mean().to_dict()
    c, b = ARMS[1], ARMS[0]
    positive_suites = sum(float(metrics.loc[metrics.suite.eq(s) & metrics.condition.eq(c), "casr"].iloc[0]) >
                          float(metrics.loc[metrics.suite.eq(s) & metrics.condition.eq(BASE_CONDITION), "casr"].iloc[0]) for s in SUITES)
    if scores[c] > scores[BASE_CONDITION] and scores[c] > scores[b] and positive_suites >= 2:
        decision = "C_ADVANCES_TO_MANUAL_SECOND_SEED"
    elif scores[b] > scores[BASE_CONDITION] and scores[c] <= scores[b]:
        decision = "OBJECTIVE_GAIN_WITHOUT_DINO_GAIN"
    else:
        decision = "STOP_THIS_READOUT_ROUTE"
    pairs = []
    for suite in SUITES:
        sub = eligible.loc[eligible.suite.eq(suite)]
        for reference in [BASE_CONDITION, b]:
            pairs.append({"suite": suite, "reference": reference,
                          "C_only_causes_failure": int((sub[c].eq(0) & sub[reference].eq(1)).sum()),
                          "reference_only_causes_failure": int((sub[c].eq(1) & sub[reference].eq(0)).sum())})
    return metrics, pd.DataFrame(tasks), pd.DataFrame(pairs), {"scores": scores, "C_improved_suites_vs_UADA": int(positive_suites), "decision": decision}


def summarize():
    prepared()
    frames = [pd.read_parquet(ROOT / "data/REUSED_BASELINE.parquet")]
    for arm in ARMS:
        require(stage_done(ROOT / "training" / arm / "STATUS.json"), "Training incomplete")
        for suite in SUITES:
            for task in range(10):
                folder = ROOT / "evaluation" / arm / suite
                require(stage_done(folder / f"task_{task:02d}_STATUS.json"), "Evaluation incomplete")
                frames.append(pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet"))
    frame = pd.concat(frames, ignore_index=True)
    require(len(frame) == 320, "Expected 160 reused + 160 new episodes")
    metrics, tasks, pairs, decision = compare(frame)
    evidence = []
    for arm in ARMS:
        folder = ROOT / "training" / arm
        inner = pd.read_parquet(folder / "inner_steps.parquet")
        diag = pd.read_parquet(folder / "diagnostics.parquet")
        require(len(inner) == 25000 and not inner.duplicated(["outer", "inner"]).any(), "Training monitor incomplete")
        require(len(diag) == 16 and np.isfinite(diag[["native_uada_before", "native_uada_after", "png_action_gap_l2"]]).all().all(), "Diagnostic trace incomplete")
        angles = diag.loc[diag.beta.gt(0), "native_helper_gradient_cosine"].dropna()
        evidence.append({"arm": arm, "native_target_mse_step_change_mean": float((diag.native_uada_after - diag.native_uada_before).mean()),
                         "png_action_gap_l2_mean": float(diag.png_action_gap_l2.mean()),
                         "helper_native_gradient_conflict_fraction": None if len(angles) == 0 else float(angles.lt(0).mean()),
                         "helper_active_update_fraction": float(inner.beta.gt(0).mean()),
                         "gradient_component_sum_relative_error_max": float(diag.component_gradient_sum_relative_error.max()),
                         "mean_saturated_pixel_fraction": float(inner.saturated_fraction.mean()),
                         "mean_png_changed_fraction": float(inner.png_changed_fraction.mean()),
                         "elapsed_h": read_json(folder / "STATUS.json")["elapsed_s"] / 3600})
    worker_costs = [read_json(p) for p in (ROOT / "reports/WORKER_COSTS").glob("*.json")]
    worker_costs = [x for x in worker_costs if x["signature"] == signature() and x["gpu_worker"]]
    total_gpu_h = sum(x["elapsed_s"] for x in worker_costs) / 3600
    write_json({"signature": signature(), "head": read_json(ROOT / "readout/DIAGNOSTICS.json"), "training": evidence,
                "gpu_worker_elapsed_h_including_failed_attempts": total_gpu_h, "worker_costs": worker_costs}, ROOT / "results/INTERNAL_EVIDENCE.json")
    write_json({"signature": signature(), **decision, "second_seed_submitted": False}, ROOT / "results/DECISION.json")
    save_frame(frame, ROOT / "results/ALL_ROLLOUTS.parquet")
    save_frame(metrics, ROOT / "results/SUITE_METRICS.parquet")
    save_frame(tasks, ROOT / "results/TASK_METRICS.parquet")
    save_frame(pairs, ROOT / "results/PAIRED_OUTCOMES.parquet")
    from run_core import atomic_write
    atomic_write(ROOT / "results/SUITE_METRICS.csv", lambda p: metrics.to_csv(p, index=False))
    atomic_write(ROOT / "results/PAIRED_OUTCOMES.csv", lambda p: pairs.to_csv(p, index=False))
    lines = ["# DINO action readout: first-seed paired results", "", "CASR: clean-success states only; task-equal within suite. Higher is stronger.", "",
             "|Method|Object|Spatial|Goal|Long|Four-suite equal mean|", "|---|---:|---:|---:|---:|---:|"]
    for condition, label in [(BASE_CONDITION, "A reused UADA"), (ARMS[0], "B native energy"), (ARMS[1], "C DINO readout")]:
        values = [float(metrics.loc[metrics.condition.eq(condition) & metrics.suite.eq(s), "casr"].iloc[0]) for s in SUITES]
        lines.append("|" + "|".join([label, *[f"{100*x:.2f}%" for x in values], f"{100*np.mean(values):.2f}%"]) + "|")
    lines += ["", "|Suite|C minus UADA (pp)|C minus B (pp)|", "|---|---:|---:|"]
    for suite in SUITES:
        values = metrics.loc[metrics.suite.eq(suite)].set_index("condition").casr
        lines.append(f"|{suite}|{100*(values[ARMS[1]]-values[BASE_CONDITION]):+.2f}|{100*(values[ARMS[1]]-values[ARMS[0]]):+.2f}|")
    lines += ["", "Decision: **" + decision["decision"] + "**", "", "Fixed development initial states 40/41; one patch seed 42.",
              f"Measured new GPU-worker wall time, including loading/failed attempts, excluding queue wait: {total_gpu_h:.3f} GPUh.",
              "Teacher supervision uses additional source-train windows; patch update counts match, total compute differs.",
              "Read results/INTERNAL_EVIDENCE.json, readout/DIAGNOSTICS.json and training/*/diagnostics.parquet for measured internal evidence.",
              "No second seed is submitted automatically."]
    path = ROOT / "results/RESULTS.md"
    atomic_write(path, lambda p: p.write_text("\n".join(lines) + "\n"))
    artifacts = list((ROOT / "results").glob("*.parquet")) + list((ROOT / "results").glob("*.csv")) + [path, ROOT / "results/DECISION.json", ROOT / "results/INTERNAL_EVIDENCE.json"]
    finish(ROOT / "results/STATUS.json", artifacts, new_rollouts=160, reused_rollouts=160, **decision)
    print("RESULTS_COMPLETE", decision["decision"], flush=True)


if __name__ == "__main__":
    summarize()

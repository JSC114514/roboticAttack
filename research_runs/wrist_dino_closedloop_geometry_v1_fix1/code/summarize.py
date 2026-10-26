"""Paired complete-episode outcomes, query costs, and internal search evidence."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from loop_core import (CONDITIONS, ROOT, SUITES, cfg, finish, locked, prepared, read_json,
                       require, save_frame, signature, stage_done, write_json)


def metrics(rows):
    keys = ["suite", "task_id", "state_index"]
    require(not rows.duplicated(keys + ["condition"]).any(), "Duplicate formal episodes")
    clean = rows.loc[rows.condition.eq("clean"), keys + ["success"]].rename(columns={"success": "clean_success"})
    joined = rows.merge(clean, on=keys, validate="many_to_one")
    result, pairs = [], []
    for suite in SUITES:
        selected = joined.loc[joined.suite.eq(suite)]
        eligible = selected.loc[selected.clean_success.eq(1)]
        for condition in CONDITIONS:
            group = selected.loc[selected.condition.eq(condition)]
            good = eligible.loc[eligible.condition.eq(condition)]
            casr = float(good.groupby("task_id").failure.mean().mean()) if len(good) else np.nan
            result.append({"suite": suite, "condition": condition, "task_equal_casr": casr,
                           "clean_success_states": len(good), "qualified_tasks": good.task_id.nunique(),
                           "successes": int(group.success.sum()), "episodes": len(group),
                           "policy_calls": int(group.policy_calls.sum()), "episode_elapsed_s": float(group.elapsed_s.sum())})
        wide = eligible.pivot(index=["task_id", "state_index"], columns="condition", values="failure").reindex(columns=CONDITIONS)
        for control in ("uada_geometry", "uada_static"):
            pairs.append({"suite": suite, "comparison": "loop_geometry-minus-" + control,
                          "new_failures": int(((wide.loop_geometry == 1) & (wide[control] == 0)).sum()),
                          "lost_failures": int(((wide.loop_geometry == 0) & (wide[control] == 1)).sum()),
                          "casr_difference": float(wide.groupby(level="task_id").mean().loop_geometry.mean()
                                                   - wide.groupby(level="task_id").mean()[control].mean())})
    return pd.DataFrame(result), pd.DataFrame(pairs), joined


def summarize():
    preparation = prepared()
    folder = ROOT / "results"
    with locked(folder / "summary.lock"):
        if stage_done(folder / "STATUS.json"):
            print("SUMMARY ALREADY_COMPLETE"); return
        frames, plans, candidates = [], [], []
        for suite in SUITES:
            for task in cfg()["tasks"]:
                for condition in CONDITIONS:
                    status = ROOT / "evaluation" / suite / f"task_{task:02d}" / condition / "STATUS.json"
                    require(stage_done(status), "Formal episode incomplete: " + str(status))
                    receipt = read_json(status)
                    attempt = ROOT / receipt["attempt"]
                    frames.append(pd.read_parquet(attempt / "ROLLOUT.parquet"))
                    for name, destination in (("PLANS", plans), ("CANDIDATES", candidates)):
                        frame = pd.read_parquet(attempt / (name + ".parquet"))
                        if len(frame):
                            for key, value in {"suite": suite, "task_id": task, "condition": condition}.items():
                                frame[key] = value
                            destination.append(frame)
        rows = pd.concat(frames, ignore_index=True)
        require(len(rows) == 160, "Expected 160 fresh paired episodes")
        report, paired, joined = metrics(rows)
        means = {}
        for condition, group in report.groupby("condition"):
            means[condition] = float(group.task_equal_casr.mean()) if group.task_equal_casr.notna().all() else None
        supported = means["loop_geometry"] is not None and means["uada_geometry"] is not None
        decision = {"signature": signature(), "four_suite_equal_mean": means,
                    "loop_minus_matched_uada_geometry": means["loop_geometry"] - means["uada_geometry"] if supported else None,
                    "loop_minus_static_uada_reference": means["loop_geometry"] - means["uada_static"] if supported else None,
                    "decision": ("CLOSEDLOOP_SELECTOR_IMPROVES" if means["loop_geometry"] > means["uada_geometry"]
                                 else "CLOSEDLOOP_SELECTOR_DOES_NOT_IMPROVE") if supported else "NO_CLEAN_SUPPORT_IN_A_SUITE",
                    "automatic_followup": False,
                    "comparison_scope": "matched digital geometry with privileged simulator branching; static PNG is a separate reference"}
        plan_rows = pd.concat(plans, ignore_index=True)
        candidate_rows = pd.concat(candidates, ignore_index=True)
        evidence = []
        for condition, group in plan_rows.groupby("condition"):
            selected = candidate_rows.loc[candidate_rows.condition.eq(condition) & candidate_rows.selected]
            evidence.append({"condition": condition, "planning_windows": len(group),
                             "candidate_count_min": int(group.candidate_count.min()),
                             "candidate_count_max": int(group.candidate_count.max()),
                             "mean_queries_per_window": float(group.policy_calls.mean()),
                             "rollback_error_max": float(group.rollback_state_max_error.max()),
                             "committed_action_error_max": float(group.commit_action_max_error.max()),
                             "committed_physics_error_max": float(group.commit_state_max_error.max()),
                             "zero_goal_score_range_fraction": float(group.goal_score_range.eq(0).mean()),
                             "mean_dense_score_range": float(group.dense_score_range.mean()),
                             "selected_candidates": selected.candidate.value_counts().to_dict(),
                             "selected_dino_relative_l2_mean": float(selected.dino_relative_l2.mean()),
                             "selected_rgb_mae_255_mean": float(selected.rgb_mae_255.mean())})
        costs = [read_json(p) for p in (ROOT / "reports/WORKER_COSTS").glob("*.json")]
        new_gpu_h = sum(r["elapsed_s"] for r in costs if r["gpu_worker"]) / 3600
        prior_gpu_h = preparation["repair_reuse"]["prior_gpu_worker_hours_including_failed_goal"]
        gpu_h = new_gpu_h + prior_gpu_h
        artifacts = []
        for name, frame in (("ALL_ROLLOUTS", joined), ("SUITE_METRICS", report), ("PAIRED_OUTCOMES", paired),
                            ("ALL_PLANS", plan_rows), ("ALL_CANDIDATES", candidate_rows)):
            p = folder / (name + ".parquet"); save_frame(frame, p); artifacts.append(p)
        write_json(decision, folder / "DECISION.json"); artifacts.append(folder / "DECISION.json")
        write_json({"search_evidence": evidence, "gpu_worker_hours_including_failed_attempts": gpu_h,
                    "new_repair_gpu_worker_hours": new_gpu_h, "prior_attempt_gpu_worker_hours": prior_gpu_h,
                    "reused_qa_suites": list(preparation["repair_reuse"]["qa"]),
                    "progress_is_search_proxy": True, "endpoint": "complete episode success from LIBERO"}, folder / "INTERNAL_EVIDENCE.json")
        artifacts.append(folder / "INTERNAL_EVIDENCE.json")
        lines = ["# Wrist DINO closed-loop geometry pilot", "", "160 fresh paired episodes; four suites, ten tasks, initial state 40.",
                 "CASR uses fresh clean-success states, equal task weights. Higher is stronger.", "",
                 "|Condition|Object|Spatial|Goal|Long|Four-suite mean|", "|---|---:|---:|---:|---:|---:|"]
        for condition in CONDITIONS:
            values = [float(report.loc[report.condition.eq(condition) & report.suite.eq(s), "task_equal_casr"].iloc[0]) for s in SUITES]
            lines.append("|" + condition + "|" + "|".join(f"{100 * v:.2f}%" if v is not None and np.isfinite(v)
                                                          else "NA (no clean-success support)" for v in values + [means[condition]]) + "|")
        lines.extend(["", f"Decision: **{decision['decision']}**.",
                      f"Recorded GPU-worker wall time (queue excluded, failed attempts included): {gpu_h:.3f} GPUh.",
                      f"Includes {prior_gpu_h:.3f} GPUh from the original QA attempt; three QA passes are reused, not rerun.",
                      "", "Primary comparison: loop_geometry vs uada_geometry, same candidates and per-window query cap.",
                      "uada_static reuses the existing PNG but reruns its full episodes here; no UADA training.",
                      "Both adaptive arms may clone simulator state to score candidate futures. This is privileged simulation access.",
                      "Short-horizon predicate/distance progress is a search proxy; only formal full-episode outcomes enter CASR.",
                      "These are digital full-wrist transformations, with different constraints from a static 50x50 patch.",
                      "No follow-up training or jobs are automatically submitted."])
        p = folder / "RESULTS.md"; p.write_text("\n".join(lines) + "\n"); artifacts.append(p)
        finish(folder / "STATUS.json", artifacts, decision=decision["decision"], episodes=len(rows))
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    summarize()

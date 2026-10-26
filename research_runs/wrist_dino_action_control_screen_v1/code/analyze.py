"""Aggregate held-out internal-control evidence without loading the VLA model."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from common import (ROOT, atomic_json, atomic_parquet, atomic_text, config, load_json,
                    require, run_signature, sha256_file)
from screen import arm_complete, arm_paths


MEANS = [
    "initial_loss", "best_loss", "loss_improvement", "random_loss",
    "optimized_loss_advantage_over_random", "action_delta_l2", "random_action_delta_l2",
    "optimized_action_delta_advantage_over_random", "actual_relative_rms_max",
    "random_actual_relative_rms_max", "wrong_open_slot_fraction", "any_wrong_open",
    "all_wrong_open", "all_open_margin", "minimum_target_gripper", "mean_target_gripper",
    "target_direction_progress", "initial_gradient_l2",
    "shared_loss", "shared_loss_improvement", "shared_random_loss",
    "shared_loss_advantage_over_random", "shared_action_delta_l2",
    "shared_random_action_delta_l2", "shared_wrong_open_slot_fraction",
    "shared_any_wrong_open", "shared_all_wrong_open", "shared_all_open_margin",
    "shared_minimum_target_gripper", "shared_mean_target_gripper",
    "shared_target_direction_progress", "shared_actual_relative_rms_max",
    "shared_random_actual_relative_rms_max",
] + [f"{prefix}axis_{axis}_mean_abs_delta" for prefix in ("", "shared_", "shared_random_") for axis in range(7)]


def _verified_screen(signature):
    path = ROOT / "reports/SCREEN_STATUS.json"
    require(path.exists(), "GPU screen has not completed")
    status = load_json(path)
    require(status["status"] == "COMPLETE" and status["signature"] == signature
            and status["completed_arms"] == status["planned_arms"] == 54, "GPU screen is incomplete")
    for relative, expected in status["artifacts"].items():
        require(sha256_file(ROOT / relative) == expected, f"Screen artifact changed: {relative}")
    return status


def _family(site):
    if site.startswith(("q_", "k_", "v_")):
        return site[0].upper()
    if site.startswith("attn_"):
        return "attention output"
    if site.startswith("mlp_"):
        return "MLP output"
    if site.startswith("block_spatial"):
        return "block spatial"
    if site.startswith("block_cls"):
        return "CLS state"
    if site.startswith("block_register"):
        return "register state"
    return {"dino_wrist_output": "consumed DINO", "siglip_wrist_output": "consumed SigLIP",
            "projector_wrist_output": "fused projector"}[site]


def _summaries(cfg, signature):
    all_rows, summaries = [], []
    for site in cfg["candidates"]:
        for endpoint in cfg["endpoints"]:
            require(arm_complete(site, endpoint, signature), f"Missing arm: {site}/{endpoint}")
            table, report_path = arm_paths(site, endpoint)
            frame = pd.read_parquet(table)
            report = load_json(report_path)
            require(len(frame) == 40 and frame.uid.nunique() == 40
                    and set(frame.split) == {"train", "validation"}, "Invalid arm rows")
            all_rows.append(frame)
            for split, part in frame.groupby("split", sort=False):
                row = {"site": site, "family": _family(site), "endpoint": endpoint, "split": split,
                       "records": len(part), "tasks": part.task_id.nunique(),
                       "shared_direction_selected_rms": report["shared_direction_selected_rms"],
                       "train_validation_gradient_cosine": report["train_validation_gradient_cosine"],
                       "gradient_cancellation_ratio": report["gradient_consensus"][split]["cancellation_ratio"],
                       "gradient_element_sign_agreement": report["gradient_consensus"][split]["element_sign_agreement"]}
                for column in MEANS:
                    if column in part:
                        row[column + "_mean"] = float(part[column].astype(float).mean())
                summaries.append(row)
    raw = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    require(len(raw) == 2160 and len(summary) == 108, "Aggregate grid changed")
    return raw, summary


def _rank(summary, endpoint):
    value = summary[(summary.endpoint == endpoint) & (summary.split == "validation")].copy()
    if endpoint == "wrong_open":
        order = ["shared_wrong_open_slot_fraction_mean", "shared_loss_improvement_mean",
                 "shared_loss_advantage_over_random_mean", "wrong_open_slot_fraction_mean",
                 "train_validation_gradient_cosine"]
    else:
        order = ["shared_loss_improvement_mean", "shared_loss_advantage_over_random_mean",
                 "shared_target_direction_progress_mean", "loss_improvement_mean",
                 "train_validation_gradient_cosine"]
    value = value.sort_values(order + ["site"], ascending=[False] * len(order) + [True]).reset_index(drop=True)
    value.insert(0, "rank", range(1, len(value) + 1))
    value["beats_random_loss"] = value.shared_loss_advantage_over_random_mean > 0
    value["shared_improves_endpoint"] = value.shared_loss_improvement_mean > 0
    value["shared_gradient_sign"] = value.train_validation_gradient_cosine > 0
    return value


def _format(number, percent=False):
    if pd.isna(number):
        return "n/a"
    return f"{100*number:.1f}%" if percent else f"{number:.4f}"


def _table(frame, endpoint, limit=12):
    if endpoint == "wrong_open":
        header = "|Rank|Site|Family|Shared val wrong-open slots|Shared val any-open|Shared loss gain|Shared−random advantage|Per-observation upper-bound open slots|Train/val grad cos|\n|---:|---|---|---:|---:|---:|---:|---:|---:|"
        rows = [f"|{r['rank']}|{r['site']}|{r['family']}|{_format(r['shared_wrong_open_slot_fraction_mean'], True)}|{_format(r['shared_any_wrong_open_mean'], True)}|{_format(r['shared_loss_improvement_mean'])}|{_format(r['shared_loss_advantage_over_random_mean'])}|{_format(r['wrong_open_slot_fraction_mean'], True)}|{_format(r['train_validation_gradient_cosine'])}|"
                for _, r in frame.head(limit).iterrows()]
    else:
        header = "|Rank|Site|Family|Shared val loss gain|Shared−random advantage|Shared target progress|Shared action ΔL2|Per-observation upper-bound gain|Train/val grad cos|\n|---:|---|---|---:|---:|---:|---:|---:|---:|"
        rows = [f"|{r['rank']}|{r['site']}|{r['family']}|{_format(r['shared_loss_improvement_mean'])}|{_format(r['shared_loss_advantage_over_random_mean'])}|{_format(r['shared_target_direction_progress_mean'])}|{_format(r['shared_action_delta_l2_mean'])}|{_format(r['loss_improvement_mean'])}|{_format(r['train_validation_gradient_cosine'])}|"
                for _, r in frame.head(limit).iterrows()]
    return "\n".join([header] + rows)


def analyze():
    cfg, signature = config(), run_signature()
    _verified_screen(signature)
    raw, summary = _summaries(cfg, signature)
    open_rank = _rank(summary, "wrong_open")
    uada_rank = _rank(summary, "uada_opposite")
    result_dir = ROOT / "results"
    atomic_parquet(raw, result_dir / "SCREEN_ROWS.parquet")
    summary.to_csv(result_dir / ".MECHANISM_SUMMARY.csv.tmp", index=False)
    (result_dir / ".MECHANISM_SUMMARY.csv.tmp").replace(result_dir / "MECHANISM_SUMMARY.csv")
    open_rank.to_csv(result_dir / ".WRONG_OPEN_RANKING.csv.tmp", index=False)
    (result_dir / ".WRONG_OPEN_RANKING.csv.tmp").replace(result_dir / "WRONG_OPEN_RANKING.csv")
    uada_rank.to_csv(result_dir / ".UADA_OPPOSITE_RANKING.csv.tmp", index=False)
    (result_dir / ".UADA_OPPOSITE_RANKING.csv.tmp").replace(result_dir / "UADA_OPPOSITE_RANKING.csv")
    candidates = {}
    for endpoint, ranking in (("wrong_open", open_rank), ("uada_opposite", uada_rank)):
        eligible = ranking[ranking.beats_random_loss & ranking.shared_improves_endpoint]
        candidates[endpoint] = eligible.head(3).site.tolist()
    lines = [
        "# Wrist internal action-control screen", "",
        "This report is an internal causal screen on 40 frozen Spatial observations. It trained no RGB patch, ran no simulator episode, and is not CASR evidence.", "",
        "## Wrong-open endpoint (validation ranking)", "",
        _table(open_rank, "wrong_open"), "",
        "The native gripper threshold is strictly `g > 0.5`. The endpoint optimized mean hinge to 0.6 only on expert-and-clean dual-close slots.", "",
        "## UADA opposite-action endpoint (validation ranking)", "",
        _table(uada_rank, "uada_opposite"), "",
        "The opposite target is the expert-sign-opposite normalized boundary on action axis 0 over eight slots, matching the Spatial UADA mask and target construction.", "",
        "## Interpretation boundary", "",
        f"Candidates whose fixed train-derived direction both improves the held-out endpoint and beats a fixed matched-random direction: wrong-open `{', '.join(candidates['wrong_open']) or 'none'}`; UADA-opposite `{', '.join(candidates['uada_opposite']) or 'none'}`.", "",
        "A positive result identifies an internally controllable site. A deployable 50×50 universal RGB patch still has to reach a shared change at that site while passing a full native RGB forward. The next stage should test RGB reachability only for these leaders and retain SigLIP/projector controls.", "",
        "All 27 candidates, both splits, random controls, per-axis action changes, gradient cancellation ratios and sign-agreement measures are in `MECHANISM_SUMMARY.csv`; all 2,160 observation-level rows are in `SCREEN_ROWS.parquet`.", "",
    ]
    atomic_text("\n".join(lines), result_dir / "RESULTS.md")
    artifacts = {path.name: sha256_file(path) for path in result_dir.iterdir() if path.is_file() and path.name != "STATUS.json"}
    status = {"status": "COMPLETE", "signature": signature, "scientific_scope": "INTERNAL_INTERVENTION_ONLY",
              "records": 40, "candidates": 27, "endpoints": 2, "rows": len(raw),
              "simulator_episodes": 0, "deployment_png_produced": False,
              "rgb_reachability_tested": False, "nominations": candidates, "artifacts": artifacts}
    atomic_json(status, result_dir / "STATUS.json")
    print("ACTION_CONTROL_ANALYSIS_COMPLETE", candidates, flush=True)

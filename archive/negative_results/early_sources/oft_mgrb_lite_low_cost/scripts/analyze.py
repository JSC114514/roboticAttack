#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import CONFIG_PATH, ROOT, atomic_json, atomic_parquet, load_json, require, sha256_file


KEYS = ["suite", "task_id", "state_index"]


def load_outputs(cfg: dict):
    rollout_frames, action_frames, trajectory_frames = [], [], []
    for mode, suites in (("object", [cfg["primary_suite"]]), ("scope", cfg["scope_suites"])):
        for suite in suites:
            for task in range(10):
                base = ROOT / "outputs" / mode / suite
                paths = {kind: base / f"task_{task:02d}_{kind}.parquet" for kind in ("rollout", "action", "trajectory")}
                require(all(path.is_file() for path in paths.values()), f"Missing {mode}/{suite}/task_{task:02d}")
                rollout_frames.append(pd.read_parquet(paths["rollout"]))
                action_frames.append(pd.read_parquet(paths["action"]))
                trajectory_frames.append(pd.read_parquet(paths["trajectory"]))
    return tuple(pd.concat(frames, ignore_index=True) for frames in (rollout_frames, action_frames, trajectory_frames))


def recovery_auc(trajectories: pd.DataFrame, action_scale: np.ndarray) -> pd.DataFrame:
    rows = []
    for keys, state in trajectories.groupby(KEYS, sort=False):
        clean = state[state.condition == "clean"].set_index("decision_block")
        for condition, attack in state[state.condition != "clean"].groupby("condition"):
            values = []
            for decision in range(3, 8):
                if decision not in clean.index or decision not in set(attack.decision_block):
                    continue
                left = np.asarray(json.loads(clean.loc[decision, "action_chunk"]), dtype=np.float64)
                right = np.asarray(json.loads(attack.set_index("decision_block").loc[decision, "action_chunk"]), dtype=np.float64)
                scale = action_scale.reshape(1, 7) if action_scale.shape == (7,) else action_scale
                require(scale.shape in {(1, 7), left.shape}, f"Action scale shape mismatch: {scale.shape} vs {left.shape}")
                values.append(float(np.linalg.norm((right - left) / scale)))
            rows.append({**dict(zip(KEYS, keys)), "condition": condition,
                         "recovery_auc5": float(np.mean(values)) if values else np.nan,
                         "recovery_observed_blocks": len(values)})
    return pd.DataFrame(rows)


def paired_table(rollouts: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    clean = rollouts[rollouts.condition == "clean"][KEYS + ["success", "episode_steps"]].rename(
        columns={"success": "clean_success", "episode_steps": "clean_episode_steps"}
    )
    attack = rollouts[rollouts.condition != "clean"].rename(
        columns={"success": "attack_success", "episode_steps": "attack_episode_steps"}
    )
    paired = attack.merge(clean, on=KEYS, validate="many_to_one").merge(recovery, on=KEYS + ["condition"], validate="one_to_one")
    paired["conversion"] = ((paired.clean_success == 1) & (paired.attack_success == 0)).astype(int)
    paired["rescue"] = ((paired.clean_success == 0) & (paired.attack_success == 1)).astype(int)
    return paired


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    eligible = frame[frame.clean_success == 1]
    return {
        "clean_sr": float(frame.clean_success.mean()),
        "attack_sr": float(frame.attack_success.mean()),
        "paired_casr": float(eligible.conversion.mean()) if len(eligible) else float("nan"),
        "clean_success_attack_failure": float(frame.conversion.sum()),
        "clean_failure_attack_success": float(frame.rescue.sum()),
        "clean_success_denominator": float(len(eligible)),
        "termination_step": float(frame.attack_episode_steps.mean()),
        "recovery_auc5": float(frame.recovery_auc5.mean()),
    }


def bootstrap(frame: pd.DataFrame, evaluator: Callable[[pd.DataFrame], dict[str, float]], draws: int, seed: int):
    rng = np.random.default_rng(seed)
    tasks = sorted(frame.task_id.unique())
    output: dict[str, list[float]] = {}
    for _ in range(draws):
        pieces = []
        for draw_task, task in enumerate(rng.choice(tasks, size=len(tasks), replace=True)):
            group = frame[frame.task_id == task]
            states = group.state_index.unique()
            chosen = rng.choice(states, size=len(states), replace=True)
            sample = pd.concat([group[group.state_index == state] for state in chosen], ignore_index=True)
            sample = sample.assign(_bootstrap_task=draw_task)
            pieces.append(sample)
        values = evaluator(pd.concat(pieces, ignore_index=True))
        for name, value in values.items():
            if np.isfinite(value):
                output.setdefault(name, []).append(float(value))
    return {name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) for name, values in output.items()}


def summarize_closed_loop(paired: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for (suite, condition), group in paired.groupby(["suite", "condition"], sort=True):
        point = metrics(group)
        ci = bootstrap(group, metrics, int(cfg["bootstrap_draws"]), int(cfg["bootstrap_seed"]))
        for metric, estimate in point.items():
            low, high = ci.get(metric, (float("nan"), float("nan")))
            rows.append({"suite": suite, "condition": condition, "metric": metric,
                         "estimate": estimate, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def summarize_actions(actions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    metric_names = ["action_l2", "translation_l2", "dx_abs", "dy_abs", "dz_abs", "rotation_l2", "gripper_abs"]
    state = actions.groupby(KEYS + ["condition"], as_index=False)[metric_names].mean()
    rows = []
    for (suite, condition), group in state.groupby(["suite", "condition"], sort=True):
        evaluator = lambda x: {metric: float(x[metric].mean()) for metric in metric_names}
        confidence = bootstrap(group, evaluator, int(cfg["bootstrap_draws"]), int(cfg["bootstrap_seed"]))
        for name in metric_names:
            point = float(group[name].mean())
            low, high = confidence[name]
            rows.append({"suite": suite, "condition": condition, "metric": name,
                         "estimate": point, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def primary_contrast(paired: pd.DataFrame, cfg: dict) -> dict:
    obj = paired[paired.suite == cfg["primary_suite"]]
    left = obj[obj.condition == "mgrb_200"][KEYS + ["clean_success", "conversion"]].rename(columns={"conversion": "mgrb"})
    right = obj[obj.condition == "generic_200"][KEYS + ["conversion"]].rename(columns={"conversion": "generic"})
    frame = left.merge(right, on=KEYS, validate="one_to_one")

    def evaluate(x):
        eligible = x[x.clean_success == 1]
        return {"delta_casr_200": float((eligible.mgrb - eligible.generic).mean()) if len(eligible) else float("nan")}

    point = evaluate(frame)["delta_casr_200"]
    low, high = bootstrap(frame, evaluate, int(cfg["bootstrap_draws"]), int(cfg["bootstrap_seed"]))["delta_casr_200"]
    signs = []
    for task, group in frame.groupby("task_id"):
        value = evaluate(group)["delta_casr_200"]
        signs.append(value)
    return {
        "metric": "CASR(MGRB-200-K3)-CASR(Generic-200-K3)", "estimate": point,
        "ci95_low": low, "ci95_high": high,
        "positive_tasks": int(sum(value > 0 for value in signs if np.isfinite(value))),
        "negative_tasks": int(sum(value < 0 for value in signs if np.isfinite(value))),
        "zero_tasks": int(sum(value == 0 for value in signs if np.isfinite(value))),
        "clean_success_denominator": int(frame.clean_success.sum()),
    }


def summarize_readout(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for task in range(10):
        path = ROOT / "outputs" / "readout" / f"task_{task:02d}.parquet"
        require(path.is_file(), f"Missing readout task {task}")
        frames.append(pd.read_parquet(path))
    raw = pd.concat(frames, ignore_index=True)
    require(len(raw) == 120 and not raw.duplicated(["task_id", "state_index", "condition"]).any(), "Readout completeness failure")
    metric_names = [
        "action_l2", "translation_l2", "dx_abs", "dy_abs", "dz_abs", "rotation_l2", "gripper_abs",
        "dino_displacement", "siglip_displacement", "geometry_alignment_cosine",
        "dino_restore_reduction", "siglip_restore_reduction",
    ]
    rows = []
    for condition, group in raw.groupby("condition"):
        evaluator = lambda x: {metric: float(x[metric].mean()) for metric in metric_names}
        confidence = bootstrap(group, evaluator, int(cfg["bootstrap_draws"]), int(cfg["bootstrap_seed"]))
        for name in metric_names:
            point = float(group[name].mean())
            low, high = confidence[name]
            rows.append({"condition": condition, "metric": name, "estimate": point, "ci95_low": low, "ci95_high": high})
    return raw, pd.DataFrame(rows)


def training_and_cost(cfg: dict, closed_summary: pd.DataFrame, action_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows, trace_rows, cost_rows = [], [], []
    for method in ("mgrb", "generic"):
        frame = pd.read_parquet(ROOT / "outputs" / f"training_{method}.parquet")
        require(len(frame) == 500 and set(frame.step) == set(range(1, 501)), f"Training completeness: {method}")
        status = load_json(ROOT / "reports" / f"TRAIN_{method.upper()}_STATUS.json")
        curve_rows.append({"method": method, "step": 0, "heldout_action_l2": float("nan"),
                           "paired_casr": float("nan"), "training_elapsed_s": 0.0})
        trace_rows.append({"method": method, "step": 0, "training_sample_action_l2": float("nan"),
                           "training_sample_alignment_cosine": float("nan"), "elapsed_s": 0.0})
        for step in (50, 200, 500):
            row = frame[frame.step == step].iloc[0]
            condition = f"{method}_{step:03d}"
            action_match = action_summary[(action_summary.suite == cfg["primary_suite"]) &
                                          (action_summary.condition == condition) &
                                          (action_summary.metric == "action_l2")]
            casr_match = closed_summary[(closed_summary.suite == cfg["primary_suite"]) &
                                        (closed_summary.condition == condition) &
                                        (closed_summary.metric == "paired_casr")]
            require(len(action_match) == 1 and len(casr_match) == 1, f"Missing held-out curve point: {condition}")
            curve_rows.append({"method": method, "step": step,
                               "heldout_action_l2": float(action_match.iloc[0].estimate),
                               "paired_casr": float(casr_match.iloc[0].estimate),
                               "training_elapsed_s": float(row.elapsed_s)})
            trace_rows.append({"method": method, "step": step,
                               "training_sample_action_l2": float(row.action_l2),
                               "training_sample_alignment_cosine": float(row.alignment_cosine),
                               "elapsed_s": float(row.elapsed_s)})
            cost_rows.append({
                "method": method, "trainable_params": 192, "training_states": 20,
                "optimization_steps": step, "forward_passes": step, "backward_passes": step,
                "clean_reference_precompute_forwards": 20,
                "anchor_precompute_forwards": 20 if method == "mgrb" else 0,
                "total_construction_forwards": step + 20 + (20 if method == "mgrb" else 0),
                "gpu_hours": float(row.elapsed_s) / 3600.0,
                "peak_gpu_memory_bytes": int(row.peak_gpu_memory_bytes),
                "test_adaptive_queries": 0, "attack_blocks": 3, "patch_area": 2500,
                "modified_pixel_fraction": 2500 / (224 * 224), "cost_source": "measured current run",
            })
    for method in ("exploring_uada_1to3", "when_official"):
        cost_rows.append({
            "method": method, "trainable_params": 7500, "training_states": np.nan,
            "optimization_steps": np.nan, "forward_passes": np.nan, "backward_passes": np.nan,
            "clean_reference_precompute_forwards": np.nan, "anchor_precompute_forwards": 0,
            "total_construction_forwards": np.nan, "gpu_hours": np.nan, "peak_gpu_memory_bytes": np.nan,
            "test_adaptive_queries": 0, "attack_blocks": 3, "patch_area": 2500,
            "modified_pixel_fraction": 2500 / (224 * 224),
            "cost_source": "released asset; original construction cost not measured in this run",
        })
    return pd.DataFrame(curve_rows), pd.DataFrame(trace_rows), pd.DataFrame(cost_rows)


def main() -> None:
    cfg = load_json(CONFIG_PATH)
    rollouts, actions, trajectories = load_outputs(cfg)
    require(len(rollouts) == 1650, f"Expected 1650 rollouts, got {len(rollouts)}")
    require(not rollouts.duplicated(KEYS + ["condition"]).any(), "Duplicate rollout cell")
    action_scale = np.asarray(load_json(cfg["paths"]["action_scale"])["scale"], dtype=np.float64)
    recovery = recovery_auc(trajectories, action_scale)
    paired = paired_table(rollouts, recovery)
    closed_summary = summarize_closed_loop(paired, cfg)
    action_summary = summarize_actions(actions, cfg)
    primary = primary_contrast(paired, cfg)
    readout_raw, readout_summary = summarize_readout(cfg)
    curve, optimization_trace, cost = training_and_cost(cfg, closed_summary, action_summary)
    results = ROOT / "results"
    atomic_parquet(rollouts, results / "CLOSED_LOOP_RAW.parquet")
    atomic_parquet(paired, results / "CLOSED_LOOP_PAIRED.parquet")
    atomic_parquet(rollouts[rollouts.suite == cfg["primary_suite"]], results / "OBJECT_CLOSED_LOOP_RAW.parquet")
    closed_summary.to_csv(results / "OBJECT_AND_SCOPE_CLOSED_LOOP_SUMMARY.csv", index=False)
    closed_summary[closed_summary.suite == cfg["primary_suite"]].to_csv(results / "OBJECT_CLOSED_LOOP_SUMMARY.csv", index=False)
    closed_summary[closed_summary.suite.isin(cfg["scope_suites"])].to_csv(results / "CROSS_SUITE_SCOPE.csv", index=False)
    action_summary.to_csv(results / "ACTION_SUMMARY.csv", index=False)
    pd.DataFrame([primary]).to_csv(results / "PRIMARY_CONTRAST.csv", index=False)
    atomic_parquet(readout_raw, results / "OBJECT_MECHANISM_READOUT_RAW.parquet")
    readout_summary.to_csv(results / "OBJECT_MECHANISM_READOUT.csv", index=False)
    curve.to_csv(results / "MGRB_TRAINING_CURVE.csv", index=False)
    optimization_trace.to_csv(results / "TRAINING_OPTIMIZATION_TRACE_CHECKPOINTS.csv", index=False)
    cost.to_csv(results / "COST_TABLE.csv", index=False)
    diagnostic = pd.concat([pd.read_parquet(ROOT / "outputs/anchors" / f"task_{task:02d}_diagnostic.parquet") for task in range(10)], ignore_index=True)
    diagnostic.to_csv(results / "PHASE_A_DIAGNOSTIC.csv", index=False)
    status = {
        "status": "COMPLETE", "stop_type": "RETURN_TO_USER_AND_WEB_GPT",
        "rollouts": len(rollouts), "paired_attack_rows": len(paired), "readout_rows": len(readout_raw),
        "primary": primary, "duplicate_rollouts": 0,
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    atomic_json(status, ROOT / "reports" / "MGRB_STATUS.json")
    (ROOT / "RESULTS.md").write_text(
        "# MGRB-Lite results\n\n## Observed Results\n\n"
        f"Primary ΔCASR_200 = {primary['estimate']:.6f} "
        f"[{primary['ci95_low']:.6f}, {primary['ci95_high']:.6f}].\n\n"
        "See `results/` for complete closed-loop, mechanism, training-curve, and cost tables.\n\n"
        "## Execution Notes\n\nAll model forwards, activation captures, patch optimization and rollouts used CUDA/NVIDIA. "
        "All attack conditions used Wrist-only K3 exposure.\n\n"
        "## Pre-registered Interpretation\n\nClassify against Cases A-F in the frozen scientific plan; no follow-up was auto-started.\n",
        encoding="utf-8",
    )
    handoff = (
        "# Web GPT handoff: OFT-MGRB-LITE-01\n\n"
        "Status: COMPLETE; STOP and adjudicate before any extension.\n\n"
        f"Primary MGRB-200 minus Generic-200 paired CASR: {primary['estimate']:.6f} "
        f"(95% hierarchical-bootstrap CI {primary['ci95_low']:.6f}, {primary['ci95_high']:.6f}).\n\n"
        "Review `PRIMARY_CONTRAST.csv`, `OBJECT_AND_SCOPE_CLOSED_LOOP_SUMMARY.csv`, "
        "`MGRB_TRAINING_CURVE.csv`, `OBJECT_MECHANISM_READOUT.csv`, and `COST_TABLE.csv`.\n\n"
        "Questions: Is there an early-budget CASR advantage? Is it Action-only or closed-loop? "
        "Does DINO alignment/restore support the registered mechanism? How far is MGRB-500 from released baselines? "
        "Does suite heterogeneity match recovery-boundary evidence? What is the strongest allowed claim?\n\n"
        "Forbidden claims: physical-world validity, black-box construction, highest-ASR attack, cross-model transfer, "
        "or zero total discovery cost.\n"
    )
    (ROOT / "reports" / "WEB_GPT_HANDOFF_MGRB_LITE.md").write_text(handoff, encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

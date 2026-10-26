#!/usr/bin/env python3
"""Checkpoint-explicit task/state paired bootstrap for Batch8/WHEN-DINO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dino_batch8_common import CONFIG, CHECKPOINTS, config, digest_json, external_baselines, require_preparation
from evaluate import (
    BASE_CONDITIONS, CONDITIONS, DENOMINATORS, EXPERIMENT_ID, METHODS, REFERENCES, KEY, load_config, patches,
    validate_new_shard, validate_runtime, require, sha256_file,
)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet

PAIRS = [
    ("original_uada_b8_minus_original_uada_b1", "original_uada_b8", "original_uada_b1"),
    ("uniform_replay_b8_minus_uniform_replay_b1", "uniform_replay_b8", "uniform_replay_b1"),
    ("dino_sampling_b8_minus_dino_sampling_b1", "dino_sampling_b8", "dino_sampling_b1"),
    ("dino_sampling_b8_minus_uniform_replay_b8", "dino_sampling_b8", "uniform_replay_b8"),
    ("when_dino_b8_minus_uniform_replay_b8", "when_dino_b8", "uniform_replay_b8"),
    ("when_dino_b8_minus_dino_sampling_b8", "when_dino_b8", "dino_sampling_b8"),
    ("when_dino_b8_minus_original_uada_b8", "when_dino_b8", "original_uada_b8"),
]


def contrast_vectors():
    vectors = {}
    for name, left, right in PAIRS:
        value = np.zeros(len(BASE_CONDITIONS))
        value[BASE_CONDITIONS.index(left)] = 1
        value[BASE_CONDITIONS.index(right)] = -1
        vectors[name] = (value, left, right)
    return vectors


def hierarchical_samples(eligible, draws, rng):
    """Every task/state draw is shared across both checkpoints and all methods."""
    require(draws >= 1 and not eligible.duplicated(
        ["checkpoint_outer", "task_id", "state_index", "condition"]).any(), "Duplicate paired bootstrap key")
    require(all(int(row.checkpoint_outer) == int(row.condition.rsplit("_",1)[1])
                for row in eligible.itertuples()), "Bootstrap checkpoint/condition mismatch")
    wide = eligible.pivot(index=["task_id", "state_index"],
                          columns=["checkpoint_outer", "condition"], values="casr")
    columns = [(cp, f"{m}_{cp}") for cp in CHECKPOINTS for m in BASE_CONDITIONS]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(columns))
    require(not wide.isna().any().any() and np.isfinite(wide.to_numpy(float)).all()
            and wide.isin([0,1]).all().all(), "Unpaired/nonfinite/binary CASR data")
    tasks = sorted(wide.index.get_level_values("task_id").unique())
    require(tasks == list(range(10)), "All ten tasks require clean-success states")
    arrays = [wide.xs(task, level="task_id").to_numpy(float) for task in tasks]
    selected = rng.integers(0,10,size=(draws,10))
    samples = np.zeros((draws,len(CONDITIONS)))
    for position in range(10):
        for task, values in enumerate(arrays):
            ids = np.flatnonzero(selected[:,position] == task)
            states = rng.integers(0,len(values),size=(len(ids),len(values)))
            samples[ids] += values[states].mean(axis=1)/10
    per_task = np.stack([values.mean(axis=0) for values in arrays])
    return per_task.mean(axis=0), samples, per_task


def paired_statistics(eligible, draws=10000, seed=20260907):
    rng = np.random.default_rng(seed)
    points, samples, tasks = {}, {}, {}
    for suite in DENOMINATORS:
        point, sample, per_task = hierarchical_samples(eligible.loc[eligible.suite.eq(suite)],draws,rng)
        points[suite], samples[suite], tasks[suite] = point, sample, per_task
    groups = {suite:[suite] for suite in DENOMINATORS}
    groups["spatial_goal_macro"] = ["libero_spatial","libero_goal"]
    groups["four_suite_macro"] = list(DENOMINATORS)
    summaries, comparisons = [], []
    for cp_index, cp in enumerate(CHECKPOINTS):
        selected = slice(cp_index*len(BASE_CONDITIONS),(cp_index+1)*len(BASE_CONDITIONS))
        for group, suites in groups.items():
            point = np.mean([points[s][selected] for s in suites],axis=0)
            sample = np.mean([samples[s][:,selected] for s in suites],axis=0)
            task_values = np.concatenate([tasks[s][:,selected] for s in suites],axis=0)
            for j,method in enumerate(BASE_CONDITIONS):
                low,high = np.percentile(sample[:,j],[2.5,97.5])
                summaries.append({"checkpoint_outer":cp,"suite_or_endpoint":group,"condition":method,
                    "task_equal_casr":float(point[j]),"ci95_low":float(low),"ci95_high":float(high),
                    "clean_success_states":sum(DENOMINATORS[s] for s in suites),
                    "tasks":10*len(suites),"suite_equal":len(suites)>1})
            for name,(vector,left,right) in contrast_vectors().items():
                delta,distribution = task_values@vector,sample@vector
                low,high = np.percentile(distribution,[2.5,97.5])
                comparisons.append({"checkpoint_outer":cp,"suite_or_endpoint":group,"contrast":name,
                    "left":left,"right":right,"difference_left_minus_right":float(point@vector),
                    "ci95_low":float(low),"ci95_high":float(high),"positive_tasks":int((delta>0).sum()),
                    "negative_tasks":int((delta<0).sum()),"tied_tasks":int((delta==0).sum()),
                    "tasks":len(delta),"paired_clean_success_states":sum(DENOMINATORS[s] for s in suites)})
    return pd.DataFrame(summaries),pd.DataFrame(comparisons)


def validate_raw(frame, cfg):
    expected_rows = len(CONDITIONS) * 400
    require(len(frame) == expected_rows and set(frame.condition) == set(CONDITIONS)
            and set(frame.suite) == set(DENOMINATORS) and set(frame.checkpoint_outer) == set(CHECKPOINTS)
            and not frame.duplicated(KEY).any(), "Incomplete/duplicate episode raw grid")
    require(frame.groupby(["checkpoint_outer","suite","condition"]).size().eq(100).all()
            and frame.groupby(["checkpoint_outer","suite","task_id","condition"]).size().eq(10).all()
            and frame.success.isin([0,1]).all() and frame.failure.eq(1-frame.success).all()
            and frame.frozen_clean_success.isin([0,1]).all() and not frame.reused.astype(bool).any()
            and np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(),
            "Raw checkpoint coverage/success/finite data failed")
    require(all(int(row.checkpoint_outer) == int(row.condition.rsplit("_",1)[1]) for row in frame.itertuples()),
            "Raw patch checkpoint/condition mismatch")
    require(frame.groupby(["suite","task_id","state_index"]).frozen_clean_success.nunique().eq(1).all()
            and frame.groupby(["suite","task_id","state_index"]).rollout_seed.nunique().eq(1).all(),
            "Unpaired clean qualification or seeds across methods/checkpoints")
    for suite,expected in DENOMINATORS.items():
        eligible = frame.loc[frame.suite.eq(suite)&frame.frozen_clean_success.eq(1)]
        require(eligible.groupby(["checkpoint_outer","condition"]).size().eq(expected).all(),
                "Frozen eligible count changed")
    return frame.loc[frame.frozen_clean_success.eq(1)].assign(casr=lambda f:f.failure.astype(float)).copy()


def analyze(draws=10000):
    train,cfg = config(),load_config()
    require(draws == train["bootstrap_draws"] == 10000,"Formal bootstrap is frozen at 10000 draws")
    preparation = require_preparation(train)
    _,hashes,_ = patches(cfg,"cpu")
    output = Path(cfg["output_root"])
    require(json.loads((output/"EVAL_CONFIG.json").read_text()) == cfg,"Final evaluation config changed")
    parts,trace_parts,artifact_hashes = [],[],{}
    for task in range(10):
        receipt = json.loads((output/f"task_{task:02d}_STATUS.json").read_text())
        require(receipt["status"] == "COMPLETE" and receipt["task_id"] == task
                and receipt["config_signature"] == digest_json(cfg) and receipt["patches"] == hashes
                and receipt["evaluation_checkpoint_outer_iterations"] == list(CHECKPOINTS),
                "Task receipt incomplete or checkpoint identity changed")
        for suite in cfg["suites"]:
            rp,tp = [output/suite/f"task_{task:02d}_{kind}.parquet" for kind in ("rollouts","traces")]
            require(sha256_file(rp) == receipt["suites"][suite]["rollouts_sha256"]
                    and sha256_file(tp) == receipt["suites"][suite]["traces_sha256"]
                    and receipt["suites"][suite]["evaluation_checkpoint_outer_iterations"] == list(CHECKPOINTS),
                    "Committed evaluation shard bytes/checkpoints changed")
            rows,traces = pd.read_parquet(rp),pd.read_parquet(tp)
            validate_new_shard(rows,traces,suite,task,cfg,hashes)
            validate_runtime(rows,traces,cfg,suite,task)
            for frame,path in ((rows,rp),(traces,tp)):
                frame["source_path"],frame["source_sha256"] = str(path),sha256_file(path)
                artifact_hashes[str(path)] = sha256_file(path)
            parts.append(rows);trace_parts.append(traces)
    frame = pd.concat(parts,ignore_index=True)
    frame["eligibility_reason"] = np.where(frame.frozen_clean_success.eq(1),"eligible_clean_success","frozen_clean_failure")
    eligible = validate_raw(frame,cfg)
    require(len(eligible) == len(CONDITIONS) * sum(DENOMINATORS.values()),
            "Expected all checkpoint/condition/eligible repeated-measure rows")
    summary,contrasts = paired_statistics(eligible,draws,train["bootstrap_seed"])
    suite_summary = summary.loc[summary.suite_or_endpoint.isin(DENOMINATORS)].copy()
    endpoints = summary.loc[~summary.suite_or_endpoint.isin(DENOMINATORS)].copy()
    require(len(suite_summary) == len(CHECKPOINTS) * len(BASE_CONDITIONS) * len(DENOMINATORS),
            "Expected all checkpoint/condition/suite summaries")
    per_task = eligible.groupby(["checkpoint_outer","suite","task_id","condition"],as_index=False).agg(
        clean_success_states=("state_index","nunique"),casr=("casr","mean"),attacked_success=("success","mean"))
    prefix = "BATCH8_WHEN2000_XSUITE"
    atomic_parquet(frame,output/f"{prefix}_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(trace_parts,ignore_index=True),output/f"{prefix}_TRACES.parquet")
    atomic_parquet(eligible,output/f"{prefix}_ELIGIBLE.parquet")
    for suffix,data in (("SUMMARY",suite_summary),("ENDPOINTS",endpoints),("CONTRASTS",contrasts),("PER_TASK",per_task)):
        data.to_csv(output/f"{prefix}_{suffix}.csv",index=False)
    costs,training_metrics = [],[]
    for method in METHODS:
        folder = Path(train["output_root"])/"training"/method
        status = json.loads((folder/"status.json").read_text())
        history = pd.read_parquet(folder/"training.parquet")
        require(len(history) == 2000 and np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all(),
                "Training metrics incomplete/nonfinite")
        for cp in CHECKPOINTS:
            attack = frame.loc[frame.condition.eq(f"{method}_{cp}")]
            row = history.loc[history.outer_iteration.eq(cp)].iloc[0]
            costs.append({"checkpoint_outer":cp,"condition":method,"fresh_start_outer":0,
                "optimizer_updates":cp*50,"sample_backprops":cp*400,"cumulative_training_elapsed_s":float(row.elapsed_s),
                "whole_training_elapsed_s":status["elapsed_s"],"whole_training_costs":status["costs"],
                "evaluation_elapsed_s":float(attack.elapsed_s.sum()),"evaluation_episodes":len(attack),
                "training_seed_count":1})
            metrics = {"checkpoint_outer":cp,"condition":method,"training_seed_count":1,
                "hard_constraints":False,"appearance_regularization":False}
            for name in ("tv","initial_tv","tv_ratio","init_correlation","init_rms_delta",
                         "saturation_fraction","effective_rgb_updates",
                         "rgb_step_mean_abs_sum","rgb_outer_mean_abs","rgb_outer_changed_pixel_fraction",
                         "zero_lr_warmup_outer","rgb_init_mean_abs","rgb_init_changed_pixel_fraction",
                         "action_axis","selected_action_remaining_score","selected_dino_remaining_score",
                         "selected_patched_error","selected_restored_error",
                         "main_loss","patch_gradient_norm","rgb_step_max_abs","rgb_outer_max_abs",
                         "feature_l1","feature_contrast","feature_valid_tokens"):
                if name in history:
                    available = ("selected_restored_error_available"
                                 if name == "selected_restored_error"
                                 else "selected_score_available"
                                 if name.startswith("selected_") else None)
                    prefix_history = history.loc[history.outer_iteration.le(cp)]
                    if available:
                        # Clean placeholders do not contribute to score summaries.
                        prefix_history = prefix_history.loc[prefix_history[available]]
                        if bool(row[available]):
                            metrics[f"checkpoint_{name}"] = float(row[name])
                    else:
                        metrics[f"checkpoint_{name}"] = float(row[name])
                    if not prefix_history.empty:
                        metrics[f"mean_to_checkpoint_{name}"] = float(prefix_history[name].mean())
            if "effective_rgb_updates" in history:
                prefix_history = history.loc[history.outer_iteration.le(cp)]
                metrics["effective_rgb_update_step_fraction"] = float(prefix_history.effective_rgb_updates.sum())/(cp*50)
            training_metrics.append(metrics)
    for condition,item in external_baselines(train).items():
        attack = frame.loc[frame.condition.eq(condition)]
        costs.append({"checkpoint_outer":item["checkpoint_outer"],"condition":condition.rsplit("_",1)[0],
            "new_training_elapsed_s":0.,"optimizer_updates":item["checkpoint_outer"]*50,
            "sample_backprops":item["checkpoint_outer"]*50,
            "recorded_training_elapsed_s":item["recorded_training_elapsed_s"],
            "evaluation_elapsed_s":float(attack.elapsed_s.sum()),"evaluation_episodes":len(attack),
            "pt_sha256":item["pt"],"note":"Existing matched checkpoint; reevaluated without retraining."})
    atomic_json(costs,output/f"{prefix}_COSTS.json")
    pd.DataFrame(training_metrics).to_csv(output/f"{prefix}_TRAINING_METRICS.csv",index=False)
    report = {"status":"COMPLETE","experiment_id":EXPERIMENT_ID,"config_sha256":sha256_file(CONFIG),
        "preparation_signature":preparation["signature"],"evaluation_config_signature":digest_json(cfg),
        "checkpoints":hashes,"evaluation_checkpoint_outer_iterations":list(CHECKPOINTS),
        "attack_episodes":len(CONDITIONS)*400,
        "new_patch_attack_episodes":len(METHODS)*len(CHECKPOINTS)*400,
        "reference_attack_episodes":len(REFERENCES)*len(CHECKPOINTS)*400,
        "reused_attack_episodes":0,"eligible_rows":len(eligible),
        "summary_rows":len(suite_summary),
        "frozen_clean_episodes":400,"denominators":DENOMINATORS,"bootstrap_draws":draws,
        "bootstrap_seed":20260907,"bootstrap_unit":"task then clean-success state",
        "paired_resampling_all_conditions_and_checkpoints":True,"task_equal_then_suite_equal":True,
        "primary_endpoint":"libero_spatial",
        "primary_comparison":"when_dino_b8_minus_uniform_replay_b8",
        "joint_endpoint":"spatial_goal_macro","training_seed_count":1,"long_secondary":True,
        "raw_shard_hashes":artifact_hashes,"stop_after_outer":2000,"no_next_stage_submission":True,
        "interpretation":"Single-seed development screening at both predetermined checkpoints; episode "
        "replication does not establish training replication, visual quality or a general causal mechanism."}
    atomic_json(report,output/"STATUS.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws",type=int,default=10000)
    print(json.dumps(analyze(parser.parse_args().draws),indent=2))

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_exploring_oft_dino_alternating_refine200_cross_suite as evaluation
import run_exploring_oft_scale500_cross_suite as shared
import analyze_exploring_oft_dino_alternating_refine200_cross_suite as analysis

CONFIG = ROOT / "configs/exploring_oft_dino/object_dino_alternating_refine200_cross_suite_v1.json"


def local_config(tmp_path):
    overlay = json.loads(CONFIG.read_text())
    overlay["output_root"] = str(tmp_path / "evaluation")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(overlay))
    return path, evaluation.load_config(path)


def rows_for_task(cfg, suite, task, controls=False):
    clean = evaluation.frozen_clean(cfg, suite)
    methods = evaluation.CONTROLS if controls else evaluation.CONDITIONS
    old = evaluation.checked_previous_config(cfg)
    rows = []
    for item in clean.loc[clean.task_id == task].itertuples():
        for order, method in enumerate(methods, 1):
            source = old["patch_sources"][method] if controls else cfg["patch_sources"][method]
            failure = int(method == evaluation.CONDITIONS[0])
            rows.append({"experiment_id": evaluation.previous.EXPERIMENT_ID if controls else evaluation.EXPERIMENT_ID,
                         "suite": suite, "task_id": task, "state_index": item.state_index,
                         "condition": method, "condition_order": order, "rollout_seed": item.rollout_seed,
                         "frozen_clean_success": item.success, "success": 1 - failure, "failure": failure,
                         "patch_sha256": source["sha256"], "renderer_vendor": "NVIDIA", "episode_steps": 10,
                         "training_suite": "libero_object", "checkpoint": cfg["checkpoints"][suite],
                         "unnorm_key": cfg["unnorm_keys"][suite]})
    rollouts = pd.DataFrame(rows)
    traces = rollouts[["experiment_id", "suite", "task_id", "state_index", "condition"]].copy()
    traces["decision_block"] = 0
    return rollouts, traces


def test_real_preflight_checks_training_and_frozen_controls_without_model():
    cfg = evaluation.load_config(CONFIG)
    result = evaluation.preflight(cfg)
    assert result["new_attack_rollouts"] == 800 and result["reused_attack_rollouts"] == 1200
    assert result["combined_attack_rollouts"] == 2000 and result["combined_eligible_rows_expected"] == 1910
    assert result["new_eligible_rows_expected"] == 764 and result["clean_success_denominators"] == evaluation.DENOMINATORS
    assert result["model_loaded"] is False and result["sbatch_submitted"] is False
    assert "manual_visual_review" not in cfg
    assert "NOT_ACCEPTED" in result["appearance_assessment"][evaluation.CONDITIONS[1]]


def test_test_authorization_cannot_be_replaced_with_visual_pass(tmp_path):
    overlay = json.loads(CONFIG.read_text())
    overlay["evaluation_authorization"] = "PASS"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(overlay))
    with pytest.raises(RuntimeError, match="evaluation authorization missing"):
        evaluation.load_config(path)
    overlay["evaluation_authorization"] = evaluation.AUTHORIZATION
    overlay["appearance_assessment"][evaluation.CONDITIONS[1]] = "PASS"
    path.write_text(json.dumps(overlay))
    with pytest.raises(RuntimeError, match="Appearance assessment drift"):
        evaluation.load_config(path)


@pytest.mark.parametrize("key,match", [("sha256", "Patch hash mismatch"),
                                       ("runtime_alignment_sha256", "Training evidence hash mismatch")])
def test_patch_and_training_evidence_drift_rejected(key, match):
    cfg = copy.deepcopy(evaluation.load_config(CONFIG))
    cfg["patch_sources"][evaluation.CONDITIONS[0]][key] = "0" * 64
    with pytest.raises(RuntimeError, match=match):
        evaluation.load_refinement_patches(cfg, torch.device("cpu"))


def test_reused_data_and_runtime_drift_rejected():
    cfg = evaluation.load_config(CONFIG)
    changed = copy.deepcopy(cfg)
    changed["reused_controls"]["rollouts_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Reused evidence hash mismatch: rollouts"):
        evaluation.frozen_controls(changed)
    changed = copy.deepcopy(cfg)
    changed["num_open_loop_steps"] = 7
    with pytest.raises(RuntimeError, match="Reused runtime/model/state grid drift"):
        evaluation.checked_previous_config(changed)


def test_result_checks_seed_hash_label_metadata_and_trace():
    cfg = evaluation.load_config(CONFIG)
    rows, traces = rows_for_task(cfg, "libero_object", 0)
    evaluation.validate_result(rows, traces, "libero_object", 0, cfg)
    for col, value, match in [("rollout_seed", -1, "Frozen clean linkage mismatch"),
                              ("patch_sha256", "bad", "Rollout patch hash mismatch"),
                              ("success", 9, "Invalid success/failure labels"),
                              ("checkpoint", "bad", "model/runtime metadata drift")]:
        broken = rows.copy()
        broken.loc[0, col] = value
        with pytest.raises(RuntimeError, match=match):
            evaluation.validate_result(broken, traces, "libero_object", 0, cfg)
    broken = traces.copy()
    broken.loc[0, "state_index"] = 999
    with pytest.raises(RuntimeError, match="Rollout/trace linkage mismatch"):
        evaluation.validate_result(rows, broken, "libero_object", 0, cfg)


def test_synthetic_full_analysis_counts_contrast_signs_and_preserves_provenance(tmp_path, monkeypatch):
    config, cfg = local_config(tmp_path)
    output = Path(cfg["output_root"])
    controls, control_traces = [], []
    evaluation.evaluation_manifest(cfg, create=True)
    for suite in cfg["suites"]:
        folder = output / suite
        folder.mkdir(parents=True)
        for task in range(10):
            r, t = rows_for_task(cfg, suite, task)
            r.to_parquet(folder / f"task_{task:02d}_rollouts.parquet", index=False)
            t.to_parquet(folder / f"task_{task:02d}_traces.parquet", index=False)
            r, t = rows_for_task(cfg, suite, task, controls=True)
            controls.append(r)
            control_traces.append(t)
    monkeypatch.setattr(analysis, "frozen_controls", lambda cfg: (pd.concat(controls), pd.concat(control_traces)))
    status = analysis.analyze(config, draws=16)
    assert status["new_attack_episodes"] == 800 and status["reused_attack_episodes"] == 1200
    assert status["combined_attack_episodes"] == 2000 and status["eligible_attack_rows"] == 1910
    summary = pd.read_csv(output / f"{analysis.PREFIX}_PER_SUITE.csv")
    contrasts = pd.read_csv(output / f"{analysis.PREFIX}_CONTRASTS.csv")
    assert len(summary) == 20 and len(contrasts) == 28
    assert set(summary.suite) == set(cfg["suites"])
    assert contrasts.loc[contrasts.contrast_role.eq("ROLLING_MINUS_FIXED"), "difference_left_minus_right"].eq(-1).all()
    assert contrasts.loc[contrasts.contrast_role.str.startswith("FIXED_MINUS"), "difference_left_minus_right"].eq(1).all()
    assert contrasts.loc[contrasts.contrast_role.str.startswith("ROLLING_MINUS") &
                         ~contrasts.contrast_role.eq("ROLLING_MINUS_FIXED"), "difference_left_minus_right"].eq(0).all()
    combined = pd.read_parquet(output / f"{analysis.PREFIX}_ROLLOUTS.parquet")
    reused = combined.loc[combined.condition.isin(evaluation.CONTROLS)]
    assert reused.experiment_id.eq(evaluation.previous.EXPERIMENT_ID).all()
    assert reused.evaluation_origin.eq("REUSED_ILA_V2_EVALUATION").all()
    rolling = combined.loc[combined.condition.eq(evaluation.CONDITIONS[1])]
    assert rolling.appearance_assessment.eq("NOISY_USER_REPORTED_NOT_ACCEPTED").all()
    (output / "libero_object/task_00_rollouts.parquet").unlink()
    with pytest.raises(RuntimeError, match="Missing evaluation shard"):
        analysis.analyze(config, draws=16)


@pytest.mark.parametrize("interrupt", [False, True])
def test_real_shared_runner_runs_only_new_patches_resumes_and_skips_final(tmp_path, monkeypatch, interrupt):
    config, cfg = local_config(tmp_path)
    calls, closed = [], []
    monkeypatch.setattr(shared, "load_wrapper", lambda cfg: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(shared, "make_env", lambda cfg, task: (SimpleNamespace(close=lambda: closed.append(True)),
                         SimpleNamespace(language="toy"), list(range(100))))
    monkeypatch.setattr(shared, "renderer_metadata", lambda cfg: ("toy", "NVIDIA"))

    def episode(env, wrapper, task, init_state, seed, condition, patch, digest, cfg, renderer, vendor, order):
        calls.append((init_state, seed, condition))
        if interrupt and len(calls) == 5:
            raise RuntimeError("mock interruption")
        assert patch.shape == (3, 50, 50) and cfg["num_open_loop_steps"] == 8
        return {"condition": condition, "success": 1, "failure": 0, "rollout_seed": seed,
                "condition_order": order, "patch_sha256": digest, "renderer_vendor": vendor,
                "episode_steps": 1}, [{"condition": condition, "decision_block": 0, "step": 0}]

    monkeypatch.setattr(shared, "run_episode", episode)
    if interrupt:
        with pytest.raises(RuntimeError, match="mock interruption"):
            evaluation.run_checked_shard(0, config)
    evaluation.run_checked_shard(0, config)
    assert len(calls) == (21 if interrupt else 20)
    assert {m for _, _, m in calls} == set(evaluation.CONDITIONS)
    for state in range(40, 50):
        assert len({seed for s, seed, _ in calls if s == state}) == 1
    previous_count = len(calls)
    evaluation.run_checked_shard(0, config)
    assert len(calls) == previous_count and len(closed) == (2 if interrupt else 1)


def test_resume_manifest_drift_and_manual_submit_gates(tmp_path):
    _, cfg = local_config(tmp_path)
    evaluation.evaluation_manifest(cfg, create=True)
    changed = copy.deepcopy(cfg)
    changed["order_seed"] += 1
    with pytest.raises(RuntimeError, match="config/source changed during resume"):
        evaluation.evaluation_manifest(changed)
    submit = (ROOT / "submit_exploring_oft_dino_alternating_refine200_cross_suite.sh").read_text()
    assert submit.index("CHECK_ONLY") < submit.index("sbatch --parsable")
    assert "afterok:${eval_id}" in submit and "57_train" not in submit and "55_dino" not in submit
    job = (ROOT / "slurm/exploring_oft_dino/58_dino_alternating_refine200_cross_suite_3090.sbatch").read_text()
    assert "#SBATCH --array=0-39%5" in job and "#SBATCH --gres=gpu:3090:1" in job
    assert "sbatch " not in job

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# Reproduce the audit's real collision: stdlib statistics is already imported.
stdlib_spec = importlib.util.spec_from_file_location("statistics", Path(sysconfig.get_path("stdlib")) / "statistics.py")
stdlib_statistics = importlib.util.module_from_spec(stdlib_spec)
sys.modules["statistics"] = stdlib_statistics
stdlib_spec.loader.exec_module(stdlib_statistics)

import run_exploring_oft_correspondence500_cross_suite as evaluation
import run_exploring_oft_scale500_cross_suite as shared
from analyze_exploring_oft_correspondence500_cross_suite import CONTRASTS, PREFIX, analyze, stats

CONFIG = ROOT / "configs/exploring_oft_dino/object_correspondence500_cross_suite.json"


def rows_for_task(cfg, suite, task):
    clean = evaluation.frozen_clean(cfg, suite)
    rows = []
    failures = {"exploring_oft_uada": 0, "uada_tv": 0, "correspondence": 1,
                "correspondence_tv": 1, "feature_tv": 0}
    for item in clean.loc[clean.task_id == task].itertuples():
        for order, method in enumerate(cfg["conditions"], 1):
            failure = failures[method]
            rows.append({"experiment_id": evaluation.EXPERIMENT_ID, "suite": suite,
                         "task_id": task, "state_index": item.state_index, "condition": method,
                         "condition_order": order, "rollout_seed": item.rollout_seed,
                         "frozen_clean_success": item.success, "success": 1 - failure, "failure": failure,
                         "patch_sha256": cfg["patch_sources"][method]["sha256"],
                         "renderer_vendor": "NVIDIA", "episode_steps": 10})
    rollouts = pd.DataFrame(rows)
    traces = rollouts[["experiment_id", "suite", "task_id", "state_index", "condition"]].copy()
    traces["decision_block"] = 0
    return rollouts, traces


def test_approved_patches_and_frozen_clean_are_ready():
    cfg = evaluation.load_config(CONFIG)
    result = evaluation.preflight(cfg)
    assert result["new_attack_rollouts"] == 2000
    assert result["eligible_attack_rows_expected"] == 1910
    assert result["clean_success_denominators"] == evaluation.DENOMINATORS
    assert len(set(result["patch_sha256"].values())) == 5
    assert result["model_loaded"] is False


def test_patch_hash_and_visual_marker_drift_are_rejected(tmp_path):
    cfg = evaluation.load_config(CONFIG)
    changed = copy.deepcopy(cfg)
    changed["patch_sources"]["correspondence"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Patch hash mismatch"):
        evaluation.load_correspondence_patches(changed, torch.device("cpu"))
    overlay = json.loads(CONFIG.read_text())
    overlay["manual_visual_review"] = "missing"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(overlay))
    with pytest.raises(RuntimeError, match="visual approval missing"):
        evaluation.load_config(path)


def test_validation_checks_seed_hash_success_labels_and_trace_identity():
    cfg = evaluation.load_config(CONFIG)
    rows, traces = rows_for_task(cfg, "libero_object", 0)
    evaluation.validate_result(rows, traces, "libero_object", 0, cfg)
    broken = rows.copy()
    broken.loc[0, "rollout_seed"] += 1
    with pytest.raises(RuntimeError, match="Paired rollout seed drift"):
        evaluation.validate_result(broken, traces, "libero_object", 0, cfg)
    broken = rows.copy()
    broken.loc[0, "patch_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="Rollout patch hash mismatch"):
        evaluation.validate_result(broken, traces, "libero_object", 0, cfg)
    broken = rows.copy()
    broken.loc[0, "failure"] = broken.loc[0, "success"]
    with pytest.raises(RuntimeError, match="success/failure labels"):
        evaluation.validate_result(broken, traces, "libero_object", 0, cfg)
    broken_traces = traces.copy()
    broken_traces.loc[0, "state_index"] = 999
    with pytest.raises(RuntimeError, match="Trace without matching rollout"):
        evaluation.validate_result(rows, broken_traces, "libero_object", 0, cfg)


def test_statistics_import_is_independent_of_stdlib_name():
    assert sys.modules["statistics"].mean([1, 3]) == 2
    assert stats.__name__ == "_oft_correspondence_bootstrap"
    assert hasattr(stats, "paired_contrast_bootstrap")
    assert [role for role, _, _ in CONTRASTS] == ["C_MINUS_A", "D_MINUS_B", "D_MINUS_E",
                                                "D_MINUS_A", "B_MINUS_A", "D_MINUS_C"]


def test_full_analysis_on_synthetic_rollouts_keeps_suites_and_contrast_signs(tmp_path):
    overlay = json.loads(CONFIG.read_text())
    overlay["output_root"] = str(tmp_path / "outputs")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(overlay))
    cfg = evaluation.load_config(config)
    output = Path(cfg["output_root"])
    for suite in cfg["suites"]:
        folder = output / suite
        folder.mkdir(parents=True)
        for task in range(10):
            rows, traces = rows_for_task(cfg, suite, task)
            rows.to_parquet(folder / f"task_{task:02d}_rollouts.parquet", index=False)
            traces.to_parquet(folder / f"task_{task:02d}_traces.parquet", index=False)
    status = analyze(config, draws=16)
    assert status["new_attack_episodes"] == 2000 and status["eligible_attack_rows"] == 1910
    summary = pd.read_csv(output / f"{PREFIX}_PER_SUITE.csv")
    contrasts = pd.read_csv(output / f"{PREFIX}_CONTRASTS.csv")
    assert len(summary) == 20 and len(contrasts) == 24
    assert set(summary.suite) == set(cfg["suites"])
    positive = contrasts.contrast_role.isin(["C_MINUS_A", "D_MINUS_B", "D_MINUS_E", "D_MINUS_A"])
    assert contrasts.loc[positive, "difference_left_minus_right"].eq(1).all()
    assert contrasts.loc[~positive, "difference_left_minus_right"].eq(0).all()
    # Missing shards fail before analysis can emit a new COMPLETE status.
    (output / "libero_object/task_00_rollouts.parquet").unlink()
    with pytest.raises(RuntimeError, match="Missing evaluation shard"):
        analyze(config, draws=16)


def test_real_shared_runner_uses_five_patches_and_skips_completed_shard(tmp_path, monkeypatch):
    overlay = json.loads(CONFIG.read_text())
    overlay["output_root"] = str(tmp_path / "evaluation")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(overlay))
    calls = []
    closed = []
    monkeypatch.setattr(shared, "load_wrapper", lambda cfg: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(shared, "make_env", lambda cfg, task: (SimpleNamespace(close=lambda: closed.append(True)),
                         SimpleNamespace(language="toy task"), list(range(100))))
    monkeypatch.setattr(shared, "renderer_metadata", lambda cfg: ("toy", "NVIDIA"))

    def episode(env, wrapper, task, init_state, seed, condition, patch, digest, cfg, renderer, vendor, order):
        calls.append((init_state, seed, condition, digest))
        assert patch.shape == (3, 50, 50)
        assert cfg["patch_xy"] == [26, 137] and cfg["num_open_loop_steps"] == 8
        return {"condition": condition, "success": 1, "failure": 0, "rollout_seed": seed,
                "condition_order": order, "patch_sha256": digest, "renderer_vendor": vendor,
                "episode_steps": 1}, [{"condition": condition, "decision_block": 0, "step": 0}]

    monkeypatch.setattr(shared, "run_episode", episode)
    evaluation.run_checked_shard(0, config)
    assert len(calls) == 50 and len(closed) == 1
    assert {condition for _, _, condition, _ in calls} == set(evaluation.CONDITIONS)
    for state in range(40, 50):
        assert len({seed for s, seed, _, _ in calls if s == state}) == 1
    evaluation.run_checked_shard(0, config)
    assert len(calls) == 50 and len(closed) == 1


def test_submit_only_runs_when_check_only_is_disabled():
    submit = (ROOT / "submit_exploring_oft_correspondence500_cross_suite.sh").read_text()
    assert submit.index("CHECK_ONLY") < submit.index("sbatch --parsable")
    assert "afterok:${eval_id}" in submit
    job = (ROOT / "slurm/exploring_oft_dino/51_correspondence500_cross_suite_3090.sbatch").read_text()
    assert "#SBATCH --array=0-39%5" in job and "#SBATCH --gres=gpu:3090:1" in job
    assert "sbatch " not in job

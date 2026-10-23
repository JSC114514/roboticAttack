"""CPU-only repair workflow contracts; all scheduler submissions are mocked."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker"), str(ROOT / "scripts")]
import dino_repair_common as common
import run_exploring_oft_dino_repair_cross_suite as runner
import analyze_exploring_oft_dino_repair_cross_suite as analyzer


def test_frozen_budget_and_prefix_full_optimizer_state():
    cfg = common.config()
    state, digest, patch = common.prefix_state(cfg)
    assert state["outer"] == 250 and len(state["rows"]) == 250
    assert state["scheduler"]["last_epoch"] == 250
    assert all(int(v["step"]) == 12500 for v in state["optimizer"]["state"].values())
    assert all(k in state for k in ("torch_rng", "cuda_rng", "numpy_rng", "python_rng"))
    assert patch.name == "outer_0250.pt"
    assert cfg["outer_iterations"] == 1000 and cfg["scheduler_total_outer"] == 2000
    assert cfg["checkpoint_outer_iterations"] == [0, 250, 500, 750, 1000]
    assert cfg["refresh_after_outer"] == [500]
    assert len(digest) == 64


def test_repair_sampler_matches_frozen_task_phase_pairing():
    cfg = common.config()
    clean = [{"task_id": i % 10, "id": i} for i in range(300)]
    replay = [{"task_id": task, "phase_proxy": phase, "dino_weight": float(weight)}
              for task in range(10) for phase in ("early", "middle", "late") for weight in (1, 2)]
    for outer in (251, 252, 500, 750, 1000):
        for method in common.METHODS:
            actual = common.draw_record(clean, replay, method, outer, cfg)
            expected = common.old_draw(clean, replay,
                "uniform" if method == "uniform_clear" else "diverse", outer, cfg)
            assert actual == expected
    with pytest.raises(Exception, match="Invalid repair"):
        common.draw_record(clean, replay, "clear", 1001, cfg)


def test_config_rejects_changed_budget_and_trained_warm_start(tmp_path):
    cfg = common.config()
    cfg["scheduler_total_outer"] = 1000
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(Exception, match="schedule"):
        common.config(path)
    cfg = common.config()
    cfg["baseline_patch"] = "outer_1000.pt"
    path.write_text(json.dumps(cfg))
    with pytest.raises(Exception, match="warm start"):
        common.config(path)


def test_normal_eval_grid_sanitized_and_no_training_code_import():
    cfg = runner.load_config()
    assert tuple(cfg["conditions"]) == common.METHODS
    assert not (set(cfg) & runner.STALE_KEYS)
    assert len(cfg["suites"]) * cfg["task_count"] * len(cfg["state_indices"]) * len(cfg["conditions"]) == 1600
    assert 6 * sum(runner.DENOMINATORS.values()) == 2292
    source = (ROOT / "scripts/run_exploring_oft_dino_repair_cross_suite.py").read_text()
    assert "from white_patch.UADA_OFT_DINO_REPAIR" not in source
    assert "clean_siglip" not in source
    eval_shell = (ROOT / "slurm/exploring_oft_dino/105_repair1000_eval_3090.sbatch").read_text()
    train_shell = (ROOT / "slurm/exploring_oft_dino/103_repair1000_train_4090.sbatch").read_text()
    assert "#SBATCH --array=0-9%8" in eval_shell
    assert "#SBATCH --array=0-3%4" in train_shell
    assert "a800" not in train_shell.lower()
    assert "reset" not in eval_shell.lower()


def test_frozen_reference_rows_keep_actual_provenance():
    cfg = runner.load_config()
    rows, traces = runner.reference_frames(cfg)
    assert len(rows) == 800 and rows.reused.all() and traces.reused.all()
    assert set(rows.condition) == set(runner.REFERENCES)
    assert rows.source_experiment_id.eq("OFT-DINO-TRANSFER2000-XSUITE-V1").all()
    assert rows.experiment_id.eq(runner.EXPERIMENT_ID).all()
    assert rows.groupby(["suite", "condition"]).size().eq(100).all()
    assert rows.source_sha256.str.len().eq(64).all()


def submit_config(tmp_path):
    return {**common.config(), "output_root": str(tmp_path / "run")}


def test_failed_train_submission_never_submits_eval(tmp_path, monkeypatch):
    cfg = submit_config(tmp_path)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if len([c for c in calls if c[0] == "sbatch"]) == 1:
            return SimpleNamespace(returncode=0, stdout="12345;cluster\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="training submit failed")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="no downstream"):
        common.submit_chain(cfg)
    submitted = [c for c in calls if c[0] == "sbatch"]
    assert len(submitted) == 2
    assert "--dependency=afterok:12345" in submitted[1]
    receipt = json.loads((Path(cfg["output_root"]) / "reports/SUBMISSION_RECEIPT.json").read_text())
    assert receipt["status"] == "SUBMISSION_FAILED" and receipt["failed_stage"] == "train"
    assert "DependencyNeverSatisfied" in receipt["dependency_failure_help"]


def test_successful_submit_dependency_chain_and_gpu_release(tmp_path, monkeypatch):
    cfg = submit_config(tmp_path)
    submitted = []
    def fake(command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        submitted.append(command)
        return SimpleNamespace(returncode=0, stdout=str(1000 + len(submitted)), stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    receipt = common.submit_chain(cfg)
    assert [r["stage"] for r in receipt["jobs"]] == ["calibrate", "train", "validate", "eval", "stats"]
    assert "--array=0,1,2,3%4" in submitted[1]
    assert "--dependency=afterok:1002" in submitted[2]
    assert "--dependency=afterok:1003" in submitted[3]
    assert "--dependency=afterok:1004" in submitted[4]


def test_queue_query_failure_aborts_before_any_submission(tmp_path, monkeypatch):
    cfg = submit_config(tmp_path)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="scheduler unavailable")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(Exception, match="squeue query failed"):
        common.submit_chain(cfg)
    assert len(calls) == 1 and calls[0][0] == "squeue"


def test_complete_chain_rerun_is_idempotent_without_scheduler_calls(tmp_path, monkeypatch):
    cfg = submit_config(tmp_path)
    status = Path(cfg["output_root"]) / "evaluation/four_suite/STATUS.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"status": "COMPLETE"}))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("No scheduler call after completion"))
    assert common.submit_chain(cfg)["status"] == "ALREADY_COMPLETE"


def fake_shard(cfg, suite, task, hashes):
    rows, traces = [], []
    for state in cfg["state_indices"]:
        for order, method in enumerate(common.METHODS, 1):
            metadata = {"experiment_id": runner.EXPERIMENT_ID, "training_suite": "libero_object",
                        "suite": suite, "task_id": task, "state_index": state,
                        "checkpoint": cfg["checkpoints"][suite], "unnorm_key": cfg["unnorm_keys"][suite],
                        "frozen_clean_success": 1, "condition": method}
            rows.append({**metadata, "success": 0, "failure": 1, "condition_order": order,
                         "rollout_seed": 1000 + state, "patch_sha256": hashes[method]["pt"],
                         "renderer_vendor": "NVIDIA Corporation", "decision_blocks": 1})
            traces.append({**metadata, "decision_block": 0, "step": 0})
    return pd.DataFrame(rows), pd.DataFrame(traces)


def test_four_suites_sequential_then_completed_shards_skip(tmp_path, monkeypatch):
    cfg = runner.load_config()
    cfg["output_root"] = str(tmp_path / "evaluation/four_suite")
    train = common.config()
    train["output_root"] = str(tmp_path)
    hashes = {m: {"pt": str(i) * 64, "png": "f" * 64} for i, m in enumerate(common.METHODS)}
    validation = {"status": "COMPLETE", "config_sha256": runner.sha256_file(common.CONFIG),
                  "evaluation_hooks": False, "checkpoints": hashes}
    (tmp_path / "AUTO_VALIDATION.json").write_text(json.dumps(validation))
    monkeypatch.setattr(runner, "config", lambda: train)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    monkeypatch.setattr(runner, "preflight", lambda base: {"status": "PASS"})
    monkeypatch.setattr(runner, "patches", lambda base, device: ({}, hashes, {}))
    monkeypatch.setattr(runner, "frozen_clean", lambda base, suite: pd.DataFrame(
        [{"task_id": 3, "state_index": s, "success": 1, "rollout_seed": 1000+s} for s in cfg["state_indices"]]))
    calls = []
    def fake_run(index, expanded, patch_loader=None):
        calls.append(index)
        suite, task = cfg["suites"][index // 10], index % 10
        rows, traces = fake_shard(cfg, suite, task, hashes)
        folder = Path(cfg["output_root"]) / suite
        folder.mkdir(parents=True)
        rows.to_parquet(folder / f"task_{task:02d}_rollouts.parquet", index=False)
        traces.to_parquet(folder / f"task_{task:02d}_traces.parquet", index=False)
    monkeypatch.setattr(runner, "run_shard", fake_run)
    runner.run_checked(3)
    assert calls == [3, 13, 23, 33]
    runner.run_checked(3)
    assert calls == [3, 13, 23, 33]
    path = Path(cfg["output_root"]) / "libero_spatial/task_03_rollouts.parquet"
    rows = pd.read_parquet(path)
    rows.loc[0, "patch_sha256"] = "wrong"
    rows.to_parquet(path, index=False)
    with pytest.raises(Exception, match="Patch bytes"):
        runner.run_checked(3)


def test_focus_bootstrap_paired_macro_and_worst_domain():
    rows = []
    for suite in ("libero_spatial", "libero_goal"):
        for task in range(10):
            for state in (40, 41):
                for condition in runner.CONDITIONS:
                    rows.append({"suite": suite, "task_id": task, "state_index": state,
                                 "condition": condition, "casr": float(condition == "dino_repair_clear")})
    focus, contrasts = analyzer.focus_statistics(pd.DataFrame(rows), 100, 42)
    assert set(focus.metric) == {"spatial_goal_macro", "spatial_goal_worst_suite"}
    matched = contrasts.loc[contrasts.contrast.eq("E_minus_D")]
    assert matched.difference_left_minus_right.eq(1.).all()
    assert np.allclose(matched.ci95_low, 1.) and np.allclose(matched.ci95_high, 1.)

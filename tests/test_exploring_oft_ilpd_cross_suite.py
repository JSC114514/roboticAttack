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
import run_exploring_oft_dino_ilpd_refine200_cross_suite as evaluation
import run_exploring_oft_scale500_cross_suite as shared
from analyze_exploring_oft_dino_ilpd_refine200_cross_suite import CONTRASTS, PREFIX, analyze

CONFIG = ROOT / "configs/exploring_oft_dino/object_dino_ilpd_refine200_cross_suite_v2.json"


def rows_for_task(cfg: dict, suite: str, task_id: int):
    clean = evaluation.frozen_clean(cfg, suite)
    rows = []
    for item in clean.loc[clean.task_id == task_id].itertuples():
        for order, method in enumerate(evaluation.CONDITIONS, 1):
            failure = int(method == "dino_ilpd_bounded")
            rows.append({"experiment_id": evaluation.EXPERIMENT_ID, "suite": suite,
                         "task_id": task_id, "state_index": item.state_index, "condition": method,
                         "condition_order": order, "rollout_seed": item.rollout_seed,
                         "frozen_clean_success": item.success, "success": 1 - failure,
                         "failure": failure, "patch_sha256": cfg["patch_sha256"][method],
                         "renderer_vendor": "NVIDIA", "episode_steps": 10})
    rollouts = pd.DataFrame(rows)
    traces = rollouts[["experiment_id", "suite", "task_id", "state_index", "condition"]].copy()
    traces["decision_block"] = 0
    return rollouts, traces


def test_preflight_freezes_training_evidence_and_clean_grid():
    cfg = evaluation.load_config(CONFIG)
    result = evaluation.preflight(cfg)
    assert result["status"] == "PASS" and result["new_attack_episodes"] == 1600
    assert result["eligible_attack_rows_expected"] == 1528
    assert result["frozen_clean_success_denominators"] == evaluation.DENOMINATORS
    assert len(set(result["patch_sha256"].values())) == 4 and not result["gpu_jobs_submitted"]


def test_patch_hash_and_training_evidence_drift_are_rejected():
    cfg = evaluation.load_config(CONFIG)
    changed = copy.deepcopy(cfg)
    changed["patch_sha256"]["dino_ilpd_bounded"] = "0" * 64
    with pytest.raises(RuntimeError, match="Final patch hash mismatch"):
        evaluation.load_ilpd_patches(changed, torch.device("cpu"))
    changed = copy.deepcopy(cfg)
    changed["evidence_sha256"]["dino_ilpd_bounded"][0] = "0" * 64
    with pytest.raises(RuntimeError, match="Training evidence hash mismatch"):
        evaluation.load_ilpd_patches(changed, torch.device("cpu"))


def test_evaluation_validates_pairing_hashes_and_labels():
    cfg = evaluation.load_config(CONFIG)
    rows, traces = rows_for_task(cfg, "libero_object", 0)
    evaluation.validate_result(rows, traces, "libero_object", 0, cfg)
    bad = rows.copy()
    bad.loc[0, "rollout_seed"] += 1
    with pytest.raises(RuntimeError, match="Paired rollout seed drift"):
        evaluation.validate_result(bad, traces, "libero_object", 0, cfg)
    bad = rows.copy()
    bad.loc[0, "patch_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="Rollout patch mismatch"):
        evaluation.validate_result(bad, traces, "libero_object", 0, cfg)
    bad = rows.copy()
    bad.loc[0, "failure"] = bad.loc[0, "success"]
    with pytest.raises(RuntimeError, match="success/failure labels"):
        evaluation.validate_result(bad, traces, "libero_object", 0, cfg)


def test_full_synthetic_analysis_preserves_contrast_signs(tmp_path, monkeypatch):
    overlay = json.loads(CONFIG.read_text())
    overlay["output_root"] = str(tmp_path / "evaluation")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(overlay))
    # Only the synthetic output root is outside the production directory.
    monkeypatch.setattr(evaluation.Path, "is_relative_to", lambda self, other: True)
    cfg = evaluation.load_config(config)
    output = Path(cfg["output_root"])
    for suite in cfg["suites"]:
        folder = output / suite
        folder.mkdir(parents=True)
        for task_id in range(10):
            rows, traces = rows_for_task(cfg, suite, task_id)
            rows.to_parquet(folder / f"task_{task_id:02d}_rollouts.parquet", index=False)
            traces.to_parquet(folder / f"task_{task_id:02d}_traces.parquet", index=False)
    result = analyze(config, draws=16)
    assert result["status"] == "COMPLETE" and result["eligible_attack_rows"] == 1528
    summary = pd.read_csv(output / f"{PREFIX}_PER_SUITE.csv")
    contrasts = pd.read_csv(output / f"{PREFIX}_CONTRASTS.csv")
    assert len(summary) == 16 and len(contrasts) == 20
    positive = contrasts.contrast_role.isin(["DINO_ILPD_MINUS_UADA",
                                             "DINO_ILPD_MINUS_DINO_GRADSCALE",
                                             "DINO_ILPD_MINUS_SIGLIP_ILPD"])
    assert contrasts.loc[positive, "difference_left_minus_right"].eq(1).all()
    assert contrasts.loc[~positive, "difference_left_minus_right"].eq(0).all()
    (output / "libero_object/task_00_rollouts.parquet").unlink()
    with pytest.raises(RuntimeError, match="Missing evaluation shard"):
        analyze(config, draws=16)


def test_real_shared_runner_uses_four_patches_and_resumes(tmp_path, monkeypatch):
    overlay = json.loads(CONFIG.read_text())
    overlay["output_root"] = str(tmp_path / "evaluation")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(overlay))
    monkeypatch.setattr(evaluation.Path, "is_relative_to", lambda self, other: True)
    calls = []
    closed = []
    monkeypatch.setattr(shared, "load_wrapper", lambda cfg: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(shared, "make_env", lambda cfg, task: (
        SimpleNamespace(close=lambda: closed.append(True)), SimpleNamespace(language="toy task"), list(range(100))))
    monkeypatch.setattr(shared, "renderer_metadata", lambda cfg: ("toy", "NVIDIA"))

    def episode(env, wrapper, task, init_state, seed, method, patch, digest, cfg, renderer, vendor, order):
        calls.append((init_state, seed, method, digest))
        assert patch.shape == (3, 50, 50) and cfg["patch_xy"] == [26, 137]
        return {"condition": method, "success": 1, "failure": 0, "rollout_seed": seed,
                "condition_order": order, "patch_sha256": digest, "renderer_vendor": vendor,
                "episode_steps": 1}, [{"condition": method, "decision_block": 0, "step": 0}]

    monkeypatch.setattr(shared, "run_episode", episode)
    evaluation.run_checked_shard(0, config)
    assert len(calls) == 40 and len(closed) == 1
    assert {method for _, _, method, _ in calls} == set(evaluation.CONDITIONS)
    for state in range(40, 50):
        assert len({seed for s, seed, _, _ in calls if s == state}) == 1
    evaluation.run_checked_shard(0, config)
    assert len(calls) == 40 and len(closed) == 1


def test_submit_helper_only_submits_after_check_only_gate():
    helper = (ROOT / "submit_exploring_oft_dino_ilpd_refine200_cross_suite.sh").read_text()
    assert helper.index("CHECK_ONLY") < helper.index("sbatch --parsable")
    assert "afterok:${eval_id}" in helper
    job = (ROOT / "slurm/exploring_oft_dino/62_dino_ilpd_refine200_cross_suite_3090.sbatch").read_text()
    assert "#SBATCH --array=0-39%5" in job and "#SBATCH --gres=gpu:3090:1" in job
    assert "sbatch " not in job
    assert len(CONTRASTS) == 5

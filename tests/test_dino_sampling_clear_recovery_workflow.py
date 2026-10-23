"""CPU checks for bounded source migration, restart safety, and manual submission."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "VLAAttacker")]
import dino_sampling_clear_common as common
import dino_sampling_clear_recovery_common as recovery
import run_exploring_oft_dino_sampling_clear_recovery as runner
import white_patch.UADA_OFT_DINO_SAMPLING_CLEAR as trainer


@pytest.fixture
def transition(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "ADDED_FILES", ("recovery.py",))
    baseline, current, approved = {}, {}, {}
    for name in ("trainer.py", "common.py"):
        path = tmp_path / name
        before = hashlib.sha256((name + " original").encode()).hexdigest()
        path.write_text(name + " reviewed")
        after = recovery.sha256_file(path)
        baseline[str(path)] = before
        current[str(path)] = after
        approved[str(path)] = {"before": before, "after": after}
    unchanged = tmp_path / "model.py"
    unchanged.write_text("frozen model")
    baseline[str(unchanged)] = current[str(unchanged)] = recovery.sha256_file(unchanged)
    added = tmp_path / "recovery.py"
    added.write_text("explicit recovery entry")
    current[str(added)] = recovery.sha256_file(added)
    monkeypatch.setattr(recovery, "APPROVED_CHANGES", approved)
    return baseline, current


def test_only_exact_reviewed_source_bytes_and_named_additions_pass(transition):
    recovery.verify_source_transition(*transition)


@pytest.mark.parametrize("drift", ["modified_model", "modified_reviewed", "unlisted_addition", "removed_source", "missing_addition"])
def test_source_migration_rejects_every_other_drift(transition, tmp_path, drift):
    baseline, current = copy.deepcopy(transition)
    if drift == "modified_model":
        path = tmp_path / "model.py"
        path.write_text("changed model")
        current[str(path)] = recovery.sha256_file(path)
    elif drift == "modified_reviewed":
        path = tmp_path / "trainer.py"
        path.write_text("extra unreviewed change")
        current[str(path)] = recovery.sha256_file(path)
    elif drift == "unlisted_addition":
        path = tmp_path / "other.py"
        path.write_text("extra")
        current[str(path)] = recovery.sha256_file(path)
    elif drift == "removed_source":
        current.pop(str(tmp_path / "model.py"))
    else:
        current.pop(str(tmp_path / "recovery.py"))
    with pytest.raises(RuntimeError):
        recovery.verify_source_transition(baseline, current)


def test_ledger_cannot_authorize_a_different_reviewed_digest(transition, tmp_path, monkeypatch):
    cfg = {"output_root": str(tmp_path / "run")}
    config_path = tmp_path / "config.json"
    config_path.write_text("frozen config")
    monkeypatch.setattr(common, "CONFIG", config_path)
    identity = {"config_sha256": recovery.sha256_file(config_path),
                "reviewed_changes": recovery.APPROVED_CHANGES,
                "actual_cpu_source": transition[1]}
    ledger = {"status": "PREPARED", "recovery_id": recovery.RECOVERY_ID,
              "identity": identity, "signature": common.digest_json(identity)}
    recovery.ledger_path(cfg).parent.mkdir(parents=True)
    recovery.ledger_path(cfg).write_text(json.dumps(ledger))
    path = str(tmp_path / "trainer.py")
    recovery.verify_historical_source(cfg, path, transition[0][path])
    with pytest.raises(RuntimeError, match="outside audited"):
        recovery.verify_historical_source(cfg, path, "unreviewed old bytes")
    ledger["identity"]["reviewed_changes"][path]["after"] = "unreviewed new bytes"
    ledger["signature"] = common.digest_json(ledger["identity"])
    recovery.ledger_path(cfg).write_text(json.dumps(ledger))
    # Independent reviewed constants still reject a self-consistent ledger.
    monkeypatch.setattr(recovery, "APPROVED_CHANGES", {
        n: {"before": transition[0][n], "after": transition[1][n]}
        for n in transition[0] if n.endswith(("trainer.py", "common.py"))})
    with pytest.raises(RuntimeError, match="ledger identity"):
        recovery._load_ledger(cfg)


def _saved_state(cfg, method, outer, source, data):
    patch = torch.full(tuple(cfg["patch_shape"]), .5)
    return {"config": cfg, "method": method, "outer": outer,
            "signature": recovery.run_signature(cfg, method, data, source), "source": source, "data": data,
            "patch": patch, "initial": patch, "initial_raw": patch,
            "raw_random_patch_sha256": trainer.patch_tensor_sha256(patch),
            "initial_patch_sha256": trainer.patch_tensor_sha256(patch),
            "rows": [{"outer_iteration": i, "optimizer_updates": i * 50,
                      "learning_rate": 0., "rgb_outer_max_abs": 0.} for i in range(1, outer + 1)],
            "optimizer": {"state": {0: {"step": outer * 50, "exp_avg": patch,
                                          "exp_avg_sq": patch}}, "param_groups": []},
            "scheduler": {"last_epoch": outer}, "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "cuda_rng": [torch.zeros(1, dtype=torch.uint8)], "score_after": 1500,
            "score_refresh_history": [{"outer": i} for i in cfg["refresh_after_outer"]],
            "replay_scoring": [{} for _ in range(258)]}


@pytest.fixture
def interrupted_arm(tmp_path):
    cfg = {"output_root": str(tmp_path / "run"), "outer_iterations": 2000,
           "inner_loop": 50, "patch_shape": [3, 2, 2],
           "checkpoint_outer_iterations": [], "refresh_after_outer": [250, 500, 1000, 1500]}
    method = "dynamic_hard"
    folder = Path(cfg["output_root"]) / "training" / method
    folder.mkdir(parents=True)
    source_file = tmp_path / "source.py"
    source_file.write_text("frozen source")
    source = {str(source_file): recovery.sha256_file(source_file)}
    data = {"config": "frozen"}
    initial = _saved_state(cfg, method, 1998, source, data)
    manifest = {"method": method, "config": cfg, "data": data,
                "source": source, "signature": initial["signature"]}
    (folder / "RUN_MANIFEST.json").write_text(json.dumps(manifest))
    with zipfile.ZipFile(folder / "source_snapshot.zip", "w") as archive:
        archive.write(source_file, str(source_file).lstrip("/"))
        archive.writestr("RUN_MANIFEST.json", json.dumps(manifest))
    status = {"method": method, "status": "FAILED", "outer": 1998, "optimizer_updates": 99900,
              "attempted_outer": 1999, "signature": initial["signature"],
              "error": "RuntimeError('Nonzero LR but unchanged complete outer RGB')",
              "source_snapshot_sha256": recovery.sha256_file(folder / "source_snapshot.zip")}
    (folder / "status.json").write_text(json.dumps(status))
    torch.save(initial, folder / "resume.pt")
    pd.DataFrame(initial["rows"]).to_parquet(folder / "training.parquet")
    evidence = recovery.recovery_root(cfg) / "evidence" / method
    evidence.mkdir(parents=True)
    shutil.copy2(folder / "resume.pt", evidence / "resume.pt")
    preparation = {"identity": {"config": cfg, "data": data, "source": source}}
    return cfg, method, folder, initial, preparation


def test_atomic_resume_may_advance_before_derived_status_and_parquet(interrupted_arm):
    cfg, method, folder, initial, preparation = interrupted_arm
    advanced = _saved_state(cfg, method, 1999, initial["source"], initial["data"])
    torch.save(advanced, folder / "resume.pt")
    arm = recovery._audit_arm(cfg, method, preparation, original=False)
    assert arm["outer"] == 1999 and arm["remaining_optimizer_updates"] == 50
    assert json.loads((folder / "status.json").read_text())["outer"] == 1998
    assert len(pd.read_parquet(folder / "training.parquet")) == 1998


def test_retry_rejects_changed_original_rows_even_with_valid_adam_state(interrupted_arm):
    cfg, method, folder, initial, preparation = interrupted_arm
    advanced = _saved_state(cfg, method, 1999, initial["source"], initial["data"])
    advanced["rows"][0]["rgb_outer_max_abs"] = 1.
    torch.save(advanced, folder / "resume.pt")
    with pytest.raises(RuntimeError, match="valid committed prefix"):
        recovery._audit_arm(cfg, method, preparation, original=False)


def test_completed_a_is_rejected_before_model_or_cuda_access(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA touched"))
    monkeypatch.setattr(recovery, "preflight", lambda cfg: pytest.fail("A recovery preflight entered"))
    with pytest.raises(RuntimeError, match="completed A"):
        runner.run("dynamic_soft", {})


@pytest.fixture
def submission(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text("frozen")
    monkeypatch.setattr(common, "CONFIG", config_path)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    cfg = {"output_root": str(tmp_path / "run"), "experiment_id": "test"}
    Path(cfg["output_root"]).mkdir()
    report = {"identity": {"config_sha256": recovery.sha256_file(config_path)},
              "pending_methods": list(recovery.PENDING)}
    monkeypatch.setattr(recovery, "preflight", lambda cfg: report)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)) + "\n", stderr="")
    monkeypatch.setattr(recovery.subprocess, "run", execute)
    return cfg, calls, execute


def test_manual_submission_preserves_stage_receipts_and_exact_gpu_limits(submission):
    cfg, calls, _ = submission
    receipt = recovery.submit_chain(cfg)
    assert receipt["status"] == "SUBMITTED"
    assert [job["stage"] for job in receipt["jobs"]] == ["prepare", "train", "validate", "eval", "stats"]
    assert "--array=1,2,3%3" in calls[2]
    assert "--array=0-9%8" in calls[4]
    for previous, following in zip(receipt["jobs"], receipt["jobs"][1:]):
        assert "--dependency=afterok:" + previous["job_id"] in following["command"]
    assert all(command[0] != "scancel" for command in calls)
    assert not (Path(cfg["output_root"]) / "reports/SUBMISSION_RECEIPT.json").exists()


def test_submission_failure_stops_downstream_and_retains_successful_ids(submission, monkeypatch):
    cfg, calls, execute = submission
    def fail_train(command, **kwargs):
        if command[-1].endswith("118_clear_sampling_recovery_train_4090.sbatch"):
            calls.append(command)
            return SimpleNamespace(returncode=1, stdout="", stderr="scheduler rejected")
        return execute(command, **kwargs)
    monkeypatch.setattr(recovery.subprocess, "run", fail_train)
    with pytest.raises(RuntimeError, match="failed at train"):
        recovery.submit_chain(cfg)
    receipt = json.loads((recovery.recovery_root(cfg) / "SUBMISSION_RECEIPT.json").read_text())
    assert receipt["status"] == "SUBMISSION_FAILED"
    assert [job["stage"] for job in receipt["jobs"]] == ["prepare"]
    assert len(calls) == 3


def test_queued_original_chain_blocks_recovery_without_sbatch(submission, monkeypatch):
    cfg, calls, _ = submission
    def occupied(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="123 running\n", stderr="")
    monkeypatch.setattr(recovery.subprocess, "run", occupied)
    with pytest.raises(RuntimeError, match="still queued"):
        recovery.submit_chain(cfg)
    assert len(calls) == 1 and calls[0][0] == "squeue"


def test_completed_recovery_is_a_noop_before_queue_or_gpu_submission(submission, monkeypatch):
    cfg, calls, _ = submission
    monkeypatch.setattr(recovery, "preflight", lambda cfg: {"identity": {}, "pending_methods": []})
    ledger = {"signature": "ledger", "identity": {"preparation_signature": "original"}}
    monkeypatch.setattr(recovery, "verify_preserved_artifacts", lambda cfg: ledger)
    validation = Path(cfg["output_root"]) / "AUTO_VALIDATION.json"
    validation.write_text("validated patches")
    root = recovery.recovery_root(cfg)
    root.mkdir(parents=True)
    (root / "VALIDATION.json").write_text(json.dumps({"status": "COMPLETE", "ledger_signature": "ledger",
        "validation_sha256": recovery.sha256_file(validation)}))
    final = Path(cfg["output_root"]) / "evaluation/four_suite/STATUS.json"
    final.parent.mkdir(parents=True)
    final.write_text(json.dumps({"status": "COMPLETE", "experiment_id": "test",
        "config_sha256": recovery.sha256_file(common.CONFIG), "preparation_signature": "original",
        "raw_shard_hashes": {}}))
    assert recovery.submit_chain(cfg)["status"] == "ALREADY_COMPLETE"
    assert calls == []


def test_completed_a_artifact_tampering_fails_preservation_check(tmp_path):
    cfg = {"output_root": str(tmp_path / "run")}
    root = Path(cfg["output_root"])
    prep = root / "reports/PREPARATION.json"
    prep.parent.mkdir(parents=True)
    prep.write_text("original preparation")
    folder = root / "training/dynamic_soft"
    folder.mkdir(parents=True)
    (folder / "outer_2000.pt").write_bytes(b"completed A")
    ledger = {"identity": {"preparation_sha256": recovery.sha256_file(prep),
                          "arms": {"dynamic_soft": {"artifacts": recovery._artifact_hashes(folder)}}}}
    recovery.verify_preserved_artifacts(cfg, ledger)
    (folder / "outer_2000.pt").write_bytes(b"changed A")
    with pytest.raises(RuntimeError, match="Preserved artifact changed"):
        recovery.verify_preserved_artifacts(cfg, ledger)


def test_prepare_retry_rechecks_completed_a_scientific_validation(tmp_path, monkeypatch):
    cfg = {"output_root": str(tmp_path / "run")}
    path = recovery.ledger_path(cfg)
    path.parent.mkdir(parents=True)
    path.write_text("already published")
    monkeypatch.setattr(recovery, "preflight", lambda cfg: {"identity": {}})
    monkeypatch.setattr(recovery, "_load_ledger", lambda cfg: {"signature": "ledger"})
    calls = []
    monkeypatch.setattr(common, "validate_training", lambda cfg, methods, write_report:
                        calls.append((methods, write_report)))
    assert recovery.prepare(cfg)["status"] == "ALREADY_PREPARED"
    assert calls == [(("dynamic_soft",), False)]


def test_preflight_helper_has_no_submission_side_effect(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\"\n")
    fake_python.chmod(0o755)
    env = dict(__import__("os").environ, OFT_SAMPLING_CLEAR_PROJECT_ROOT=str(ROOT),
               OFT_SAMPLING_CLEAR_PYTHON=str(fake_python))
    result = subprocess.run(["bash", str(ROOT / "submit_exploring_oft_dino_sampling_clear_recovery_v1.sh"),
                             "--preflight"], check=True, capture_output=True, text=True, env=env)
    assert result.stdout.strip().endswith("dino_sampling_clear_recovery_common.py --preflight")
    assert "--submit" not in result.stdout


def test_slurm_recovery_excludes_a_and_keeps_normal_eval_entry():
    folder = ROOT / "slurm/exploring_oft_dino"
    train = (folder / "118_clear_sampling_recovery_train_4090.sbatch").read_text()
    evaluation = (folder / "120_clear_sampling_recovery_eval_3090.sbatch").read_text()
    assert "#SBATCH --array=1,2,3%3" in train and "#SBATCH --gres=gpu:4090:1" in train
    assert "#SBATCH --array=0-9%8" in evaluation and "#SBATCH --gres=gpu:3090:1" in evaluation
    assert "run_exploring_oft_dino_sampling_clear_cross_suite.py" in evaluation
    assert "--check-evidence" in evaluation
    for number, stage in ((117, "prepare"), (119, "validate"), (121, "stats")):
        text = (folder / f"{number}_clear_sampling_recovery_{stage}_cpu.sbatch").read_text()
        assert "#SBATCH --partition=cpu" in text and "#SBATCH --gres" not in text

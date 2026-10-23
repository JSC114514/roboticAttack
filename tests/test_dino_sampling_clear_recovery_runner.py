"""CPU runner mocks: no actual CUDA, VLA loading, Slurm, or training."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "VLAAttacker")]

import dino_sampling_clear_recovery_common as recovery
import run_exploring_oft_dino_sampling_clear_recovery as runner
import white_patch.UADA_OFT_DINO_SAMPLING_CLEAR as trainer


@pytest.fixture
def runner_setup(tmp_path, monkeypatch):
    cfg = {"output_root": str(tmp_path / "run"), "patch_xy": [26, 137],
           "outer_iterations": 2000, "inner_loop": 50, "scheduler_total_outer": 2000}
    cfg_before = copy.deepcopy(cfg)
    original_bytes = {}
    original_source, actual_cpu = {}, {}
    for name, old, new in (("model.py", b"frozen model", b"frozen model"),
                           ("trainer.py", b"original trainer", b"reviewed trainer"),
                           ("runtime_import.py", b"frozen remote code", b"frozen remote code")):
        path = tmp_path / name
        path.write_bytes(new)
        original_bytes[str(path)] = old
        original_source[str(path)] = hashlib.sha256(old).hexdigest()
        if old != new:
            actual_cpu[str(path)] = runner.sha256_file(path)
    addition = tmp_path / "recovery_entry.py"
    addition.write_text("reviewed recovery entry")
    actual_cpu[str(addition)] = runner.sha256_file(addition)
    expected_actual = dict(original_source, **actual_cpu)
    data = {"clean_bank": "old-clean", "window_bank": "old-replay", "config": "old-config"}
    ledger = {"signature": "audited-ledger", "identity": {"actual_cpu_source": actual_cpu}}
    calls = {"preflight": [], "verify": [], "load": [], "cuda": [], "train": []}
    control = {"fail": False}
    root = recovery.recovery_root(cfg)
    root.mkdir(parents=True)
    for method in recovery.PENDING:
        outer = recovery.ORIGINAL_OUTERS[method]
        folder = Path(cfg["output_root"]) / "training" / method
        folder.mkdir(parents=True)
        rows = [{"outer_iteration": i, "optimizer_updates": i * 50, "origin": "preserved"}
                for i in range(1, outer + 1)]
        saved = {"outer": outer, "rows": rows, "source": original_source, "data": data}
        trainer.atomic_torch(saved, folder / "resume.pt")
        trainer.atomic_json({"status": "FAILED", "outer": outer}, folder / "status.json")
        trainer.atomic_json({"source": original_source, "data": data, "signature": "original-" + method},
                            folder / "RUN_MANIFEST.json")
        with zipfile.ZipFile(folder / "source_snapshot.zip", "w") as archive:
            for name, content in original_bytes.items():
                archive.writestr(name.lstrip("/"), content)
        evidence = root / "evidence" / method
        evidence.mkdir(parents=True)
        trainer.atomic_torch(saved, evidence / "resume.pt")

    clean, replay = [{"bank": "clean"}], [{"bank": "replay"}]

    def preflight(configuration):
        calls["preflight"].append(copy.deepcopy(configuration))
        return {"status": "PASS", "identity": ledger["identity"]}

    def verify(configuration, existing=None):
        assert configuration == cfg_before and (existing is None or existing == ledger)
        calls["verify"].append(existing)
        return ledger

    def load_wrapper(configuration):
        calls["load"].append(copy.deepcopy(configuration))
        return SimpleNamespace(device=torch.device("cpu"), vla=torch.nn.Linear(1, 1).requires_grad_(False))

    def forbidden(*args, **kwargs):
        pytest.fail("Real CUDA/model startup is forbidden in recovery runner CPU mocks")

    class DummyAttacker:
        def __init__(self, wrapper, output_root, patch_xy):
            assert wrapper.device.type == "cpu" and output_root == cfg["output_root"] and patch_xy == cfg["patch_xy"]

        def train_sampling_clear(self, passed_clean, passed_replay, configuration, method, source, passed_data, load_s):
            calls["train"].append({"cfg": copy.deepcopy(configuration), "method": method,
                                   "source": copy.deepcopy(source), "data": copy.deepcopy(passed_data)})
            assert passed_clean is clean and passed_replay is replay and load_s >= 0
            if control["fail"]:
                raise RuntimeError("mock tail interrupted")
            folder = Path(configuration["output_root"]) / "training" / method
            saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
            saved["rows"].extend({"outer_iteration": i, "optimizer_updates": i * 50, "origin": "mock-tail"}
                                 for i in range(saved["outer"] + 1, 2001))
            saved["outer"] = 2000
            trainer.atomic_torch(saved, folder / "resume.pt")
            trainer.atomic_json({"status": "COMPLETE", "outer": 2000}, folder / "status.json")

    monkeypatch.setenv("SLURM_JOB_ID", "CPU-MOCK-ONLY")
    monkeypatch.setattr(recovery, "preflight", preflight)
    monkeypatch.setattr(recovery, "verify_preserved_artifacts", verify)
    monkeypatch.setattr(runner.common, "source_records", lambda configuration: clean)
    monkeypatch.setattr(runner.common, "replay_records", lambda configuration: (replay, "old-replay"))
    monkeypatch.setitem(sys.modules, "UADA_OFT_DINO_wrapper", SimpleNamespace(load_wrapper=load_wrapper))
    monkeypatch.setattr(trainer, "source_manifest", lambda configuration, wrapper: dict(expected_actual))
    monkeypatch.setattr(trainer, "ExploringOFTDINOSamplingClearAttacker", DummyAttacker)
    for name in ("init", "_lazy_init", "current_device", "set_device", "get_rng_state", "get_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: calls["cuda"].append("is_available") or True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: calls["cuda"].append("device_count") or 1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls["cuda"].append("synchronize"))
    return SimpleNamespace(cfg=cfg, cfg_before=cfg_before, source=original_source, data=data,
                           ledger=ledger, actual=expected_actual, calls=calls, control=control, forbidden=forbidden)


def receipt_path(state, method):
    return recovery.recovery_root(state.cfg) / f"{method}_RUNTIME.json"


@pytest.mark.parametrize("method", recovery.PENDING)
def test_bcd_runner_completes_exact_tail_and_records_actual_union_source(method, runner_setup):
    state = runner_setup
    folder = Path(state.cfg["output_root"]) / "training" / method
    evidence = recovery.recovery_root(state.cfg) / "evidence" / method / "resume.pt"
    original = torch.load(evidence, map_location="cpu", weights_only=False)
    evidence_hash, snapshot_hash = runner.sha256_file(evidence), runner.sha256_file(folder / "source_snapshot.zip")
    result = runner.run(method, state.cfg)
    assert result == {"status": "COMPLETE", "method": method, "to_outer": 2000}
    assert state.cfg == state.cfg_before
    assert state.calls["train"] == [{"cfg": state.cfg, "method": method, "source": state.source, "data": state.data}]
    final = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    assert final["outer"] == 2000 and len(final["rows"]) == 2000
    assert final["rows"][:recovery.ORIGINAL_OUTERS[method]] == original["rows"]
    assert final["source"] == state.source and final["data"] == state.data
    assert runner.sha256_file(evidence) == evidence_hash and runner.sha256_file(folder / "source_snapshot.zip") == snapshot_hash
    receipt = json.loads(receipt_path(state, method).read_text())
    assert receipt["status"] == "COMPLETE" and receipt["to_outer"] == 2000
    assert receipt["resumed_from_outer"] == receipt["from_outer"] == recovery.ORIGINAL_OUTERS[method]
    assert receipt["actual_runtime_source"] == state.actual
    assert receipt["ledger_signature"] == state.ledger["signature"]
    assert receipt["final_resume_sha256"] == runner.sha256_file(folder / "resume.pt")
    snapshot = recovery.recovery_root(state.cfg) / f"{method}_actual_source_snapshot.zip"
    assert receipt["actual_source_snapshot_sha256"] == runner.sha256_file(snapshot)
    with zipfile.ZipFile(snapshot) as archive:
        assert set(archive.namelist()) == {name.lstrip("/") for name in state.actual}
        assert all(hashlib.sha256(archive.read(name.lstrip("/"))).hexdigest() == digest
                   for name, digest in state.actual.items())
    assert state.calls["cuda"] == ["is_available", "device_count", "synchronize"]
    assert len(state.calls["verify"]) == 2 and not torch.cuda.is_initialized()


def test_runner_excludes_a_and_rejects_actual_runtime_source_drift(runner_setup, monkeypatch):
    state = runner_setup
    with pytest.raises(RuntimeError, match="completed A"):
        runner.run("dynamic_soft", state.cfg)
    assert not any(state.calls.values())
    changed = dict(state.actual)
    changed[next(iter(changed))] = "unaudited loaded source"
    monkeypatch.setattr(trainer, "source_manifest", lambda cfg, wrapper: changed)
    with pytest.raises(RuntimeError, match="Actually loaded recovery runtime differs"):
        runner.run("dynamic_hard", state.cfg)
    assert state.calls["train"] == [] and not receipt_path(state, "dynamic_hard").exists()
    assert not (recovery.recovery_root(state.cfg) / "dynamic_hard_actual_source_snapshot.zip").exists()


def test_completed_recovered_arm_skips_model_and_all_cuda_access(runner_setup, monkeypatch):
    state = runner_setup
    method = "stable_hard"
    runner.run(method, state.cfg)
    state.calls["load"].clear()
    state.calls["cuda"].clear()
    state.calls["train"].clear()
    monkeypatch.setitem(sys.modules, "UADA_OFT_DINO_wrapper", SimpleNamespace(load_wrapper=state.forbidden))
    for name in ("is_available", "device_count", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, state.forbidden)
    result = runner.run(method, state.cfg)
    assert result == {"status": "ALREADY_COMPLETE", "method": method}
    assert state.calls["load"] == state.calls["cuda"] == state.calls["train"] == []
    receipt = json.loads(receipt_path(state, method).read_text())
    assert receipt["status"] == "COMPLETE" and receipt["to_outer"] == 2000


def test_failed_runner_receipt_and_original_commit_are_preserved_on_retry(runner_setup):
    state = runner_setup
    method = "shuffled_hard"
    folder = Path(state.cfg["output_root"]) / "training" / method
    original_resume_hash = runner.sha256_file(folder / "resume.pt")
    state.control["fail"] = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="mock tail interrupted"):
            runner.run(method, state.cfg)
        assert runner.sha256_file(folder / "resume.pt") == original_resume_hash
    receipt = json.loads(receipt_path(state, method).read_text())
    assert receipt["status"] == receipt["previous_attempt"]["status"] == "FAILED"
    assert receipt["error"] == receipt["previous_attempt"]["error"] == "RuntimeError('mock tail interrupted')"
    assert receipt["to_outer"] == recovery.ORIGINAL_OUTERS[method]
    assert receipt["actual_runtime_source"] == receipt["previous_attempt"]["actual_runtime_source"] == state.actual
    assert receipt["input_resume_sha256"] == original_resume_hash
    assert (recovery.recovery_root(state.cfg) / f"{method}_actual_source_snapshot.zip").is_file()

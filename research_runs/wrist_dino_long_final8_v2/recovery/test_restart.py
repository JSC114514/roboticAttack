"""CPU/mocked scheduler tests; never submit real jobs or load a VLA."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("final8_restart", Path(__file__).with_name("restart.py"))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_selective_train_array_and_serial_dependencies():
    command = recovery.build_command("train", "07_train_4090.sbatch", [0, 7], external=["90"])
    assert "--array=0,7%8" in command and "--dependency=afterany:90" in command
    command = recovery.build_command("eval", "09_eval_3090.sbatch", [0, 7], dependency="92", external=["90"])
    assert "--dependency=afterok:92" in command
    assert not any("afterany" in part or part.startswith("--array") for part in command)


@pytest.mark.parametrize("queue,owned", [
    ("138718_[0-7]|PENDING|gres/gpu:4090:1|Resources|oft_long8v2_train\n", set()),
    ("138720_0|RUNNING|gres/gpu:3090:1|aias-compute-4|other_name\n", {"138720"}),
    ("138800|PENDING|gres/gpu:4090:1|DependencyNeverSatisfied|unrelated\n", set()),
    ("unrecognized-format\n", set()),
])
def test_unsafe_queue_blocks_submission(queue, owned):
    with pytest.raises(RuntimeError):
        recovery.check_queue(queue, owned)


def test_unrelated_gpu_waits_but_cpu_does_not():
    queue = "20|RUNNING|gres/gpu:4090:1|node|unrelated\n21|RUNNING|N/A|node|cpu_job\n"
    assert recovery.check_queue(queue, set()) == ["20"]
    assert recovery.check_queue("", set()) == []
    assert recovery.receipt_jobs({"jobs": [{"job_id": "20"}], "previous_receipt": {"jobs": [{"job_id": "19"}]}}) == {"19", "20"}


def mock_submission(monkeypatch, tmp_path, pending=(0, 7), responses=()):
    root = tmp_path / "project/research_runs/batch"
    (root / "reports").mkdir(parents=True)
    (root.parents[1] / "submit_exploring_oft_long_final8_2000_resume_v2.sh").touch()
    monkeypatch.setattr(recovery, "ROOT", root)
    monkeypatch.setattr(recovery, "recovery_preflight", lambda: {
        "signature": "CPU_MOCK_ONLY", "pending": list(pending), "checkpoints": {}, "already_complete": False})
    calls, items = [], iter(responses)
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command[0] == "sbatch"
        code, stdout, stderr = next(items, (0, str(900 + len(calls)) + "\n", ""))
        return subprocess.CompletedProcess(command, code, stdout, stderr)
    monkeypatch.setattr(recovery.subprocess, "run", run)
    return root, calls


def test_only_recovery_stages_submitted_and_receipted(monkeypatch, tmp_path):
    root, calls = mock_submission(monkeypatch, tmp_path)
    recovery.submit()
    receipt = json.loads((root / "reports" / recovery.RECEIPT_NAME).read_text())
    assert receipt["status"] == "SUBMITTED"
    assert [job["stage"] for job in receipt["jobs"]] == ["train", "validate", "eval", "stats"]
    assert "--array=0,7%8" in calls[1]
    for before, after in zip(receipt["jobs"], receipt["jobs"][1:]):
        assert "--dependency=afterok:" + before["job_id"] in after["command"]
    assert not (root / "reports/SUBMISSION_RECEIPT.json").exists()


def test_partial_submission_failure_preserves_ids(monkeypatch, tmp_path):
    root, calls = mock_submission(monkeypatch, tmp_path, responses=((0, "901\n", ""), (1, "", "mock unavailable")))
    with pytest.raises(RuntimeError, match="Failed at validate"):
        recovery.submit()
    receipt = json.loads((root / "reports" / recovery.RECEIPT_NAME).read_text())
    assert receipt["status"] == "SUBMISSION_FAILED" and receipt["jobs"][0]["job_id"] == "901"
    assert len(calls) == 3 and all(command[0] != "scancel" for command in calls)


def test_completed_training_skips_gpu_train(monkeypatch, tmp_path):
    root, calls = mock_submission(monkeypatch, tmp_path, pending=())
    recovery.submit()
    receipt = json.loads((root / "reports" / recovery.RECEIPT_NAME).read_text())
    assert [job["stage"] for job in receipt["jobs"]] == ["validate", "eval", "stats"]


def test_unavailable_controller_submits_nothing(monkeypatch, tmp_path):
    root, calls = mock_submission(monkeypatch, tmp_path)
    monkeypatch.setattr(recovery.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "mock offline"))
    with pytest.raises(RuntimeError, match="squeue unavailable"):
        recovery.submit()
    assert not (root / "reports" / recovery.RECEIPT_NAME).exists()


def test_preflight_never_calls_scheduler(monkeypatch, tmp_path):
    mock_submission(monkeypatch, tmp_path)
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected scheduler call"))
    recovery.main(["--preflight"])
    assert not (recovery.ROOT / "reports" / recovery.RECEIPT_NAME).exists()

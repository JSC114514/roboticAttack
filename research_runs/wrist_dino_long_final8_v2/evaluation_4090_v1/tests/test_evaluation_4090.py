"""CPU/mocked behavior checks only; never start a model, renderer or Slurm job."""
import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import eval_common


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


worker = module("final8_4090_worker", "evaluate.py")
pipeline = module("final8_4090_pipeline", "pipeline.py")


def function_ast(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
                    include_attributes=False)


def test_scientific_core_and_graphics_copy_are_unchanged():
    source = ROOT.parent / "code"
    assert function_ast(source / "evaluate.py", "episode") == function_ast(ROOT / "code/evaluate.py", "episode")
    for name in ("task_casr", "paired_bootstrap"):
        assert function_ast(source / "analyze.py", name) == function_ast(ROOT / "code/analyze.py", name)
    assert (source / "model_runtime.py").read_bytes() == (ROOT / "code/model_runtime_4090.py").read_bytes()
    assert eval_common.config() == eval_common.original.config()
    assert eval_common.runtime_config("libero_10") == eval_common.original.runtime_config("libero_10")
    assert eval_common.EVALUATION_RESOURCES["max_concurrent"] == 8


@pytest.mark.parametrize("gpu,allowed", [("NVIDIA GeForce RTX 4090", True),
                                         ("NVIDIA GeForce RTX 3090", False), ("cpu", False)])
def test_real_device_guard_not_fake_3090(monkeypatch, gpu, allowed):
    monkeypatch.setenv("SLURM_JOB_ID", "CPU_MOCK_ONLY")
    monkeypatch.setenv("OFT_STAGE_GPU", "3090")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda n: gpu)
    if allowed:
        assert eval_common.require_gpu("evaluation") == gpu
        assert eval_common.os.environ["OFT_STAGE_GPU"] == "4090"
    else:
        with pytest.raises(RuntimeError):
            eval_common.require_gpu("evaluation")
    with pytest.raises(RuntimeError):
        eval_common.require_gpu("training")


def test_copy_never_reverts_to_high_level_gl_string_query():
    source = (ROOT / "code/model_runtime_4090.py").read_text()
    assert "GL.glGetString(" not in source
    assert "ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint)" in source
    assert 'eglBindAPI(backend.EGL.EGL_OPENGL_API)' in source
    assert 'eglGetCurrentContext()' in source


@pytest.mark.parametrize("line", [
    "138820_[0-9%8]|PENDING|gres:gpu:3090:1|ReqNodeNotAvail|oft_long8v2_eval",
    "138821|PENDING|N/A|Dependency|oft_long8v2_stats",
    "200_[0-9%8]|RUNNING|gres:gpu:4090:1|node|oft_long8_4090_eval",
    "210|PENDING|gres:gpu:4090:1|DependencyNeverSatisfied|other_job",
])
def test_obsolete_or_duplicate_or_invalid_gpu_jobs_block_submission(line):
    with pytest.raises(RuntimeError):
        pipeline.queue_dependencies(line, set())


def test_unrelated_gpu_waits_and_serial_commands():
    assert pipeline.queue_dependencies("200|RUNNING|gres:gpu:4090:1|node|other_job", set()) == ["200"]
    assert pipeline.queue_dependencies("", set()) == []
    first = pipeline.command_for("01_prepare_cpu.sbatch", existing=["200"])
    second = pipeline.command_for("02_eval_4090.sbatch", dependency="201", existing=["200"])
    assert "--dependency=afterany:200" in first
    assert "--dependency=afterok:201" in second
    assert not any("afterany" in item for item in second)
    assert [stage for stage, _ in pipeline.STAGES] == ["prepare", "eval", "stats"]
    sbatch = (ROOT / "slurm/02_eval_4090.sbatch").read_text()
    assert "#SBATCH --gres=gpu:4090:1" in sbatch and "#SBATCH --array=0-9%8" in sbatch
    assert "--qos" not in sbatch and "3090" not in sbatch
    with pytest.raises(RuntimeError):
        pipeline.queue_dependencies("200|RUNNING|gres:gpu:4090:1|node|other_job", {"200"})


def fake_suite(monkeypatch, tmp_path):
    conditions = ["clean"] + [f"mock_{n}" for n in range(24)]
    cfg = {"evaluation": {"seed_sources": {"libero_object": "CPU_MOCK.parquet"}, "order_seed": 7}}
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "config", lambda: cfg)
    monkeypatch.setattr(worker, "run_signature", lambda: "CPU_MOCK_ONLY")
    monkeypatch.setattr(worker, "source_signature", lambda: "CPU_MOCK_PARENT")
    monkeypatch.setattr(worker, "prepared", lambda: {})
    monkeypatch.setattr(worker, "patches", lambda: ({c: None for c in conditions}, {c: "mock_hash" for c in conditions}))
    monkeypatch.setattr(worker, "require_gpu", lambda stage: "CPU_MOCK_4090")
    monkeypatch.setattr(worker, "runtime_config", lambda suite: {})
    monkeypatch.setattr(worker, "deployment_conditions", lambda: conditions)
    seeds = pd.DataFrame({"suite": ["libero_object"] * 10, "condition": ["clean"] * 10,
                          "task_id": [0] * 10, "state_index": list(range(40, 50)), "rollout_seed": list(range(10))})
    monkeypatch.setattr(worker.pd, "read_parquet", lambda path: seeds)
    return tmp_path / "evaluation/libero_object/task_00_STATUS.json"


def test_egl_preload_before_model_and_failure_phase_is_saved(monkeypatch, tmp_path):
    status_path = fake_suite(monkeypatch, tmp_path)
    order = []
    monkeypatch.setattr(worker, "prepare_graphics_backend", lambda: order.append("egl"))
    def fail_model(*args):
        order.append("model")
        raise RuntimeError("CPU mock model error")
    monkeypatch.setattr(worker, "load_wrapper", fail_model)
    with pytest.raises(RuntimeError, match="CPU mock model error"):
        worker.run_suite(0, "libero_object")
    assert order == ["egl", "model"]
    status = json.loads(status_path.read_text())
    assert status["status"] == "FAILED" and status["phase"] == "load_model"
    assert status["episodes_complete"] == 0


def test_close_failure_not_complete_and_saved_episodes_resume(monkeypatch, tmp_path):
    status_path = fake_suite(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(worker, "prepare_graphics_backend", lambda: None)
    monkeypatch.setattr(worker, "load_wrapper", lambda *args: object())
    monkeypatch.setattr(worker, "make_env", lambda *args, **kwargs: (object(), object(), {n: [] for n in range(40, 50)}))
    monkeypatch.setattr(worker, "renderer_metadata", lambda env: ("CPU_MOCK_4090", "CPU_MOCK_NVIDIA"))
    def episode(*args):
        calls.append(1)
        return {"success": 1, "failure": 0, "executed_steps": 1, "decision_blocks": 1}, [{"decision_block": 0}]
    monkeypatch.setattr(worker, "episode", episode)
    def failed_close(env):
        assert json.loads(status_path.read_text())["status"] != "COMPLETE"
        raise RuntimeError("CPU mock close error")
    monkeypatch.setattr(worker, "close_environment", failed_close)
    with pytest.raises(RuntimeError, match="CPU mock close error"):
        worker.run_suite(0, "libero_object")
    assert len(calls) == 250
    assert json.loads(status_path.read_text())["status"] == "FAILED"
    monkeypatch.setattr(worker, "close_environment", lambda env: None)
    worker.run_suite(0, "libero_object")
    assert len(calls) == 250  # no already-saved episode is rerun
    assert json.loads(status_path.read_text())["status"] == "COMPLETE"


def test_native_suite_exit_code_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, -11))
    with pytest.raises(subprocess.CalledProcessError):
        worker.run_task(0)
    status = json.loads((tmp_path / "evaluation/libero_object/task_00_STATUS.json").read_text())
    assert status["status"] == "FAILED" and status["child_returncode"] == -11


def fake_pipeline(monkeypatch, tmp_path, failing_stage=None):
    child = tmp_path / "source/evaluation_4090_v1"
    (child.parent / "reports").mkdir(parents=True)
    monkeypatch.setattr(pipeline, "ROOT", child)
    monkeypatch.setattr(pipeline, "SOURCE_ROOT", child.parent)
    monkeypatch.setattr(pipeline, "preflight", lambda: {"signature": "CPU_MOCK_ONLY", "source_training_signature": "CPU_MOCK_PARENT"})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command[0] == "sbatch"
        if failing_stage and failing_stage in command[-1]:
            return subprocess.CompletedProcess(command, 1, "", "CPU mock unavailable")
        return subprocess.CompletedProcess(command, 0, str(900 + len(calls)) + "\n", "")
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    return child, calls


def test_three_stage_submission_no_training_or_smoke(monkeypatch, tmp_path):
    child, calls = fake_pipeline(monkeypatch, tmp_path)
    pipeline.submit()
    receipt = json.loads((child / "reports/SUBMISSION_RECEIPT.json").read_text())
    assert receipt["status"] == "SUBMITTED" and receipt["training_restarted"] is False
    assert [job["stage"] for job in receipt["jobs"]] == ["prepare", "eval", "stats"]
    assert len(calls) == 4
    for left, right in zip(receipt["jobs"], receipt["jobs"][1:]):
        assert "--dependency=afterok:" + left["job_id"] in right["command"]
    assert (child / "logs").is_dir()


def test_partial_failure_keeps_ids_and_does_not_cancel(monkeypatch, tmp_path):
    child, calls = fake_pipeline(monkeypatch, tmp_path, failing_stage="02_eval")
    with pytest.raises(RuntimeError, match="Failed at eval"):
        pipeline.submit()
    receipt = json.loads((child / "reports/SUBMISSION_RECEIPT.json").read_text())
    assert receipt["status"] == "SUBMISSION_FAILED" and len(receipt["jobs"]) == 1
    assert all(call[0] != "scancel" for call in calls)


def test_preflight_does_not_query_or_submit(monkeypatch):
    monkeypatch.setattr(pipeline, "preflight", lambda: {"signature": "CPU_MOCK_ONLY"})
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected Slurm call"))
    pipeline.main(["--preflight"])


def test_cpu_preparation_has_real_entrypoint():
    source = (ROOT / "code/eval_common.py").read_text()
    assert 'if __name__ == "__main__":\n    prepare()' in source

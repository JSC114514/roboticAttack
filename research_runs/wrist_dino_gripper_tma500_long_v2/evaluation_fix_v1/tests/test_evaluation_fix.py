"""CPU-only regression tests; mocked renderer checks are NOT a GPU pass."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

FIX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIX / "code"))
import eval_common as common
import evaluate_fixed as evaluation
import model_runtime_fixed as runtime
import pipeline_fixed as pipeline


def fake_environment(events):
    def context(label):
        return SimpleNamespace(_context=object(), make_current=lambda: events.append("bind:" + label))
    env = SimpleNamespace(sim=SimpleNamespace(_render_context_offscreen=SimpleNamespace(gl_ctx=context("old"))))
    raw = {key: np.zeros((256, 256, 3), np.uint8)
           for key in ("agentview_image", "robot0_eye_in_hand_image")}
    env.seed = lambda seed: events.append(("seed", seed))
    def reset():
        events.append("reset")
        env.sim._render_context_offscreen.gl_ctx = context("new")
        return raw
    env.reset = reset
    env.set_init_state = lambda state: events.append("initial_state") or raw
    env.get_sim_state = lambda: np.array([1., 2., 3.])
    env.step = lambda action: (events.append("wait") or raw, 0., False, {})
    env.close = lambda: events.append("close")
    return env, raw


def fake_gl(monkeypatch, events, renderer=b"NVIDIA GeForce RTX4090", vendor=b"NVIDIA Corporation"):
    package = ModuleType("OpenGL")
    def query(key):
        events.append("query:" + key)
        return renderer if key == "renderer" else vendor
    package.GL = SimpleNamespace(GL_RENDERER="renderer", GL_VENDOR="vendor", glGetString=query)
    monkeypatch.setitem(sys.modules, "OpenGL", package)


def test_renderer_binds_the_environment_before_query(monkeypatch):
    events = []
    env, _ = fake_environment(events)
    fake_gl(monkeypatch, events)
    assert runtime.renderer_metadata(env) == ("NVIDIA GeForce RTX4090", "NVIDIA Corporation")
    assert events == ["bind:old", "query:renderer", "query:vendor"]


@pytest.mark.parametrize("renderer,vendor", [(None, b"NVIDIA"), (b"NVIDIA", None), (b"", b"NVIDIA")])
def test_empty_renderer_is_a_diagnostic_error_not_decode(monkeypatch, renderer, vendor):
    events = []
    env, _ = fake_environment(events)
    fake_gl(monkeypatch, events, renderer, vendor)
    with pytest.raises(RuntimeError, match="OpenGL returned empty"):
        runtime.renderer_metadata(env)


@pytest.mark.parametrize("renderer,vendor", [(b"llvmpipe", b"Mesa"), (b"softpipe", b"NVIDIA")])
def test_cpu_renderer_is_not_a_fix(monkeypatch, renderer, vendor):
    events = []
    env, _ = fake_environment(events)
    fake_gl(monkeypatch, events, renderer, vendor)
    with pytest.raises(RuntimeError):
        runtime.renderer_metadata(env)


@pytest.mark.parametrize("broken", ["missing", "freed", "unbound"])
def test_missing_freed_unbindable_context_rejected(monkeypatch, broken):
    events = []
    env, _ = fake_environment(events)
    if broken == "missing":
        env.sim._render_context_offscreen = None
    elif broken == "freed":
        env.sim._render_context_offscreen.gl_ctx._context = None
    else:
        def fail():
            raise RuntimeError("eglMakeCurrent failed")
        env.sim._render_context_offscreen.gl_ctx.make_current = fail
    fake_gl(monkeypatch, events)
    with pytest.raises(RuntimeError):
        runtime.renderer_metadata(env)
    assert not any(item.startswith("query:") for item in events)


def install_dummy_action(monkeypatch):
    module = ModuleType("VLAAttacker.multiview.openvla_oft_experiments")
    module.get_libero_dummy_action = lambda: [0., 0., 0., 0., 0., 0., -1.]
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_reset_rebinds_old_then_new_context_and_preserves_waits(monkeypatch):
    events = []
    env, raw = fake_environment(events)
    fake_gl(monkeypatch, events)
    install_dummy_action(monkeypatch)
    monkeypatch.setattr(runtime.gc, "collect", lambda: events.append("collect"))
    assert runtime.reset_wait(env, np.array([1., 2., 3.]), 12345, 10) is raw
    assert events[:5] == [("seed", 12345), "bind:old", "reset", "collect", "bind:new"]
    assert events.index("initial_state") > events.index("bind:new")
    assert events.count("reset") == 1 and events.count("wait") == 10
    assert events[-3:] == ["bind:new", "query:renderer", "query:vendor"]


def test_reset_still_rejects_wrong_initial_state(monkeypatch):
    events = []
    env, _ = fake_environment(events)
    fake_gl(monkeypatch, events)
    install_dummy_action(monkeypatch)
    with pytest.raises(RuntimeError, match="Initial state reset mismatch"):
        runtime.reset_wait(env, np.zeros(3), 42, 10)
    assert "wait" not in events


def test_invalid_camera_frame_rejected(monkeypatch):
    events = []
    env, raw = fake_environment(events)
    raw["robot0_eye_in_hand_image"] = np.zeros((224, 224, 3), np.uint8)
    fake_gl(monkeypatch, events)
    install_dummy_action(monkeypatch)
    with pytest.raises(RuntimeError, match="camera frame invalid"):
        runtime.reset_wait(env, np.array([1., 2., 3.]), 42, 10)


def test_close_binds_before_native_free(monkeypatch):
    events = []
    env, _ = fake_environment(events)
    monkeypatch.setattr(runtime.gc, "collect", lambda: events.append("collect"))
    runtime.close_environment(env)
    assert events == ["bind:old", "close", "collect"]


def fake_randomization_error(monkeypatch):
    module = ModuleType("robosuite.utils.errors")
    class RandomizationError(RuntimeError):
        pass
    module.RandomizationError = RandomizationError
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return RandomizationError


def test_render_reset_error_is_not_silently_retried_forever(monkeypatch):
    fake_randomization_error(monkeypatch)
    calls = []
    def reset():
        calls.append(True)
        raise RuntimeError("EGL reset failed")
    env = SimpleNamespace(env=SimpleNamespace(reset=reset))
    with pytest.raises(RuntimeError, match="EGL reset failed"):
        runtime.reset_environment(env)
    assert len(calls) == 1


def test_original_placement_randomization_retry_is_preserved(monkeypatch):
    error = fake_randomization_error(monkeypatch)
    calls = []
    expected = {"actual_observation": True}
    def reset():
        calls.append(True)
        if len(calls) < 3:
            raise error("random placement rejected")
        return expected
    env = SimpleNamespace(env=SimpleNamespace(reset=reset))
    assert runtime.reset_environment(env) is expected and len(calls) == 3


def test_environment_constructor_does_not_add_unseeded_reset(monkeypatch):
    events, constructor_args = [], []
    env, _ = fake_environment(events)
    fake_gl(monkeypatch, events)
    package = ModuleType("libero.libero")
    task = SimpleNamespace(problem_folder="suite", bddl_file="task.bddl")
    bench = SimpleNamespace(get_task=lambda _: task, get_task_init_states=lambda _: ["state"])
    package.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_object": lambda: bench})
    package.get_libero_path = lambda _: "/mock/bddl"
    env_package = ModuleType("libero.libero.envs")
    env_package.OffScreenRenderEnv = lambda **kwargs: constructor_args.append(kwargs) or env
    monkeypatch.setitem(sys.modules, "libero.libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", env_package)
    monkeypatch.setattr(runtime, "add_paths", lambda _: None)
    actual, _, states = runtime.make_env({"env_image_resolution": 256}, "libero_object", 0)
    assert actual is env and states == ["state"] and "reset" not in events
    assert constructor_args[0]["camera_names"] == ["agentview", "robot0_eye_in_hand"]
    assert "hard_reset" not in constructor_args[0]  # original True default preserved


def function_ast(path, name):
    return ast.dump(next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)


@pytest.mark.parametrize("function", ["frozen_clean", "paste_png", "run_episode"])
def test_scientific_episode_functions_unchanged(function):
    assert function_ast(common.ORIGIN_ROOT / "code/evaluate.py", function) == function_ast(
        FIX / "code/evaluate_fixed.py", function)


@pytest.mark.parametrize("function", ["observation", "load_wrapper", "quat2axisangle"])
def test_model_preprocessing_and_weights_loader_unchanged(function):
    assert function_ast(common.ORIGIN_ROOT / "code/model_runtime_copy.py", function) == function_ast(
        FIX / "code/model_runtime_fixed.py", function)


def test_bootstrap_statistics_unchanged():
    assert function_ast(common.ORIGIN_ROOT / "code/analyze.py", "bootstrap_suite") == function_ast(
        FIX / "code/analyze_fixed.py", "bootstrap_suite")


def test_original_patch_loading_is_exact_and_read_only():
    cfg = common.config()
    before = common.sha256_file(common.ORIGIN_ROOT / "reports/AUTO_VALIDATION.json")
    patches, hashes = evaluation.load_patches(cfg)
    validation = common.load_json(common.ORIGIN_ROOT / "reports/AUTO_VALIDATION.json")
    assert set(patches) == set(common.CONDITIONS)
    for method in common.METHODS:
        assert patches[method].dtype == np.uint8 and patches[method].shape == (50, 50, 3)
        assert {key: hashes[method][key] for key in ("pt", "png")} == validation["checkpoints"][method]["500"]
    assert before == common.sha256_file(common.ORIGIN_ROOT / "reports/AUTO_VALIDATION.json")
    assert common.ROOT != common.ORIGIN_ROOT


@pytest.fixture
def mocked_rollout(monkeypatch, tmp_path):
    cfg = common.config()
    prep = {"fixed": "mock-preparation"}
    calls, cleanup = [], []
    patches = {name: None if name == "clean" else np.zeros((50, 50, 3), np.uint8)
               for name in common.CONDITIONS}
    hashes = {name: {"deployment_rgb": name} for name in common.CONDITIONS}
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "config", lambda: cfg)
    monkeypatch.setattr(evaluation, "preparation", lambda: prep)
    monkeypatch.setattr(evaluation, "load_patches", lambda _: (patches, hashes))
    monkeypatch.setattr(evaluation, "source_manifest", lambda *_: {})
    monkeypatch.setattr(evaluation, "require_smoke", lambda *_: None)
    monkeypatch.setattr(evaluation, "require_stage_gpu", lambda *_: {"gpu": "MOCK-4090-NOT-A-REAL-GPU"})
    monkeypatch.setattr(evaluation.torch.cuda, "empty_cache", lambda: None)
    wrapper = SimpleNamespace(gen_cfg=SimpleNamespace(unnorm_key="key"),
                              vla=SimpleNamespace(get_action_stats=lambda _: {"mask": [True]*6+[False]}))
    monkeypatch.setattr(evaluation, "load_wrapper", lambda *_: wrapper)
    task = SimpleNamespace(language="mock task")
    states = [np.array([index]) for index in range(50)]
    monkeypatch.setattr(evaluation, "make_env", lambda *_: (object(), task, states))
    monkeypatch.setattr(evaluation, "renderer_metadata", lambda *_: ("MOCK-4090", "NVIDIA mock"))
    monkeypatch.setattr(evaluation, "close_environment", lambda *_: cleanup.append("close"))
    def clean(_, suite):
        return pd.DataFrame({"task_id": [0]*10, "state_index": list(range(40, 50)),
                             "success": [1]*10, "rollout_seed": list(range(400, 410))})
    monkeypatch.setattr(evaluation, "frozen_clean", clean)
    def episode(env, model, instruction, state, seed, rgb, runtime_cfg, cfg):
        calls.append((int(state[0]), seed))
        row = {"success": 1, "failure": 0, "executed_steps": 1,
               "eligible_close_slots": 1, "wrong_open_slots": 0}
        trace = [{"step": 1, "decision_block": 0, "action_slot": 0, "clean_gripper": 0.,
                  "attacked_gripper": 0., "executed_gripper": 1., "clean_close": 1, "wrong_open": 0}]
        return row, trace
    monkeypatch.setattr(evaluation, "run_episode", episode)
    return cfg, prep, hashes, calls, cleanup, tmp_path


def test_smoke_pairs_reused_without_repeating_twenty_episodes(mocked_rollout):
    cfg, _, hashes, calls, cleanup, root = mocked_rollout
    suite = "libero_object"
    evaluation.run_suite(0, suite, smoke=True)
    folder = root / "evaluation" / suite
    assert len(calls) == 5 and not (folder / "task_00_STATUS.json").exists()
    assert len(torch.load(folder / "task_00_resume.pt", weights_only=False)["rows"]) == 5
    evaluation.run_suite(0, suite, smoke=True)
    assert len(calls) == 5  # paired smoke retry also resumes
    evaluation.run_suite(0, suite)
    assert len(calls) == 50 and sum(state == 40 for state, seed in calls) == 5
    assert cleanup == ["close", "close"]
    rows = pd.read_parquet(folder / "task_00_rollouts.parquet")
    traces = pd.read_parquet(folder / "task_00_traces.parquet")
    evaluation.validate_frame(rows, traces, cfg, 0, suite, hashes)
    receipt = common.load_json(folder / "task_00_STATUS.json")
    assert receipt["episodes"] == 50 and receipt["repair_id"] == common.REPAIR_ID
    evaluation.run_suite(0, suite)
    assert len(calls) == 50


def test_failure_keeps_previous_pairs_and_does_not_commit_partial_state(mocked_rollout, monkeypatch):
    _, _, _, calls, _, root = mocked_rollout
    suite = "libero_object"
    evaluation.run_suite(0, suite, smoke=True)
    count = 0
    prior = evaluation.run_episode
    def fail_third(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("mock render failure")
        return prior(*args)
    monkeypatch.setattr(evaluation, "run_episode", fail_third)
    with pytest.raises(RuntimeError, match="mock render failure"):
        evaluation.run_suite(0, suite)
    folder = root / "evaluation" / suite
    saved = torch.load(folder / "task_00_resume.pt", weights_only=False)
    assert len(saved["rows"]) == 5 and {row["state_index"] for row in saved["rows"]} == {40}
    assert not (folder / "task_00_STATUS.json").exists()
    progress = common.load_json(folder / "task_00_PROGRESS.json")
    assert progress["status"] == "FAILED" and progress["stage"] == "episode"


def test_retry_after_close_failure_rechecks_real_setup_without_duplicate_episode(mocked_rollout, monkeypatch):
    _, _, _, calls, cleanup, root = mocked_rollout
    suite = "libero_object"
    evaluation.run_suite(0, suite, smoke=True)
    path = root / "evaluation" / suite / "task_00_PROGRESS.json"
    previous = common.load_json(path)
    previous.update(status="FAILED", stage="environment_close")
    common.atomic_json(previous, path)
    loads = []
    prior = evaluation.load_wrapper
    monkeypatch.setattr(evaluation, "load_wrapper", lambda *args: loads.append(True) or prior(*args))
    evaluation.run_suite(0, suite, smoke=True)
    assert len(loads) == 1 and len(calls) == 5 and cleanup == ["close", "close"]
    assert common.load_json(path)["status"] == "SMOKE_STATE_COMPLETE"


def test_worker_sigsegv_is_recorded_and_smoke_not_marked_pass(mocked_rollout, monkeypatch):
    _, _, _, _, _, root = mocked_rollout
    calls = []
    def worker(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=-11)
    monkeypatch.setattr(evaluation.subprocess, "run", worker)
    with pytest.raises(RuntimeError, match="returncode=-11"):
        evaluation.run_task(0, smoke=True)
    assert len(calls) == 1 and calls[0][1:4] == ["-X", "faulthandler", "-B"]
    assert not (root / "reports/GPU_SMOKE.json").exists()
    progress = common.load_json(root / "evaluation/libero_object/task_00_PROGRESS.json")
    assert progress["worker_exit_code"] == -11


def test_smoke_all_four_workers_serial_and_same_original_grid(mocked_rollout, monkeypatch):
    _, _, _, _, cleanup, root = mocked_rollout
    commands = []
    def worker(command, **kwargs):
        commands.append(command)
        evaluation.run_suite(0, command[command.index("--suite")+1], smoke=True)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(evaluation.subprocess, "run", worker)
    evaluation.run_task(0, smoke=True)
    assert [command[command.index("--suite")+1] for command in commands] == list(common.DENOMINATORS)
    assert all("--smoke" in command and "sbatch" not in command for command in commands)
    assert len(cleanup) == 4
    receipt = common.load_json(root / "reports/GPU_SMOKE.json")
    assert receipt["status"] == "GPU_SMOKE_PASS" and receipt["paired_episodes_reused"] == 20


def test_stats_gate_requires_actual_smoke_report(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluation.require_smoke(common.config(), {}, {})


def test_manual_submission_only_four_stages_and_five4090_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "preflight", lambda: {"status": "MOCK_CPU"})
    commands = []
    next_job = 9000
    def fake_run(command, **kwargs):
        nonlocal next_job
        commands.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert command[0] == "sbatch"
        next_job += 1
        return SimpleNamespace(returncode=0, stdout=str(next_job), stderr="")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline.submit()  # mocked sbatch only; no real scheduler calls
    receipt = common.load_json(tmp_path / "reports/SUBMISSION_RECEIPT.json")
    assert [job["stage"] for job in receipt["jobs"]] == ["prepare", "smoke", "eval", "stats"]
    assert receipt["training_restarted"] is False
    gpu_jobs = [job for job in receipt["jobs"] if job["stage"] in {"smoke", "eval"}]
    assert all("--partition=rtx" in job["command"] and "--gres=gpu:4090:1" in job["command"] for job in gpu_jobs)
    assert "--array=0,1,2,3,4,5,6,7,8,9%5" in gpu_jobs[1]["command"]
    assert "--dependency=afterok:9002" in gpu_jobs[1]["command"]
    assert "--dependency=afterok:9003" in receipt["jobs"][-1]["command"]
    assert not any("a800" in arg or "3090" in arg or "scancel" in arg
                   for command in commands for arg in command)


def test_preflight_path_never_queries_or_submits(monkeypatch):
    calls = []
    monkeypatch.setattr(common.original, "preparation", lambda: {"status": "CPU_PREFLIGHT_PASS"})
    expected = {"status": "COMPLETE"}
    monkeypatch.setattr(common.original, "load_json", lambda _: expected)
    import manage
    monkeypatch.setattr(manage, "validated_training", lambda write: calls.append(write) or expected)
    monkeypatch.setattr(common, "trained_inventory", lambda: {})
    monkeypatch.setattr(common, "source_manifest", lambda: {})
    def forbidden(*args, **kwargs):
        raise AssertionError("No scheduler or subprocess in preflight")
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = common.preflight()
    assert calls == [False] and result["gpu_started"] is False
    assert result["training_restarted"] is False


def test_all_shell_syntax_and_no_training_commands():
    scripts = [common.ENTRY, FIX / "slurm/environment.sh", *sorted((FIX / "slurm").glob("*.sbatch"))]
    for path in scripts:
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
    assert len(scripts) == 6
    for script in (FIX / "slurm").glob("*.sbatch"):
        text = script.read_text()
        assert "train.py" not in text and "collect_labels.py" not in text
        assert "--kill-on-invalid-dep=yes" in text
    assert "--array=0-9%5" in (FIX / "slurm/156_eval_4090.sbatch").read_text()

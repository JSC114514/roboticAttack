"""CPU-only regression: real loss code on a small frozen stand-in, never OFT/GPU."""
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common
import train
import analyze
import pipeline
from batch_runtime import prepare_references, pixel_batch_backward, generator_batch_backward
from feature_objectives import new_generator, high_component
from remaining_objectives import batch_feature_loss, cached_generator_backward, rgb_terms

torch.set_num_threads(2)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(12, 36, bias=False)
        self.num_heads = 2
        self.q_norm = self.k_norm = torch.nn.Identity()
        self.scale = (6.) ** -.5

    def forward(self, x):
        qkv = self.qkv(x)
        return x + qkv[..., 24:] * .01


class Dino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([SimpleBlock() for _ in range(24)])
        self.num_prefix_tokens = 5

    def forward(self, rgb):
        spatial = F.adaptive_avg_pool2d(rgb, (16, 16)).flatten(2).transpose(1, 2)
        # Spatially varying offsets keep cosine and affinity objectives nontrivial.
        offsets = torch.arange(256, device=rgb.device).float()[None, :, None] / 512
        x = torch.cat((spatial, spatial.square(), spatial.sin(), spatial.cos()), -1) + offsets
        prefix = x.mean(1, keepdim=True).expand(-1, 5, -1)
        x = torch.cat((prefix, x), 1)
        for block in self.blocks:
            x = block(x)
        return x[:, 5:]


class SimpleBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()

    def forward(self, x):
        return self.attn(x)


class Engine:
    def __init__(self, wrapper=None):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            self.dino = Dino().eval().requires_grad_(False)
        self.wrapper = SimpleNamespace(device=torch.device("cpu"), vla=self.dino,
                                       action_head=None, proprio_projector=None)

    def base(self, record):
        image = torch.as_tensor(record["observation"]["wrist_image"]).permute(2, 0, 1)[None].float() / 255
        return {"pixel_values": image}, np.zeros(8)

    def pixels(self, record, rgb, seed, diverse=True, return_rgb=False):
        original = self.base(record)[0]["pixel_values"]
        padded = F.pad(rgb[None], (0, 174, 0, 174))
        mask = F.pad(torch.ones_like(rgb[None]), (0, 174, 0, 174))
        rendered = original * (1 - mask) + padded
        footprint = torch.zeros(256)
        footprint[:16] = 1
        return (rendered, footprint, rendered) if return_rgb else (rendered, footprint)

    def direct_dino(self, pixels):
        return self.dino(pixels)

    def forward(self, inputs, proprio, pixels):
        value = pixels.mean().expand(1, 8, 7)
        return value, {}, {}

    def validate(self, record, rgb):
        return {"status": "CPU_TOY_ONLY", "not_real_model": True}


def records(n):
    return [{"uid": f"e{i}:0", "episode_id": f"e{i}", "task_id": i % 10, "task_name": f"task{i}",
             "expert_actions": torch.full((1, 8, 7), (-1.) ** i * .4),
             "observation": {"wrist_image": np.full((224, 224, 3), 50 + (i * 11) % 185, np.uint8)}} for i in range(n)]


@pytest.mark.parametrize("method", common.METHODS)
def test_selected_arm_has_connected_real_objective(method):
    cfg, info, engine = common.config(), common.arm_info(method), Engine()
    batch = records(info["batch"])
    rgb = torch.rand(3, 50, 50, generator=torch.Generator().manual_seed(17), requires_grad=True)
    references = prepare_references(engine, batch, info["objective"], cfg)
    if info["id"] in {4, 9}:
        value, detail = generator_batch_backward(engine, batch, references, rgb, list(range(len(batch))), info["objective"], cfg)
        if info["id"] == 9 and info["batch"] == 1:
            assert detail["feature_negative_nce"] == 0 and detail["nce_degenerate"] == 1
    else:
        center = torch.zeros(1, 256, 12) if info["id"] == 3 else None
        value, detail = pixel_batch_backward(engine, batch, references, rgb, list(range(len(batch))), info["objective"], cfg, center)
    assert np.isfinite(value) and rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert rgb.grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in engine.dino.parameters())


@pytest.mark.parametrize("b", common.BATCHES)
def test_pixel_gradient_is_batch_mean_not_sum(b):
    cfg, engine, batch = common.config(), Engine(), records(b)
    initial = torch.rand(3, 50, 50, generator=torch.Generator().manual_seed(2))
    reference = prepare_references(engine, batch, "uada_original", cfg)
    rgb = initial.clone().requires_grad_(True)
    pixel_batch_backward(engine, batch, reference, rgb, range(b), "uada_original", cfg)
    expected = []
    for record, ref in zip(batch, reference):
        leaf = initial.clone().requires_grad_(True)
        pixel_batch_backward(engine, [record], [ref], leaf, [0], "uada_original", cfg)
        expected.append(leaf.grad)
    torch.testing.assert_close(rgb.grad, torch.stack(expected).mean(0), rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize("b", common.BATCHES)
@pytest.mark.parametrize("criterion", ["cos", "nce"])
def test_exact_cache_matches_full_coupled_batch(b, criterion):
    cfg, engine, batch = common.config(), Engine(), records(b)
    refs = prepare_references(engine, batch, "advencoder_cos", cfg)
    clean = torch.cat([r["clean"] for r in refs])
    originals = [r["clean_rgb"] for r in refs]
    initial = torch.rand(3, 50, 50, generator=torch.Generator().manual_seed(7))
    cached = initial.clone().requires_grad_(True)
    value, detail = cached_generator_backward(engine, batch, clean, originals, cached, list(range(b)), criterion, cfg)
    direct = initial.clone().requires_grad_(True)
    features, aux = [], []
    for record, original in zip(batch, originals):
        pixels, _, rendered = engine.pixels(record, direct, 0, return_rgb=True)
        features.append(engine.direct_dino(pixels))
        aux.append(rgb_terms(original, rendered, cfg["generator"])[0])
    loss = cfg["generator"]["feature_weight"] * batch_feature_loss(clean, torch.cat(features), criterion, cfg["nce_temperature"])[0] + torch.stack(aux).mean()
    loss.backward()
    assert value == pytest.approx(float(loss.detach()), abs=2e-6)
    torch.testing.assert_close(cached.grad, direct.grad, rtol=3e-5, atol=2e-8)
    assert detail["recompute_relative_max_error"] == 0


def test_nce_b1_has_exactly_zero_feature_gradient():
    clean = torch.randn(1, 256, 12)
    attacked = torch.randn_like(clean, requires_grad=True)
    loss, detail = batch_feature_loss(clean, attacked, "nce", .1)
    gradient = torch.autograd.grad(loss, attacked)[0]
    assert float(loss) == 0 and torch.equal(gradient, torch.zeros_like(gradient))
    assert detail["nce_negative_count"] == 0 and detail["nce_degenerate"] == 1


def test_grid_schedule_clocks_and_original_transform():
    cfg = common.config()
    workers = common.worker_schedule()
    assert len(common.METHODS) == 32 and len(workers) == 6
    assert sorted(m for w in workers for m in w) == sorted(common.METHODS)
    assert common.arm_info(workers[0][0])["id"] == 12 and common.arm_info(workers[0][0])["batch"] == 8
    assert common.arm_info(workers[1][0])["id"] == 1 and common.arm_info(workers[1][0])["batch"] == 8
    assert cfg["learning_rate"] == .002 and cfg["schedule_outer_iterations"] == 2000
    assert cfg["generator"]["learning_rate"] == .0002
    assert cfg["eot"] == common.base.config()["eot"]
    schedule = [[f"u{i}"] for i in range(250)]
    for method in common.METHODS:
        b = common.arm_info(method)["batch"]
        assert len(common.batch_uids(schedule, 500, method)) == b
        assert 500 * cfg["inner_loop"] * b == 25000 * b
    with pytest.raises(RuntimeError):
        common.batch_uids(schedule, 501, common.METHODS[0])
    assert common.eot_seeds(2, 3, 8)[0] == cfg["data_seed"]*1000000 + 20000 + 3
    assert len(set(common.eot_seeds(2, 3, 8))) == 8


@pytest.mark.parametrize("candidate", list(common.FAMILIES))
def test_real_train_loop_resume_and_final_receipt_recovery(tmp_path, monkeypatch, candidate):
    cfg = copy.deepcopy(common.config())
    cfg.update(outer_iterations=2, inner_loop=2, checkpoint_outer_iterations=[0, 1, 2], resume_every_outer=1)
    method = f"m{candidate:02d}_{common.FAMILIES[candidate]}_b2"
    batch = records(250)
    bank = {"records": {r["uid"]: r for r in batch}, "candidates": [[r["uid"]] for r in batch]}
    monkeypatch.setattr(train, "config", lambda: cfg)
    monkeypatch.setattr(train, "run_signature", lambda: "cpu-toy")
    monkeypatch.setattr(train, "prepared", lambda: None)
    monkeypatch.setattr(train, "require_gpu", lambda stage: "CPU_TOY")
    monkeypatch.setattr(train, "load_bank", lambda: bank)
    monkeypatch.setattr(train, "patch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train, "load_wrapper", lambda *args: None)
    monkeypatch.setattr(train, "OFTEngine", Engine)
    monkeypatch.setattr(train, "fixed_center", lambda *args: torch.zeros(1, 256, 12))
    monkeypatch.setattr(train, "new_generator", lambda settings, seed, device: new_generator(settings, seed, device, filters=(16, 8, 4, 2)))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.get_rng_state()])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: torch.set_rng_state(states[0]))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    monkeypatch.setattr(train, "ROOT", continuous)
    train.train(method)
    expected = torch.load(continuous / "training" / method / "resume.pt", weights_only=False)
    monkeypatch.setattr(train, "ROOT", resumed)
    real_atomic = train.atomic_torch
    def interrupt(value, path):
        real_atomic(value, path)
        if Path(path).name == "resume.pt" and value["outer"] == 1:
            raise RuntimeError("CPU forced stop after committed outer1")
    monkeypatch.setattr(train, "atomic_torch", interrupt)
    with pytest.raises(RuntimeError, match="forced stop"):
        train.train(method)
    monkeypatch.setattr(train, "atomic_torch", real_atomic)
    train.train(method)
    actual = torch.load(resumed / "training" / method / "resume.pt", weights_only=False)
    assert torch.equal(expected["patch"], actual["patch"])
    assert expected["optimizer"]["state"].keys() == actual["optimizer"]["state"].keys()
    for key in expected["optimizer"]["state"]:
        for moment in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(expected["optimizer"]["state"][key][moment], actual["optimizer"]["state"][key][moment], rtol=0, atol=0)
    monkeypatch.setattr(train, "require_gpu", lambda stage: pytest.fail("Final recovery must not touch GPU"))
    (resumed / "training" / method / "status.json").unlink()
    train.train(method)
    assert json.loads((resumed / "training" / method / "status.json").read_text())["status"] == "COMPLETE"


def test_resource_dependency_and_safe_cli():
    train_script = (ROOT / "slurm/02_train_4090.sbatch").read_text()
    eval_script = (ROOT / "slurm/04_eval_4090.sbatch").read_text()
    assert "--array=0-5%6" in train_script and "--gres=gpu:4090:1" in train_script
    assert "--array=0-9%8" in eval_script and "--gres=gpu:4090:1" in eval_script
    assert "--dependency=afterok:123" in pipeline.command_for("04_eval_4090.sbatch", "123", array="0-9%8")
    assert "--dependency=afterany:12:13" in pipeline.command_for("01_prepare_cpu.sbatch", external=["12", "13"])
    queue = "12|RUNNING|gres/gpu:4090:1|None\n13_[0-5%6]|PENDING|gres/gpu:4090:1|Resources\n14|PENDING|gres/gpu:4090:1|DependencyNeverSatisfied\n"
    assert pipeline.existing_gpu_jobs(queue) == ["12", "13"]
    assert pipeline.ignored_invalid_jobs(queue) == ["14"]
    assert len(analyze.comparisons()) == 80


def test_pairing_and_per_suite_statistics_reject_incomplete_data():
    rows = []
    for suite in common.SUITES:
        for task in range(10):
            for state in range(40, 50):
                for name in common.deployment_conditions():
                    failure = int(name != "clean" and (task + state) % 2 == 0)
                    rows.append({"suite": suite, "task_id": task, "state_index": state, "condition": name,
                                 "success": 1-failure, "failure": failure, "rollout_seed": task*100+state})
    raw = pd.DataFrame(rows)
    summaries, contrasts, tasks = analyze.summarize(raw, draws=20)
    assert len(summaries) == 132 and len(contrasts) == 320 and len(tasks) == 1280
    assert summaries.loc[summaries.method.ne("clean"), "task_equal_casr"].eq(.5).all()
    assert contrasts.casr_difference.eq(0).all()
    assert not summaries.suite.eq("average").any()
    with pytest.raises(RuntimeError):
        analyze.summarize(raw.iloc[:-1], draws=2)
    damaged = raw.copy()
    damaged.loc[0, "rollout_seed"] = -1
    with pytest.raises(RuntimeError):
        analyze.summarize(damaged, draws=2)


def test_submit_is_only_explicit_and_serial_capped_mock(tmp_path, monkeypatch):
    # Test the real orchestrator without ever running squeue or sbatch.
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "preflight", lambda: {"signature": "mock"})
    monkeypatch.setattr(pipeline, "run_signature", lambda: "mock")
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="77|RUNNING|gres/gpu:4090:1|None\n", stderr="")
        assert command[0] == "sbatch"
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline.submit()
    submissions = calls[1:]
    assert len(submissions) == 5
    assert "--dependency=afterany:77" in submissions[0]
    for previous, current in zip(submissions, submissions[1:]):
        previous_id = str(100 + calls.index(previous) + 1)
        assert f"--dependency=afterok:{previous_id}" in current
    assert "--array=0,1,2,3,4,5%6" in submissions[1]
    assert "--array=0,1,2,3,4,5,6,7,8,9%8" in submissions[3]


def test_pending_arrays_avoid_allocating_complete_gpu_phases(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "run_signature", lambda: "mock")
    assert pipeline.pending_arrays() == {"train": list(range(6)), "eval": list(range(10))}
    for method in common.METHODS:
        folder = tmp_path / "training" / method
        folder.mkdir(parents=True)
        common.atomic_json({"signature": "mock", "status": "COMPLETE", "outer": 500}, folder / "status.json")
        (folder / "resume.pt").touch()
        (folder / "outer_0500.png").touch()
    for suite in common.SUITES:
        folder = tmp_path / "evaluation" / suite
        folder.mkdir(parents=True)
        for task in range(10):
            common.atomic_json({"signature": "mock", "status": "COMPLETE", "episodes": 330}, folder / f"task_{task:02d}_STATUS.json")
            (folder / f"task_{task:02d}_rollouts.parquet").touch()
            (folder / f"task_{task:02d}_traces.parquet").touch()
    assert pipeline.pending_arrays() == {"train": [], "eval": []}
    (tmp_path / "evaluation" / common.SUITES[0] / "task_04_STATUS.json").unlink()
    assert pipeline.pending_arrays() == {"train": [], "eval": [4]}

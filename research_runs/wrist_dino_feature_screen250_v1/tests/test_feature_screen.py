"""Focused CPU/toy tests. Never load OFT weights, render, or submit Slurm."""
from __future__ import annotations
import ast
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
import data
import analyze
import evaluate
import feature_objectives as feature
import pipeline
import train
import engine as engine_module
from engine import OFTEngine
from UADA_SOURCE_COPY import build_uada_adamw
from source_transform import RandomPatchTransform

torch.set_num_threads(2)


def test_frozen_contract_budget_resources_and_no_selection():
    cfg = common.config()
    assert common.METHODS == ("uada_original", "dino_key", "dino_center", "advencoder_cos")
    assert cfg["outer_iterations"] * cfg["inner_loop"] == 12500
    assert cfg["schedule_outer_iterations"] == 2000
    assert cfg["center_reference_count"] == 1024
    assert cfg["evaluation_checkpoints"] == [250]
    assert len(common.deployment_conditions()) * 4 * 10 * 2 == 400
    assert all(not common.flags(m)["sampling"] for m in common.METHODS)
    assert all(c["max_concurrent"] == 4 for c in cfg["resources"].values())
    assert cfg["eot"]["fixed_inner_count"] == 0


def test_common_training_schedule_is_ordinary_and_prefix_compatible():
    episodes = [{"episode_id": "a", "eligible_windows": 2}, {"episode_id": "b", "eligible_windows": 3}]
    short = data.candidate_schedule(episodes, 250, 1, 19)
    assert short == data.candidate_schedule(episodes, 2000, 1, 19)[:250]
    assert all(len(row) == 1 for row in short)
    assert {x for row in short for x in row} == {"a:0", "a:1", "b:0", "b:1", "b:2"}


def test_expert_actions_are_normalized_without_modification():
    raw = np.array([[2, -2, .2, .3, .4, .5, -1], [-.2, .2, 0, .3, .4, .5, 1]], np.float32)
    before = raw.copy()
    stats = {"q01": [-1]*6+[0], "q99": [1]*7, "min": [-1]*6+[0], "max": [1]*7, "mask": [True]*6+[False]}
    actual = data.normalized_expert(raw, stats)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(actual[:, 0], [1, -.2], atol=1e-6)
    np.testing.assert_array_equal(actual[:, 6], [1, 0])


def test_native_key_uses_qk_norm_scale_and_real_prefix():
    qkv = torch.randn(1, 261, 24, generator=torch.Generator().manual_seed(8), requires_grad=True)
    attn = SimpleNamespace(num_heads=2, q_norm=lambda q: q*1.3, k_norm=lambda k: k*.7, scale=.4)
    key, saliency = feature.key_and_saliency(qkv, attn)
    q, k, _ = qkv.reshape(1, 261, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
    expected = ((q[:, :, :1]*1.3*attn.scale) @ (k*.7).transpose(-2, -1)).softmax(-1)[:, :, 0, 5:].mean(1)
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(key, (k*.7)[:, :, 5:])
    torch.testing.assert_close(saliency, expected)
    assert key.shape == (1, 2, 256, 4) and not saliency.requires_grad
    changed = qkv.detach().clone()
    changed[:, 1:5, 8:16] *= 100
    _, modified = feature.key_and_saliency(changed, attn)
    # Prefix keys remain in the native softmax, influencing head averaging.
    assert not torch.allclose(saliency, modified)


def test_key_weights_do_not_cancel_and_gradients_remain():
    clean = torch.randn(1, 2, 256, 4, generator=torch.Generator().manual_seed(7))
    adv = (clean + .3*torch.randn(clean.shape, generator=torch.Generator().manual_seed(9))).requires_grad_()
    weights = torch.arange(1, 257).float()[None]; weights /= weights.sum()
    weights.requires_grad_()
    loss = feature.weighted_key_loss(clean, adv, weights)
    expected = F.cosine_similarity((clean*weights[:, None, :, None]).flatten(1),
                                   (adv*weights[:, None, :, None]).flatten(1)).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(adv.grad).all() and adv.grad.norm() > 0 and weights.grad is None
    assert abs(float(loss-feature.flattened_cosine(clean, adv))) > 1e-5


def test_centering_uses_fixed_actual_dense_mean():
    clean = torch.randn(1, 256, 8, generator=torch.Generator().manual_seed(1)) + 10
    adv = (clean + .4*torch.randn(clean.shape, generator=torch.Generator().manual_seed(2))).requires_grad_()
    mean = torch.ones_like(clean)*10
    loss = feature.centered_loss(clean, adv, mean)
    torch.testing.assert_close(loss, feature.flattened_cosine(clean-mean, adv-mean))
    assert float(loss) < float(feature.flattened_cosine(clean, adv))
    loss.backward()
    assert adv.grad.norm() > 0 and not mean.requires_grad
    with pytest.raises(RuntimeError):
        feature.centered_loss(clean, adv, mean[:, :1])


def test_qkv_hook_is_read_only_and_always_removed():
    attention = SimpleNamespace(qkv=torch.nn.Linear(2, 6))
    value = torch.ones(1, 2)
    with feature.capture_qkv(attention) as outputs:
        actual = attention.qkv(value)
    assert outputs[0] is actual and not attention.qkv._forward_hooks
    with pytest.raises(ValueError):
        with feature.capture_qkv(attention):
            raise ValueError("test exception")
    assert not attention.qkv._forward_hooks


def test_key_forward_hooks_executed_penultimate_block_only():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv = torch.nn.Linear(4,12)
            self.attn.num_heads, self.attn.scale = 2, .5
            self.attn.q_norm = self.attn.k_norm = torch.nn.Identity()
        def forward(self,x): return x+self.attn.qkv(x)[...,:4]
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.blocks=torch.nn.ModuleList([Block() for _ in range(4)]); self.num_prefix_tokens=5
        def forward(self,x):
            for block in self.blocks[:-1]: x=block(x)
            return x[:,5:]
    dino = Encoder().eval().requires_grad_(False)
    engine = SimpleNamespace(dino=dino, direct_dino=dino)
    x = torch.randn(1,261,4,requires_grad=True)
    dense,key,weights,layer = feature.key_forward(engine,x)
    assert layer==2 and dense.shape==(1,256,4) and key.shape==(1,2,256,2)
    assert weights.shape==(1,256) and not weights.requires_grad
    key.square().mean().backward()
    assert x.grad.norm()>0 and all(not b.attn.qkv._forward_hooks for b in dino.blocks)


def test_generator_raw_rgb_is_exact_uint8_forward_with_connected_ste(monkeypatch):
    engine = object.__new__(OFTEngine)
    engine.transform=RandomPatchTransform("cpu",resize_patch=False)
    engine.wrapper=SimpleNamespace(processor=SimpleNamespace(image_processor=None),_official_openvla_utils=None,
                                   gen_cfg=None,dtype=torch.float32)
    monkeypatch.setattr(engine_module,"aligned_pixels",lambda front,wrist,*args: torch.cat((front,front,wrist,wrist))[None])
    record={"observation":{"full_image":np.full((224,224,3),128,np.uint8),"wrist_image":np.full((224,224,3),25,np.uint8)}}
    patch=torch.rand(3,50,50,requires_grad=True)
    pixels,footprint,rgb=engine.pixels(record,patch,17,diverse=True,return_rgb=True)
    assert torch.equal(rgb,rgb.detach().mul(255).round().div(255))
    assert torch.equal(pixels[:,:3],torch.full_like(pixels[:,:3],128/255))
    rgb.mean().backward()
    assert torch.isfinite(patch.grad).all() and patch.grad.norm()>0 and footprint.shape==(256,)


def test_fft_highpass_matches_author_radial_math():
    image = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(3), requires_grad=True)
    mask = torch.tensor([[((i-8)**2+(j-8)**2)**.5 >= 3 for j in range(16)] for i in range(16)])
    expected = torch.fft.ifftn(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fftn(image, dim=(-2,-1)), dim=(-2,-1))*mask,
                                               dim=(-2,-1)), dim=(-2,-1)).real
    actual = feature.high_component(image, 3)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert image.grad.norm() > 0
    assert feature.high_component(torch.ones_like(image), 3).abs().max() == 0


def test_author_generator_produces_static_rgb_and_all_parameters_learn():
    cfg = common.config()["generator"]
    generator, z = feature.new_generator(cfg, 42, "cpu", filters=(8, 8, 4, 4))
    rgb = feature.generated_rgb(generator, z)
    assert rgb.shape == (3, 50, 50) and rgb.min() >= 0 and rgb.max() <= 1
    assert not z.requires_grad and rgb.std() > 0
    pixels = F.interpolate(rgb[None], size=(64, 64), mode="bilinear", align_corners=False)
    clean = torch.linspace(.1, .9, 64*64*3).reshape_as(pixels)
    loss, detail = feature.generator_loss(clean.flatten(2).transpose(1, 2), pixels.flatten(2).transpose(1, 2), clean, pixels, cfg)
    expected = 5*detail["feature_cosine"] + detail["rgb_quality_mse"] - detail["rgb_high_frequency_mse"]
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in generator.parameters())
    assert sum(float(p.grad.norm()) for p in generator.parameters()) > 0


def test_pixel_schedule_is_not_zero_at_screen_end():
    pixel = torch.ones(3, 50, 50, requires_grad=True)
    optimizer, scheduler = build_uada_adamw(pixel, .002, 20, 2000, 0)
    for _ in range(250):
        optimizer.zero_grad(); pixel.mean().backward(); optimizer.step(); scheduler.step()
    assert .0018 < optimizer.param_groups[0]["lr"] < .002
    assert scheduler.last_epoch == 250


def test_original_eot_source_and_deployment_lifecycle_unchanged():
    old = ROOT.parent / "wrist_dino_official_long4_2000_v1"
    for name in ("source_transform.py", "preprocessing.py", "model_runtime.py", "readonly_model.py", "deployment.py"):
        assert (ROOT/"code"/name).read_bytes() == (old/"code"/name).read_bytes()
    def function_ast(path, name):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name))
    assert function_ast(ROOT/"code/evaluate.py", "episode") == function_ast(old/"code/evaluate.py", "episode")
    assert RandomPatchTransform("cpu", resize_patch=False).angle == 30


class ToyEngine:
    """Differentiable miniature encoder/action readout, never real OFT."""
    pixel_calls = 0
    fail_on_call = None
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.dino = SimpleNamespace(training=False)
    def base(self, record):
        x = float(record["observation"]["wrist_image"][0,0,0])/255
        return {"pixel_values": torch.full((1, 12, 224, 224), x)}, np.zeros(8)
    def pixels(self, record, patch, seed, diverse=False, return_rgb=False):
        ToyEngine.pixel_calls += 1
        if ToyEngine.fail_on_call == ToyEngine.pixel_calls:
            raise RuntimeError("simulated interruption")
        random.seed(seed); np.random.seed(seed % (2**32-1))
        clean = torch.as_tensor(record["observation"]["wrist_image"]).permute(2,0,1).float()/255
        rgb = clean.clone(); rgb[:, :50, :50] = patch
        inputs, _ = self.base(record)
        pixels = inputs["pixel_values"].clone(); pixels[:, 6:9] = rgb
        result = (pixels, torch.zeros(256))
        return (*result, rgb[None]) if return_rgb else result
    def direct_dino(self, pixels):
        x = pixels[:,6:9].mean((1,2,3))[:,None]
        token = torch.linspace(.01, .9, 256)[None]
        return torch.stack((torch.sin(x*3+token), torch.cos(x*5+token), x*token+1, (x-.5).square()+token), -1)
    def forward(self, inputs, proprio, pixels):
        return pixels[:,6:9].mean().expand(1,8,7), {}, {}
    def validate(self, record, patch):
        return {"status": "CPU_TOY_ONLY", "not_real_OFT": True}


def toy_key(engine, pixels):
    dense = engine.direct_dino(pixels)
    key = dense.reshape(1,256,2,2).permute(0,2,1,3)
    weights = torch.arange(1,257).float()[None]; weights /= weights.sum()
    return dense, key, weights, 22


@pytest.fixture
def toy_setup(monkeypatch):
    settings = copy.deepcopy(common.config())
    settings.update(outer_iterations=2, inner_loop=2, checkpoint_outer_iterations=[0,1,2],
                    evaluation_checkpoints=[2], resume_every_outer=1, center_reference_count=2)
    records = {}
    for i, value in enumerate((25,130,205)):
        uid = f"demo-{i}:0"
        records[uid] = {"uid":uid, "episode_id":f"demo-{i}", "step_index":0, "task_id":i, "task_name":f"task-{i}",
                        "expert_actions":torch.ones(1,8,7)*.2,
                        "observation":{"full_image":np.full((224,224,3),value,np.uint8),
                                       "wrist_image":np.full((224,224,3),value,np.uint8), "state":np.ones(8)}}
    bank = {"candidates":[["demo-0:0"],["demo-1:0"]], "center_reference_uids":["demo-0:0","demo-2:0"], "records":records}
    for module in (train, analyze):
        monkeypatch.setattr(module, "config", lambda: settings)
        monkeypatch.setattr(module, "run_signature", lambda: "CPU_TOY_ONLY")
    monkeypatch.setattr(train, "prepared", lambda: None)
    monkeypatch.setattr(train, "require_gpu", lambda stage: "CPU_TOY_ONLY")
    monkeypatch.setattr(train, "load_bank", lambda: bank)
    monkeypatch.setattr(train, "patch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train, "runtime_config", lambda: {})
    def wrapper(*args):
        module = torch.nn.Linear(1,1); module.requires_grad_(False)
        return SimpleNamespace(vla=module,action_head=None,proprio_projector=None)
    monkeypatch.setattr(train, "load_wrapper", wrapper)
    monkeypatch.setattr(train, "OFTEngine", ToyEngine)
    monkeypatch.setattr(train, "key_forward", toy_key)
    monkeypatch.setattr(train, "new_generator", lambda cfg,seed,device: feature.new_generator(cfg,seed,device,filters=(8,8,4,4)))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.get_rng_state()])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: None)
    ToyEngine.pixel_calls, ToyEngine.fail_on_call = 0, None
    return settings, bank


@pytest.mark.parametrize("method", common.METHODS)
def test_complete_training_and_interrupted_resume_are_equivalent(method, toy_setup, monkeypatch, tmp_path):
    cfg, _ = toy_setup
    full, restarted = tmp_path/"full", tmp_path/"restarted"
    monkeypatch.setattr(train, "ROOT", full)
    train.train(method)
    expected = torch.load(full/"training"/method/"resume.pt", weights_only=False)
    assert expected["outer"] == 2 and expected["rows"][-1]["effective_rgb_updates"] > 0
    monkeypatch.setattr(train, "ROOT", restarted)
    ToyEngine.pixel_calls, ToyEngine.fail_on_call = 0, 3
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.train(method)
    prior = torch.load(restarted/"training"/method/"resume.pt", weights_only=False)
    assert prior["outer"] == 1
    ToyEngine.pixel_calls, ToyEngine.fail_on_call = 0, None
    train.train(method)
    actual = torch.load(restarted/"training"/method/"resume.pt", weights_only=False)
    torch.testing.assert_close(actual["patch"], expected["patch"], rtol=0, atol=0)
    for key, state in expected["optimizer"]["state"].items():
        for moment in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual["optimizer"]["state"][key][moment], state[moment], rtol=0, atol=0)
    if method == "advencoder_cos":
        assert actual["scheduler"] is None and not actual["noise"].requires_grad
        for key, value in expected["generator"].items():
            torch.testing.assert_close(actual["generator"][key], value, rtol=0, atol=0)
    assert (restarted/"training"/method/"outer_0002.png").exists()
    # A repeated completed invocation must not retrain.
    previous_calls = ToyEngine.pixel_calls
    train.train(method)
    assert ToyEngine.pixel_calls == previous_calls
    broken = copy.deepcopy(actual); broken["optimizer"]["state"][next(iter(broken["optimizer"]["state"]))]["step"] = 1
    with pytest.raises(RuntimeError, match="counter"):
        train.validate_resume(broken, method, cfg, "CPU_TOY_ONLY")


def test_all_four_asset_validation_after_toy_training(toy_setup, monkeypatch, tmp_path):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(analyze, "ROOT", tmp_path)
    for method in common.METHODS:
        train.train(method)
    analyze.validate_training()
    receipt = json.loads((tmp_path/"reports/AUTO_VALIDATION.json").read_text())
    assert receipt["status"] == "COMPLETE" and len(receipt["patches"]) == 4
    assert not receipt["model_parameters_updated"] and not receipt["manual_appearance_gate"]


def toy_rollouts():
    rows = []
    for suite in common.SUITES:
        for task in range(10):
            for state in (40,41):
                for condition in common.deployment_conditions():
                    failure = int(condition != "clean" and (condition.startswith("dino_key") or task < 3))
                    rows.append({"suite":suite, "task_id":task, "state_index":state, "condition":condition,
                                 "rollout_seed":1000+state, "success":1-failure, "failure":failure})
    return pd.DataFrame(rows)


def test_per_suite_summary_and_paired_baseline_differences():
    summary, differences, tasks = analyze.summarize(toy_rollouts(), draws=16)
    assert len(summary) == 20 and len(differences) == 12 and len(tasks) == 160
    keys = differences.loc[differences.left.eq("dino_key")]
    np.testing.assert_allclose(keys.casr_difference, .7)
    assert keys.positive_tasks.eq(7).all() and keys.negative_tasks.eq(0).all()
    assert set(summary.suite) == set(common.SUITES)
    assert not differences.contrast_type.eq("training_rounds").any()


def test_complete_mock_statistics_writes_four_suite_results(monkeypatch,tmp_path):
    monkeypatch.setattr(analyze,"ROOT",tmp_path)
    monkeypatch.setattr(analyze,"run_signature",lambda: "CPU_TOY_ONLY")
    bootstrap=analyze.paired_bootstrap
    monkeypatch.setattr(analyze,"paired_bootstrap",lambda pairs,draws=10000,seed=20260907: bootstrap(pairs,draws=8,seed=seed))
    raw=toy_rollouts().assign(signature="CPU_TOY_ONLY",evaluation_training_hooks=False,
                             renderer_vendor="NVIDIA",renderer="RTX4090")
    for suite in common.SUITES:
        for task in range(10):
            folder=tmp_path/"evaluation"/suite; folder.mkdir(parents=True,exist_ok=True)
            rp,tp=folder/f"task_{task:02d}_rollouts.parquet",folder/f"task_{task:02d}_traces.parquet"
            raw.loc[raw.suite.eq(suite)&raw.task_id.eq(task)].to_parquet(rp,index=False)
            pd.DataFrame([{"suite":suite,"task_id":task,"decision_block":0}]).to_parquet(tp,index=False)
            common.atomic_json({"status":"COMPLETE","signature":"CPU_TOY_ONLY",
                                "rollouts_sha256":common.sha256_file(rp),"traces_sha256":common.sha256_file(tp)},
                               folder/f"task_{task:02d}_STATUS.json")
    analyze.analyze()
    status=json.loads((tmp_path/"results/STATUS.json").read_text())
    assert status["status"]=="COMPLETE" and status["episodes"]==400
    assert pd.read_csv(tmp_path/"results/SCREEN250_SUMMARY.csv").shape[0]==20
    assert pd.read_csv(tmp_path/"results/SCREEN250_CONTRASTS.csv").shape[0]==12
    report=(tmp_path/"results/RESULTS.md").read_text()
    assert "本轮无补丁清洁对照" in report and "advencoder_cos" in report


@pytest.mark.parametrize("mutation", ["duplicate", "seed", "state"])
def test_summary_rejects_misaligned_grids(mutation):
    frame = toy_rollouts()
    if mutation == "duplicate":
        frame = pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True)
    elif mutation == "seed":
        frame.loc[1,"rollout_seed"] += 1
    else:
        frame.loc[frame.state_index.eq(41),"state_index"] = 42
    with pytest.raises(RuntimeError):
        analyze.summarize(frame, draws=4)


def test_manual_pipeline_stage_order_caps_and_dependencies():
    assert [name for name,_ in pipeline.STAGES] == ["prepare","train","validate","eval","stats"]
    command = pipeline.command_for("02_train_4090.sbatch", "123")
    assert "--dependency=afterok:123" in command and "--kill-on-invalid-dep=yes" in command
    command = pipeline.command_for("01_prepare_cpu.sbatch", external=["100","101"])
    assert "--dependency=afterany:100:101" in command
    assert pipeline.existing_gpu_jobs("100_[0-3]|PENDING|gres/gpu:4090:1|Resources\n101|RUNNING|gpu:1|node\n102|RUNNING|N/A|node") == ["100","101"]
    with pytest.raises(RuntimeError, match="invalid dependency"):
        pipeline.existing_gpu_jobs("100|PENDING|gpu:1|DependencyNeverSatisfied")
    assert "--array=0-3%4" in (ROOT/"slurm/02_train_4090.sbatch").read_text()
    assert "--array=0-9%4" in (ROOT/"slurm/04_eval_4090.sbatch").read_text()
    assert all("wrist_dino_official_long4_2000_v1" not in p.read_text() for p in (ROOT/"slurm").iterdir())


def test_manual_submit_would_chain_five_jobs_without_gpu_overlap(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,"ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"preflight",lambda: {"signature":"CPU_TOY_ONLY"})
    calls=[]
    def command(args,**kwargs):
        calls.append(args)
        if args[0]=="squeue": return SimpleNamespace(returncode=0,stdout="99|RUNNING|gres/gpu:4090:1|node",stderr="")
        assert args[0]=="sbatch"
        return SimpleNamespace(returncode=0,stdout=str(100+len(calls)),stderr="")
    monkeypatch.setattr(pipeline.subprocess,"run",command)
    pipeline.submit()
    assert len(calls)==6 and "--dependency=afterany:99" in calls[1]
    for prior,current in zip(calls[1:],calls[2:]):
        previous_job=str(100+calls.index(prior)+1)
        assert "--dependency=afterok:"+previous_job in current
    receipt=json.loads((tmp_path/"reports/SUBMISSION_RECEIPT.json").read_text())
    assert receipt["serial_phase_cap"]==receipt["training_gpu_cap"]==receipt["evaluation_gpu_cap"]==4


def test_evaluator_pairs_two_states_five_conditions_and_skips_complete(monkeypatch,tmp_path):
    suite,task_id=common.SUITES[0],3
    settings=copy.deepcopy(common.config())
    seeds=pd.DataFrame([{"suite":suite,"task_id":task_id,"state_index":state,"condition":"clean","rollout_seed":1000+state}
                        for state in (40,41,42)])
    seed_path=tmp_path/"seeds.parquet"; seeds.to_parquet(seed_path,index=False)
    settings["evaluation"]["seed_sources"][suite]=str(seed_path)
    conditions=common.deployment_conditions()
    monkeypatch.setattr(evaluate,"ROOT",tmp_path)
    monkeypatch.setattr(evaluate,"config",lambda: settings)
    monkeypatch.setattr(evaluate,"run_signature",lambda: "CPU_TOY_ONLY")
    monkeypatch.setattr(evaluate,"prepared",lambda: None)
    monkeypatch.setattr(evaluate,"require_gpu",lambda stage: "CPU_TOY_ONLY")
    monkeypatch.setattr(evaluate,"runtime_config",lambda suite: {})
    monkeypatch.setattr(evaluate,"patches",lambda: ({c:None if c=="clean" else np.zeros((50,50,3),np.uint8) for c in conditions},
                                                   {c:"mock_png" for c in conditions}))
    monkeypatch.setattr(evaluate,"prepare_graphics_backend",lambda: None)
    monkeypatch.setattr(evaluate,"load_wrapper",lambda *args: object())
    monkeypatch.setattr(evaluate,"make_env",lambda *args,**kwargs: (object(),SimpleNamespace(language="toy"),list(range(50))))
    monkeypatch.setattr(evaluate,"renderer_metadata",lambda env: ("mock RTX4090","NVIDIA"))
    monkeypatch.setattr(evaluate,"close_environment",lambda env: None)
    calls=[]
    def episode(env,wrapper,task,initial,seed,rgb,runtime,cfg):
        calls.append((initial,seed,rgb is None))
        return {"success":int(rgb is None),"failure":int(rgb is not None),"executed_steps":8,"decision_blocks":1}, [{"decision_block":0}]
    monkeypatch.setattr(evaluate,"episode",episode)
    evaluate.run_suite(task_id,suite)
    frame=pd.read_parquet(tmp_path/"evaluation"/suite/"task_03_rollouts.parquet")
    assert len(frame)==len(calls)==10 and set(frame.state_index)=={40,41}
    assert frame.groupby("state_index").rollout_seed.nunique().eq(1).all()
    assert frame.groupby("state_index").condition.nunique().eq(5).all()
    assert not frame.evaluation_training_hooks.any()
    evaluate.run_suite(task_id,suite)
    assert len(calls)==10


def test_record_cache_uses_only_official_expert_split(monkeypatch, tmp_path):
    class Tensor:
        def __init__(self,value): self.value=np.asarray(value)
        @property
        def shape(self): return self.value.shape
        def numpy(self): return self.value.item() if self.value.ndim==0 else self.value
        def __getitem__(self,index): return Tensor(self.value[index])
    raw = np.tile(np.arange(7,dtype=np.float32)[None]/20,(8,1))
    episodes = [{"tfds_id":Tensor(f"demo-{i}".encode()), "episode_metadata":{"file_path":Tensor(f"expert-{i}.hdf5".encode())},
                 "steps":{"action":Tensor(raw),"language_instruction":Tensor([f"task {i%10:02d}".encode()]*8),
                          "observation":{"state":Tensor(np.ones((8,8))),"image":Tensor([b"jpeg"]*8),"wrist_image":Tensor([b"jpeg"]*8)}}} for i in range(379)]
    image = np.zeros((224,224,3),np.uint8)
    monkeypatch.setattr(data,"ROOT",tmp_path)
    monkeypatch.setattr(data,"run_signature",lambda: "CPU_TOY_ONLY")
    monkeypatch.setattr(data,"preflight",lambda: {"status":"CPU_TOY_ONLY","signature":"CPU_TOY_ONLY"})
    monkeypatch.setattr(data,"dataset_files",lambda: [])
    monkeypatch.setattr(data,"episode_iterator",lambda: iter(episodes))
    monkeypatch.setattr(data,"resize_policy_jpeg",lambda encoded: image)
    data.prepare()
    bank = torch.load(tmp_path/"data/CANDIDATE_BANK.pt",weights_only=False)
    assert len(bank["candidates"]) == 250 and len(bank["center_reference_uids"]) == 1024
    assert all(len(row)==1 for row in bank["candidates"])
    for record in bank["records"].values():
        assert record["source_split"]=="train" and "clean_action" not in record
        torch.testing.assert_close(record["expert_actions"],torch.from_numpy(data.normalized_expert(raw,bank["statistics"])).unsqueeze(0))

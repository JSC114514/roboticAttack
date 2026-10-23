"""CPU-only contracts and interrupted-workflow simulations; Slurm is always mocked."""
from __future__ import annotations

import copy
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
import dino_action_counter_common as common
import run_exploring_oft_dino_action_counter_cross_suite as runner
import analyze_exploring_oft_dino_action_counter_cross_suite as analyzer
import white_patch.UADA_OFT_DINO_ACTION_COUNTER as trainer


@pytest.fixture(scope="module")
def frozen_cfg():
    return common.config()


def test_fresh_budget_schedule_common_records_and_external_500(frozen_cfg):
    cfg = frozen_cfg
    assert cfg["base_outer_iterations"] == 0 and cfg["outer_iterations"] == 500
    assert cfg["scheduler_total_outer"] == 2000 and cfg["checkpoint_outer_iterations"] == [0, 250, 500]
    assert cfg["refresh_after_outer"] == [] and not set(cfg) & {"prefix_resume", "prefix_patch", "calibration"}
    clean, (replay, digest) = common.source_records(cfg), common.replay_records(cfg)
    schedule = common.schedule_records(cfg)
    assert len(clean) == 300 and len(replay) == 258 and len(schedule) == 500
    assert digest == common.REPLAY_SHA
    for outer in (1, 250, 251, 252, 254, 500):
        draws = [common.draw_record(clean, replay, m, outer, cfg) for m in common.METHODS]
        assert all(d[0] is draws[0][0] and d[1:] == draws[0][1:] for d in draws)
        row = schedule[outer-1]
        assert draws[0][1:] == (row["origin"], row["record_index"], row["base_record_index"])
    baselines = common.external_baselines(cfg)
    assert set(baselines) == set(runner.REFERENCES)
    assert all(v["patch"].shape == (3, 50, 50) for v in baselines.values())


@pytest.mark.parametrize("field,value,match", [
    ("outer_iterations", 1000, "budget"), ("scheduler_total_outer", 500, "schedule"),
    ("counter_epsilon", 8/255, "Counter"), ("tv_scale", .3, "appearance"),
    ("refresh_after_outer", [250], "schedule"), ("seed", 1, "seed"),
    ("prefix_resume", "old.pt", "warm start"),
])
def test_config_rejects_scientific_drift(frozen_cfg, tmp_path, field, value, match):
    cfg = copy.deepcopy(frozen_cfg)
    cfg[field] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(Exception, match=match):
        common.config(path)


def test_schedule_is_cached_and_record_metadata_drift_fails(frozen_cfg, monkeypatch):
    cfg = frozen_cfg
    rows = common.schedule_records(cfg)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Frozen500 schedule reread"))
    assert common.schedule_records(cfg) is rows
    clean = [{"task_id": 0, "state_index": 10} for _ in range(300)]
    replay = [{"task_id": 0, "state_index": 10, "phase_proxy": "early"} for _ in range(258)]
    with pytest.raises(Exception, match="pairing"):
        common.draw_record(clean, replay, common.METHODS[0], 1, cfg)
    with pytest.raises(Exception, match="method/outer"):
        common.draw_record(clean, replay, common.METHODS[0], 501, cfg)


def test_frozen_asset_hash_checked_before_reusing_clean(frozen_cfg, tmp_path, monkeypatch):
    cfg = copy.deepcopy(frozen_cfg)
    init_file = tmp_path / "task.pruned_init"
    init_file.write_text("frozen init order")
    cfg["frozen_inputs"] = {}
    cfg["frozen_eval_assets"] = {str(init_file): common.sha256_file(init_file)}
    init_file.write_text("changed init order")
    monkeypatch.setattr(trainer, "source_manifest", lambda *a: pytest.fail("Changed asset must fail first"))
    with pytest.raises(Exception, match="Frozen input missing/changed"):
        common.source_preflight(cfg)


def test_target_metadata_and_alias_hash_contract(frozen_cfg, tmp_path):
    assert len(frozen_cfg["frozen_eval_assets"]) == 168
    paths = list(frozen_cfg["frozen_eval_assets"])
    assert sum("dataset_statistics.json" in p for p in paths) == 4
    assert sum("model.safetensors.index.json" in p for p in paths) == 4
    assert any(".pruned_init" in p for p in paths) and any(".bddl" in p for p in paths)
    value = tmp_path / "dataset_statistics.json"
    value.write_text('{"q01":0}')
    cfg = {"frozen_inputs": {}, "frozen_eval_assets": {str(value): common.sha256_file(value)}}
    common.frozen_digest(cfg, value)
    value.write_text('{"q01":1}')
    with pytest.raises(Exception, match="changed"):
        common.frozen_digest(cfg, value)
    bank = Path(frozen_cfg["training_bank"])
    assert common.frozen_digest(frozen_cfg, bank.resolve()) == common.BANK_SHA


def test_normal_eval_eight_condition_grid_no_stale_fields(frozen_cfg):
    cfg = runner.load_config(frozen_cfg)
    assert tuple(cfg["conditions"]) == runner.CONDITIONS and len(runner.CONDITIONS) == 8
    assert not set(cfg) & runner.STALE_KEYS and cfg["eval_hooks"] is False and cfg["eval_eta"] is False
    assert len(cfg["suites"]) * 10 * 10 * 8 == 3200 and sum(runner.DENOMINATORS.values()) * 8 == 3056
    rows, traces = runner.reference_frames(cfg, frozen_cfg)
    assert rows.empty and traces.empty
    train = (ROOT / "slurm/exploring_oft_dino/108_counter500_train_4090.sbatch").read_text()
    test = (ROOT / "slurm/exploring_oft_dino/110_counter500_eval_3090.sbatch").read_text()
    assert "#SBATCH --array=0-5%5" in train and "#SBATCH --array=0-9%8" in test
    assert "TRAIN_ORDER=(" + " ".join(common.TRAIN_ORDER) + ")" in train
    for number, stage in ((107, "prepare"), (109, "validate"), (111, "stats")):
        text = next((ROOT / "slurm/exploring_oft_dino").glob(f"{number}_counter500_*.sbatch")).read_text()
        assert "#SBATCH --partition=cpu" in text and "#SBATCH --gres" not in text
    source = (ROOT / "scripts/run_exploring_oft_dino_action_counter_cross_suite.py").read_text()
    assert "register_forward_hook" not in source and "action_counter_eta(" not in source


def submission_cfg(frozen_cfg, tmp_path):
    return {**frozen_cfg, "output_root": str(tmp_path / "run")}


def test_submit_afterok_chain_receipts_and_concurrency(frozen_cfg, tmp_path, monkeypatch):
    cfg = submission_cfg(frozen_cfg, tmp_path)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        n = sum(c[0] == "sbatch" for c in calls)
        return SimpleNamespace(returncode=0, stdout="" if command[0] == "squeue" else f"{1000+n};cluster\n", stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    receipt = common.submit_chain(cfg)
    assert [j["stage"] for j in receipt["jobs"]] == ["prepare", "train", "validate", "eval", "stats"]
    submitted = [c for c in calls if c[0] == "sbatch"]
    assert "--array=0-5%5" in submitted[1] and "--array=0-9%8" in submitted[3]
    assert all("--kill-on-invalid-dep=yes" in c for c in submitted)
    for index in range(1, 5):
        assert f"--dependency=afterok:{1000+index}" in submitted[index]
    assert json.loads((Path(cfg["output_root"]) / "reports/SUBMISSION_RECEIPT.json").read_text())["status"] == "SUBMITTED"


@pytest.mark.parametrize("failed_stage", ["prepare", "train", "validate", "eval", "stats"])
def test_partial_submission_failure_never_submits_downstream(frozen_cfg, tmp_path, monkeypatch, failed_stage):
    cfg = submission_cfg(frozen_cfg, tmp_path)
    calls = []
    stages = ["prepare", "train", "validate", "eval", "stats"]
    stop = stages.index(failed_stage) + 1
    def fake(command, **kwargs):
        calls.append(command)
        count = sum(c[0] == "sbatch" for c in calls)
        bad = command[0] == "sbatch" and count == stop
        return SimpleNamespace(returncode=int(bad), stdout="" if bad or command[0] == "squeue" else str(9000+count),
                               stderr="mock submit failed" if bad else "")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="no downstream"):
        common.submit_chain(cfg)
    assert sum(c[0] == "sbatch" for c in calls) == stop
    receipt = json.loads((Path(cfg["output_root"]) / "reports/SUBMISSION_RECEIPT.json").read_text())
    assert receipt["status"] == "SUBMISSION_FAILED" and receipt["failed_stage"] == failed_stage
    assert len(receipt["jobs"]) == stop-1


@pytest.mark.parametrize("code,stdout", [(1, ""), (0, "123 GPU queued")])
def test_duplicate_or_unknown_queue_aborts(frozen_cfg, tmp_path, monkeypatch, code, stdout):
    cfg = submission_cfg(frozen_cfg, tmp_path)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=code, stdout=stdout, stderr="scheduler unavailable")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(Exception):
        common.submit_chain(cfg)
    assert len(calls) == 1 and calls[0][0] == "squeue"


def test_completed_methods_map_to_priority_array_indices(frozen_cfg, tmp_path, monkeypatch):
    cfg = submission_cfg(frozen_cfg, tmp_path)
    for method in ("dino_original", "none_original"):
        path = Path(cfg["output_root"]) / "training" / method / "status.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"status":"COMPLETE"}')
    checked = []
    monkeypatch.setattr(common, "validate_training", lambda cfg, methods, **k: checked.extend(methods))
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="" if command[0] == "squeue" else str(1000+len(calls)), stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    common.submit_chain(cfg)
    assert set(checked) == {"dino_original", "none_original"}
    train = next(c for c in calls if c[-1].endswith("108_counter500_train_4090.sbatch"))
    assert "--array=0,1,3,5%5" in train


def eval_fixture(frozen_cfg, tmp_path, monkeypatch, task=3):
    cfg = runner.load_config(frozen_cfg)
    cfg["output_root"] = str(tmp_path / "eval")
    train = {**frozen_cfg, "output_root": str(tmp_path)}
    hashes = {m: {"pt": f"{i+1:064x}", "png": "f"*64} for i, m in enumerate(runner.CONDITIONS)}
    asset = tmp_path / "actual_modeling.py"
    asset.write_text("loaded runtime implementation")
    source = {str(asset): runner.sha256_file(asset)}
    runtime = {"source": source, "config_signature": runner.digest_json(cfg),
               "evaluation_hooks": False, "evaluation_eta": False}
    monkeypatch.setattr(runner, "config", lambda: train)
    monkeypatch.setattr(runner, "load_config", lambda *a: cfg)
    monkeypatch.setattr(runner, "source_preflight", lambda *a: {})
    monkeypatch.setattr(runner, "require_preparation", lambda *a: {"signature": "prepared"})
    monkeypatch.setattr(runner, "patches", lambda *a: ({}, hashes, {}))
    monkeypatch.setattr(runner, "reference_frames", lambda *a, **k: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(runner, "frozen_clean", lambda base, suite: pd.DataFrame([
        {"task_id": task, "state_index": s, "success": 1, "rollout_seed": 1000+s} for s in cfg["state_indices"]]))
    (tmp_path / "AUTO_VALIDATION.json").write_text(json.dumps({"status": "COMPLETE",
        "config_sha256": runner.sha256_file(common.CONFIG), "preparation_signature": "prepared", "evaluation_hooks": False}))
    return cfg, train, hashes, asset, runtime


def fake_frames(cfg, suite, task, hashes, runtime):
    rows, traces = [], []
    for state in cfg["state_indices"]:
        for method in runner.CONDITIONS:
            meta = {"experiment_id": runner.EXPERIMENT_ID, "training_suite": "libero_object",
                "suite": suite, "task_id": task, "state_index": state, "condition": method,
                "checkpoint": cfg["checkpoints"][suite], "unnorm_key": cfg["unnorm_keys"][suite],
                "frozen_clean_success": 1, "config_signature": runner.digest_json(cfg),
                "runtime_source_digest": runner.digest_json(runtime["source"]), "reused": False}
            rows.append({**meta, "success": 0, "failure": 1, "condition_order": runner._order(cfg,suite,task,state).index(method)+1,
                "rollout_seed": 1000+state, "patch_sha256": hashes[method]["pt"], "renderer_vendor": "NVIDIA Corporation",
                "decision_blocks": 1, "elapsed_s": 0.})
            traces.append({**meta, "decision_block": 0, "step": 0})
    return pd.DataFrame(rows), pd.DataFrame(traces)


def test_four_suites_complete_shards_skip_and_runtime_drift_fails(frozen_cfg, tmp_path, monkeypatch):
    cfg, train, hashes, asset, runtime = eval_fixture(frozen_cfg, tmp_path, monkeypatch)
    calls = []
    def fake_run(index, expanded, **kwargs):
        calls.append(index)
        suite, task = cfg["suites"][index // 10], index % 10
        rows, traces = fake_frames(cfg, suite, task, hashes, runtime)
        folder = Path(cfg["output_root"]) / suite
        folder.mkdir(parents=True)
        rows.to_parquet(folder / f"task_{task:02d}_rollouts.parquet", index=False)
        traces.to_parquet(folder / f"task_{task:02d}_traces.parquet", index=False)
        runner.atomic_json(runtime, Path(cfg["output_root"]) / f"task_{task:02d}_{suite}_RUNTIME.json")
    monkeypatch.setattr(runner, "run_shard", fake_run)
    runner.run_checked(3)
    runner.run_checked(3)
    assert calls == [3,13,23,33]
    asset.write_text("changed HF runtime cache implementation")
    with pytest.raises(Exception, match="Actual loaded evaluation runtime source changed"):
        runner.run_checked(3)


@pytest.mark.parametrize("field,value,match", [
    ("rollout_seed", -1, "seed"), ("frozen_clean_success", 0, "eligibility"),
    ("patch_sha256", "wrong", "Patch bytes"), ("checkpoint", "wrong", "metadata"),
    ("elapsed_s", float("inf"), "Nonfinite"),
])
def test_eval_metadata_failure_is_closed(frozen_cfg, tmp_path, monkeypatch, field, value, match):
    cfg, _, hashes, _, runtime = eval_fixture(frozen_cfg, tmp_path, monkeypatch)
    rows, traces = fake_frames(cfg, "libero_object", 3, hashes, runtime)
    rows.loc[0, field] = value
    with pytest.raises(Exception, match=match):
        runner.validate_new_shard(rows, traces, "libero_object", 3, cfg, hashes)
    rows, traces = fake_frames(cfg, "libero_object", 3, hashes, runtime)
    with pytest.raises(Exception, match="Duplicate"):
        runner.validate_metadata(pd.concat([rows,rows.iloc[:1]]), traces, "libero_object",3,cfg,hashes)
    changed = {**cfg, "patch_xy": [0,0]}
    with pytest.raises(Exception, match="configuration signature"):
        runner.validate_new_shard(rows, traces, "libero_object",3,changed,hashes)


def test_joint_checkpoint_recovers_fault_between_parquet_writes(frozen_cfg, tmp_path, monkeypatch):
    cfg, train, hashes, _, runtime = eval_fixture(frozen_cfg, tmp_path, monkeypatch)
    loaded = {m: (torch.zeros(3,50,50), h["pt"]) for m,h in hashes.items()}
    monkeypatch.setattr(trainer, "source_manifest", lambda *a: runtime["source"])
    monkeypatch.setattr(runner, "load_wrapper", lambda *a: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(runner, "make_env", lambda *a: (SimpleNamespace(close=lambda: None), SimpleNamespace(language="task"), list(range(50))))
    monkeypatch.setattr(runner, "renderer_metadata", lambda *a: ("EGL", "NVIDIA Corporation"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    episode_calls = []
    def episode(env, wrapper, task, state, seed, method, patch, digest, base, renderer, vendor, order):
        episode_calls.append((state,method))
        return {"condition":method,"success":0,"failure":1,"condition_order":order,
            "rollout_seed":seed,"patch_sha256":digest,"renderer_vendor":vendor,"decision_blocks":1,"elapsed_s":0.}, [
            {"condition":method,"decision_block":0,"step":0}]
    monkeypatch.setattr(runner, "run_episode", episode)
    original = runner.atomic_parquet
    failed = False
    def fault(frame,path):
        nonlocal failed
        if str(path).endswith("traces.partial.parquet") and not failed:
            failed = True
            raise OSError("fault between paired parquet writes")
        return original(frame,path)
    monkeypatch.setattr(runner, "atomic_parquet", fault)
    loader = lambda *a: (loaded,hashes,{})
    with pytest.raises(OSError,match="fault between"):
        runner.run_shard(3,cfg,patch_loader=loader)
    assert len(episode_calls) == 8
    runner.run_shard(3,cfg,patch_loader=loader)
    assert len(episode_calls) == 80 and len(set(episode_calls)) == 80
    folder = Path(cfg["output_root"]) / "libero_object"
    rows = pd.read_parquet(folder / "task_03_rollouts.parquet")
    traces = pd.read_parquet(folder / "task_03_traces.parquet")
    runner.validate_new_shard(rows,traces,"libero_object",3,cfg,hashes)
    # A fault between final writes also uses the same full joint transaction.
    (folder / "task_03_traces.parquet").unlink()
    runner.run_shard(3,cfg,patch_loader=loader)
    assert len(episode_calls) == 80 and (folder / "task_03_traces.parquet").exists()


def test_paired_bootstrap_task_equal_and_planned_interaction():
    rows=[]
    for suite in runner.DENOMINATORS:
        for task in range(10):
            for state in ([40] if task == 0 else range(40,50)):
                for method in runner.CONDITIONS:
                    rows.append({"suite":suite,"task_id":task,"state_index":state,"condition":method,
                                 "casr":float(method=="dino_clear")})
    frame=pd.DataFrame(rows)
    summary,contrasts=analyzer.paired_statistics(frame,draws=100,seed=20260907)
    assert set(summary.suite_or_endpoint) == set(runner.DENOMINATORS)|{"spatial_goal_macro","four_suite_macro"}
    match=contrasts.loc[contrasts.contrast.eq("appearance_x_direction_interaction")]
    assert np.allclose(match.difference_left_minus_right,1) and np.allclose(match.ci95_low,1) and np.allclose(match.ci95_high,1)
    frame["casr"]=(frame.task_id.eq(0)&frame.condition.eq("dino_clear")).astype(float)
    point,_,_=analyzer.hierarchical_samples(frame.loc[frame.suite.eq("libero_object")],100,np.random.default_rng(42))
    assert point[runner.CONDITIONS.index("dino_clear")] == pytest.approx(.1)
    assert frame.loc[frame.condition.eq("dino_clear")].casr.mean() != pytest.approx(.1)
    missing=frame.loc[~(frame.task_id.eq(0)&frame.condition.eq("none_clear"))]
    with pytest.raises(Exception,match="Unpaired"):
        analyzer.paired_statistics(missing,100,42)


def test_1000_results_cannot_be_relabelled_500(frozen_cfg):
    cfg = copy.deepcopy(frozen_cfg)
    cfg["legacy_same500_reuse"] = {"diverse_reference_500": {"checkpoint_outer":1000,
        "patch_sha256":cfg["external_baselines"]["diverse_reference_500"]["pt_sha256"]}}
    with pytest.raises(Exception,match="same 500"):
        runner.reference_frames(runner.load_config(cfg),cfg)


def test_mocked_complete_grid_statistics_and_raw_failures(frozen_cfg,tmp_path,monkeypatch):
    cfg = runner.load_config(frozen_cfg)
    cfg["output_root"] = str(tmp_path / "eval")
    train = {**frozen_cfg,"output_root":str(tmp_path)}
    hashes = {m:{"pt":f"{i+1:064x}","png":"f"*64} for i,m in enumerate(runner.CONDITIONS)}
    asset = tmp_path / "actual_modeling.py"
    asset.write_text("loaded class implementation")
    runtime = {"source":{str(asset):runner.sha256_file(asset)},"config_signature":runner.digest_json(cfg),
        "evaluation_hooks":False,"evaluation_eta":False}
    output=Path(cfg["output_root"])
    output.mkdir()
    runner.atomic_json(cfg,output/"EVAL_CONFIG.json")
    def clean(base,suite):
        denied=100-runner.DENOMINATORS[suite]
        return pd.DataFrame([{"task_id":task,"state_index":state,"success":int((state-40)*10+task>=denied),
            "rollout_seed":1000+state} for task in range(10) for state in cfg["state_indices"]])
    monkeypatch.setattr(runner,"frozen_clean",clean)
    frames=[]
    for task in range(10):
        receipt={"status":"COMPLETE","task_id":task,"config_signature":runner.digest_json(cfg),"patches":hashes,"suites":{}}
        for suite in cfg["suites"]:
            rows,traces=fake_frames(cfg,suite,task,hashes,runtime)
            qualification=clean(cfg,suite).loc[lambda f:f.task_id.eq(task)].set_index("state_index").success
            for frame in (rows,traces):
                frame["frozen_clean_success"]=frame.state_index.map(qualification)
                frame["source_experiment_id"]=runner.EXPERIMENT_ID
                frame["source_condition"]=frame.condition
            folder=output/suite
            folder.mkdir(exist_ok=True)
            rp,tp=[folder/f"task_{task:02d}_{kind}.parquet" for kind in ("rollouts","traces")]
            rows.to_parquet(rp,index=False)
            traces.to_parquet(tp,index=False)
            receipt["suites"][suite]={"rollouts_sha256":runner.sha256_file(rp),"traces_sha256":runner.sha256_file(tp)}
            runner.atomic_json(runtime,output/f"task_{task:02d}_{suite}_RUNTIME.json")
            frames.append(rows)
        runner.atomic_json(receipt,output/f"task_{task:02d}_STATUS.json")
    raw=pd.concat(frames,ignore_index=True)
    assert len(analyzer.validate_raw(raw,cfg))==3056
    bad=pd.concat([raw.iloc[:-1],raw.iloc[:1]],ignore_index=True)
    with pytest.raises(Exception,match="duplicate"):
        analyzer.validate_raw(bad,cfg)
    bad=raw.copy()
    bad.loc[0,"elapsed_s"]=float("inf")
    with pytest.raises(Exception,match="finite"):
        analyzer.validate_raw(bad,cfg)
    for method in common.METHODS:
        folder=tmp_path/"training"/method
        folder.mkdir(parents=True)
        runner.atomic_json({"elapsed_s":1.,"costs":{"outer_forward_calls":25000}},folder/"status.json")
        pd.DataFrame({"outer_iteration":range(1,501),"tv":np.full(500,.4),"hf_energy":np.full(500,.1),
                      "zero_update_outer":np.zeros(500,dtype=bool)}).to_parquet(folder/"training.parquet",index=False)
    monkeypatch.setattr(analyzer,"config",lambda:train)
    monkeypatch.setattr(analyzer,"load_config",lambda:cfg)
    monkeypatch.setattr(analyzer,"require_preparation",lambda *a:{"signature":"prepared"})
    monkeypatch.setattr(analyzer,"patches",lambda *a:({},hashes,{}))
    monkeypatch.setattr(analyzer,"external_baselines",lambda *a:{m:{"recorded_training_elapsed_s":1.,"pt":hashes[m]["pt"]} for m in runner.REFERENCES})
    report=analyzer.analyze()
    assert report["status"]=="COMPLETE" and report["attack_episodes"]==3200 and report["eligible_rows"]==3056
    assert report["bootstrap_seed"]==20260907 and report["bootstrap_draws"]==10000
    assert report["new_attack_episodes"]==3200 and report["stop_after_outer"]==500
    assert (output/"DINO_ACTION_COUNTER500_XSUITE_CONTRASTS.csv").exists()
    with pytest.raises(Exception,match="frozen"):
        analyzer.analyze(draws=100)

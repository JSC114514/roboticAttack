"""CPU-only workflow, paired-checkpoint identity and interrupted state transactions."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"VLAAttacker"),str(ROOT/"scripts")]
import dino_sampling_clear_common as common
import run_exploring_oft_dino_sampling_clear_cross_suite as runner
import analyze_exploring_oft_dino_sampling_clear_cross_suite as analyzer
import white_patch.UADA_OFT_DINO_SAMPLING_CLEAR as trainer
from white_patch.oft_dino_sampling_clear_objectives import apply_score_weights


@pytest.fixture(scope="module")
def cfg():
    return common.config()


@pytest.fixture(scope="module")
def banks(cfg):
    clean = common.source_records(cfg)
    replay,_ = common.replay_records(cfg)
    return clean,[dict(r,dino_weight=1.,shuffled_weight=1.) for r in replay]


def test_fresh_live_2000_contract_and_historical_matched_refs(cfg,banks):
    assert cfg["outer_iterations"] == cfg["scheduler_total_outer"] == 2000
    assert cfg["inner_loop"] == 50 and cfg["gradient_clip"] is None
    assert cfg["refresh_after_outer"] == [250,500,1000,1500]
    assert "sampling_schedule" not in cfg and "prefix_patch" not in cfg
    clean,replay = banks
    for outer in (1,250,251,252,254,500,1000,1500,2000):
        for method in common.METHODS:
            record,origin,index,base = common.draw_record(clean,replay,method,outer,cfg)
            assert origin == ("clean" if outer <= 250 or outer%2 else "replay")
            assert int(record["task_id"]) == int(clean[base]["task_id"])
            if origin == "replay":
                assert record["phase_proxy"] == ("early","middle","late")[((outer-250)//2-1)%3]
    refs = common.external_baselines(cfg)
    assert set(refs) == {f"{m}_{cp}" for cp in common.CHECKPOINTS for m in runner.REFERENCES}
    assert {item["checkpoint_outer"] for item in refs.values()} == {1000,2000}
    assert common.data_identity(cfg).keys() == {"clean_bank","window_bank","config"}


@pytest.mark.parametrize("key,value,match",[
    ("outer_iterations",1500,"budget"),("scheduler_total_outer",500,"schedule"),
    ("refresh_after_outer",[500,1000,1500],"schedule"),("seed",1,"seed"),
    ("tv_threshold",.4,"appearance"),("hf_epsilon",32/255,"appearance"),
    ("gaussian_sigma",1.,"appearance"),("tv_quantization_margin",0.,"appearance"),
    ("gradient_clip",.001,"optimizer"),("prefix_resume","old.pt","warm start"),
    ("sampling_schedule","old.parquet","frozen schedule"),
])
def test_scientific_drift_fails(cfg,tmp_path,key,value,match):
    changed = copy.deepcopy(cfg);changed[key] = value
    path = tmp_path/"changed.json";path.write_text(json.dumps(changed))
    with pytest.raises(Exception,match=match):
        common.config(path)


def test_live_sampler_uses_own_shuffled_weight_field(cfg,banks):
    clean,replay = banks
    # A fixed local draw responds to the correspondence of weights, preserving task/phase.
    changes = 0
    for outer in range(252,2001,2):
        modified = [dict(r,dino_weight=2. if i%2 else 1.,shuffled_weight=1. if i%2 else 2.)
                    for i,r in enumerate(replay)]
        a = common.draw_record(clean,modified,"stable_hard",outer,cfg)
        b = common.draw_record(clean,modified,"shuffled_hard",outer,cfg)
        changes += a[2] != b[2]
        assert a[3] == b[3] and a[0]["task_id"] == b[0]["task_id"]
        assert a[0]["phase_proxy"] == b[0]["phase_proxy"]
    assert changes > 0
    with pytest.raises(Exception,match="method/outer"):
        common.draw_record(clean,replay,"stable_hard",2001,cfg)


def test_eval_contract_4800_two_checkpoints_and_resource_limits(cfg):
    evaluation = runner.load_config(cfg)
    assert len(runner.CONDITIONS) == 12 and len(runner.BASE_CONDITIONS) == 6
    assert "checkpoint_outer" in runner.KEY
    assert evaluation["evaluation_checkpoint_outer_iterations"] == [1000,2000]
    assert not set(evaluation)&runner.STALE_KEYS
    assert evaluation["eval_hooks"] is False and evaluation["eval_eta"] is False
    assert 4*10*10*len(runner.CONDITIONS) == 4800
    assert sum(runner.DENOMINATORS.values())*12 == 4584
    assert runner.reference_frames(evaluation,cfg)[0].empty
    train = (ROOT/"slurm/exploring_oft_dino/113_clear_sampling_train_4090.sbatch").read_text()
    test = (ROOT/"slurm/exploring_oft_dino/115_clear_sampling_eval_3090.sbatch").read_text()
    assert "#SBATCH --array=0-3%4" in train and "#SBATCH --gres=gpu:4090:1" in train
    assert "TRAIN_ORDER=("+" ".join(common.TRAIN_ORDER)+")" in train
    assert "#SBATCH --array=0-9%8" in test and "#SBATCH --gres=gpu:3090:1" in test
    for stage in ("112_clear_sampling_prepare_cpu","114_clear_sampling_validate_cpu","116_clear_sampling_stats_cpu"):
        content = (ROOT/"slurm/exploring_oft_dino"/(stage+".sbatch")).read_text()
        assert "#SBATCH --partition=cpu" in content and "#SBATCH --gres" not in content


def test_changed_frozen_asset_fails_before_sources(cfg,tmp_path,monkeypatch):
    item = tmp_path/"dataset_statistics.json";item.write_text('{"q01":0}')
    changed = {**cfg,"frozen_inputs":{},"frozen_eval_assets":{str(item):common.sha256_file(item)}}
    item.write_text('{"q01":1}')
    monkeypatch.setattr(trainer,"source_manifest",lambda *a:pytest.fail("must check assets first"))
    with pytest.raises(Exception,match="Frozen input missing/changed"):
        common.source_preflight(changed)
    assert len(cfg["frozen_eval_assets"]) == 168


def submit_cfg(cfg,tmp_path):
    return {**cfg,"output_root":str(tmp_path/"run")}


def test_mock_submit_dependency_chain_receipts_and_limits(cfg,tmp_path,monkeypatch):
    changed = submit_cfg(cfg,tmp_path);calls = []
    def fake(command,**kwargs):
        calls.append(command);n = sum(c[0] == "sbatch" for c in calls)
        return SimpleNamespace(returncode=0,stdout="" if command[0] == "squeue" else f"{1000+n};cluster\n",stderr="")
    monkeypatch.setattr(subprocess,"run",fake)
    report = common.submit_chain(changed)
    assert [j["stage"] for j in report["jobs"]] == ["prepare","train","validate","eval","stats"]
    commands = [c for c in calls if c[0] == "sbatch"]
    assert "--array=0-3%4" in commands[1] and "--array=0-9%8" in commands[3]
    assert all("--kill-on-invalid-dep=yes" in c for c in commands)
    for i in range(1,5):
        assert f"--dependency=afterok:{1000+i}" in commands[i]
    assert json.loads((Path(changed["output_root"])/"reports/SUBMISSION_RECEIPT.json").read_text())["status"] == "SUBMITTED"


@pytest.mark.parametrize("failed_stage",["prepare","train","validate","eval","stats"])
def test_partial_submit_stops_and_persists_ids(cfg,tmp_path,monkeypatch,failed_stage):
    changed = submit_cfg(cfg,tmp_path);calls = []
    stop = ["prepare","train","validate","eval","stats"].index(failed_stage)+1
    def fake(command,**kwargs):
        calls.append(command);n = sum(c[0] == "sbatch" for c in calls)
        bad = command[0] == "sbatch" and n == stop
        return SimpleNamespace(returncode=int(bad),stdout="" if bad or command[0] == "squeue" else str(9000+n),stderr="mock failure" if bad else "")
    monkeypatch.setattr(subprocess,"run",fake)
    with pytest.raises(RuntimeError,match="no downstream"):
        common.submit_chain(changed)
    assert sum(c[0] == "sbatch" for c in calls) == stop
    receipt = json.loads((Path(changed["output_root"])/"reports/SUBMISSION_RECEIPT.json").read_text())
    assert receipt["status"] == "SUBMISSION_FAILED" and len(receipt["jobs"]) == stop-1


@pytest.mark.parametrize("code,stdout",[(1,""),(0,"123 queued")])
def test_unknown_or_duplicate_queue_fails_closed(cfg,tmp_path,monkeypatch,code,stdout):
    calls = []
    def fake(command,**kwargs):
        calls.append(command);return SimpleNamespace(returncode=code,stdout=stdout,stderr="queue unavailable")
    monkeypatch.setattr(subprocess,"run",fake)
    with pytest.raises(Exception):
        common.submit_chain(submit_cfg(cfg,tmp_path))
    assert len(calls) == 1 and calls[0][0] == "squeue"


def test_completed_methods_skip_only_validated_array_indices(cfg,tmp_path,monkeypatch):
    changed = submit_cfg(cfg,tmp_path)
    folder = Path(changed["output_root"])/"training/stable_hard";folder.mkdir(parents=True)
    (folder/"status.json").write_text('{"status":"COMPLETE"}')
    checked,calls = [],[]
    monkeypatch.setattr(common,"validate_training",lambda cfg,methods,**k:checked.extend(methods))
    def fake(command,**kwargs):
        calls.append(command);return SimpleNamespace(returncode=0,stdout="" if command[0] == "squeue" else str(1000+len(calls)),stderr="")
    monkeypatch.setattr(subprocess,"run",fake)
    common.submit_chain(changed)
    assert checked == ["stable_hard"]
    command = next(c for c in calls if c[-1].endswith("113_clear_sampling_train_4090.sbatch"))
    assert "--array=0,1,3%4" in command


def eval_fixture(cfg,tmp_path,monkeypatch,task=3):
    evaluation = runner.load_config(cfg);evaluation["output_root"] = str(tmp_path/"eval")
    train = {**cfg,"output_root":str(tmp_path)}
    hashes = {m:{"pt":f"{i+1:064x}","png":"f"*64} for i,m in enumerate(runner.CONDITIONS)}
    asset = tmp_path/"actual_modeling.py";asset.write_text("loaded runtime implementation")
    runtime = {"source":{str(asset):runner.sha256_file(asset)},"config_signature":runner.digest_json(evaluation),
        "evaluation_hooks":False,"evaluation_eta":False,"evaluation_checkpoint_outer_iterations":[1000,2000]}
    monkeypatch.setattr(runner,"config",lambda:train)
    monkeypatch.setattr(runner,"load_config",lambda *a:evaluation)
    monkeypatch.setattr(runner,"source_preflight",lambda *a:{})
    monkeypatch.setattr(runner,"require_preparation",lambda *a:{"signature":"prepared"})
    monkeypatch.setattr(runner,"patches",lambda *a:({},hashes,{}))
    monkeypatch.setattr(runner,"frozen_clean",lambda base,suite:pd.DataFrame([
        {"task_id":task,"state_index":s,"success":1,"rollout_seed":1000+s} for s in evaluation["state_indices"]]))
    (tmp_path/"AUTO_VALIDATION.json").write_text(json.dumps({"status":"COMPLETE",
        "config_sha256":runner.sha256_file(common.CONFIG),"preparation_signature":"prepared","evaluation_hooks":False}))
    return evaluation,train,hashes,asset,runtime


def fake_frames(cfg,suite,task,hashes,runtime):
    rows,traces = [],[]
    for state in cfg["state_indices"]:
        for condition in runner.CONDITIONS:
            cp = int(condition.rsplit("_",1)[1])
            meta = {"experiment_id":runner.EXPERIMENT_ID,"training_suite":"libero_object",
                "suite":suite,"task_id":task,"state_index":state,"condition":condition,"checkpoint_outer":cp,
                "checkpoint":cfg["checkpoints"][suite],"unnorm_key":cfg["unnorm_keys"][suite],
                "frozen_clean_success":1,"config_signature":runner.digest_json(cfg),
                "runtime_source_digest":runner.digest_json(runtime["source"]),"reused":False}
            rows.append({**meta,"success":0,"failure":1,"condition_order":runner._order(cfg,suite,task,state).index(condition)+1,
                "rollout_seed":1000+state,"patch_sha256":hashes[condition]["pt"],"renderer_vendor":"NVIDIA Corporation",
                "decision_blocks":1,"elapsed_s":0.})
            traces.append({**meta,"decision_block":0,"step":0})
    return pd.DataFrame(rows),pd.DataFrame(traces)


def test_complete_task_skips_validated_suites_and_rejects_runtime_drift(cfg,tmp_path,monkeypatch):
    evaluation,_,hashes,asset,runtime = eval_fixture(cfg,tmp_path,monkeypatch);calls = []
    def fake_run(index,expanded,**kwargs):
        calls.append(index);suite,task = evaluation["suites"][index//10],index%10
        rows,traces = fake_frames(evaluation,suite,task,hashes,runtime)
        folder = Path(evaluation["output_root"])/suite;folder.mkdir(parents=True)
        rows.to_parquet(folder/f"task_{task:02d}_rollouts.parquet",index=False)
        traces.to_parquet(folder/f"task_{task:02d}_traces.parquet",index=False)
        runner.atomic_json(runtime,Path(evaluation["output_root"])/f"task_{task:02d}_{suite}_RUNTIME.json")
    monkeypatch.setattr(runner,"run_shard",fake_run)
    runner.run_checked(3);runner.run_checked(3)
    assert calls == [3,13,23,33]
    asset.write_text("changed loaded runtime")
    with pytest.raises(Exception,match="Actual loaded evaluation runtime source changed"):
        runner.run_checked(3)


@pytest.mark.parametrize("field,value,match",[
    ("checkpoint_outer",1500,"checkpoint metadata"),("rollout_seed",-1,"seed"),
    ("frozen_clean_success",0,"eligibility"),("patch_sha256","wrong","Patch bytes"),
    ("checkpoint","wrong","metadata"),("condition_order",99,"order"),
])
def test_eval_checkpoint_seed_patch_and_qualification_drift_fails(cfg,tmp_path,monkeypatch,field,value,match):
    evaluation,_,hashes,_,runtime = eval_fixture(cfg,tmp_path,monkeypatch)
    rows,traces = fake_frames(evaluation,"libero_object",3,hashes,runtime);rows.loc[0,field] = value
    with pytest.raises(Exception,match=match):
        runner.validate_new_shard(rows,traces,"libero_object",3,evaluation,hashes)


def test_joint_transaction_recovers_interrupt_between_parquet_writes(cfg,tmp_path,monkeypatch):
    evaluation,_,hashes,_,runtime = eval_fixture(cfg,tmp_path,monkeypatch)
    loaded = {m:(torch.zeros(3,50,50),h["pt"]) for m,h in hashes.items()}
    monkeypatch.setattr(trainer,"source_manifest",lambda *a:runtime["source"])
    monkeypatch.setattr(runner,"load_wrapper",lambda *a:SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(runner,"make_env",lambda *a:(SimpleNamespace(close=lambda:None),SimpleNamespace(language="task"),list(range(50))))
    monkeypatch.setattr(runner,"renderer_metadata",lambda *a:("EGL","NVIDIA Corporation"))
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    calls = []
    def episode(env,wrapper,task,state,seed,method,patch,digest,base,renderer,vendor,order):
        calls.append((state,method))
        return {"condition":method,"success":0,"failure":1,"condition_order":order,"rollout_seed":seed,
            "patch_sha256":digest,"renderer_vendor":vendor,"decision_blocks":1,"elapsed_s":0.},[
            {"condition":method,"decision_block":0,"step":0}]
    monkeypatch.setattr(runner,"run_episode",episode)
    original = runner.atomic_parquet;failed = False
    def fault(frame,path):
        nonlocal failed
        if str(path).endswith("traces.partial.parquet") and not failed:
            failed = True;raise OSError("fault between paired writes")
        return original(frame,path)
    monkeypatch.setattr(runner,"atomic_parquet",fault)
    loader = lambda *a:(loaded,hashes,{})
    with pytest.raises(OSError,match="fault between"):
        runner.run_shard(3,evaluation,patch_loader=loader)
    assert len(calls) == 12
    runner.run_shard(3,evaluation,patch_loader=loader)
    assert len(calls) == len(set(calls)) == 120
    folder = Path(evaluation["output_root"])/"libero_object"
    (folder/"task_03_traces.parquet").unlink()
    runner.run_shard(3,evaluation,patch_loader=loader)
    assert len(calls) == 120 and (folder/"task_03_traces.parquet").exists()


def test_checkpoint_paired_bootstrap_shared_draws_and_task_equal():
    rows = []
    for suite in runner.DENOMINATORS:
        for task in range(10):
            for state in ([40] if task == 0 else range(40,50)):
                for condition in runner.CONDITIONS:
                    cp = int(condition.rsplit("_",1)[1]);method = condition.rsplit("_",1)[0]
                    rows.append({"checkpoint_outer":cp,"suite":suite,"task_id":task,"state_index":state,
                        "condition":condition,"casr":float(method == "stable_hard" and task == 0)})
    frame = pd.DataFrame(rows)
    point,samples,_ = analyzer.hierarchical_samples(frame.loc[frame.suite.eq("libero_object")],100,np.random.default_rng(42))
    assert point[runner.CONDITIONS.index("stable_hard_1000")] == pytest.approx(.1)
    assert np.array_equal(samples[:,runner.CONDITIONS.index("stable_hard_1000")],samples[:,runner.CONDITIONS.index("stable_hard_2000")])
    summary,contrasts = analyzer.paired_statistics(frame,100,20260907)
    assert len(summary.loc[summary.suite_or_endpoint.isin(runner.DENOMINATORS)]) == 48
    assert set(summary.checkpoint_outer) == {1000,2000}
    assert {"B_minus_A","C_minus_B","C_minus_D"} <= set(contrasts.contrast)
    match = contrasts.loc[contrasts.contrast.eq("C_minus_D")]
    assert np.allclose(match.difference_left_minus_right,.1)
    missing = frame.loc[~(frame.task_id.eq(0)&frame.condition.eq("stable_hard_2000"))]
    with pytest.raises(Exception,match="Unpaired"):
        analyzer.paired_statistics(missing,100,42)


def test_own_score_snapshots_reconstruct_exact_draw_and_reject_weight_tampering(cfg,banks,tmp_path):
    clean,replay = banks
    fields = ("dino_score","dino_delta_rms","patched_target_mse","restored_target_mse",
              "dino_weight","shuffled_weight","score_view_count","score_full_forward_calls")
    history,intervals = [],{}
    for cp in cfg["refresh_after_outer"]:
        patch = torch.full((3,50,50),.4+cp/10000)
        torch.save({"patch":patch},tmp_path/f"outer_{cp:04d}.pt")
        scored = [dict(record,dino_score=float((i%7)-3),dino_delta_rms=.2,
                       patched_target_mse=1.,restored_target_mse=1.+float((i%7)-3),
                       score_view_count=1,score_full_forward_calls=3) for i,record in enumerate(replay)]
        scored = apply_score_weights(scored,cfg["data_seed"],cp)
        values = [{k:record[k] for k in fields} for record in scored]
        identity = {"outer":cp,"records":258,"full_forward_calls":258*3,
            "patch_tensor_sha256":common._tensor_digest(patch),
            "quantized_patch_sha256":common._tensor_digest(patch.mul(255).round().div(255))}
        history.append(identity);intervals[cp] = scored
        (tmp_path/f"scores_after_{cp:04d}.json").write_text(json.dumps({**identity,"method":"dynamic_hard","records":values}))
    rows = []
    for outer in range(1,2001):
        score_after = max([0]+[cp for cp in cfg["refresh_after_outer"] if cp < outer])
        record,origin,index,base = common.draw_record(clean,intervals.get(score_after,replay),"dynamic_hard",outer,cfg)
        rows.append({"outer_iteration":outer,"origin":origin,"record_index":index,"base_record_index":base,
                     "task_id":record["task_id"],"state_index":record["state_index"],
                     "score_after_outer":score_after,"dino_score":record.get("dino_score",0.)})
    saved = {"score_refresh_history":history,"replay_scoring":values}
    frame = pd.DataFrame(rows)
    common.validate_sampling_history(cfg,"dynamic_hard",tmp_path,saved,frame,clean,replay)
    changed = frame.copy();changed.loc[251,"record_index"] = (int(changed.loc[251,"record_index"])+1)%258
    with pytest.raises(Exception,match="Exact live own-weight"):
        common.validate_sampling_history(cfg,"dynamic_hard",tmp_path,saved,changed,clean,replay)
    path = tmp_path/"scores_after_0250.json";score = json.loads(path.read_text())
    score["records"][0]["dino_weight"] = 3.;path.write_text(json.dumps(score))
    with pytest.raises(Exception,match="top-half weights"):
        common.validate_sampling_history(cfg,"dynamic_hard",tmp_path,saved,frame,clean,replay)


def test_complete_4800_grid_stats_and_checkpoint_drift(cfg,tmp_path,monkeypatch):
    evaluation = runner.load_config(cfg);evaluation["output_root"] = str(tmp_path/"eval")
    train = {**cfg,"output_root":str(tmp_path)}
    hashes = {m:{"pt":f"{i+1:064x}","png":"f"*64} for i,m in enumerate(runner.CONDITIONS)}
    asset = tmp_path/"actual_modeling.py";asset.write_text("actual loaded class")
    runtime = {"source":{str(asset):runner.sha256_file(asset)},"config_signature":runner.digest_json(evaluation),
        "evaluation_hooks":False,"evaluation_eta":False,"evaluation_checkpoint_outer_iterations":[1000,2000]}
    output = Path(evaluation["output_root"]);output.mkdir()
    runner.atomic_json(evaluation,output/"EVAL_CONFIG.json")
    def clean(base,suite):
        denied = 100-runner.DENOMINATORS[suite]
        return pd.DataFrame([{"task_id":task,"state_index":state,"success":int((state-40)*10+task>=denied),
            "rollout_seed":1000+state} for task in range(10) for state in evaluation["state_indices"]])
    monkeypatch.setattr(runner,"frozen_clean",clean)
    frames = []
    for task in range(10):
        receipt = {"status":"COMPLETE","task_id":task,"config_signature":runner.digest_json(evaluation),
            "patches":hashes,"suites":{},"evaluation_checkpoint_outer_iterations":[1000,2000]}
        for suite in evaluation["suites"]:
            rows,traces = fake_frames(evaluation,suite,task,hashes,runtime)
            qualification = clean(evaluation,suite).loc[lambda f:f.task_id.eq(task)].set_index("state_index").success
            for frame in (rows,traces):
                frame["frozen_clean_success"] = frame.state_index.map(qualification)
            folder = output/suite;folder.mkdir(exist_ok=True)
            rp,tp = [folder/f"task_{task:02d}_{kind}.parquet" for kind in ("rollouts","traces")]
            rows.to_parquet(rp,index=False);traces.to_parquet(tp,index=False)
            receipt["suites"][suite] = {"rollouts_sha256":runner.sha256_file(rp),"traces_sha256":runner.sha256_file(tp),
                                       "evaluation_checkpoint_outer_iterations":[1000,2000]}
            runner.atomic_json(runtime,output/f"task_{task:02d}_{suite}_RUNTIME.json");frames.append(rows)
        runner.atomic_json(receipt,output/f"task_{task:02d}_STATUS.json")
    raw = pd.concat(frames,ignore_index=True)
    assert len(analyzer.validate_raw(raw,evaluation)) == 4584
    bad = raw.copy();bad.loc[0,"checkpoint_outer"] = 1500
    with pytest.raises(Exception,match="raw grid"):
        analyzer.validate_raw(bad,evaluation)
    bad = pd.concat([raw.iloc[:-1],raw.iloc[:1]],ignore_index=True)
    with pytest.raises(Exception,match="duplicate"):
        analyzer.validate_raw(bad,evaluation)
    for method in common.METHODS:
        folder = tmp_path/"training"/method;folder.mkdir(parents=True)
        runner.atomic_json({"elapsed_s":1.,"costs":{"score_refresh_s":.2,"runtime_validation_s":.1}},folder/"status.json")
        pd.DataFrame({"outer_iteration":range(1,2001),"elapsed_s":np.arange(1,2001,dtype=float),
            "tv":np.full(2000,.2),"hf_max":np.full(2000,.03),"png_tv":np.full(2000,.2),"png_hf_max":np.full(2000,.03),
            "correction_updates":np.zeros(2000),"effective_rgb_updates":np.full(2000,50)}).to_parquet(folder/"training.parquet",index=False)
    monkeypatch.setattr(analyzer,"config",lambda:train)
    monkeypatch.setattr(analyzer,"load_config",lambda:evaluation)
    monkeypatch.setattr(analyzer,"require_preparation",lambda *a:{"signature":"prepared"})
    monkeypatch.setattr(analyzer,"patches",lambda *a:({},hashes,{}))
    monkeypatch.setattr(analyzer,"external_baselines",lambda *a:{f"{m}_{cp}":{"checkpoint_outer":cp,
        "recorded_training_elapsed_s":1.,"pt":hashes[f"{m}_{cp}"]["pt"]} for cp in common.CHECKPOINTS for m in runner.REFERENCES})
    report = analyzer.analyze()
    assert report["status"] == "COMPLETE" and report["attack_episodes"] == 4800 and report["eligible_rows"] == 4584
    assert report["bootstrap_seed"] == 20260907 and report["bootstrap_draws"] == 10000
    assert report["summary_rows"] == 48 and report["reused_attack_episodes"] == 0
    summary = pd.read_csv(output/"DINO_SAMPLING_CLEAR2000_XSUITE_SUMMARY.csv")
    assert len(summary) == 48 and set(summary.checkpoint_outer) == {1000,2000}
    endpoints = pd.read_csv(output/"DINO_SAMPLING_CLEAR2000_XSUITE_ENDPOINTS.csv")
    assert len(endpoints.loc[endpoints.suite_or_endpoint.eq("spatial_goal_macro")]) == 12
    with pytest.raises(Exception,match="frozen"):
        analyzer.analyze(draws=100)


def test_helper_preflight_in_independent_process_never_models_cuda_queue_or_submit(tmp_path):
    executable = tmp_path/"cpu_trap_python"
    executable.write_text("#!"+sys.executable+"\n"+'''
import runpy,sys,subprocess,torch
from types import SimpleNamespace
def forbidden(*args,**kwargs):
    raise AssertionError("preflight touched model/CUDA/queue/submit")
original_run,original_popen=subprocess.run,subprocess.Popen
def check_command(command):
    words=command if isinstance(command,(list,tuple)) else str(command).split()
    if any(str(word).rsplit("/",1)[-1] in ("squeue","sbatch","nvidia-smi") for word in words): forbidden()
def guarded_run(command,*args,**kwargs):
    check_command(command)
    return original_run(command,*args,**kwargs)
def guarded_popen(command,*args,**kwargs):
    check_command(command)
    return original_popen(command,*args,**kwargs)
subprocess.run=guarded_run
subprocess.Popen=guarded_popen
torch.cuda._lazy_init=lambda *a,**k:forbidden()
torch.cuda.init=lambda *a,**k:forbidden()
torch.cuda.set_device=lambda *a,**k:forbidden()
torch.cuda.synchronize=lambda *a,**k:forbidden()
sys.modules["UADA_OFT_DINO_wrapper"]=SimpleNamespace(load_wrapper=forbidden)
sys.argv.pop(0)
if sys.argv[:1]==["-B"]: sys.argv.pop(0)
script=sys.argv.pop(0)
sys.argv.insert(0,script)
runpy.run_path(script,run_name="__main__")
''')
    executable.chmod(0o755)
    env = {**os.environ,"OFT_SAMPLING_CLEAR_PROJECT_ROOT":str(ROOT),"OFT_SAMPLING_CLEAR_PYTHON":str(executable),
        "PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    result = subprocess.run(["bash",str(ROOT/"submit_exploring_oft_dino_sampling_clear2000_v1.sh"),"--preflight"],
                            cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode == 0,result.stdout+result.stderr
    assert "SAMPLING_CLEAR2000_PREFLIGHT_PASS" in result.stdout and '"model_loaded": false' in result.stdout

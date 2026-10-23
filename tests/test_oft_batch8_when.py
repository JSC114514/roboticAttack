"""CPU behavior checks: actual averaging, fresh training/resume, and bounded submission."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/"research_runs/wrist_dino_batch8_when_v1/code"
sys.path[:0]=[str(CODE),str(ROOT/"scripts"),str(ROOT/"VLAAttacker")]
import dino_batch8_common as common
from objectives import mean_gradient,feature_objective
from train import Batch8Attacker,validate_resume

def fake_banks():
    clean=[{"task_id":i//30,"state_index":10+i%5,"task_name":"test",
            "observation":{"value":i/300.}} for i in range(300)]
    replay=[]
    for i in range(258):
        obs={"value":i/258.}
        replay.append({"task_id":i%10,"state_index":10+i%5,"task_name":"test",
                       "phase_proxy":("early","middle","late")[(i//10)%3],
                       "observation":obs,"observations":[obs,dict(value=obs["value"]+.01),
                                                        dict(value=obs["value"]+.02)]})
    return clean,replay

def small_cfg(tmp_path):
    cfg=json.loads((CODE.parent/"config.json").read_text())
    cfg.update(output_root=str(tmp_path),outer_iterations=4,scheduler_total_outer=4,
               inner_loop=3,bootstrap_outer=1,diverse_after_outer=1,
               refresh_after_outer=[1,3],checkpoint_outer_iterations=[0,1,3,4],
               warmup_outer_iterations=1)
    return cfg

def test_eight_state_gradient_is_mean_not_eight_steps():
    p=torch.randn(12,requires_grad=True)
    states=torch.randn(8,12)
    expected=torch.autograd.grad(((p[None]-states)**2).mean(),p)[0]
    got,value=mean_gradient(p,(((p-x)**2).mean() for x in states))
    torch.testing.assert_close(got,expected)
    assert value==pytest.approx(float(((p.detach()[None]-states)**2).mean()))
    opt=torch.optim.AdamW([p],lr=.01)
    p.grad=got;opt.step()
    assert int(opt.state[p]["step"])==1
    with pytest.raises(ValueError): mean_gradient(p,((p*p).sum() for _ in range(7)))

def test_when_mask_sign_and_grad_accumulation():
    clean=torch.randn(1,16,12)
    adv=(clean+.2*torch.randn_like(clean)).requires_grad_()
    coverage=torch.zeros(16);coverage[:3]=1
    loss,metrics=feature_objective(clean,adv,coverage)
    grad=torch.autograd.grad(loss,adv)[0]
    assert loss.item()<0 and metrics["feature_valid_tokens"]==13
    assert torch.equal(grad[:,:3],torch.zeros_like(grad[:,:3]))
    assert torch.isfinite(grad).all() and float(grad[:,3:].norm())>0
    updated=adv.detach()-.001*grad
    assert feature_objective(clean,updated,coverage)[0] < loss.detach()
    with pytest.raises(ValueError): feature_objective(clean,adv,torch.ones(16))

def test_when_core_is_author_copy():
    import ast
    original=ast.parse((CODE/"source_when.py").read_text())
    actual=ast.parse((CODE/"when_core.py").read_text())
    get=lambda module:next(node for node in module.body
                          if isinstance(node,ast.ClassDef) and node.name=="BiContrastiveLoss")
    assert ast.dump(get(original))==ast.dump(get(actual))

def test_when_and_uniform_have_same_data_and_feature_batch_mean():
    clean_bank,replay=fake_banks()
    cfg=json.loads((CODE.parent/"config.json").read_text())
    for outer in (1,250,251,252,1000,2000):
        assert common.draw_batch(clean_bank,replay,"uniform_replay_b8",outer,cfg)==common.draw_batch(
            clean_bank,replay,"when_dino_b8",outer,cfg)
    p=torch.randn(5,requires_grad=True)
    features=torch.randn(8,1,16,5)
    coverage=torch.zeros(16)
    losses=[feature_objective(x,x+p[None,None,:],coverage)[0] for x in features]
    direct=torch.autograd.grad(torch.stack(losses).mean(),p)[0]
    accumulated,_=mean_gradient(p,(feature_objective(x,x+p[None,None,:],coverage)[0] for x in features))
    torch.testing.assert_close(accumulated,direct)

def test_clean_batch_distinct_across_shuffle_boundary():
    cfg=json.loads((CODE.parent/"config.json").read_text())
    clean,replay=fake_banks()
    for outer in (38,75,113,150):
        for method in common.METHODS:
            batch=common.draw_batch(clean,replay,method,outer,cfg)
            assert len({b["index"] for b in batch})==8

@pytest.mark.parametrize("method",common.METHODS)
def test_batch_unique_and_replay_policy(method,tmp_path):
    clean,replay=fake_banks();cfg=small_cfg(tmp_path)
    for outer in range(1,5):
        batch=common.draw_batch(clean,replay,method,outer,cfg)
        assert len(batch)==len({b["index"] for b in batch})==8
        assert batch==common.draw_batch(clean,replay,method,outer,cfg)
        expected="clean" if method=="original_uada_b8" or outer<=1 or (outer-1)%2 else "replay"
        assert all(b["origin"]==expected for b in batch)
        if method!="dino_sampling_b8": assert all(b["axis"]==0 for b in batch)

def test_dino_score_changes_selection(tmp_path):
    clean,replay=fake_banks();cfg=small_cfg(tmp_path)
    uniform=[common.draw_batch(clean,replay,"dino_sampling_b8",i,cfg) for i in (3,)]
    for r in replay:
        r["axis_scores"]=[{"axis":j,"dino_remaining_score":1000. if j==2 else 0.} for j in range(3)]
    selected=common.draw_batch(clean,replay,"dino_sampling_b8",3,cfg)
    assert [b["axis"] for b in selected]!=[b["axis"] for b in uniform[0]]

class FakeAttacker(Batch8Attacker):
    def __init__(self,fail_after=None):
        self.wrapper=SimpleNamespace(device="cpu",vla=torch.nn.Linear(1,1))
        self.site=(26,137);self.transfer_diverse=False;self.calls=0;self.fail_after=fail_after
    def _base_inputs(self,record):
        return {"value":record["observation"]["value"]},np.zeros(7)
    def _forward(self,inputs,proprio,observation,patch,geometry,**kwargs):
        if patch is not None:
            self.calls+=1
            if self.fail_after==self.calls: raise RuntimeError("injected interruption")
        value=torch.zeros(()) if patch is None else .2*patch.mean()
        return (torch.zeros(1,8,7)+.01*inputs["value"]+value),torch.zeros(1,256,8),None
    def _features(self,observation,patch,geometry):
        if patch is not None:
            self.calls+=1
            if self.fail_after==self.calls: raise RuntimeError("injected interruption")
        basis=torch.sin(torch.arange(256*8).reshape(1,256,8).float()/23.)
        feat=basis+observation["value"]
        if patch is not None: feat=feat+.1*patch.mean()*basis
        cov=torch.zeros(256);cov[:10]=int(patch is not None)
        return feat,cov
    def _validate_runtime(self,*args): return {"status":"PASS"}
    def _refresh_axis_scores(self,replay,patch,cfg,method,refresh):
        assert method=="dino_remaining"
        return [dict(r,axis_scores=[{"axis":j,"dino_remaining_score":(j+1)*float(patch.mean())}
                                    for j in range(3)],score_full_forward_calls=5) for r in replay]

@pytest.mark.parametrize("method",common.METHODS)
def test_fresh_training_updates_and_resume_exact(method,tmp_path):
    clean,replay=fake_banks()
    cfg=small_cfg(tmp_path/"complete")
    FakeAttacker().train(clean,replay,cfg,method,{}, {})
    complete=torch.load(Path(cfg["output_root"])/"training"/method/"resume.pt",weights_only=False)
    assert complete["outer"]==4 and int(next(iter(complete["optimizer"]["state"].values()))["step"])==12
    assert not torch.equal(complete["initial"],complete["patch"])
    assert torch.equal(complete["initial"],torch.rand((3,50,50),generator=torch.Generator().manual_seed(42)))
    cfg2=small_cfg(tmp_path/"resumed")
    with pytest.raises(RuntimeError,match="injected"):
        FakeAttacker(fail_after=29).train(clean,replay,cfg2,method,{}, {})
    FakeAttacker().train(clean,replay,cfg2,method,{}, {})
    resumed=torch.load(Path(cfg2["output_root"])/"training"/method/"resume.pt",weights_only=False)
    assert torch.equal(complete["patch"],resumed["patch"])
    for key in ("exp_avg","exp_avg_sq","step"):
        left=next(iter(complete["optimizer"]["state"].values()))[key]
        right=next(iter(resumed["optimizer"]["state"].values()))[key]
        assert torch.equal(left,right) if isinstance(left,torch.Tensor) else left==right
    assert [{k:v for k,v in h.items() if k!="elapsed_s"} for h in complete["score_history"]]==[
        {k:v for k,v in h.items() if k!="elapsed_s"} for h in resumed["score_history"]]
    validate_resume(resumed,cfg2)
    rows=pd.read_parquet(Path(cfg2["output_root"])/"training"/method/"training.parquet")
    assert rows.sample_backprops.tolist()==[24,48,72,96]
    assert np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy()).all()

def test_submit_dependencies_caps_and_no_overlap(monkeypatch,tmp_path):
    monkeypatch.setattr(common,"RUN",tmp_path)
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        if cmd[0]=="squeue": return SimpleNamespace(returncode=0,stdout="123|oft_old|gres/gpu:1|None\n",stderr="")
        return SimpleNamespace(returncode=0,stdout=str(700+len(calls)),stderr="")
    monkeypatch.setattr(common.subprocess,"run",run)
    result=common.submit_chain({"methods":list(common.METHODS)})
    jobs=[x["command"] for x in result["jobs"]]
    assert len(jobs)==5 and "--dependency=afterany:123" in jobs[0]
    assert "--array=0,1,2,3%4" in jobs[1]
    assert "--array=0-9%8" in jobs[3]
    for i in range(1,5): assert "--dependency=afterok:"+result["jobs"][i-1]["job_id"] in jobs[i]
    assert result["training_gpu_limit"]<=5

def test_queue_failure_submits_nothing(monkeypatch,tmp_path):
    monkeypatch.setattr(common,"RUN",tmp_path)
    called=[]
    def run(cmd,**kwargs):
        called.append(cmd)
        return SimpleNamespace(returncode=1,stdout="",stderr="unavailable")
    monkeypatch.setattr(common.subprocess,"run",run)
    with pytest.raises(RuntimeError,match="Queue query failed"): common.submit_chain({})
    assert len(called)==1 and called[0][0]=="squeue"

def test_partial_submit_failure_stops_chain(monkeypatch,tmp_path):
    monkeypatch.setattr(common,"RUN",tmp_path);called=[]
    def run(cmd,**kwargs):
        called.append(cmd)
        if cmd[0]=="squeue": return SimpleNamespace(returncode=0,stdout="",stderr="")
        if len(called)==3: return SimpleNamespace(returncode=1,stdout="",stderr="QOS")
        return SimpleNamespace(returncode=0,stdout="800",stderr="")
    monkeypatch.setattr(common.subprocess,"run",run)
    with pytest.raises(RuntimeError): common.submit_chain({})
    assert len(called)==3
    assert json.loads((tmp_path/"reports/SUBMISSION_RECEIPT.json").read_text())["failed_stage"]=="train"

def test_matched_statistics_both_budgets():
    import analyze
    from evaluate import BASE_CONDITIONS,DENOMINATORS
    rows=[]
    for cp in (1000,2000):
        for suite,n in DENOMINATORS.items():
            for task in range(10):
                count=n//10+(task<n%10)
                for state in range(count):
                    for j,m in enumerate(BASE_CONDITIONS):
                        rows.append({"checkpoint_outer":cp,"suite":suite,"task_id":task,"state_index":state+40,
                                     "condition":f"{m}_{cp}","casr":float((task+state+j)%2)})
    frame=pd.DataFrame(rows)
    summary,contrasts=analyze.paired_statistics(frame,draws=25)
    assert len(summary)==2*6*7
    assert set(contrasts.contrast)=={p[0] for p in analyze.PAIRS}
    assert len(frame)==5348

def test_full_evaluation_grid_validation():
    import analyze
    from evaluate import BASE_CONDITIONS,DENOMINATORS
    rows=[]
    for cp in (1000,2000):
        for suite,n in DENOMINATORS.items():
            for task in range(10):
                count=n//10+(task<n%10)
                for state in range(10):
                    for j,m in enumerate(BASE_CONDITIONS):
                        success=(task+state+j)%2
                        rows.append({"checkpoint_outer":cp,"suite":suite,"task_id":task,
                            "state_index":state+40,"condition":f"{m}_{cp}","success":success,
                            "failure":1-success,"reused":False,"frozen_clean_success":int(state<count),
                            "rollout_seed":task*100+state})
    frame=pd.DataFrame(rows)
    assert len(frame)==5600
    assert len(analyze.validate_raw(frame,{}))==5348
    with pytest.raises(RuntimeError): analyze.validate_raw(frame.iloc[:-1],{})

def test_completed_artifacts_validate_and_reject_failed_alignment(monkeypatch,tmp_path):
    """Exercise the actual pre-evaluation gate without a model or GPU."""
    import zipfile
    from source_axis import _save_checkpoint
    cfg=json.loads((CODE.parent/"config.json").read_text())
    cfg["output_root"]=str(tmp_path)
    config_path=tmp_path/"config.json"
    config_path.write_text(json.dumps(cfg))
    clean,replay=fake_banks()
    monkeypatch.setattr(common,"RUN",tmp_path)
    monkeypatch.setattr(common,"CONFIG",config_path)
    monkeypatch.setattr(common,"source_records",lambda _:clean)
    monkeypatch.setattr(common,"replay_records",lambda _:(replay,{}))
    monkeypatch.setattr(common,"data_identity",lambda _: {})
    monkeypatch.setattr(common,"require_preparation",lambda _:{"signature":"fake-prepared"})
    initial=torch.rand((3,50,50),generator=torch.Generator().manual_seed(42))
    for method in common.METHODS:
        folder=tmp_path/"training"/method
        folder.mkdir(parents=True)
        signature=method+"-fake"
        rows=[];live=copy.deepcopy(replay);refreshes=[]
        for outer in range(1,2001):
            if method=="dino_sampling_b8" and outer-1 in cfg["refresh_after_outer"]:
                record_scores=[{"axis_scores":[{"axis":j,"dino_remaining_score":float(j)}
                                               for j in range(3)]} for _ in live]
                live=[dict(r,**s) for r,s in zip(replay,record_scores,strict=True)]
                (folder/f"scores_after_{outer-1:04d}.json").write_text(json.dumps(
                    {"outer":outer-1,"records":record_scores,"full_forward_calls":258*5}))
                refreshes.append(outer-1)
            batch=common.draw_batch(clean,live,method,outer,cfg)
            rows.append({"outer_iteration":outer,"optimizer_updates":outer*50,
                         "sample_backprops":outer*400,"batch_json":json.dumps([
                             {k:b[k] for k in ("origin","index","base_index","axis","slot")}
                             for b in batch])})
        for cp in cfg["checkpoint_outer_iterations"]:
            value=initial*(1-cp/100000.)+.005*cp/2000.
            _save_checkpoint(folder/f"outer_{cp:04d}.pt",value,
                {"method":method,"signature":signature,"total_outer":cp,
                 "total_optimizer_updates":cp*50,"sample_backprops":cp*400})
        saved={"config":cfg,"method":method,"outer":2000,"rows":rows,
               "scheduler":{"last_epoch":2000},"optimizer":{"state":{0:{"step":100000,
                   "exp_avg":torch.zeros_like(initial),"exp_avg_sq":torch.zeros_like(initial)}}},
               "patch":value,"initial":initial,"source":{},"data":{},"signature":signature,
               "score_history":[{"outer":cp} for cp in refreshes],"scores":live,
               "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),
               "python_rng":None,"cuda_rng":None}
        torch.save(saved,folder/"resume.pt")
        pd.DataFrame(rows).to_parquet(folder/"training.parquet",index=False)
        with zipfile.ZipFile(folder/"source_snapshot.zip","w") as archive:
            archive.writestr("fixture.txt","CPU fixture")
        (folder/"RUN_MANIFEST.json").write_text(json.dumps({"config":cfg,"source":{},
            "signature":signature,"fresh_start_outer":0,"initial_optimizer_state_empty":True,
            "feature_loss":method=="when_dino_b8","appearance_regularization":False}))
        (folder/"RUNTIME_ALIGNMENT.json").write_text(json.dumps({"status":"PASS",
            "full_forward_calls":2,"records":[{"pixel_exact":True,"feature_close":True}]*2}))
        status={"status":"COMPLETE","signature":signature,"outer":2000,
                "optimizer_updates":100000,"sample_backprops":800000,"score_refresh_outer":refreshes}
        for file,key in (("outer_2000.pt","final_patch_sha256"),("outer_2000.png","final_png_sha256"),
                         ("resume.pt","resume_sha256"),("source_snapshot.zip","source_snapshot_sha256")):
            status[key]=common.sha256_file(folder/file)
        (folder/"status.json").write_text(json.dumps(status))
    report=common.validate_training(cfg)
    assert report["status"]=="COMPLETE" and set(report["checkpoints"])==set(common.METHODS)
    assert (tmp_path/"AUTO_VALIDATION.json").exists()
    alignment_path=tmp_path/"training/when_dino_b8/RUNTIME_ALIGNMENT.json"
    alignment=json.loads(alignment_path.read_text())
    alignment["records"][0]["pixel_exact"]=False
    alignment_path.write_text(json.dumps(alignment))
    with pytest.raises(RuntimeError,match="alignment missing or failed"):
        common.validate_training(cfg,("when_dino_b8",),write_report=False)

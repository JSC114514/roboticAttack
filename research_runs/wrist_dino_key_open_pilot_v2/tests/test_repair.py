"""Compute-node timing regression and full manual-chain tests; all scheduler calls are mocked."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
from common import ROOT

@pytest.mark.parametrize("name,stage", [
    ("01_prepare_cpu.sbatch","prepare"), ("02_train_4090.sbatch","train"),
    ("03_validate_cpu.sbatch","validate"), ("04_eval_4090.sbatch","eval"), ("05_stats_cpu.sbatch","stats")])
def test_exact_stage_body_runs_without_external_time_or_path_tools(tmp_path,name,stage):
    script=ROOT/"slurm"/name
    assert "/usr/bin/time" not in script.read_text()
    stub=tmp_path/"python-stub";capture=tmp_path/"arguments"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$KEY_TEST_CAPTURE"\n');stub.chmod(0o755)
    env={**os.environ,"PATH":"/definitely-no-tools","KEY_TEST_PY":str(stub),"KEY_TEST_ROOT":str(tmp_path),
         "KEY_TEST_CAPTURE":str(capture),"SLURM_ARRAY_TASK_ID":"0"}
    # Substitute only environment setup; execute the exact submitted stage body with a harmless Python stand-in.
    body='source() { export PYTHON_BIN="$KEY_TEST_PY" EXPERIMENT_ROOT="$KEY_TEST_ROOT"; }; . "$1"'
    subprocess.run(["/bin/bash","-c",body,"test",str(script)],env=env,check=True,capture_output=True,text=True)
    args=capture.read_text().splitlines()
    assert args[:4]==["-B",str(tmp_path/"code/worker.py"),"--stage",stage]
    if stage=="eval":assert args[-2:]==["--index","0"]

def test_user_entry_submits_entire_chain_and_rejects_duplicate(tmp_path,monkeypatch):
    import pipeline as p
    calls=[];active=False
    monkeypatch.setattr(p,"ROOT",tmp_path);monkeypatch.setattr(p,"preflight",lambda **kw:None)
    monkeypatch.setattr(p,"run_signature",lambda:"test");monkeypatch.setattr(p,"config",lambda:{"resources":{}})
    def fake_run(command,**kwargs):
        if command[0]=="squeue":return SimpleNamespace(returncode=0,stdout="101|RUNNING|gres/gpu:4090:1|None\n" if active else "",stderr="")
        assert command[0]=="sbatch";calls.append(command)
        return SimpleNamespace(returncode=0,stdout=str(100+len(calls)-1),stderr="")
    monkeypatch.setattr(p.subprocess,"run",fake_run)
    p.submit()
    receipt=json.loads((tmp_path/"reports/FULL_SUBMISSION.json").read_text())
    assert [j["stage"] for j in receipt["jobs"]]==["prepare","train","validate","eval","stats"]
    assert "--dependency=afterok:102" in calls[3] and "--array=0,1,2,3%2" in calls[3]
    assert "--dependency=afterok:103" in calls[4]
    active=True
    with pytest.raises(RuntimeError,match="active jobs"):p.submit()
    assert len(calls)==5

def test_scientific_stop_blocks_afterok_without_submitting_or_cancelling(tmp_path,monkeypatch):
    import worker as w
    (tmp_path/"reports").mkdir()
    (tmp_path/"reports/AUTO_VALIDATION.json").write_text(json.dumps({"evaluation_allowed":False}))
    called=[]
    monkeypatch.setitem(sys.modules,"analyze",SimpleNamespace(validate_training=lambda:None,stats=lambda:None,stop_report=lambda:called.append("stop")))
    monkeypatch.setattr(w,"ROOT",tmp_path);monkeypatch.setattr(w,"snapshot",lambda:{})
    monkeypatch.setattr(w.signal,"signal",lambda *args:None)
    monkeypatch.setenv("SLURM_JOB_ID","123");monkeypatch.setattr(sys,"argv",["worker","--stage","validate"])
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:pytest.fail("Worker must not call a scheduler"))
    assert w.main()==3 and called==["stop"]
    report=json.loads(next((tmp_path/"reports/WORKER_COSTS").glob("*.json")).read_text())
    assert report["status"]=="SCIENTIFIC_STOP"
    assert not (tmp_path/"reports/FAILURES").exists()
    assert "from pipeline" not in (ROOT/"code/worker.py").read_text()

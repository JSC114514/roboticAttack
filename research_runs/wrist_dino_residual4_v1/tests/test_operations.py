import json
import subprocess
import sys
from types import SimpleNamespace
import pytest
import residual4_common as common
import residual4_pipeline as pipeline


def test_default_entrypoint_never_submits(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'prepare', lambda:calls.append('prepare'))
    monkeypatch.setattr(pipeline, 'submit', lambda:pytest.fail('Automatic submission'))
    monkeypatch.setattr(sys, 'argv', ['run'])
    pipeline.main()
    assert calls == ['prepare']


def mock_submit(tmp_path, monkeypatch, previous=None):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'prepare', lambda:{'signature':'sig'})
    monkeypatch.setattr(pipeline, 'operational_hashes', lambda:{})
    monkeypatch.setattr(pipeline, 'queue', lambda:[])
    monkeypatch.setattr(pipeline, 'pending_shards', lambda sig:[0,3])
    if previous:
        common.atomic_json(previous,tmp_path/'reports/SUBMISSION.json')


def test_only_missing_shards_and_dependent_cpu_stages_under_scheduler_mock(tmp_path, monkeypatch):
    mock_submit(tmp_path,monkeypatch)
    commands = []
    def fake(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0,stdout=str(100+len(commands)),stderr='')
    monkeypatch.setattr(pipeline.subprocess,'run',fake)
    pipeline.submit()
    assert len(commands)==3 and '--array=0,3%4' in commands[1]
    assert '--dependency=afterok:101' in commands[1] and '--dependency=afterok:102' in commands[2]
    assert [r['stage'] for r in common.load_json(tmp_path/'reports/SUBMISSION.json')['jobs']]==['prepare','validate','analyze']


def test_uncertain_submission_does_not_retry(tmp_path, monkeypatch):
    mock_submit(tmp_path,monkeypatch,{'signature':'sig','status':'SUBMISSION_UNCERTAIN','jobs':[]})
    monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**k:pytest.fail('Ambiguous retry'))
    with pytest.raises(RuntimeError,match='ambiguous'):
        pipeline.submit()


def test_timeout_preserves_ambiguous_submission_without_retry(tmp_path, monkeypatch):
    mock_submit(tmp_path,monkeypatch)
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command,40)
    monkeypatch.setattr(pipeline.subprocess,'run',fake)
    with pytest.raises(RuntimeError,match='Ambiguous submission'):
        pipeline.submit()
    assert len(calls)==1
    assert common.load_json(tmp_path/'reports/SUBMISSION.json')['status']=='SUBMISSION_UNCERTAIN'


def test_completed_cases_are_not_scheduled_again(monkeypatch):
    monkeypatch.setattr(pipeline,'completed_episode',lambda *a:{'status':'COMPLETE'})
    assert pipeline.pending_shards('sig')==[]


def test_trace_integrity_is_checked_before_resume(tmp_path):
    assert common.completed_episode(tmp_path,'sig') is None
    p=tmp_path/'trace.jsonl';p.write_text('complete\n')
    common.atomic_json({'status':'COMPLETE','signature':'sig','assets':{'trace.jsonl':common.sha256(p)}},tmp_path/'COMPLETE.json')
    assert common.completed_episode(tmp_path,'sig')
    p.write_text('tampered')
    with pytest.raises(RuntimeError,match='assets changed'):
        common.completed_episode(tmp_path,'sig')

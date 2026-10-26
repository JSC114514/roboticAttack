import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import functional77_common as common
import functional77_pipeline as pipeline


def test_complete_episode_resume_verifies_assets_and_never_uses_partial_trace(tmp_path):
    folder = tmp_path / 'episode'
    (folder / 'unfinished').mkdir(parents=True)
    (folder / 'unfinished/trace.jsonl').write_text('{"partial": true}\n')
    assert common.completed_episode(folder, 'sig') is None
    path = folder / 'trace.jsonl'
    path.write_text('{"finished": true}\n')
    common.atomic_json({'status': 'COMPLETE', 'signature': 'sig', 'row': {},
                        'assets': {'trace.jsonl': common.sha256(path)}}, folder / 'COMPLETE.json')
    assert common.completed_episode(folder, 'sig')['status'] == 'COMPLETE'
    path.write_text('corrupted')
    with pytest.raises(RuntimeError, match='assets changed'):
        common.completed_episode(folder, 'sig')
    assert (folder / 'unfinished/trace.jsonl').read_text() == '{"partial": true}\n'


def test_default_and_status_entrypoints_cannot_submit(monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, 'prepare', lambda: called.append('prepare'))
    monkeypatch.setattr(pipeline, 'status', lambda: called.append('status'))
    monkeypatch.setattr(pipeline, 'submit', lambda: pytest.fail('Automatic submission'))
    monkeypatch.setattr(sys, 'argv', ['run'])
    pipeline.main()
    monkeypatch.setattr(sys, 'argv', ['run', '--status'])
    pipeline.main()
    assert called == ['prepare', 'status']


def test_failed_cpu_preflight_invalidates_previous_pass_and_records_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'preflight', lambda: {'signature': 'synthetic'})
    def fake_run(command, **kwargs):
        if command[0] == 'bash':
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        return SimpleNamespace(returncode=1, stdout='synthetic CPU test failure', stderr='')
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    common.atomic_json({'status': 'CPU_PREFLIGHT_PASS', 'signature': 'synthetic'}, tmp_path / 'reports/PREPARATION.json')
    with pytest.raises(RuntimeError, match='CPU tests failed'):
        pipeline.prepare()
    assert common.load_json(tmp_path / 'reports/CPU_PREFLIGHT.json')['status'] == 'CPU_TESTS_FAILED'
    assert 'synthetic CPU test failure' in common.load_json(tmp_path / 'reports/PREPARE_FAILED.json')['error']
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='Latest CPU preflight failed'):
        common.prepared()


def mocked_submit(tmp_path, monkeypatch, previous=None, queue=None):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'prepare', lambda: {'signature': 'sig'})
    monkeypatch.setattr(pipeline, 'operational_hashes', lambda: {})
    monkeypatch.setattr(pipeline, 'pending_shards', lambda _sig: [2, 4])
    monkeypatch.setattr(pipeline, 'queue', lambda: queue or [])
    if previous:
        common.atomic_json(previous, tmp_path / 'reports/SUBMISSION.json')


@pytest.mark.parametrize('state', ['SUBMITTING', 'SUBMISSION_UNCERTAIN'])
def test_ambiguous_submission_never_retries(tmp_path, monkeypatch, state):
    mocked_submit(tmp_path, monkeypatch, {'signature': 'sig', 'status': state, 'jobs': []})
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Retried ambiguous submission'))
    with pytest.raises(RuntimeError, match='ambiguous'):
        pipeline.submit()


def test_duplicate_active_batch_and_other_gpu_batch_never_submit(tmp_path, monkeypatch):
    mocked_submit(tmp_path, monkeypatch, {'signature': 'sig', 'status': 'SUBMITTED', 'jobs': [{'job_id': '42'}]},
                  queue=[{'job_id': '42', 'gres': 'gpu:4090:1'}])
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Duplicate allocation'))
    with pytest.raises(RuntimeError, match='already has active'):
        pipeline.submit()
    monkeypatch.setattr(pipeline, 'queue', lambda: [{'job_id': '99', 'gres': 'gpu:4090:1'}])
    with pytest.raises(RuntimeError, match='Another own GPU batch'):
        pipeline.submit()


def test_explicit_submission_constructs_dependent_missing_shards_only_under_mock(tmp_path, monkeypatch):
    mocked_submit(tmp_path, monkeypatch)
    commands = []
    def fake_run(command, **kwargs):
        assert command[0] == 'sbatch'
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(commands)) + '\n', stderr='')
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    pipeline.submit()
    assert len(commands) == 3
    assert '--array=2,4%4' in commands[1]
    assert '--dependency=afterok:101' in commands[1]
    assert '--dependency=afterok:102' in commands[2]
    assert all('--kill-on-invalid-dep=yes' in c for c in commands)
    receipt = common.load_json(tmp_path / 'reports/SUBMISSION.json')
    assert receipt['status'] == 'SUBMITTED'
    assert [r['stage'] for r in receipt['jobs']] == ['prepare', 'validate', 'analyze']


def test_timeout_does_not_repeat_or_cancel_uncertain_jobs(tmp_path, monkeypatch):
    mocked_submit(tmp_path, monkeypatch)
    calls = []
    def timeout(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 40)
    monkeypatch.setattr(pipeline.subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='Ambiguous submission'):
        pipeline.submit()
    assert len(calls) == 1
    receipt = common.load_json(tmp_path / 'reports/SUBMISSION.json')
    assert receipt['status'] == 'SUBMISSION_UNCERTAIN' and receipt['jobs'] == []


def test_completed_shards_do_not_request_another_gpu(monkeypatch):
    monkeypatch.setattr(pipeline, 'completed_episode', lambda *args: {'status': 'COMPLETE'})
    assert pipeline.pending_shards('sig') == []


def test_status_is_read_only_and_does_not_contact_scheduler(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'queue', lambda: pytest.fail('Status contacted scheduler'))
    pipeline.status()
    value = json.loads(capsys.readouterr().out)
    assert value['submission'] == 'NOT_SUBMITTED' and not list(tmp_path.iterdir())


def test_status_keeps_full_escaped_traceback_in_file_and_displays_short_reason(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    error = repr(RuntimeError('CPU tests failed:\n' + 'synthetic traceback line\n' * 100))
    path = tmp_path / 'reports/PREPARE_FAILED.json'
    common.atomic_json({'error': error}, path)
    pipeline.status()
    value = json.loads(capsys.readouterr().out)
    assert len(value['prepare_failure']['error']) < 50
    assert 'traceback line' not in value['prepare_failure']['error']
    assert common.load_json(path)['error'] == error

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import pilot_common as common
import pilot_pipeline as pipeline


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path); monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'preflight',lambda:{'signature':'abcdef0123456789'})
    monkeypatch.setattr(pipeline,'config',lambda:{'resources':{'test':True}})
    monkeypatch.setattr(pipeline,'remaining',lambda prep:list(pipeline.STAGES))
    common.directories()
    return tmp_path


def test_gpu_dependencies_parse_arrays_and_ignore_cpu_jobs():
    output='123_[1-4]|PENDING|gpu:4090:1|Resources|old\n124|RUNNING|N/A|node|cpu\n'
    assert pipeline.gpu_dependencies(output)==['123']
    with pytest.raises(RuntimeError,match='failed dependency'):
        pipeline.gpu_dependencies('125|PENDING|gpu:4090:1|DependencyNeverSatisfied|bad')


def test_first_gpu_depends_on_preparation_and_existing_gpus():
    command=pipeline.command_for('mechanism','abcdef01',previous='100',existing=['90','91'])
    assert '--dependency=afterany:100:90:91' in command
    assert not any('srun' in value or 'scancel' in value for value in command)


def test_submission_is_serial_and_can_only_submit_declared_stages(isolated,monkeypatch):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='squeue': return SimpleNamespace(returncode=0,stdout='77|RUNNING|gpu:4090:1|node|old\n',stderr='')
        return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+';cluster\n',stderr='')
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    pipeline.submit()
    submitted=calls[1:]
    assert len(submitted)==4 and all(c[0]=='sbatch' for c in submitted)
    assert '--dependency=afterany:102:77' in submitted[1]
    assert '--dependency=afterany:103' in submitted[2]
    assert '--dependency=afterany:104' in submitted[3]
    receipt=json.loads((isolated/'reports/SUBMISSION.json').read_text())
    assert receipt['status']=='SUBMITTED' and len(receipt['jobs'])==4


def test_active_chain_blocks_duplicates_before_sbatch(isolated,monkeypatch):
    common.atomic_json({'status':'SUBMITTED','signature':'abcdef0123456789','jobs':[{'job_id':'80'}]},isolated/'reports/SUBMISSION.json')
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        assert command[0]=='squeue'
        return SimpleNamespace(returncode=0,stdout='80|RUNNING|gpu:4090:1|node|own\n',stderr='')
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='active'): pipeline.submit()
    assert len(calls)==1


def test_unavailable_scheduler_never_submits(isolated,monkeypatch):
    calls=[]
    def run(command,**kwargs):
        calls.append(command); return SimpleNamespace(returncode=1,stdout='',stderr='controller unavailable')
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='unavailable'): pipeline.submit()
    assert len(calls)==1 and not (isolated/'reports/SUBMISSION.json').exists()


def test_ambiguous_sbatch_response_is_recorded_and_not_retried(isolated,monkeypatch):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='squeue': return SimpleNamespace(returncode=0,stdout='',stderr='')
        raise TimeoutError('unknown whether controller accepted')
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='not automatically'): pipeline.submit()
    value=json.loads((isolated/'reports/SUBMISSION.json').read_text())
    assert value['status']=='AMBIGUOUS_SBATCH_RESPONSE' and len(calls)==2


def test_dry_run_does_not_call_any_scheduler_or_prepare_mutations(isolated,monkeypatch):
    monkeypatch.setattr(pipeline,'signature',lambda:'abcdef0123456789')
    monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**k:pytest.fail('scheduler invoked'))
    monkeypatch.setattr(pipeline,'preflight',lambda:pytest.fail('dry run mutated preparation'))
    monkeypatch.setattr('sys.argv',['pipeline','--dry-run'])
    pipeline.main()
    assert not (isolated/'reports/SUBMISSION.json').exists()


def test_already_complete_does_not_query_slurm(isolated,monkeypatch):
    monkeypatch.setattr(pipeline,'remaining',lambda prep:[])
    monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**k:pytest.fail('scheduler invoked'))
    pipeline.submit()


def test_failed_prerequisite_rejects_before_importing_engine(tmp_path,monkeypatch):
    import pilot_worker
    monkeypatch.setattr(pilot_worker,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='Upstream stage'): pilot_worker.prerequisites('pilot','signature')


def test_resource_requests_match_fixed_budget():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'config.json').read_text())
    for index,stage in enumerate(pipeline.STAGES,1):
        script=(root/f'slurm/{index:02d}_{stage}.sbatch').read_text()
        resource=c['resources'][stage]
        assert f'--cpus-per-task={resource["cpus"]}' in script
        assert f'--mem={resource["host_memory_gib"]}G' in script
        assert f'--time={resource["time"]}' in script
        if resource['gpus']: assert '--gres=gpu:4090:1' in script
        else: assert '--gres=' not in script

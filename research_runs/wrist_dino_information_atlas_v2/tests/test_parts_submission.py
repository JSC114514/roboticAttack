import json
from types import SimpleNamespace
import numpy as np
import pytest
import atlas_pipeline as pipe
from atlas_parts import validate_annotations


def example_annotations():
    manifest={'signature':'abc','images':[{'id':str(i)} for i in range(80)]}
    value={'signature':'abc','annotations':[{'id':str(i),'reviewed':True,'polygons':[],
                                            'unobservable':list(range(1,7))} for i in range(80)]}
    return value,manifest


def test_explicit_unobservable_is_not_background_negative():
    value,manifest=example_annotations(); masks=validate_annotations(value,manifest)
    assert len(masks)==80 and all(not m.any() for m in masks.values())


@pytest.mark.parametrize('kind',['unreviewed','duplicate','missing','out_of_frame','contradiction'])
def test_bad_annotations_are_rejected(kind):
    value,manifest=example_annotations()
    if kind=='unreviewed': value['annotations'][0]['reviewed']=False
    if kind=='duplicate': value['annotations'][1]['id']='0'
    if kind=='missing': value['annotations'][0]['unobservable']=[]
    if kind in ('out_of_frame','contradiction'):
        value['annotations'][0]['polygons']=[{'class_id':1,'points':[[0,0],[20,0],[20,300 if kind=='out_of_frame' else 20]]}]
    with pytest.raises(RuntimeError): validate_annotations(value,manifest)


def test_gpu_dependencies_parse_arrays_without_double_counting():
    queue='123_1|RUNNING|gpu:4090:1|None|old\n123_2|PENDING|gpu:4090:1|Resources|old\n124|RUNNING|N/A|None|cpu\n'
    assert pipe.gpu_dependencies(queue)==['123']


def test_submission_commands_only_one_gpu_then_cpu():
    a=pipe.command_for('capture','abcdef',existing=['123'])
    b=pipe.command_for('analyze','abcdef',previous='456')
    assert '--dependency=afterany:123' in a and '--dependency=afterany:456' in b
    assert not any('srun' in s or 'scancel' in s for s in a+b)


def setup_submit(monkeypatch,tmp_path):
    monkeypatch.setattr(pipe,'ROOT',tmp_path)
    monkeypatch.setattr(pipe,'preflight',lambda:{'signature':'abc'})
    monkeypatch.setattr(pipe,'receipt_valid',lambda *a:False)
    pipe.directories()


def test_submission_creates_logs_before_queue_and_sbatch(monkeypatch,tmp_path):
    setup_submit(monkeypatch,tmp_path); calls=[]
    def run(command,**kwargs):
        assert (tmp_path/'logs').is_dir(); calls.append(command)
        return SimpleNamespace(returncode=0,stdout='' if command[0]=='squeue' else str(100+len(calls)),stderr='')
    monkeypatch.setattr(pipe.subprocess,'run',run); pipe.submit()
    assert [c[0] for c in calls]==['squeue','sbatch','sbatch']
    receipt=json.loads((tmp_path/'reports/SUBMISSION.json').read_text())
    assert receipt['status']=='SUBMITTED' and len(receipt['jobs'])==2


def test_ambiguous_submission_never_retries_or_cancels(monkeypatch,tmp_path):
    setup_submit(monkeypatch,tmp_path); calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='sbatch': raise TimeoutError('test ambiguity')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(pipe.subprocess,'run',run)
    with pytest.raises(RuntimeError): pipe.submit()
    assert json.loads((tmp_path/'reports/SUBMISSION.json').read_text())['status']=='AMBIGUOUS_SBATCH_RESPONSE'
    with pytest.raises(RuntimeError): pipe.submit()
    assert sum(c[0]=='sbatch' for c in calls)==1

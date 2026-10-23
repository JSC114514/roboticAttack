import numpy as np
import pytest
import pilot_common as common
from pilot_analysis import interval
from pilot_readouts import linear_coefficients,predict,pooled


def test_ridge_serialized_weights_reproduce_training_only_solution():
    from atlas_probes import ridge_path
    rng=np.random.default_rng(42)
    x=rng.normal(size=(35,50)); y=x[:,:3]+rng.normal(size=(35,3))*.1; test=rng.normal(size=(5,50))
    alphas=[.01,1.,10.]
    fitted=linear_coefficients(x,y,alphas)
    expected=ridge_path(x,y,test,alphas)[:,np.arange(3),np.arange(3)]
    assert np.allclose(predict(fitted,test),expected,atol=1e-9)
    altered_validation=test+100
    other=linear_coefficients(x,y,alphas)
    assert np.array_equal(fitted['coefficients'],other['coefficients'])
    assert not np.array_equal(predict(fitted,test),predict(fitted,altered_validation))


def test_grid_pool_preserves_spatial_cell_order():
    x=np.arange(256,dtype=np.float32)[:,None].repeat(3,axis=1)
    actual=pooled(x).reshape(4,4,3)
    assert actual[0,0,0]==25.5 and actual[3,3,0]==229.5


def test_eot_repeats_do_not_inflate_independent_episode_count_or_change_interval():
    rows=[{'task_id':t,'episode_id':f'{t}_{e}'} for t in range(2) for e in range(2)]
    values=[1.,2.,3.,5.]
    a=interval(values,rows)
    b=interval(np.repeat(values,5),[row for row in rows for _ in range(5)])
    assert a==b and a['episodes']==4 and a['tasks']==2


def test_nested_receipt_detects_changed_data_not_only_changed_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path)
    artifact=tmp_path/'data.json'; common.atomic_json({'a':1},artifact)
    child=tmp_path/'child/COMPLETE.json'; parent=tmp_path/'COMPLETE.json'
    common.commit(child,[artifact],'id'); common.commit(parent,[child],'id')
    assert common.complete(parent,'id')
    common.atomic_json({'a':2},artifact)
    with pytest.raises(RuntimeError,match='Committed artifact'): common.complete(parent,'id')


def test_receipts_refuse_mixed_run_identities(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path)
    common.atomic_json({'x':0},tmp_path/'x.json'); common.commit(tmp_path/'COMPLETE.json',[tmp_path/'x.json'],'first')
    with pytest.raises(RuntimeError,match='identity'): common.complete(tmp_path/'COMPLETE.json','second')

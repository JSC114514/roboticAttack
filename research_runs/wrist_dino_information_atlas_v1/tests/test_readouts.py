import numpy as np
import pytest
from atlas_probes import fit_ridge,local_ridge,metrics,ridge_path


def records(n):
    return [{'task_id':i%2,'episode_id':f'e{i//2}','step_id':i} for i in range(n)]


def test_ridge_recovers_signal_on_new_data_with_missing_targets():
    rng=np.random.default_rng(2); x=rng.normal(size=(60,4)); z=rng.normal(size=(15,4))
    w=rng.normal(size=(4,2)); y=x@w+3; y[::3,1]=np.nan
    p=ridge_path(x,y,z,[1e-6])[:,:,0]
    assert np.max(abs(p-(z@w+3)))<1e-4


def test_constant_baseline_and_unsupported_labels():
    y=np.c_[np.arange(10),np.full(10,np.nan)]
    p=ridge_path(np.zeros((10,0)),y,np.zeros((3,0)),[1])[:,:,0]
    assert np.allclose(p[:,0],4.5) and np.isnan(p[:,1]).all()


def test_cv_grouping_fit_has_no_test_targets():
    rng=np.random.default_rng(3); x=rng.normal(size=(60,3)); z=rng.normal(size=(7,3)); y=x[:,:1]*2
    r=[{'task_id':i//30,'episode_id':f'e{i//5}','step_id':i%5} for i in range(60)]
    p,fit=fit_ridge(x,y,z,r,[1e-4,.1,100])
    assert p.shape==(7,1) and fit['alpha'][0]==1e-4 and fit['cv_episode_grouped']
    assert np.max(abs(p[:,0]-z[:,0]*2))<.01


def test_primal_token_ridge_streams_prediction_and_preserves_targets():
    rng=np.random.default_rng(5); x=rng.normal(size=(80,4)); z=rng.normal(size=(2050,4)); y=x[:,:2]+4
    p,fit=local_ridge(x,y,z,[1e-5,.1],np.arange(80)%5,np.ones(80))
    assert p.shape==(2050,2) and np.max(abs(p-(z[:,:2]+4)))<1e-4


def test_binary_auc_ties_and_missingness():
    assert metrics(np.array([0,1]),np.array([.5,.5]),True)['auroc']==.5
    assert metrics(np.array([0,1]),np.array([0,1]),True)['auroc']==1
    assert metrics(np.array([1,1]),np.array([0,1]),True)['auroc'] is None
    assert metrics(np.array([np.nan]),np.array([0]),True)['n']==0


def test_counterfactual_feature_inputs_not_part_of_public_fit_api():
    import inspect
    assert list(inspect.signature(fit_ridge).parameters)==['x','y','test','rows','alphas']

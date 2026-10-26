"""Scientific contracts: actual decoder, null controls, deployment pairing, budgets."""
from pathlib import Path
import sys
import json
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from rr_math import Readout,relation_loss,gate_decision,learning_rate,sample_order
torch.set_num_threads(2)


@pytest.fixture(scope='module')
def decoder():
    return Readout(ROOT.parent/'wrist_dino_geometry_pilot_v1/readouts/b22.npz')


def test_actual_decoder_matches_independent_numpy_pool(decoder):
    p=ROOT.parent/'wrist_dino_geometry_rgb2_v1_fix1/data/cache/libero_spatial_t00_s40.pt'
    row=torch.load(p,map_location='cpu',weights_only=False)['records'][1]
    tokens=row['target_features']['translation_plus'].float()
    raw=tokens.numpy().reshape(1,16,16,1024)
    h=raw.reshape(1,4,4,4,4,1024).mean((2,4)).reshape(1,-1).astype(np.float16).astype(float)
    with np.load(ROOT.parent/'wrist_dino_geometry_pilot_v1/readouts/b22.npz',allow_pickle=False) as z:
        expected=((h-z['mean'])/z['scale'])@z['coefficients'][:,:3]+z['intercept'][:3]
    np.testing.assert_allclose(decoder(tokens).numpy(),expected,atol=1e-5,rtol=0)


def test_raw_feature_minimum_norm_pseudoinverse_and_null_controls(decoder):
    delta=torch.tensor([.06,0.,0.],dtype=torch.float64)
    dense=decoder.intervention(delta)
    grid=decoder.pool(dense,rounded=False)
    torch.testing.assert_close((grid@decoder.weight)[0],delta,atol=1e-9,rtol=0)
    # Every pixel-space null control has matching dense norm and no XYZ readout.
    for seed in range(42,46):
        control=decoder.random_null(dense.norm(),seed)
        torch.testing.assert_close(control.norm(),dense.norm(),atol=1e-9,rtol=0)
        zero=decoder.pool(control,rounded=False)@decoder.weight
        torch.testing.assert_close(zero,torch.zeros_like(zero),atol=1e-9,rtol=0)
        assert (dense+control).norm()>=dense.norm()
        assert abs(float((dense*control).sum()))<1e-8


def test_expand_and_pool_are_inverse_in_continuous_space(decoder):
    grid=torch.randn(16384,dtype=torch.float64)
    torch.testing.assert_close(decoder.pool(decoder.expand(grid),rounded=False)[0],grid)


def test_frozen_relation_loss_reaches_dense_feature_gradient(decoder):
    feature=torch.randn((1,256,1024),requires_grad=True)
    clean=decoder(feature.detach())
    delta=torch.tensor([.03,0.,0.],dtype=torch.float64)
    loss=relation_loss(decoder(feature),clean,delta)
    assert float(loss.detach())==pytest.approx(1.)
    gradient=torch.autograd.grad(loss,feature)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    assert not decoder.weight.requires_grad and decoder.weight.grad is None


@pytest.mark.parametrize('rows,expected',[
    ([],False),
    ([{'applicable':False}],False),
    ([{'applicable':True,'relation_score':0.,'random_score':-1.}],False),
    ([{'applicable':True,'relation_score':1.,'random_score':1.}],False),
    ([{'applicable':True,'relation_score':1.,'random_score':.5}],True),
    ([{'applicable':True,'relation_score':-1.,'random_score':-2.}],False),
])
def test_gate_is_fixed_direction_and_advantage_not_loss(rows,expected):
    assert gate_decision(rows)['passed'] is expected


def test_candidate_schedule_and_order_resume_exactly():
    def sequence(start,end):
        return [(sample_order(60,42,n//60)[n%60],learning_rate(n+1)) for n in range(start,end)]
    assert sequence(0,2000)==sequence(0,1000)+sequence(1000,2000)
    assert learning_rate(20)==pytest.approx(.002)
    assert learning_rate(2000)==pytest.approx(0.)


def test_legacy_clean_pointer_does_not_hide_condition_fingerprint_mismatch(tmp_path):
    from rr_evaluate import baseline_arrays_valid,fingerprints
    current={k:np.zeros((3,),dtype=np.float32) for k in ('sim_state','front','wrist','proprio')}
    start=tmp_path/'START.npz';np.savez_compressed(start,**current)
    hashes=fingerprints(current)
    meta={'start_sim_sha256':hashes['sim_state'],'start_front_sha256':hashes['front'],
          'start_wrist_sha256':hashes['wrist'],'start_proprio_sha256':hashes['proprio']}
    path=tmp_path/'EPISODE.json';path.write_text(json.dumps(meta))
    entry={'eligible':True,'conditions':{'initial':{'meta':str(path),'start':str(start)}}}
    assert baseline_arrays_valid(entry,current)[0]
    meta['start_wrist_sha256']='different_condition_start';path.write_text(json.dumps(meta))
    assert not baseline_arrays_valid(entry,current)[0]
    # Clean's own sealed arrays are sufficient; unlike patched conditions,
    # its historical metadata need not duplicate those hashes as strings.
    path.write_text('{}')
    entry['conditions']={'clean':{'meta':str(path),'start':str(start)}}
    assert baseline_arrays_valid(entry,current)[0]


def test_resource_contract_and_no_hidden_extra_gpu():
    for stage,mem,gpu in [('gate','12G',True),('train','16G',True),('evaluate','12G',True),('report','4G',False),('interim','4G',False)]:
        text=(ROOT/f'slurm/{stage}.sbatch').read_text()
        assert '#SBATCH --mem='+mem in text
        assert ('#SBATCH --gres=gpu:4090:1' in text)==gpu
        assert '#SBATCH --cpus-per-task='+str(4 if gpu else 2) in text
    from rr_cli import command
    assert '--array=0,1%2' in command('evaluate',['123'])
    assert '--dependency=afterany:123' in command('evaluate',['123'])


def test_pending_interim_does_not_block_main_resume_or_duplicate_interim(monkeypatch):
    import rr_cli
    monkeypatch.setattr(rr_cli,'signature',lambda:'12345678abcdef')
    assert rr_cli.active_jobs('1|PENDING|rrp1_12345678_interim|BeginTime')==(False,True)
    assert rr_cli.active_jobs('2|RUNNING|rrp1_12345678_gate|node')==(True,False)
    assert rr_cli.active_jobs('3|PENDING|rrp1_oldrev00_interim|BeginTime')==(True,False)
    assert rr_cli.active_jobs('4|RUNNING|other_experiment|node')==(False,False)


@pytest.mark.parametrize('stopped',[True,False])
def test_reports_preserve_negative_gate_and_exclude_missing_pairs(tmp_path,monkeypatch,stopped):
    import rr_report as report
    config={'eval_tasks':[0,1],'eval_states':[44,45],
            'conditions':['clean','initial','uada','relation'],'bootstrap_draws':40}
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(report,'ROOT',tmp_path)
    monkeypatch.setattr(report,'config',lambda:config)
    monkeypatch.setattr(report,'signature',lambda:'test')
    monkeypatch.setattr(report,'complete',lambda p:Path(p).exists())
    monkeypatch.setattr(report,'episode_path',lambda t,s,k:tmp_path/f'episodes/{t}_{s}_{k}/COMPLETE.json')
    monkeypatch.setattr(report,'finish',lambda p,artifacts,**kw:report.atomic_json(kw,p))
    class Monitor:
        def update(self,**fields):pass
    if stopped:
        report.atomic_json({'passed':False,'applicable':12,'relation_median':-.1,
                            'advantage_median':-.01},tmp_path/'reports/DIRECTION_COMPLETE.json')
    else:
        for condition,failure in [('clean',0),('initial',0),('uada',1),('relation',1)]:
            receipt=report.episode_path(0,44,condition)
            report.atomic_json({},receipt)
            report.atomic_json({'success':1-failure,'failure':failure},receipt.parent/'EPISODE.json')
        # An orphan new-patch failure has no matched clean/reference: excluded.
        receipt=report.episode_path(1,45,'relation');report.atomic_json({},receipt)
        report.atomic_json({'success':0,'failure':1},receipt.parent/'EPISODE.json')
    report.run(Monitor())
    status=report.read_json(tmp_path/'results/STATUS.json')
    if stopped:
        assert status['status']=='STOPPED_DIRECTION_NOT_SUPPORTED'
        assert status['clean_success_starts']==0
        assert all(row['casr'] is None for row in status['casr'])
    else:
        assert status['status']=='PARTIAL_OR_PENDING'
        assert status['fully_paired_starts']==1 and status['episodes']==5
        assert status['casr'][-1]['failures']==1 and status['casr'][-1]['clean_successes']==1

import numpy as np
import atlas_local as local


def test_local_controls_use_native_tokens_and_keep_unknown_pixels_out(monkeypatch,tmp_path):
    monkeypatch.setattr(local,'progress',lambda *a,**k:None)
    monkeypatch.setattr(local,'config',lambda:{'seed':42,'probe':{'ridge_alphas':[.001,.1],'shuffle_repeats':2}})
    def row(task,episode,split):
        return {'task_id':task,'episode_id':f'{split}_{task}_{episode}','step_id':0,'state_id':f'{split}_{task}_{episode}',
                'visibility':{v:{'subject_visible':True,'subject_occlusion_proxy':False} for v in ('front','wrist')},
                'truth':{'contact':0,'grasp':0}}
    train=[row(t,e,'discovery') for t in range(2) for e in range(4)]
    test=[row(t,e,'validation') for t in range(2) for e in range(2)]
    rng=np.random.default_rng(22)
    tx=rng.normal(size=(len(train),256,3)); zx=rng.normal(size=(len(test),256,3))
    labels=(tx[:,:,0]>0).astype(int); truth=(zx[:,:,0]>0).astype(int)
    full=np.eye(2)[labels]; full[:,100:]=0 # Unlabelled pixels are unknown, never negatives.
    owner=np.repeat(np.arange(len(train)),32); ids=np.tile(np.arange(32),len(train))
    summary=local.audit_tokens(tmp_path,'front__b22',tx[owner,ids],full,owner,ids,zx.reshape(-1,3),train,test,
                               truth,np.ones_like(truth,dtype=bool),['a','b'],np.ones(len(owner)))
    assert any(r['metric']=='token_accuracy' and r['mean']>.8 for r in summary)
    with np.load(tmp_path/'PREDICTIONS.npz') as p:
        assert p['shuffle_class_prediction'].shape==(2,len(test),256)
        assert p['loto_class_prediction'].shape==(len(test),256)
        assert p['valid_tokens'].all()
    import json
    r=json.loads((tmp_path/'FIT.json').read_text())
    assert r['truth_regions_not_inputs'] and r['train_images']==8
    assert r['main']['per_class_iou'][0]['true_tokens']>0


def test_missing_loto_predictions_do_not_count_as_background():
    rows=[{'task_id':0,'episode_id':'a'}]
    p=np.full((256,2),np.nan); truth=np.zeros((1,256),int); valid=np.zeros_like(truth,dtype=bool)
    summary,values,_,conf=local.token_metrics(p,truth,valid,rows,['a','b'])
    assert summary['token_accuracy']['mean'] is None and not conf.any()

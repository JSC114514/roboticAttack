import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import pilot_common as common
import pilot_train as train


def test_interruption_resume_restores_adam_and_shared_patch_exactly(tmp_path,monkeypatch):
    cfg={**common.config(),'updates':4,'warmup_updates':1,'checkpoints':[0,2,4],'resume_interval':2}
    path=tmp_path/'features.npz'
    np.savez(path,clean__dino_b16=np.zeros((256,4),np.float32),
             direction_pos_large__dino_b16=np.ones((256,4),np.float32)*.9)
    monkeypatch.setattr(train,'config',lambda:cfg)
    monkeypatch.setattr(train,'features_file',lambda row:path)
    monkeypatch.setattr(train,'observation',lambda row:({},'pick'))
    monkeypatch.setattr(train,'pixels',lambda wrapper,obs,patch,seed,diverse:(patch,None,{'seed':seed}))
    monkeypatch.setattr(train,'progress',lambda *a,**k:None)
    rows=[{'state_id':str(i),'split':'discovery'} for i in range(20)]
    candidate={'candidate_id':'direction_pos_b16_s42','target':'direction_pos','layer':16,'seed':42}
    class FakeEngine:
        def __init__(self,fail=None):
            self.wrapper=SimpleNamespace(device=torch.device('cpu'),vla=torch.nn.Identity())
            self.calls=0; self.fail=fail
        def spatial(self,patch,layer):
            self.calls+=1
            if self.calls==self.fail: raise RuntimeError('interrupted')
            return patch.mean().expand(1,256,4)
    def run_root(folder):
        folder.mkdir(); monkeypatch.setattr(common,'ROOT',folder); monkeypatch.setattr(train,'ROOT',folder)
        (folder/'config.json').write_text(json.dumps(cfg))
    run_root(tmp_path/'full')
    train.train_candidate(candidate,rows,FakeEngine(),'signature')
    full=torch.load(tmp_path/'full/training'/candidate['candidate_id']/'LAST.pt',weights_only=False)
    run_root(tmp_path/'resume')
    with pytest.raises(RuntimeError,match='interrupted'):
        train.train_candidate(candidate,rows,FakeEngine(fail=3),'signature')
    saved=torch.load(tmp_path/'resume/training'/candidate['candidate_id']/'LAST.pt',weights_only=False)
    assert saved['step']==2 and saved['optimizer']['state'][0]['exp_avg'].norm()>0
    train.train_candidate(candidate,rows,FakeEngine(),'signature')
    resumed=torch.load(tmp_path/'resume/training'/candidate['candidate_id']/'LAST.pt',weights_only=False)
    assert torch.equal(full['patch'],resumed['patch'])
    for key in ('exp_avg','exp_avg_sq','step'):
        assert torch.equal(full['optimizer']['state'][0][key],resumed['optimizer']['state'][0][key])
    assert resumed['optimized']==4 and resumed['no_target']==0

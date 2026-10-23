import json
import numpy as np
import pytest
import atlas_common as common
import atlas_report as report
from atlas_labels import label_specs


def test_input_and_committed_artifact_changes_block_reuse(monkeypatch,tmp_path):
    monkeypatch.setattr(common,'ROOT',tmp_path); monkeypatch.setattr(common,'freeze_id',lambda:'abc')
    source=tmp_path/'source.npz'; source.write_bytes(b'original')
    common.atomic_json({'signature':'abc','input_hashes':{str(source):common.sha256_file(source)}},tmp_path/'reports/PREFLIGHT.json')
    assert common.prepared(verify_inputs=True)['signature']=='abc'
    common.commit(tmp_path/'COMPLETE.json',[source],'abc')
    assert common.receipt_valid(tmp_path/'COMPLETE.json','abc')
    source.write_bytes(b'changed')
    with pytest.raises(RuntimeError): common.prepared(verify_inputs=True)
    with pytest.raises(RuntimeError): common.receipt_valid(tmp_path/'COMPLETE.json','abc')


def test_report_builds_from_synthetic_predictions_without_a_model(monkeypatch,tmp_path):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    rows=[]; specs=label_specs(); labels=['object_eef_distance','gripper_opening','object_eef_relative_x',
                                        'object_eef_relative_y','contact','grasp']
    for t in (0,4,9):
        row={'state_id':f's{t}','episode_id':f'e{t}','split':'validation','task_id':t,
             'truth':{'contact':0,'grasp':0},'visibility':{v:{'subject_visible':True} for v in ('front','wrist')}}
        rows.append(row); folder=tmp_path/f'data/task_{t:02d}/states'; folder.mkdir(parents=True)
        np.savez(folder/f's{t}.npz',**{v+k:np.zeros((224,224,3),np.uint8) if k=='_rgb' else np.zeros(256,int)
                                    for v in ('front','wrist') for k in ('_rgb','_class_tokens')})
    global_rows=[]
    for view in ('front','wrist'):
        folder=tmp_path/f'readouts/local/{view}__b22'; folder.mkdir(parents=True)
        np.savez(folder/'PREDICTIONS.npz',class_prediction=np.zeros((3,256),int))
        folder=tmp_path/f'readouts/global/{view}__consumed'; folder.mkdir(parents=True)
        truth=np.ones((3,len(specs)))*.1
        np.savez(folder/'PREDICTIONS.npz',truth=truth,prediction=truth+.03)
        for site in [f'{view}__b{b:02d}_spatial' for b in range(24)]+[f'{view}__consumed']:
            for label in labels:
                global_rows.append({'site':site,'label':label,'condition':'all','r2':.1,'auroc':.5,'mae':.03})
    common.atomic_json({'labels':[]},tmp_path/'results/INVENTORY.json')
    report.write_report(rows,global_rows,[],[],[])
    assert (tmp_path/'results/REPORT.html').exists()
    assert len(list((tmp_path/'results/figures').glob('*.png')))==7
    failures=json.loads((tmp_path/'results/FAILURE_CASES.json').read_text())
    assert len(failures)==6 and all(f['error_m']==pytest.approx(.03) for f in failures)
    assert '尚待明确核对' in (tmp_path/'results/RESULTS.md').read_text()

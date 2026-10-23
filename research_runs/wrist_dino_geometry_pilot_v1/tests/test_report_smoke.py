"""Synthetic artifacts only: exercise joins, paired statistics and all report figures."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import pilot_analysis as analysis
import pilot_common as common
from pilot_geometry import action_metrics,candidates,conditions


def test_complete_report_from_synthetic_artifacts_without_vla_or_scheduler(tmp_path,monkeypatch):
    cfg={**common.config(),'bootstrap_draws':30}
    monkeypatch.setattr(common,'ROOT',tmp_path); monkeypatch.setattr(analysis,'ROOT',tmp_path)
    monkeypatch.setattr(analysis,'config',lambda:cfg)
    monkeypatch.setattr(analysis,'complete',lambda *a:True)
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    rows=[]; grid=candidates(cfg); clean=np.zeros((8,7)); clean[:,6]=.4
    rgb=np.full((224,224,3),100,np.uint8)
    monitor_clean=np.array([.04,.03,-.04,np.sqrt(.04**2+.03**2+.04**2)])
    branches=[('native_cf_rgb',22,1.),('zero',22,0.),('zero_render_rgb',22,0.),('siglip_only',None,.4),
              ('dino_only',16,.6),('both',16,1.),('random_dino',16,.02),
              ('dino_only',22,.6),('both',22,1.),('random_dino',22,.02)]
    for task in range(10):
        for split in ('discovery','validation'):
            for episode in range(2):
                state=f't{task}_{split}_{episode}'
                source=tmp_path/'synthetic_source'/f'{state}.npz'; source.parent.mkdir(exist_ok=True)
                np.savez(source,front_rgb=rgb,wrist_rgb=rgb,proprio=np.zeros(8),language=np.asarray('synthetic test only'))
                truth=dict(zip(('object_eef_relative_x','object_eef_relative_y','object_eef_relative_z'),monitor_clean[:3]))
                truth['object_eef_distance']=float(monitor_clean[3])
                row={'state_id':state,'episode_id':'ep_'+state,'task_id':task,'split':split,
                     'truth':truth,'state_file':str(source),'conditions':conditions(truth,cfg)}
                rows.append(row); base={k:row[k] for k in ('state_id','episode_id','task_id','split')}
                features=tmp_path/'data/features'/state
                meta={'actions':{'clean':clean.tolist()},'monitor_predictions_m':{'clean':{'16':monitor_clean.tolist(),'22':monitor_clean.tolist()}}}
                render_frames={'clean__rgb':rgb}; render_meta={'clean':{'subject_visible':True}}
                mechanism=[]
                for condition in row['conditions']:
                    name=condition['condition_id']; action=clean.copy(); action[0,:3]=np.asarray(condition['translation_m'])*2
                    target=np.r_[condition['relative_position_m'],condition['distance_m']]
                    meta['actions'][name]=action.tolist(); meta['monitor_predictions_m'][name]={'16':target.tolist(),'22':target.tolist()}
                    render_frames[name+'__rgb']=rgb.copy(); render_meta[name]={'subject_visible':True}
                    for branch,layer,factor in branches:
                        modified=clean+(action-clean)*factor
                        mechanism.append({**base,'condition_id':name,'target':condition['target'],'strength':condition['strength'],
                                          'sign':condition['sign'],'branch':branch,'layer':layer,'cf_subject_visible':True,
                                          'clean_subject_visible':True,'actions':modified.tolist(),'clean_actions':clean.tolist(),
                                          **action_metrics(modified,clean,condition)})
                common.atomic_json(meta,features/'METADATA.json')
                common.atomic_json(mechanism,tmp_path/'data/mechanism'/state/'ROWS.json')
                common.atomic_npz(tmp_path/'data/renders'/state/'FRAMES.npz',**render_frames)
                common.atomic_json({'conditions':render_meta},tmp_path/'data/renders'/state/'METADATA.json')
                if split!='validation': continue
                baselines=[]
                for baseline in ('uada500_b1','initial_s42','initial_s43'):
                    for draw in ('fixed','eot1','eot2','eot3','eot4'):
                        baselines.append({**base,'baseline':baseline,'draw':draw,'actions':clean.tolist(),
                                          'target_feature_distances':{f'{target}|{layer}':1. for target in ('direction_pos','direction_neg','distance_near','distance_far') for layer in (16,22)}})
                common.atomic_json(baselines,tmp_path/'evaluation/baselines'/state/'ROWS.json')
                for candidate in grid:
                    condition=next(c for c in row['conditions'] if c['condition_id']==candidate['target']+'_large')
                    target=np.r_[condition['relative_position_m'],condition['distance_m']]
                    action=clean.copy(); action[0,:3]=condition['translation_m']
                    predictions={'16':((target+monitor_clean)/2).tolist(),'22':((target+monitor_clean)/2).tolist()}
                    evaluations=[]
                    for draw in ('fixed','eot1','eot2','eot3','eot4'):
                        entry={**base,**candidate,'draw':draw,'actions':action.tolist(),'monitor_predictions_m':predictions,
                               'feature_target_distance':.5,'cf_subject_visible':True,**action_metrics(action,clean,condition)}
                        if draw=='fixed':
                            entry['branch_restorations']={branch:action_metrics(clean+(action-clean)*factor,clean,condition)
                                                          for branch,factor in [('restore_dino',.2),('restore_siglip',.8)]}
                        evaluations.append(entry)
                    common.atomic_json(evaluations,tmp_path/'evaluation'/candidate['candidate_id']/state/'ROWS.json')
    for candidate in grid:
        folder=tmp_path/'training'/candidate['candidate_id']; folder.mkdir(parents=True)
        Image.fromarray(np.full((50,50,3),candidate['seed'],np.uint8)).save(folder/'step_1000.png')
    prep={'signature':'synthetic-test-only','states':rows,'candidates':grid}
    analysis.analyze(prep)
    status=json.loads((tmp_path/'results/STATUS.json').read_text())
    assert status['status']=='COMPLETE' and status['closed_loop_casr_measured'] is False
    summary=pd.read_csv(tmp_path/'results/PNG_SUMMARY.csv')
    fixed=summary[(summary.draw_group=='fixed')&(summary.visibility_subset=='all')]
    assert len(fixed)==16 and (fixed.episodes==20).all() and (fixed.tasks==10).all()
    assert np.allclose(fixed.feature_target_improvement_vs_clean_mean,.5)
    assert np.allclose(fixed.projection_minus_initial_mean,fixed.first_projection_mean)
    assert len(list((tmp_path/'results/figures').glob('*.png')))==4
    cases=json.loads((tmp_path/'results/WEAK_RESPONSE_CASES.json').read_text())
    assert len(cases['cases'])==6
    report=(tmp_path/'results/RESULTS.md').read_text()
    assert '没有测量闭环' in report and 'DINO-only投影' in report and '纯几何' in report

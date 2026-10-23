"""Meaningful CPU checks; no simulator import or GPU allocation on login."""
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from audit_math import decode_actions, visibility, frame_alignment, paired_interval, event_summary

def test_parquet_chunk_decoding_and_reject_nonfinite():
    original=np.arange(56,dtype=np.float32).reshape(8,7)
    value=np.empty(8,dtype=object)
    for i,row in enumerate(original): value[i]=row
    assert np.array_equal(decode_actions(value),original)
    original[0,1]=np.nan
    with pytest.raises(ValueError): decode_actions(original)

def test_crop_excludes_border_and_patch_occludes_only_target():
    mask=np.zeros((256,256),bool); mask[0:2,:]=True
    assert visibility(mask,False)['raw_visible_pixels']>0
    assert visibility(mask,False)['policy_fov_pixels']==0
    mask[:]=False
    # A target well inside the pinned patch rectangle after 256->224 NN resize.
    mask[170:190,40:60]=True
    before=visibility(mask,False); after=visibility(mask,True)
    assert before['policy_fov_pixels']>0
    assert after['policy_uncovered_pixels']==0
    assert after['patch_covered_pixels']==before['policy_fov_pixels']
    mask[:]=False; mask[80:100,180:200]=True
    assert visibility(mask,True)['patch_covered_pixels']==0

def test_physical_mismatch_fails_and_nan_is_not_passed():
    assert frame_alignment([0,1,2],[0,1,2],1e-9)==0
    with pytest.raises(RuntimeError): frame_alignment([0,1,2.0001],[0,1,2],1e-9)
    with pytest.raises(RuntimeError): frame_alignment([0,np.nan,2],[0,1,2],1e-9)

def test_missing_initial_visibility_not_counted_as_invisible():
    f=pd.DataFrame({'step':[0,1,2,3],'visibility_verified':[False,True,True,True],
        'policy_fov_pixels':[None,20,10,0],'policy_uncovered_pixels':[None,10,0,0],
        'contact':[False,False,True,True],'grasp':[False,False,False,True],
        'distance_m':[.1,.09,.08,.07]})
    e=event_summary(f)
    assert e['first_not_visible_in_policy_fov_step']==3
    assert e['first_fully_hidden_after_overlay_step']==2
    assert e['first_fully_occluded_by_patch_step']==2
    assert e['first_contact_step']==2 and e['first_grasp_step']==3
    assert e['visibility_verified_frames']==3

def test_bootstrap_uses_paired_differences():
    pairs=[{'suite':'goal','task_id':t,'difference':1} for t in (0,0,1,1)]
    assert paired_interval(pairs,100,42)==[1,1]
    assert paired_interval([],100,42) is None

def test_sensor_hook_restored_on_render_error(monkeypatch):
    import audit_replay as replay
    original=lambda cache:np.zeros((256,256,3),np.uint8)
    obs=types.SimpleNamespace(_sensor=original)
    def fail(**kwargs): raise RuntimeError('synthetic render failure')
    env=types.SimpleNamespace(env=types.SimpleNamespace(_observables={'robot0_eye_in_hand_image':obs}),
        sim=types.SimpleNamespace(render=fail))
    monkeypatch.setitem(sys.modules,'robosuite',types.SimpleNamespace(macros=types.SimpleNamespace(IMAGE_CONVENTION='opengl')))
    monkeypatch.setattr(replay,'fingerprint',lambda _: {})
    with pytest.raises(RuntimeError,match='synthetic render failure'):
        with replay.SensorCapture(env,{'render_ids':[1]}) as camera:
            camera.arm(1); obs._sensor({})
    assert obs._sensor is original

def test_report_lock_is_bounded_and_independent(tmp_path,monkeypatch):
    import audit_common as common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    with common.lock('report'):
        with pytest.raises(BlockingIOError):
            with common.lock('report'): pass
        with common.lock('worker_report'): pass

def test_mosaic_streaming_has_native_frame_count_and_holds_end(tmp_path,monkeypatch):
    import imageio.v2 as imageio
    import json
    import audit_report as report
    conditions=['clean','initial','uada','relation','far']; paths={}; entries={}
    monkeypatch.setattr(report,'config',lambda:{'video_conditions':conditions})
    for index,c in enumerate(conditions):
        p=tmp_path/c; p.mkdir(); paths[c]=p; length=index+2
        entries[c]={'episode':{'executed_steps':length-1,'success':index%2}}
        (p/'REPLAY.json').write_text(json.dumps({'fps':20}))
        with imageio.get_writer(p/'video.mp4',fps=20,codec='libx264',ffmpeg_params=['-threads','1']) as out:
            for i in range(length): out.append_data(np.full((288,448,3),30+index*30,np.uint8))
    out=tmp_path/'paired'; out.mkdir(); report.compose('synthetic',entries,paths,out)
    reader=imageio.get_reader(out/'comparison.mp4',ffmpeg_params=['-threads','1'])
    frames=list(reader.iter_data()); reader.close()
    assert len(frames)==6 and frames[-1].shape==(576,1344,3)
    # The short clip must be retained, labeled, rather than black or dropped.
    assert frames[-1][100:200,20:200].mean()>10

def test_stages_do_not_request_gpu_after_replay_completion(monkeypatch):
    import audit_cli as cli
    monkeypatch.setattr(cli,'complete',lambda p: str(p).endswith('REPLAY_COMPLETE.json'))
    calls=[]
    cli.stages(lambda stage,deps: calls.append((stage,deps)) or '123')
    assert calls==[('report',[])]

def test_resume_eta_excludes_previously_completed_work(tmp_path,monkeypatch):
    import audit_common as common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monitor=common.Monitor('report'); monitor.started=10
    monitor.update(completed_steps=100,resumed_steps=80,total_steps=200)
    monkeypatch.setattr(common.time,'monotonic',lambda:20)
    monitor.write()
    value=common.read(tmp_path/'reports/PROGRESS/report.json')
    assert value['steps_per_second']==2
    assert value['estimated_remaining_s']==50


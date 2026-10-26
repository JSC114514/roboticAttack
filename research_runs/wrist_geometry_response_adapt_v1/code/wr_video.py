"""CPU-only single-camera MP4/WebM delivery, resumable by source checksums."""
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from wr_common import ROOT, PROJECT, require, read, sha, atomic, complete, signature

def run(command):
    result=subprocess.run(command,capture_output=True,text=True)
    require(result.returncode==0,'Video command failed: '+result.stderr[-2000:])
    return result.stdout

def metadata(path):
    reader=imageio_ffmpeg.read_frames(str(path))
    try:return next(reader)
    finally:reader.close()

def export(stage,monitor=None):
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    base=PROJECT/'research_videos'/f'{ROOT.name}_singleview'/stage
    records=sorted((ROOT/f'evaluation/{stage}').glob('*/*/*/RESULT.json'))
    done=0
    for path in records:
        receipt=path.with_name('COMPLETE.json')
        if not complete(receipt):continue
        result=read(path);files=read(receipt)['artifacts']
        for view in ('front','wrist'):
            source=next(ROOT/p for p in files if Path(p).name==view+'.mp4')
            stem=result['start']+f'_success_{int(result["success"])}'
            folder=base/result['candidate']/result['profile']/view;folder.mkdir(parents=True,exist_ok=True)
            mp4=folder/(stem+'.mp4');webm=folder/'IDE_WebM'/(stem+'.webm')
            proof=folder/'IDE_WebM'/(stem+'.json');h=sha(source)
            if proof.is_file():
                old=read(proof)
                if old['source_sha256']==h and mp4.is_file() and webm.is_file() and sha(mp4)==old['mp4_sha256'] and sha(webm)==old['webm_sha256']:
                    done+=1;continue
            import os
            if not mp4.exists():os.link(source,mp4)
            require(sha(mp4)==h,'Central MP4 changed')
            webm.parent.mkdir(parents=True,exist_ok=True)
            tmp=webm.with_name('.'+webm.stem+'.tmp.webm')
            run([ffmpeg,'-v','error','-xerror','-nostdin','-y','-threads','2','-i',str(mp4),
                 '-an','-c:v','libvpx','-deadline','realtime','-cpu-used','6','-threads','2',
                 '-b:v','500k','-pix_fmt','yuv420p',str(tmp)])
            tmp.replace(webm)
            for value in (mp4,webm):run([ffmpeg,'-v','error','-xerror','-nostdin','-threads','2','-i',str(value),'-f','null','-'])
            a,b=metadata(mp4),metadata(webm)
            require(a['size']==b['size'] and abs(a['fps']-b['fps'])<1e-6,'Video view/fps mismatch')
            require(abs(a['duration']-b['duration'])<=1/a['fps']+.001,'Video duration mismatch')
            atomic(dict(source=str(source),source_sha256=h,mp4_sha256=sha(mp4),webm_sha256=sha(webm),
                        format_checks='both_full_decode_same_dimensions_fps_duration'),proof)
            done+=1
            if monitor:monitor.update(phase='CPU_video_export',completed=done,total=len(records)*2)
    atomic(dict(signature=signature(),stage=stage,completed_view_files=done,video_root=str(base)),
           ROOT/f'reports/VIDEO_{stage}.json')

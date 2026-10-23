#!/usr/bin/env python3
"""Single-episode, single-camera MP4 and IDE WebM; no HTML or compositions.

Sealed legacy videos are cropped only at their documented camera rectangles.
Raw-camera videos from repaired evaluators are copied unchanged when possible.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import imageio_ffmpeg


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def verify_receipt(root,path,signature=None):
    r=read(path)
    if r.get('status')!='COMPLETE' or (signature and r.get('signature')!=signature):
        raise RuntimeError('Invalid source completion: '+str(path))
    for name,expected in r['artifacts'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=expected:
            raise RuntimeError('Source artifact changed: '+str(p))
    return r


def metadata(path):
    reader=imageio_ffmpeg.read_frames(str(path))
    try:return next(reader)
    finally:reader.close()


def codec_args(extension):
    if extension=='.mp4':
        return ['-c:v','libx264','-preset','veryfast','-crf','20','-profile:v','baseline','-level:v','3.0',
                '-pix_fmt','yuv420p','-movflags','+faststart']
    if extension=='.webm':
        return ['-c:v','libvpx','-deadline','realtime','-cpu-used','6','-crf','12','-b:v','600k','-pix_fmt','yuv420p']
    raise ValueError(extension)


def encode(source,target,crop=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    temp=target.with_name(f'.{target.stem}.{os.getpid()}{target.suffix}')
    cmd=[ffmpeg,'-v','error','-xerror','-nostdin','-n','-threads','2','-i',str(source),'-map','0:v:0','-an']
    if crop:cmd+=['-vf',crop,'-filter_threads','1']
    cmd+=codec_args(target.suffix)+['-threads','2',str(temp)]
    try:
        if crop is None and target.suffix=='.mp4' and Path(source).suffix=='.mp4':
            shutil.copy2(source,temp)
        else:subprocess.run(cmd,check=True,timeout=120,capture_output=True)
        subprocess.run([ffmpeg,'-v','error','-xerror','-nostdin','-threads','2','-i',str(temp),'-f','null','-'],
                       check=True,timeout=120,capture_output=True)
        m=metadata(temp);original=metadata(source)
        if tuple(m['size'])!=(224,224) or abs(m['fps']-original['fps'])>1e-6:
            raise RuntimeError('Camera dimensions/fps mismatch')
        if abs(m['duration']-original['duration'])>max(.06,1/original['fps']):
            raise RuntimeError('Video duration changed')
        temp.replace(target)
        return {'sha256':sha(target),'decoded_complete':True,'metadata':m}
    finally:temp.unlink(missing_ok=True)


def export_run(root,output,limit=None):
    root=Path(root).resolve();output=Path(output).resolve()
    if root.is_relative_to(output) or output.is_relative_to(root):raise ValueError('Use an independent video root')
    output.mkdir(parents=True,exist_ok=True)
    manifest=read(root/'data/MANIFEST.json');signature=manifest['signature']
    receipts=sorted((root/'episodes').glob('*/*/COMPLETE.json'))
    if limit is not None:receipts=receipts[:limit]
    started=time.monotonic();exports=[];count=0;skipped=0
    try:
        for receipt in receipts:
            saved=verify_receipt(root,receipt,signature)
            info=[root/k for k in saved['artifacts'] if k.endswith('/EPISODE.json')]
            if len(info)!=1:raise RuntimeError('Ambiguous successful attempt')
            meta=read(info[0]);condition=meta['condition'];start=receipt.parents[1].name
            label=f'{start}__task_success_{int(meta["success"])}'
            folder=info[0].parent
            raw=(folder/'front.mp4').exists() and (folder/'wrist.mp4').exists()
            legacy=folder/'video.mp4'
            if not raw and tuple(metadata(legacy)['size'])!=(448,288):
                raise RuntimeError('Legacy camera rectangle is not the audited 448x288 layout')
            for view,x in [('front',0),('wrist',224)]:
                source=folder/f'{view}.mp4' if raw else legacy
                crop=None if raw else f'crop=224:224:{x}:64'
                source_hash=sha(source)
                for extension in ('.mp4','.webm'):
                    prefix='MP4' if extension=='.mp4' else 'IDE_WebM'
                    target=output/prefix/condition/view/(label+extension)
                    proof=target.with_suffix(extension+'.json')
                    identity={'source':str(source),'source_sha256':source_hash,'source_receipt':str(receipt),
                              'source_signature':signature,'crop':crop,'view':view,'task_success':int(meta['success']),
                              'encoded_legacy_derivative':not raw,'codec_args':codec_args(extension)}
                    if proof.exists():
                        old=read(proof)
                        if old['identity']!=identity or not target.exists() or sha(target)!=old['sha256']:
                            raise RuntimeError('Existing export differs: '+str(target))
                        result=old;skipped+=1
                    else:
                        if target.exists():raise RuntimeError('Unsealed output exists: '+str(target))
                        result={'identity':identity,**encode(source,target,crop)}
                        atomic(proof,result)
                    exports.append({'file':str(target.relative_to(output)),**result})
            count+=1
            if count==1 or count%10==0 or count==len(receipts):
                elapsed=time.monotonic()-started
                progress={'status':'RUNNING','episodes':count,'total':len(receipts),'elapsed_s':elapsed,
                          'estimated_remaining_s':(len(receipts)-count)*elapsed/count,'files':len(exports),'gpu_used':False,
                          'process_rss_hwm_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          'child_rss_hwm_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024}
                atomic(output/'PROGRESS.json',progress)
                print(f'videos {count}/{len(receipts)} | {elapsed:.1f}s | remaining {progress["estimated_remaining_s"]:.1f}s',flush=True)
    except BaseException as error:
        atomic(output/'FAILURE.json',{'status':'FAILED','error':repr(error),'completed_episodes':count,'elapsed_s':time.monotonic()-started})
        raise
    report={'status':'COMPLETE','source_run':str(root),'source_signature':signature,'episodes':count,
            'files':exports,'reused_files':skipped,'gpu_used':False,'elapsed_s':time.monotonic()-started,
            'process_rss_hwm_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'child_rss_hwm_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024}
    atomic(output/'MANIFEST.json',report);atomic(output/'PROGRESS.json',{k:v for k,v in report.items() if k!='files'})
    (output/'README.txt').write_text('单回合、单视角视频；无拼接、无文字栏、无长合集。\n'
        '在IDE中直接打开 IDE_WebM/方法/front或wrist/ 下的.webm文件。\n'
        '普通播放器使用MP4/；文件名task_success_1代表任务成功，0代表任务失败，不等于无条件攻击成功。\n'
        '旧回合从已编码视频的相机区域裁出，因此是有损派生副本；新回合直接保存独立相机视频。\n'
        '保持源控制时间与帧率；来源、校验和及裁剪定义见MANIFEST.json。\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if hasattr(os,'sched_getaffinity'):
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2])
    result=export_run(args.run,args.output,args.limit)
    print(json.dumps({k:result[k] for k in ('status','episodes','elapsed_s','gpu_used')},ensure_ascii=False))


if __name__=='__main__':main()

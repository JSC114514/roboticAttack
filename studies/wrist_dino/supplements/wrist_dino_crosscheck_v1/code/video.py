# 发布阅读说明：原评测器所需的独立相机编码依赖。这里只发布源码；现有MP4/WebM及视频索引不随仓库发布。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Stream single-camera H264 files; no extra render/sensor/simulation calls.

One frame per actual control step at the environment control frequency. Wrist
frames show the actual deployed PNG. WebM conversion runs later on CPU only.
"""
from pathlib import Path
import subprocess
import numpy as np
from common import ROOT,require,sha,atomic

def decode_measure(ffmpeg,path):
    """Fully decode once, retaining actual frame count and output time.

    ffprobe is not installed on this server. ffmpeg's progress protocol gives
    exact decoded counts without copying all RGB frames through Python.
    """
    r=subprocess.run([ffmpeg,'-v','error','-threads','2','-progress','pipe:1','-nostats','-i',str(path),'-f','null','-'],capture_output=True,text=True,check=True)
    values={}
    for line in r.stdout.splitlines():
        if '=' in line:
            k,v=line.split('=',1);values[k]=v.strip()
    require(values.get('progress')=='end' and int(values.get('frame','0'))>0,'Incomplete video decode: '+str(path))
    return dict(frames=int(values['frame']),out_time_us=int(values.get('out_time_us','0')))

class Videos:
    def __init__(self,category,condition,stem,fps):
        import imageio_ffmpeg
        self.ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();self.folder=ROOT/'videos'/category/condition;self.stem=stem;self.fps=float(fps);self.frames=0;self.processes={};self.paths={}
        for view in ['front','wrist']:
            folder=self.folder/view;folder.mkdir(parents=True,exist_ok=True)
            path=folder/(stem+'.partial.mp4');self.paths[view]=path
            cmd=[self.ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','224x224','-r',str(self.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(path)]
            self.processes[view]=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    def append(self,observation):
        for view,key in [('front','full_image'),('wrist','wrist_image')]:
            a=np.asarray(observation[key]);require(a.shape==(224,224,3) and a.dtype==np.uint8,'Invalid video frame')
            self.processes[view].stdin.write(np.ascontiguousarray(a).tobytes())
        self.frames+=1
    def close(self,success):
        paths=[]
        for view,p in self.processes.items():
            p.stdin.close();error=p.stderr.read().decode();rc=p.wait();require(rc==0,'H264 encoding failed: '+error)
            require(self.frames>0,'Empty episode video')
            path=self.paths[view];final=path.with_name(self.stem+('_success' if success else '_failure')+'.mp4');path.replace(final);paths.append(final)
        return dict(frames=self.frames,fps=self.fps,files={str(p.relative_to(ROOT)):sha(p) for p in paths})
    def abort(self):
        for p in self.processes.values():
            if p.poll() is None:p.terminate();p.wait()

def export_all(monitor):
    """Independent CPU stage; complete decode is checked before marking each file."""
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();files=sorted((ROOT/'videos').rglob('*.mp4'));files=[p for p in files if '.partial.' not in p.name]
    monitor.update(total=len(files),completed=0,phase='webm_export')
    for i,p in enumerate(files):
        out=ROOT/'videos/IDE_WebM'/p.relative_to(ROOT/'videos').with_suffix('.webm');receipt=out.with_suffix('.json');source=sha(p)
        if receipt.exists() and out.exists():
            from common import read
            a=read(receipt)
            if a.get('source_sha256')==source and a.get('webm_sha256')==sha(out):monitor.update(completed=i+1);continue
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.stem+'.partial.webm')
        subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(p),'-an','-c:v','libvpx','-deadline','realtime','-cpu-used','6','-crf','20','-b:v','0','-threads','2',str(tmp)],check=True)
        before,after=decode_measure(ffmpeg,p),decode_measure(ffmpeg,tmp)
        require(before['frames']==after['frames'] and abs(before['out_time_us']-after['out_time_us'])<=100000,'Video conversion changed frames/time')
        tmp.replace(out);atomic(dict(source_sha256=source,webm_sha256=sha(out),status='DECODED',mp4_decode=before,webm_decode=after),receipt);monitor.update(completed=i+1)
    from common import done
    done(ROOT/'reports/units/cpu-webm-export.json',files=[ROOT/'reports/VIDEO_INDEX.json'] if (ROOT/'reports/VIDEO_INDEX.json').exists() else [],videos=len(files))

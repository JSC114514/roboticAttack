# 发布阅读说明：确定性构造按套件/任务平衡的专家候选池，记录轨迹身份与时序采样。1920是候选数，实际访问数另计。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""P01: deterministic240-trajectory/1920-state RLDS expert bank, streamed on CPU.

Use ten distinct source trajectories per training task, eight time-bin midpoint
anchors with eight real future actions. RGB is stored as raw decoded256px, then
processed with the same native OFT resize at model time. No simulation rollouts
or old checkpoint predictions enter the bank.
"""
import io
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from .common import ROOT,config,read,atomic,require,sha,digest,finish,complete,save_npz,task_map


def anchors(length,count=8,chunk=8):
    maximum=int(length)-chunk
    require(maximum>=count-1,'Trajectory too short for unique future-action anchors')
    values=np.floor((np.arange(count)+.5)*(maximum+1)/count).astype(int)
    require(len(set(values.tolist()))==count,'Duplicate selected timeline state')
    return values


def dataset(suite):
    """TensorFlow must not discover GPUs; limit CPU pools and avoid image decoding."""
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='1'
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([],'GPU')
    b=tfds.builder_from_directory(str(Path(config()['expert_root'])/(suite+'_no_noops')/'1.0.0'))
    dec={'steps':{'observation':{'image':tfds.decode.SkipDecoding(),'wrist_image':tfds.decode.SkipDecoding()}}}
    ds=b.as_dataset(split='train',shuffle_files=False,decoders=dec,read_config=tfds.ReadConfig(try_autocache=False))
    opts=tf.data.Options();opts.threading.private_threadpool_size=2;return ds.with_options(opts)


def episode_arrays(ep):
    path=ep['episode_metadata']['file_path'].numpy().decode();actions=[];states=[];front=[];wrist=[];lang=[]
    for s in ep['steps']:
        actions.append(s['action'].numpy());states.append(s['observation']['state'].numpy());front.append(s['observation']['image'].numpy());wrist.append(s['observation']['wrist_image'].numpy());lang.append(s['language_instruction'].numpy().decode())
    require(len(set(lang))==1,'Language changes within trajectory')
    return dict(file_path=path,actions=np.asarray(actions,np.float32),states=np.asarray(states,np.float32),front=front,wrist=wrist,language=lang[0])


def trajectory_id(ep):
    h=__import__('hashlib').sha256(ep['file_path'].encode())
    for a in [ep['actions'],ep['states']]:
        a=np.ascontiguousarray(a,dtype='<f4');h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def decoded(b):
    with Image.open(io.BytesIO(b)) as im:a=np.asarray(im.convert('RGB'),np.uint8)
    require(a.shape==(256,256,3),'Unexpected RLDS raw image resolution');return a


def build(monitor):
    receipt=ROOT/'reports/P01_COMPLETE.json'
    if complete(receipt):return
    c=config();old=Path(c['project'])/'research_runs/wrist_geometry_response_adapt_v1/data/EXPERT_INDEX.json';meta=read(old)
    # Existing metadata can avoid a full repeated indexing pass only when ALL
    # original dataset shard sizes/mtimes still agree. Selected content is rehashed.
    for name,st in meta['inputs'].items():
        p=Path(name);require(p.is_file() and [p.stat().st_size,p.stat().st_mtime_ns]==st,'RLDS source metadata changed: '+name)
    selected=[];shortages=[]
    for suite in c['suites']:
        for task in c['splits'][suite]['train']:
            rows=sorted({r['episode_id']:r for r in meta['episodes'] if r['suite']==suite and r['task_id']==task and r['length']>=15}.values(),key=lambda r:(r['file_path'],r['episode_id']))
            if len(rows)<10:shortages.append(dict(suite=suite,task=task,available=len(rows)))
            selected+=rows[:10]
    atomic(dict(selected=selected,shortages=shortages,source_index_sha256=sha(old),source_shards=meta['inputs']),ROOT/'data/TRAJECTORIES.json')
    records=[];files=[];seen_hashes=set();total=sum(len(anchors(r['length'])) for r in selected)
    monitor.update(phase='extract_RLDS',completed=0,total=total)
    for suite in c['suites']:
        wanted={r['episode_id']:r for r in selected if r['suite']==suite};ordinals={r['ordinal'] for r in wanted.values()}
        for ordinal,ep in enumerate(dataset(suite)):
            if ordinal not in ordinals:continue
            a=episode_arrays(ep);identity=trajectory_id(a)
            require(identity in wanted,'RLDS trajectory content hash/ordering changed')
            row=wanted.pop(identity);idx=anchors(len(a['actions']));path=ROOT/'data/clean'/suite/f'{identity}.npz'
            if not path.exists():save_npz(path,front_rgb=np.stack([decoded(a['front'][i]) for i in idx]),wrist_rgb=np.stack([decoded(a['wrist'][i]) for i in idx]),proprio=a['states'][idx],expert_actions=np.stack([a['actions'][i:i+8] for i in idx]),anchors=idx,language=np.asarray(a['language']))
            file_sha=sha(path);files.append(path)
            with np.load(path,allow_pickle=False) as v:
                require(np.array_equal(v['anchors'],idx) and np.array_equal(v['proprio'],a['states'][idx]),'Cached extraction mismatch')
                for j,t in enumerate(idx):
                    # Content hash prevents accidentally counting copied states as new data.
                    sample=__import__('hashlib').sha256(v['front_rgb'][j].tobytes()+v['wrist_rgb'][j].tobytes()+v['proprio'][j].tobytes()).hexdigest()
                    require(sample not in seen_hashes,'Duplicate raw RGB/proprio state in bank');seen_hashes.add(sample)
                    records.append(dict(record_id=f'{suite}_t{row["task_id"]:02d}_{identity[:12]}_f{int(t):05d}',suite=suite,task_id=row['task_id'],trajectory_id=identity,source_file_path=a['file_path'],source_timestep=int(t),phase=('early','middle','late')[min(2,int(3*t/len(a['actions'])))],path=str(path.relative_to(ROOT)),row=j,sample_sha256=sample,shard_sha256=file_sha,language=a['language']))
            monitor.update(completed=len(records),current_suite=suite)
            if not wanted:break
        require(not wanted,'Selected RLDS trajectories missing')
    records.sort(key=lambda r:(c['suites'].index(r['suite']),r['task_id'],r['trajectory_id'],r['source_timestep']))
    require(len({(r['suite'],r['task_id']) for r in records})==24,'Missing entire training task')
    out=ROOT/'data/BANK_MANIFEST.json';atomic(dict(records=records,count=len(records),trajectories=len(selected),shortages=shortages,selected_by='10lexicographic_unique_trajectories_per_task;8time_bin_midpoints;no_outcome_selection',source_shards=meta['inputs'],bank_sha256=digest(records)),out)
    import csv
    with (ROOT/'data/BANK_MANIFEST.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    finish(receipt,[out,ROOT/'data/BANK_MANIFEST.csv',ROOT/'data/TRAJECTORIES.json',*files],records=len(records),trajectories=len(selected),shortages=shortages)


def load_record(row):
    """Read one small8-observation trajectory; no giant bank resident in RAM."""
    with np.load(ROOT/row['path'],allow_pickle=False) as a:
        j=row['row'];return dict(full_image=a['front_rgb'][j].copy(),wrist_image=a['wrist_rgb'][j].copy(),state=a['proprio'][j].copy()),str(a['language'])

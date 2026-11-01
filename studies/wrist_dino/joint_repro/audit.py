# 发布阅读说明：CPU模型/归一化/初态字节核验；不申请GPU，不查询或提交Slurm。
"""P00 CPU asset audit: actual source, checkpoint receipt and simulator starts."""
import csv
import importlib.metadata
from pathlib import Path
from .common import ROOT,config,read,atomic,require,sha,task_map,Monitor,finish


def run(args=None):
    with Monitor('P00_assets') as monitor:
        c=config();model=Path(c['model_dir']);receipt=read(model/'JOINT_DOWNLOAD.json')
        require(receipt['status']=='COMPLETE' and receipt['revision']==c['checkpoint_hf_revision'],'Wrong checkpoint revision')
        verified={}
        for name,row in receipt['files'].items():
            p=model/name;require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Invalid checkpoint file: '+name);verified[name]=row
        index=read(model/'model.safetensors.index.json');shards=sorted(set(index['weight_map'].values()))
        # Record full hashes for local large weights, not merely the index file.
        for name in shards:
            p=model/name;require(p.is_file(),'Missing model shard');monitor.update(phase='hash_model_shards',file=name);verified[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
        stats=read(model/'dataset_statistics.json');require(all(s+'_no_noops' in stats for s in c['suites']),'Missing normalization stats')
        names=task_map();starts=[]
        for suite in c['suites']:
            for tid,name in enumerate(names[suite]):
                p=Path(c['libero_root'])/'libero/libero/init_files'/suite/(name+'.pruned_init');require(p.is_file(),'Missing simulator init file: '+str(p));starts.append(dict(suite=suite,task_id=tid,path=str(p),sha256=sha(p)))
        versions={}
        for n in ['torch','torchvision','transformers','timm','mujoco','robosuite','tensorflow','tensorflow-datasets','flash-attn']:
            try:versions[n]=importlib.metadata.version(n)
            except importlib.metadata.PackageNotFoundError:versions[n]='NOT_INSTALLED'
        out=ROOT/'reports/ASSET_AUDIT.json';atomic(dict(model=verified,initial_states=starts,versions=versions,protocol_sha256=sha(ROOT/'configs/PROTOCOL.json')),out)
        (ROOT/'reports/P00_ASSET_AUDIT.md').write_text('# 本地资源与资产审计\n\n联合checkpoint、300000动作头／本体投影器、四套件归一化及40任务初态文件存在。详细字节哈希见reports/ASSET_AUDIT.json。\n\n尚需GPU原生接口、EOT及完整梯度预检。Pro6000禁用。\n')
        finish(ROOT/'reports/P00_CPU_COMPLETE.json',[out,ROOT/'reports/P00_ASSET_AUDIT.md'])

"""Readable evidence, processor-aligned examples and layer curves; no causal claims."""
from __future__ import annotations
import html
import numpy as np
from atlas_common import ROOT,atomic_text,config,load_json
from atlas_math import crop_map


def write_report(rows,global_rows,paired,local_rows,part_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'results/figures'; folder.mkdir(parents=True,exist_ok=True)
    lookup={(r['site'],r['label'],r['condition']):r for r in global_rows}
    labels=['object_eef_distance','gripper_opening','object_eef_relative_x','object_eef_relative_y','contact','grasp']
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,label in zip(axes.flat,labels):
        for view in config()['views']:
            values=[]
            for b in range(24):
                row=lookup[(f'{view}__b{b:02d}_spatial',label,'all')]
                values.append(row.get('auroc') if label in ('contact','grasp') else row.get('r2'))
            ax.plot(range(24),values,label=view)
        ax.axvline(22,color='gray',linestyle='--',label='consumed b22' if label==labels[0] else None)
        ax.set_title(label); ax.set_xlabel('DINO block (zero based)')
        ax.set_ylabel('AUROC' if label in ('contact','grasp') else 'R²'); ax.grid(alpha=.2)
    axes.flat[0].legend(); fig.savefig(folder/'layer_curves.png',dpi=160); plt.close(fig)
    test=[r for r in rows if r['split']=='validation']
    evidence=[]
    names=[s['name'] for s in __import__('atlas_labels').label_specs()]
    for view in config()['views']:
        path=ROOT/f'readouts/local/{view}__b22/PREDICTIONS.npz'
        with np.load(path,allow_pickle=False) as z: predicted=z['class_prediction'].copy()
        # Predetermined task coverage, independent of readout quality.
        chosen=[next(i for i,r in enumerate(test) if r['task_id']==t) for t in (0,4,9)]
        fig,axes=plt.subplots(3,3,figsize=(9,9),constrained_layout=True)
        for line,index in enumerate(chosen):
            row=test[index]
            with np.load(ROOT/f"data/task_{row['task_id']:02d}/states/{row['state_id']}.npz",allow_pickle=False) as z:
                image=np.clip(crop_map(z[view+'_rgb'])/255,0,1); true=z[view+'_class_tokens']
            axes[line,0].imshow(image); axes[line,0].set_title(f"{view} task {row['task_id']}")
            axes[line,1].imshow(image); axes[line,1].imshow(true.reshape(16,16),extent=(-.5,223.5,223.5,-.5),alpha=.5,
                                                          cmap='tab20',vmin=0,vmax=19,interpolation='nearest')
            axes[line,1].set_title('Simulator class labels')
            axes[line,2].imshow(image); axes[line,2].imshow(predicted[index].reshape(16,16),extent=(-.5,223.5,223.5,-.5),
                                                          alpha=.5,cmap='tab20',vmin=0,vmax=19,interpolation='nearest')
            axes[line,2].set_title('DINO-only linear prediction')
        for ax in axes.flat: ax.set_axis_off()
        fig.savefig(folder/(view+'_class_examples.png'),dpi=150); plt.close(fig)
        with np.load(ROOT/f'readouts/global/{view}__consumed/PREDICTIONS.npz',allow_pickle=False) as z:
            truth=z['truth'].copy(); p=z['prediction'].copy()
        geom=['object_eef_distance','object_eef_relative_x','object_eef_relative_y','object_eef_relative_z']
        fig,axes=plt.subplots(1,4,figsize=(14,3.5),constrained_layout=True)
        for ax,name in zip(axes,geom):
            c=names.index(name); ok=np.isfinite(truth[:,c])&np.isfinite(p[:,c])
            ax.scatter(truth[ok,c],p[ok,c],s=5,alpha=.4)
            if ok.any():
                low=min(truth[ok,c].min(),p[ok,c].min()); high=max(truth[ok,c].max(),p[ok,c].max())
                ax.plot([low,high],[low,high],color='gray',linestyle='--')
            ax.set_title(name); ax.set_xlabel('simulator truth (m)'); ax.set_ylabel('DINO prediction (m)')
        fig.savefig(folder/(view+'_geometry.png'),dpi=150); plt.close(fig)
        c=names.index('object_eef_distance'); error=abs(truth[:,c]-p[:,c]); valid=np.flatnonzero(np.isfinite(error))
        # Descriptive failures only: these examples never select training or hyperparameters.
        chosen=valid[np.argsort(error[valid])[-3:][::-1]]
        fig,axes=plt.subplots(1,3,figsize=(10,4),constrained_layout=True)
        for ax,index in zip(axes,chosen):
            row=test[index]
            with np.load(ROOT/f"data/task_{row['task_id']:02d}/states/{row['state_id']}.npz",allow_pickle=False) as z:
                ax.imshow(np.clip(crop_map(z[view+'_rgb'])/255,0,1))
            ax.set_title(f"task {row['task_id']}, error {error[index]:.3f}m"); ax.set_axis_off()
            evidence.append({'view':view,'state_id':row['state_id'],'label':'object_eef_distance',
                             'truth_m':float(truth[index,c]),'prediction_m':float(p[index,c]),'error_m':float(error[index]),
                             'selection':'largest_validation_absolute_error_descriptive_only',
                             'visibility':row['visibility'],'contact':row['truth']['contact'],'grasp':row['truth']['grasp']})
        fig.savefig(folder/(view+'_failure_examples.png'),dpi=150); plt.close(fig)
        part_path=ROOT/f'readouts/parts/{view}__b22/PREDICTIONS.npz'
        if part_rows and part_path.exists():
            with np.load(part_path,allow_pickle=False) as z:
                ids=z['state_ids'].tolist(); pp=z['class_prediction'].copy()
            byid={r['state_id']:r for r in test}; fig,axes=plt.subplots(3,3,figsize=(9,9),constrained_layout=True)
            for line,i in enumerate(range(min(3,len(ids)))):
                row=byid[ids[i]]
                with np.load(ROOT/f"data/task_{row['task_id']:02d}/states/{row['state_id']}.npz",allow_pickle=False) as z:
                    image=np.clip(crop_map(z[view+'_rgb'])/255,0,1)
                mask=np.load(ROOT/f'parts/reviewed/{ids[i]}__{view}.npy',allow_pickle=False)
                axes[line,0].imshow(image); axes[line,0].set_title(ids[i],fontsize=7)
                axes[line,1].imshow(image); axes[line,1].imshow(np.ma.masked_equal(mask,0),alpha=.6,cmap='tab10',vmin=1,vmax=6)
                axes[line,1].set_title('Explicitly reviewed regions')
                axes[line,2].imshow(image); axes[line,2].imshow(pp[i].reshape(16,16)+1,extent=(-.5,223.5,223.5,-.5),alpha=.5,cmap='tab10',vmin=1,vmax=6)
                axes[line,2].set_title('DINO-only part prediction')
            for ax in axes.flat: ax.set_axis_off()
            fig.savefig(folder/(view+'_part_examples.png'),dpi=150); plt.close(fig)
    from atlas_common import atomic_json
    atomic_json(evidence,ROOT/'results/FAILURE_CASES.json')
    specs=load_json(ROOT/'results/INVENTORY.json')
    parts_pending=not part_rows
    lines=['# Spatial DINO 信息图谱','',
           '这是干净观测的信息读出结果。可读出、被原策略使用、可被 RGB 扰动调动是不同问题。',
           '本轮没有训练补丁、修改激活或产生 CASR 结论。','',
           '数据为 240 条已成功的原轨迹、1,200 个对齐观测点；训练和验证轨迹互不重叠。',
           '这些验证状态曾用于旧研究，本轮未用其结果选择读出层，不称为全新未查看的数据。','',
           '## 原生消费输出能读出什么','',
           '|信息|前视误差/分数|腕部误差/分数|说明|','|---|---:|---:|---|']
    for label in labels:
        a=lookup[('front__consumed',label,'all')]; b=lookup[('wrist__consumed',label,'all')]
        if label in ('contact','grasp'):
            va,vb=a.get('auroc'),b.get('auroc'); unit='AUROC，越大越好；不等于直接观察到接触'
        else:
            va,vb=a.get('mae'),b.get('mae'); unit='平均绝对误差，米；越小越好'
        fmt=lambda v:'数据不足' if v is None else f'{v:.4g}'
        lines.append(f'|{label}|{fmt(va)}|{fmt(vb)}|{unit}|')
    local_fits={}
    for view in config()['views']:
        path=ROOT/f'readouts/local/{view}__b22/FIT.json'
        if path.exists(): local_fits[view]=load_json(path)
    if len(local_fits)==2:
        lines += ['','## 画面中的类别与深度','',
                  '下面是原生空间 token 的读出；覆盖当前场景实体，不是预测恒定的“目标是黑碗”。',
                  'IoU 在原生 16×16 token 网格上评价，不宣称像素级完整轮廓精度。','',
                  '|类别|前视 IoU|腕部 IoU|前视/腕部验证 token 数|','|---|---:|---:|---:|']
        for i,name in enumerate(config()['class_names']):
            a=local_fits['front']['main']['per_class_iou'][i]; b=local_fits['wrist']['main']['per_class_iou'][i]
            lines.append(f'|{name}|{fmt(a["pooled_iou"])}|{fmt(b["pooled_iou"])}|{a["true_tokens"]}/{b["true_tokens"]}|')
        for view,fit in local_fits.items():
            m=fit['main']['depth_mae_m']; ct=fit['coordinates']['token_accuracy']; base=fit['majority']['token_accuracy']
            lo=fit['leave_one_task_out']['token_accuracy']
            lines += ['',f'{view}：深度误差 {fmt(m["mean"])} 米；坐标基线准确率 {fmt(ct["mean"])}；'
                      f'多数类基线 {fmt(base["mean"])}；留一任务准确率 {fmt(lo["mean"])}。']
    lines += ['', '训练内选择的独立基线、留一任务与轨迹置乱对照见每个 readouts/global 目录；',
              'INFORMATION_ATLAS.csv 同时列出可见性、疑似遮挡、接触与抓持条件，VIEW_CONTRASTS.csv 为同状态配对差值。',
              '疑似遮挡仅指物体中心在裁剪视锥内但没有分割像素，不是完整/部分遮挡的准确标签。',
              'MAE/R²/AUROC 为观测级描述；mae_mean 及区间为任务等权、轨迹内平均的估计；条件子集下二者权重可能不同。',
              '腕部信息读出更好不等于更脆弱。旧 P1–P2 的分支敏感性结果是独立背景，本轮不重跑它。','',
              '## 图像与标签可靠性','',
              '标签来自同一已实现仿真模型中的原动作重放。缓存 RGB、精确状态、本体状态均经过逐点检查。',
              '标签在原相机实际采样回调中取得，控制步结束状态单独核对；二者的时刻、pose 和缓存年龄分别记录。',
              '未强制刷新原观测，也未把控制步结束状态误当作缓存画面的采样状态。',
              '原数据未保存原始逐点相机/物体 pose bundle；本轮记录重放 bundle，并验证标注、渲染前后没有改变。',
              '不能把没有保存的原始 bundle 描述为已经逐项直接比对。旧标签差异见 INVENTORY.json。','',
              '## 部件证据','',
              ('80 张图尚待明确核对；未生成部件准确率。打开 ../parts/review.html。' if parts_pending else
               '80 张图已明确核对。PART_READOUTS.csv 单列训练类别、图像数和轨迹区间，不与全量数据混报。'),'',
              '## 范围与缺口','',
              '类别只覆盖当前 Spatial 场景；两个外观相同的碗不按仿真编号宣称视觉身份。',
              '碗/盘姿态用表面法向量评价，轴向对称转角不作为可辨语义。',
              '颜色/纹理统计可能与类别和背景相关；接触/抓持读出也可能利用视觉相关线索，均不作因果结论。',
              '无变异、无可见标签或读出未稳定检出的项目保留在清单，不按零分或“信息不存在”处理。','',
              '![分层读出](figures/layer_curves.png)','',
              '![前视示例](figures/front_class_examples.png)','',
              '![腕部示例](figures/wrist_class_examples.png)','']
    for view in config()['views']:
        lines += [f'![{view}几何预测](figures/{view}_geometry.png)','',
                  f'![{view}典型失败](figures/{view}_failure_examples.png)','']
        if (folder/(view+'_part_examples.png')).exists(): lines += [f'![{view}部件](figures/{view}_part_examples.png)','']
    atomic_text('\n'.join(lines),ROOT/'results/RESULTS.md')
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in ('site','label','condition','mae','r2','auroc'))+'</tr>'
                  for r in global_rows if r['site'].endswith('__consumed') and r['condition']=='all')
    page='<!doctype html><meta charset="utf-8"><title>DINO 信息图谱</title><style>body{font:16px sans-serif;margin:30px}img{max-width:100%}td,th{padding:6px;border:1px solid #ccc}</style>'
    page+='<h1>Spatial DINO 信息图谱</h1><p>干净观测的可读信息；不构成因果控制或补丁效果结论。</p>'
    page+='<p>部件核对：'+('待完成' if parts_pending else '已完成')+'</p>'
    page+='<img src="figures/layer_curves.png"><table><tr><th>表示</th><th>信息</th><th>条件</th><th>MAE</th><th>R²</th><th>AUROC</th></tr>'+table+'</table>'
    page+='<img src="figures/front_class_examples.png"><img src="figures/wrist_class_examples.png">'
    for view in config()['views']:
        page+=f'<img src="figures/{view}_geometry.png"><img src="figures/{view}_failure_examples.png">'
        if (folder/(view+'_part_examples.png')).exists(): page+=f'<img src="figures/{view}_part_examples.png">'
    atomic_text(page,ROOT/'results/REPORT.html')

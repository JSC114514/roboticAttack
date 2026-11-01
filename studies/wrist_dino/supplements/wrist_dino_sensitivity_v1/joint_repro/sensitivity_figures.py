# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Four single-variable development curves from immutable completed results.

This is a CPU-only presentation step. No model, attack, simulation or statistical
selection runs here. A missing/closed cell stays missing; no interpolation is
used to invent an unexecuted batch or training endpoint.
"""
from .common import ROOT,read,atomic,sha


STYLE={
    'B01':('#0072B2','o','X action'),
    'B04':('#333333','s','7-DoF action'),
    'O03':('#D55E00','o','X action + DINO'),
    'D07':('#E69F00','s','7-DoF action + DINO'),
}


def render():
    source=ROOT/'reports/SCREEN_RESULTS.json'
    if not source.exists():return None
    rows={r['identity']:r['macro_casr']*100 for r in read(source)['rows']}
    # SCREEN_RESULTS has a changing timestamp. Cache on its numeric rows,
    # rather than redraw unchanged figures at each half-hour status review.
    from .common import digest
    signature=digest(rows);folder=ROOT/'reports/figures';folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'FIGURES.json';files=[folder/f'ablation_development_seed42.{ext}' for ext in ['pdf','png']]
    if manifest.exists():
        previous=read(manifest)
        if previous.get('data_digest')==signature and previous.get('code_sha256')==sha(__file__) and all(p.exists() for p in files):return manifest
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':11,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})
    figure,axes=plt.subplots(2,2,figsize=(10.2,7.0),layout='constrained')
    for axis in axes.flat:
        axis.set_ylim(0,105);axis.set_yticks([0,20,40,60,80,100]);axis.set_ylabel('Task/suite-equal CASR (%)');axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
    # beta0 uses the exactly paired action-only baseline. The two curves
    # represent action masks, because changing beta is this panel's variable.
    axis=axes[0,0]
    for base,aux in [('B01','O03'),('B04','D07')]:
        labels=[base,f'W025-{aux}',aux,f'W400-{aux}'];positions=[i for i,label in enumerate(labels) if label in rows]
        color,marker,_=STYLE[aux]
        axis.plot(positions,[rows[labels[i]] for i in positions],color=color,marker=marker,lw=1.7,label='X target' if base=='B01' else '7-DoF target')
    axis.set_xticks(range(4),['0','0.25','1','4']);axis.set_xlabel(r'Tested DINO weight multiplier $\beta$');axis.set_title('A  DINO weight (zero = action baseline)',loc='left');axis.legend(loc='lower left',frameon=False)
    # The other panels retain the four fixed method families.
    definitions=[(axes[0,1],['0.001','0.002','0.004'],lambda g:[f'LR001-{g}',g,f'LR004-{g}'],'Learning rate','B  Learning rate'),
                 (axes[1,0],['500','1000','2000'],lambda g:[f'T0500-{g}',f'T1000-{g}',f'T2000-{g}'],'Outer rounds (50 optimizer updates each)','C  Length (one fixed 2000-round schedule)'),
                 (axes[1,1],['1','2','4','8'],lambda g:[g,f'B02-{g}',f'B04-{g}',f'B08-{g}'],'Observation batch (fixed optimizer updates)','D  Batch size')]
    for axis,ticks,labels_for,xlabel,title in definitions:
        coordinates=[.001,.002,.004] if axis is axes[0,1] else [500,1000,2000] if axis is axes[1,0] else [1,2,4,8]
        for group,(color,marker,label) in STYLE.items():
            labels=labels_for(group);positions=[i for i,name in enumerate(labels) if name in rows]
            if positions:axis.plot([coordinates[i] for i in positions],[rows[labels[i]] for i in positions],color=color,marker=marker,lw=1.6,linestyle='--' if group in ['B01','B04'] else '-',label=label)
        axis.set_xticks(coordinates,ticks)
        if axis is axes[1,0]:axis.set_xlim(400,2100)
        else:
            axis.set_xscale('log',base=2);axis.set_xlim(coordinates[0]*.85,coordinates[-1]*1.15)
            axis.set_xticks(coordinates,ticks);xlabel+=' (log scale)'
        axis.set_xlabel(xlabel);axis.set_title(title,loc='left')
    axes[0,1].legend(loc='lower left',frameon=False,fontsize=9)
    decisions=read(ROOT/'reports/BUDGET_DECISIONS.json') if (ROOT/'reports/BUDGET_DECISIONS.json').exists() else {}
    if decisions.get('batch4',{}).get('activate') is False:
        axis=axes[1,1];axis.axvspan(2.828,9.2,color='#eeeeee',zorder=-1)
        axis.text(5.5,45,'Not run\n(no gain at batch2)',ha='center',va='center',fontsize=9,color='#666666')
    if not any(label.startswith('T2000-') for label in rows):axes[1,0].text(2000,45,'Pending',ha='center',fontsize=9,color='#666666')
    figure.suptitle('Single-variable ablations | joint OFT | development, seed42',fontsize=13)
    for path in files:figure.savefig(path,dpi=300,facecolor='white')
    plt.close(figure)
    atomic(dict(data_digest=signature,code_sha256=sha(__file__),source='reports/SCREEN_RESULTS.json',scope='development_seed42',outputs={p.name:sha(p) for p in files}),manifest)
    (folder/'README.md').write_text('''# 参数消融图

`ablation_development_seed42.pdf`为矢量稿，PNG为300dpi预览。仅展示已完成的seed42开发集结果；选择及三种子确认结果分别保存。

- A：DINO权重，β=0对应同动作mask的基线，λ=β×22.855948214853818。
- B：四组共同覆盖0.001、0.002、0.004，只改变学习率。
- C：同一2000轮调度中的500/1000/2000检查点；不混入历史500轮调度。
- D：固定25000次Adam更新，batch增加会相应增加样本反传。未触发的4/8档标为未运行。

纵轴为先任务、再套件等权CASR；不是任务成功率。点间线段仅连接已经运行的档位。图不生成三种子误差棒；正式确认另有配对区间。

来源和图像哈希见FIGURES.json。数据未变则复用图，不在每次审核重复绘图。
''')
    return manifest

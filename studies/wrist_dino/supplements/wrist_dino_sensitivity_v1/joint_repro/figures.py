# 发布阅读说明：从已完成结果生成可追溯图表，保留原始分母和来源；逐任务/套件等权，不把预测槽位当重复试验。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""CPU-only V03 figures from recorded results, with machine-readable plot data.

No model, simulator, optimizer or new statistical estimator is used here.
Final figures require every selected seed in both frozen evaluation scopes.
Development previews live separately and cannot create a V03 completion receipt.
"""
import argparse
import csv
from collections import defaultdict
import numpy as np
from .common import ROOT,read,atomic,sha,require,complete

LABELS={'B01':'B01\nWrist dx','B02':'B02\nFront dx','B03':'B03\nWrist XYZ','B04':'B04\nWrist 7DoF','O03':'O03\nDINO auxiliary'}
COLORS=['#0072B2','#D55E00','#009E73']


def selected_rows(rows,selection):
    """Keep every preselected method/seed; refuse partial or duplicate results."""
    names=[f'{m}-seed{s}' for m in selection['methods'] for s in selection['seeds']]
    values=[r for r in rows if r['condition'] in names]
    require(len(values)==len(names),'Selected figure results missing or duplicated')
    indexed={r['condition']:r for r in values}
    require(len(indexed)==len(names),'Duplicate method/seed in figure input')
    require(all(indexed[n]['completed']==indexed[n]['expected'] for n in names),'Incomplete selected figure result')
    return [indexed[n] for n in names]


def hierarchical_mean(rows,field):
    """Average starts within tasks, tasks within suites, then suites equally.

    Each input record is one complete episode, never a frame or an action slot.
    Vector fields retain their original action-dimension order and units.
    """
    tasks=defaultdict(list)
    for row in rows:tasks[(row['suite'],row['task_id'])].append(np.asarray(row[field],dtype=float))
    require(bool(tasks),'No episode records for figure')
    suites=defaultdict(list)
    for (suite,_),values in tasks.items():suites[suite].append(np.mean(values,axis=0))
    return np.mean([np.mean(values,axis=0) for values in suites.values()],axis=0)


def export(preview=False):
    """Export PNG/PDF plus exact plotting values; no missing values become zero."""
    if not preview:
        require(complete(ROOT/'reports/V02_COMPLETE.json'),'Final figure export requires V02')
        require(complete(ROOT/'reports/M03_COMPLETE.json'),'Final figure export requires M03')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    selection=read(ROOT/'reports/V_SELECTION.json')
    destination=ROOT/'reports'/('figures_preview' if preview else 'figures')
    destination.mkdir(parents=True,exist_ok=True);manifest=[];files=[]

    def save(fig,name,caption,data,sources):
        """Close each figure promptly and retain source hashes and exact values."""
        paths=[destination/f'{name}.png',destination/f'{name}.pdf',destination/f'{name}.data.json']
        try:
            fig.savefig(paths[0],dpi=300,bbox_inches='tight',facecolor='white')
            fig.savefig(paths[1],bbox_inches='tight',facecolor='white')
        finally:plt.close(fig)
        atomic(dict(caption=caption,preview=preview,data=data,sources={s:sha(ROOT/s) for s in sources}),paths[2])
        manifest.append(dict(name=name,caption=caption,assets={str(p.relative_to(ROOT)):sha(p) for p in paths}))
        files.extend(paths)

    def seed_plot(scope):
        rows=selected_rows(read(ROOT/f'reports/{scope}_SUMMARY.json')['conditions'],selection)
        fig,ax=plt.subplots(figsize=(8.4,4.2),layout='constrained')
        for j,seed in enumerate(selection['seeds']):
            for i,method in enumerate(selection['methods']):
                row=next(r for r in rows if r['condition']==f'{method}-seed{seed}')
                value=row['suite_task_equal_casr'];interval=row['casr_ci95'];x=i+(j-1)*.19
                if value is None:continue
                if interval is not None:
                    lo,hi=100*np.asarray(interval);ax.vlines(x,lo,hi,color=COLORS[j],linewidth=1.2)
                    ax.hlines([lo,hi],x-.035,x+.035,color=COLORS[j],linewidth=1.2)
                ax.scatter(x,100*value,color=COLORS[j],marker=['o','s','^'][j],s=40,label=f'Seed {seed}' if i==0 else None,zorder=3)
        ax.set(xticks=range(len(selection['methods'])),xticklabels=[LABELS.get(m,m) for m in selection['methods']],ylim=(0,105),ylabel='Suite/task-equal CASR (%)',title={'development':'Development starts (80)','full':'All 40 tasks (400 starts)','confirmation_extra':'Additional confirmation starts (80)'}[scope])
        ax.grid(axis='y',alpha=.2);ax.legend(ncols=3,loc='lower right')
        save(fig,scope+'_casr_by_seed','Each fixed training seed is separate. Whiskers are existing task-to-start bootstrap 95% intervals; no frames or action slots are independent samples.',[{k:r[k] for k in ['condition','completed','expected','clean_success','induced_failures','suite_task_equal_casr','casr_ci95']} for r in rows],[f'reports/{scope}_SUMMARY.json'])

    for scope in (['development'] if preview else ['full','confirmation_extra']):seed_plot(scope)
    development=read(ROOT/'reports/development_SUMMARY.json')['conditions']
    screen=[r for r in development if r['condition'].endswith('-seed42') and r.get('evaluation_role')=='performance' and r['completed']==r['expected']]
    fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
    for i,row in enumerate(screen):
        value=row['suite_task_equal_casr'];ci=row['casr_ci95']
        if value is None:continue
        ax.scatter(i,100*value,color='#0072B2',zorder=3)
        if ci is not None:ax.vlines(i,100*ci[0],100*ci[1],color='#0072B2')
    ax.set(xticks=range(len(screen)),xticklabels=[r['condition'].removesuffix('-seed42') for r in screen],ylim=(0,105),ylabel='Suite/task-equal CASR (%)',title='Complete development screening, fixed seed 42')
    ax.grid(axis='y',alpha=.2)
    save(fig,'development_screening_seed42','All completed seed42 performance arms are retained, including negative results. The 16-start initialization smoke test is a different cohort and is excluded.',[{k:r[k] for k in ['condition','completed','suite_task_equal_casr','casr_ci95']} for r in screen],['reports/development_SUMMARY.json'])

    scope='development' if preview else 'full'
    rows=selected_rows(read(ROOT/f'reports/{scope}_SUMMARY.json')['conditions'],selection)
    suites=['libero_spatial','libero_object','libero_goal','libero_10'];matrix=[]
    for method in selection['methods']:
        cells=[]
        for suite in suites:
            values=[v['task_equal_casr'] for r in rows if r['condition'].startswith(method+'-seed') for v in r['by_suite'] if v['suite']==suite and v['task_equal_casr'] is not None]
            cells.append(100*float(np.mean(values)) if values else None)
        matrix.append(cells)
    fig,ax=plt.subplots(figsize=(6.2,4.1),layout='constrained');im=ax.imshow(np.asarray(matrix,dtype=float),vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set(xticks=range(4),xticklabels=['Spatial','Object','Goal','Long'],yticks=range(len(matrix)),yticklabels=selection['methods'],title=('Development' if preview else 'Full evaluation')+': mean over three training seeds')
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,'NA' if value is None else f'{value:.1f}',ha='center',va='center',color='white' if value is not None and value>60 else 'black')
    fig.colorbar(im,ax=ax,label='Task-equal CASR (%)')
    save(fig,scope+'_suite_heatmap','Within each suite, tasks are equally weighted. Cells average all three fixed training seeds; undefined values remain NA.',dict(methods=selection['methods'],suites=suites,percent=matrix),[f'reports/{scope}_SUMMARY.json'])

    mechanism=read(ROOT/'reports/M03_SUMMARY.json')['records']
    native={r['condition']:r for r in development}
    clean={(r['suite'],r['task_id'],r['state']):r['success'] for r in mechanism if r['condition']=='clean'}
    cross=read(ROOT/'reports/M01_RESULTS.json')['results'];restores=read(ROOT/'reports/M02_RESULTS_FLOAT32.json')['results']
    cross_values=[]
    for method in ['B02','B01']:
        subset=[r for r in cross if f'-{method}-' in r['condition']]
        qualified=[r for r in subset if clean[(r['suite'],r['task_id'],r['state'])]]
        require(len(subset)==80 and qualified,'Cross-view diagnostic cohort changed')
        cross_values.append(100*sum(not r['success'] for r in qualified)/len(qualified))
    matrix=[[100*native['B02-seed42']['pooled_casr'],cross_values[0]],[cross_values[1],100*native['B01-seed42']['pooled_casr']]]
    recovery=[dict(condition='Clean',success=native['clean']['success'],n=native['clean']['completed']),dict(condition='PNG',success=native['B01-seed42']['success'],n=native['B01-seed42']['completed'])]
    for branch in ['dino','siglip','both']:
        subset=[r for r in restores if f'-restore-{branch}-' in r['condition']]
        require(len(subset)==80,'Float32 restoration cohort changed')
        recovery.append(dict(condition={'dino':'DINO restored','siglip':'SigLIP restored','both':'Both restored'}[branch],success=sum(r['success'] for r in subset),n=len(subset)))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');axes[0].imshow(matrix,vmin=0,vmax=100,cmap='Blues')
    axes[0].set(xticks=[0,1],xticklabels=['Front','Wrist'],yticks=[0,1],yticklabels=['Front','Wrist'],xlabel='Deployment view',ylabel='Training view',title='Cross-view diagnostic: CASR (%)')
    for i in range(2):
        for j in range(2):axes[0].text(j,i,f'{matrix[i][j]:.1f}',ha='center',va='center',color='white' if matrix[i][j]>60 else 'black')
    values=[100*r['success']/r['n'] for r in recovery];axes[1].bar(range(len(values)),values,color=['#666666','#D55E00','#0072B2','#009E73','#CC79A7'])
    axes[1].set(xticks=range(len(values)),xticklabels=[r['condition'].replace(' restored','') for r in recovery],ylim=(0,110),ylabel='Robot task success (%)',title='Privileged same-state feature restoration')
    for i,r in enumerate(recovery):axes[1].text(i,values[i]+2,f'{r["success"]}/{r["n"]}',ha='center',fontsize=9)
    save(fig,'view_and_branch_diagnostics','Seed42 matched 80-start development cohort. Left: cross-view diagnostic. Right: clean same-state feature donors are privileged mechanism probes, separate from deployable PNG scores. Only the accepted float32 M02 revision is used.',dict(cross_view_percent=matrix,restoration=recovery),['reports/development_SUMMARY.json','reports/M01_RESULTS.json','reports/M02_RESULTS_FLOAT32.json','reports/M03_SUMMARY.json'])

    action_matrix=[]
    for method in selection['methods']:
        subset=[r for r in mechanism if r['condition']==f'{method}-seed42' and r['native_png']]
        require(len(subset)==80,'Action decomposition requires the matched 80-start cohort')
        value=hierarchical_mean(subset,'action_mse_by_dim');require(value.shape==(7,) and np.isfinite(value).all(),'Invalid seven-dimensional action decomposition')
        action_matrix.append(value.tolist())
    fig,ax=plt.subplots(figsize=(8.3,4),layout='constrained');im=ax.imshow(action_matrix,cmap='viridis',aspect='auto',vmin=0)
    ax.set(xticks=range(7),xticklabels=['dx','dy','dz','rot-x','rot-y','rot-z','gripper'],yticks=range(len(action_matrix)),yticklabels=selection['methods'],title='Action deviation: matched development cohort, seed 42')
    fig.colorbar(im,ax=ax,label='Normalized action MSE')
    for i,row in enumerate(action_matrix):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='black' if im.norm(value)>.6 else 'white',fontsize=8)
    save(fig,'action_error_by_dimension','Existing M03 errors include all eight predicted slots, not only executed slots. Episode-level errors are averaged within task, then suite, then across suites. Action commands are not physical displacement distances.',dict(methods=selection['methods'],dimensions=['dx','dy','dz','rot-x','rot-y','rot-z','gripper'],normalized_mse=action_matrix),['reports/M03_SUMMARY.json'])

    costs=list(csv.DictReader((ROOT/'GPU_COST.csv').open()));totals=defaultdict(float);counts=defaultdict(int)
    for row in costs:
        label='Rejected interface' if row['result_validity']!='NOT_INVALIDATED' else 'Failed / interrupted' if row['status']!='DONE' else 'Accepted completed'
        totals[label]+=float(row['gpu_hours']);counts[label]+=1
    labels=['Accepted completed','Rejected interface','Failed / interrupted'];values=[totals[k] for k in labels]
    fig,ax=plt.subplots(figsize=(7.4,3.8),layout='constrained');ax.bar(range(3),values,color=['#0072B2','#D55E00','#999999'])
    ax.set(xticks=range(3),xticklabels=labels,ylabel='Allocated GPU hours',title='Compute cost, including unsuccessful attempts')
    for i,value in enumerate(values):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom')
    save(fig,'gpu_costs','Slurm allocation time is preferred, including startup failures before CUDA initialized. Explicitly labelled process-time fallback is retained in GPU_COST.csv. CPU-only allocation contributes zero GPU hours; this is allocation time, not measured GPU utilization.',dict(gpu_hours={k:totals[k] for k in labels},record_counts={k:counts[k] for k in labels},time_sources=dict(__import__('collections').Counter(r.get('gpu_cost_time_source','legacy') for r in costs))),['GPU_COST.csv'])

    manifest_path=destination/'MANIFEST.json';atomic(dict(preview=preview,figures=manifest),manifest_path);files.append(manifest_path)
    index=destination/'README.md';index.write_text('# '+('Development preview' if preview else 'Final recorded-result figures')+'\n\nEvery figure has PNG, PDF, and exact data/source hashes in its .data.json file. Preview output does not complete V03.\n\n'+'\n\n'.join(f'## {r["name"]}\n\n{r["caption"]}\n\n![{r["name"]}]({r["name"]}.png)' for r in manifest)+'\n');files.append(index)
    return files


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    print('\n'.join(str(p.relative_to(ROOT)) for p in export(args.preview)))

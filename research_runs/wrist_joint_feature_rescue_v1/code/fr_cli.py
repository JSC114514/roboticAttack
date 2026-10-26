"""Manual submission only; frozen joint model, two GPUs maximum, no training stage."""
import argparse
import json
import os
import signal
import sys
import time
from fr_common import ROOT,config,signature,read,require,complete,Monitor,lock,episode_receipt
import jr_cli as base
import fr_prepare
import fr_report
sys.modules['jr_prepare']=fr_prepare
base.PREFIX='featres1_'
_plans=base.plans
def plans(workers,*args,**kwargs):
    require(1<=workers<=2,'At most two GPU workers for this rescue')
    return _plans(workers,*args,**kwargs)
base.plans=plans


def worker(stage,workers):
    require(os.environ.get('JR_EXPECTED_SIGNATURE')==signature(),'Submitted revision differs')
    if stage=='__launch_check__':
        print(json.dumps(dict(root=str(ROOT),signature=signature())));return
    require(1<=workers<=2,'At most two GPU workers')
    def terminate(signum,frame):raise RuntimeError('Scheduler termination; completed episodes preserved')
    signal.signal(signal.SIGTERM,terminate)
    index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    with lock(stage+'_'+str(index)),Monitor(stage+'_'+str(index)) as monitor:
        if stage in ('clean','evaluate'):
            from fr_rollout import run
            run(stage,index,workers,monitor)
        elif stage=='report':fr_report.run(monitor)
        elif stage=='videos':fr_report.videos(monitor)
        else:raise ValueError(stage)


def status(slurm):
    out=dict(signature=signature(),prepared=False,completed=0,expected=144,clean_completed=0,training=False)
    if (ROOT/'data/MANIFEST.json').exists():
        m=read(ROOT/'data/MANIFEST.json');out['prepared']=m['signature']==signature()
        if out['prepared']:
            for c in m['cases']:
                for v in m['conditions']:
                    if complete(episode_receipt(c,v['name']),artifacts=False):
                        out['completed']+=1;out['clean_completed']+=int(v['name']=='clean')
    out['progress']=[read(p) for p in sorted((ROOT/'reports/PROGRESS').glob('*.json'))]
    if (ROOT/'results/STATUS.json').exists():out['results']=read(ROOT/'results/STATUS.json')
    if slurm:
        try:out['queue']=[v for v in base.queue() if v[2].startswith(base.PREFIX)]
        except Exception as error:out['slurm_error']=str(error)
        p=ROOT/'SUBMISSION.json'
        if p.exists():
            ids=list(read(p).get('jobs',{}).values())
            if ids:
                import subprocess
                try:
                    result=subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode','-P'],capture_output=True,text=True,timeout=30)
                    out['accounting']=result.stdout if result.returncode==0 else result.stderr
                except Exception as error:out['accounting_error']=str(error)
    print(json.dumps(out,indent=2,ensure_ascii=False));return out


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for a in ('prepare','assets','check','submit','resume','dry-run','status','watch','report','export-videos'):g.add_argument('--'+a,action='store_true')
    g.add_argument('--worker',choices=('clean','evaluate','report','videos','__launch_check__'))
    p.add_argument('--workers',type=int,choices=(1,2,3,4),default=2);p.add_argument('--slurm',action='store_true');a=p.parse_args()
    if a.worker:worker(a.worker,a.workers);return
    require(1<=a.workers<=2,'Rescue limited to two workers')
    if a.check:base.check()
    elif a.submit or a.resume or a.dry_run:base.submit(a.workers,a.dry_run)
    elif a.status:status(a.slurm)
    elif a.watch:
        try:
            while True:status(a.slurm);time.sleep(15)
        except KeyboardInterrupt:pass
    else:
        stage='prepare' if a.prepare or a.assets else 'report' if a.report else 'videos'
        with lock(stage),Monitor(stage) as monitor:
            if stage=='prepare':fr_prepare.run(monitor)
            elif stage=='report':fr_report.run(monitor)
            else:fr_report.videos(monitor)


if __name__=='__main__':main()

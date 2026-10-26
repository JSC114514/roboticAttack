"""Manual-only submission and episode continuation for the finite-pulse mechanism screen."""
import argparse
import ast
import json
import os
import shlex
import signal
import subprocess
import time
from dp_common import (ROOT, PYTHON, config, read, atomic, require, signature, complete, verify,
                       lock, Monitor, expected_receipts)
import jr_cli as scheduler
scheduler.PREFIX = 'dinopulse1_'


def plans(workers, done_eval=False, done_report=False, done_video=False):
    require(1 <= workers <= 2, 'At most two GPU workers')
    rows = []
    if not done_eval:
        rows.append(dict(name='evaluate',launcher='evaluate.sbatch',after='afterok',parents=[],array=f'0-{workers-1}%{workers}'))
    if not done_report:
        rows.append(dict(name='report',launcher='cpu.sbatch',after='afterany',parents=['evaluate'] if not done_eval else []))
    if not done_video:
        rows.append(dict(name='videos',launcher='cpu.sbatch',after='afterany',parents=['report'] if not done_report else
                         (['evaluate'] if not done_eval else [])))
    return rows


def worker(stage, workers):
    require(os.environ.get('JR_EXPECTED_SIGNATURE') == signature(), 'Submitted revision differs')
    if stage == '__launch_check__':
        print(json.dumps(dict(root=str(ROOT),signature=signature()))); return
    require(1 <= workers <= 2, 'At most two GPU workers')
    def terminate(signum, frame):raise RuntimeError('Scheduler termination; completed branches preserved')
    signal.signal(signal.SIGTERM, terminate)
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    with lock(stage+'_'+str(index)), Monitor(stage+'_'+str(index)) as monitor:
        if stage == 'evaluate':
            from dp_rollout import run
            run(index,workers,monitor)
        elif stage in ('report','videos'):
            import dp_report
            (dp_report.run if stage=='report' else dp_report.videos)(monitor)
        else:raise ValueError(stage)


def check():
    paths = [p for folder in ('code','tests') for p in (ROOT/folder).glob('*.py')]
    for p in paths:ast.parse(p.read_text(),filename=str(p))
    shells = list((ROOT/'slurm').glob('*.sbatch'))+[ROOT/'slurm/environment.sh',ROOT/'run.sh']
    for p in shells:
        result = subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
        require(result.returncode==0,'Shell syntax failure: '+result.stderr)
    env = dict(os.environ,JR_EXPECTED_SIGNATURE=signature(),CUDA_VISIBLE_DEVICES='0')
    for name in ('evaluate','cpu'):
        result = subprocess.run(['bash',str(ROOT/'slurm'/(name+'.sbatch')),'__launch_check__','2'],
                                env=env,capture_output=True,text=True,timeout=30)
        require(result.returncode==0,'Launcher failure: '+result.stderr)
        require(json.loads(result.stdout)==dict(root=str(ROOT),signature=signature()),'Wrong launcher destination')
    result = subprocess.run([PYTHON,'-B','-m','pytest','-q','-p','no:cacheprovider',str(ROOT/'tests')],
                            capture_output=True,text=True,timeout=180)
    print(result.stdout,end='')
    require(result.returncode==0,'CPU contracts failed: '+result.stderr)
    proof = dict(status='PASS',signature=signature(),tests=result.stdout.strip(),GPU_simulation_tested=False,
                 parsed_files=len(paths),checked_unix=time.time())
    atomic(proof,ROOT/'reports/CPU_CHECKS.json');print(json.dumps(proof))


def submit(workers, dry=False):
    check()
    from dp_prepare import run
    with lock('prepare'), Monitor('prepare') as monitor:m = run(monitor)
    m = verify()
    with lock('submission'):
        if not dry:
            active = [row for row in scheduler.queue() if row[2].startswith(scheduler.PREFIX)]
            if active:
                print('ALREADY_ACTIVE: no duplicate jobs submitted.');print(json.dumps(active));return
            if (ROOT/'SUBMISSION.json').exists():
                require(read(ROOT/'SUBMISSION.json').get('status')!='AMBIGUOUS_SUBMISSION_RESPONSE',
                        'Inspect recorded job IDs and queue before another submission')
        evaluated = all(complete(p) for p in expected_receipts(m))
        reported = complete(ROOT/'results/COMPLETE.json')
        video_file = ROOT/'reports/VIDEO_STATUS.json'
        video_done = evaluated and video_file.exists() and read(video_file)['status']=='COMPLETE'
        jobs = {}
        for row in plans(workers,evaluated,reported,video_done):
            cmd = scheduler.command(row,workers,[jobs[k] for k in row['parents'] if k in jobs])
            if dry:
                jobs[row['name']] = str(len(jobs)+1);print(shlex.join(cmd));continue
            state = dict(signature=signature(),jobs=dict(jobs),workers=workers,status='SUBMITTING',
                         pending_command=cmd,submitted_unix=time.time())
            atomic(state,ROOT/'SUBMISSION.json')
            try:result = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            except BaseException as error:
                atomic({**state,'status':'AMBIGUOUS_SUBMISSION_RESPONSE','error':repr(error)},ROOT/'SUBMISSION.json');raise
            if result.returncode:
                atomic({**state,'status':'SUBMISSION_FAILED','error':result.stderr},ROOT/'SUBMISSION.json')
                raise RuntimeError('sbatch failed; earlier IDs retained: '+result.stderr)
            job = result.stdout.strip().split(';')[0]
            if not job.isdigit():
                atomic({**state,'status':'AMBIGUOUS_SUBMISSION_RESPONSE','error':result.stdout},ROOT/'SUBMISSION.json')
                raise RuntimeError('Unclear submission response; inspect queue')
            jobs[row['name']] = job
            atomic({**state,'jobs':dict(jobs),'status':'SUBMITTED','pending_command':None},ROOT/'SUBMISSION.json')
            print(row['name'],job,flush=True)
        if not jobs:print('ALL_COMPLETE: no jobs needed')


def status(slurm=False):
    out = dict(signature=signature(),prepared=False,expected=128,completed=0,training=False)
    if (ROOT/'data/MANIFEST.json').exists():
        m = read(ROOT/'data/MANIFEST.json');out['prepared'] = m['signature']==signature()
        if out['prepared']:
            out.update(completed=sum(complete(p,artifacts=False) for p in expected_receipts(m)),runnable=m['runnable'],
                       unavailable=m['expected']-m['runnable'])
    out['progress'] = [read(p) for p in sorted((ROOT/'reports/PROGRESS').glob('*.json'))]
    if (ROOT/'results/STATUS.json').exists():out['results'] = read(ROOT/'results/STATUS.json')
    if slurm:
        out['queue'] = [row for row in scheduler.queue() if row[2].startswith(scheduler.PREFIX)]
        if (ROOT/'SUBMISSION.json').exists():
            ids = list(read(ROOT/'SUBMISSION.json').get('jobs',{}).values())
            if ids:
                result = subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode','-P'],
                                        capture_output=True,text=True,timeout=30)
                out['accounting'] = result.stdout if result.returncode==0 else result.stderr
    print(json.dumps(out,indent=2,ensure_ascii=False));return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','check','submit','resume','dry-run','status','watch','report','export-videos'):
        group.add_argument('--'+name,action='store_true')
    group.add_argument('--worker',choices=('evaluate','report','videos','__launch_check__'))
    parser.add_argument('--workers',type=int,choices=(1,2),default=2);parser.add_argument('--slurm',action='store_true')
    args = parser.parse_args()
    if args.worker:worker(args.worker,args.workers)
    elif args.check:check()
    elif args.submit or args.resume or args.dry_run:submit(args.workers,args.dry_run)
    elif args.status:status(args.slurm)
    elif args.watch:
        try:
            while True:status(args.slurm);time.sleep(15)
        except KeyboardInterrupt:pass
    else:
        stage = 'prepare' if args.prepare else 'report' if args.report else 'videos'
        with lock(stage), Monitor(stage) as monitor:
            if stage=='prepare':
                from dp_prepare import run
                run(monitor)
            else:
                import dp_report
                (dp_report.run if stage=='report' else dp_report.videos)(monitor)


if __name__=='__main__':main()

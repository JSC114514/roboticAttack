"""Manual submission only; narrow arrays, deduplication and receipt-based resume."""
import argparse
import fcntl
import getpass
import os
import re
import shlex
import signal
import subprocess
import time
from sd_common import (ROOT,prepare,signature,require,complete,read_json,atomic_json,
    worker_complete,verify_inputs,Monitor,cost,resources)

def queue():
    p=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%100j|%R'],capture_output=True,text=True,timeout=20)
    require(p.returncode==0,'Slurm queue unavailable; no submission: '+p.stderr.strip())
    return p.stdout

def commands():
    result=[];deps=[]
    for stage in ('capture','mechanism'):
        indices=[i for i in (0,1) if not worker_complete(stage,i)]
        if not indices:continue
        cmd=['sbatch','--parsable','--job-name=sdm1_'+signature()[:8]+'_'+stage,
            '--export=ALL,SD_EXPECTED_SIGNATURE='+signature(),
            '--array='+','.join(map(str,indices))+'%2']
        if deps:cmd.append('--dependency=afterany:'+':'.join(deps))
        result.append((stage,cmd+[str(ROOT/f'slurm/{stage}.sbatch')]))
        deps=['<'+stage.upper()+'_JOB_ID>']
    if not complete(ROOT/'results/COMPLETE.json'):
        cmd=['sbatch','--parsable','--job-name=sdm1_'+signature()[:8]+'_report',
             '--export=ALL,SD_EXPECTED_SIGNATURE='+signature()]
        if deps:cmd.append('--dependency=afterany:'+':'.join(deps))
        result.append(('report',cmd+[str(ROOT/'slurm/report.sbatch')]))
    return result

def submit():
    prepare()
    with (ROOT/'reports/locks/submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=queue()
        if any(len(v:=r.split('|'))>=3 and v[2].strip().startswith('sdm1_') for r in active.splitlines()):
            print('ALREADY_ACTIVE: no duplicate jobs');return
        schedule=commands()
        if not schedule:print('ALREADY_COMPLETE: no resources requested');return
        path=ROOT/'reports/SUBMISSION.json'
        if path.exists():
            old=read_json(path);require(old['status']!='AMBIGUOUS_SUBMISSION_RESPONSE','Inspect uncertain previous submission before retrying')
            atomic_json(old,ROOT/f'reports/SUBMISSIONS/{time.time_ns()}.json')
        meta={'signature':signature(),'status':'SUBMITTING','jobs':[],'max_concurrent_gpus':2,
              'training_jobs':0,'automatic_cancellation':False}
        atomic_json(meta,path);ids={}
        for stage,template in schedule:
            command=[next((v.replace('<'+k.upper()+'_JOB_ID>',j) for k,j in ids.items() if '<'+k.upper()+'_JOB_ID>' in v),v) for v in template]
            meta['pending_command']=command;atomic_json(meta,path)
            try:
                p=subprocess.run(command,capture_output=True,text=True,timeout=30)
                job=p.stdout.strip().split(';')[0]
                require(p.returncode==0 and re.fullmatch(r'\d+',job),p.stderr or p.stdout)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error));atomic_json(meta,path)
                raise RuntimeError('No retry or cancellation; inspect SUBMISSION.json and Slurm') from error
            ids[stage]=job;meta['jobs'].append({'stage':stage,'job_id':job,'command':command})
            meta.pop('pending_command',None);atomic_json(meta,path);print(stage,job,flush=True)
        meta['status']='SUBMITTED';atomic_json(meta,path)

def worker(stage):
    require(bool(os.environ.get('SLURM_JOB_ID')),'Worker requires user-submitted allocation')
    require(os.environ.get('SD_EXPECTED_SIGNATURE')==signature(),'Submitted code changed')
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'CPU preparation incomplete')
    i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));require(i in (0,1),'Invalid worker index')
    def stop(n,_frame):raise SystemExit(128+n)
    signal.signal(signal.SIGTERM,stop)
    name=f'{stage}_{i}'
    with (ROOT/f'reports/locks/{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Monitor(name) as monitor,cost(name):
            verify_inputs()
            if stage=='capture':
                from sd_replay import run
                run(i,monitor)
            elif stage=='mechanism':
                from sd_mechanism import run
                run(i,monitor)
            else:
                from sd_report import run
                run(monitor)

def status(slurm=False):
    print('revision',signature()[:12],'; <=2x4090, each4CPU/12G; no PNG training')
    for stage in ('capture','mechanism'):
        print(stage,{i:worker_complete(stage,i) for i in (0,1)})
    for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read_json(p);age=round(time.time()-v['heartbeat_unix'])
        print(p.stem,{k:v.get(k) for k in ('status','phase','episode_id','case_id','condition','completed','total',
            'executed_steps','distance_cm','estimated_remaining_s','process_rss_hwm_bytes',
            'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','error')},'heartbeat_age_s',age,
            'STALE' if v.get('status')=='RUNNING' and age>90 else '')
    p=ROOT/'results/STATUS.json'
    if p.exists():print('results',read_json(p))
    if slurm:
        try:
            print('\n'.join(r for r in queue().splitlines() if 'sdm1_' in r))
            p=ROOT/'reports/SUBMISSION.json'
            if p.exists():
                ids=[j['job_id'] for j in read_json(p)['jobs']]
                if ids:
                    r=subprocess.run(['sacct','-j',','.join(ids),'-P','--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'],capture_output=True,text=True,timeout=20)
                    print(r.stdout or r.stderr)
        except (OSError,RuntimeError,subprocess.TimeoutExpired) as error:print('SLURM_UNAVAILABLE',str(error))

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for flag in ('preflight','submit','resume','dry-run','status','watch','report'):g.add_argument('--'+flag,action='store_true')
    g.add_argument('--worker',choices=('capture','mechanism','report'));p.add_argument('--slurm',action='store_true');p.add_argument('--interval',type=int,default=30)
    a=p.parse_args()
    if a.preflight:
        started=time.monotonic();m=prepare()
        atomic_json({'status':'PASS','signature':signature(),'elapsed_s':time.monotonic()-started,
            'mujoco_imported':False,'model_loaded':False,'jobs_submitted':False,**resources()},ROOT/'reports/CPU_PREFLIGHT.json')
        print('PREFLIGHT_PASS',signature(),'14 trajectories, <=28 states, <=84 short rollouts; no jobs submitted')
    elif a.submit or a.resume:submit()
    elif a.dry_run:
        prepare()
        for name,cmd in commands():print(name,shlex.join(cmd))
    elif a.worker:worker(a.worker)
    elif a.report:
        prepare()
        with (ROOT/'reports/locks/report_0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from sd_report import run
            with Monitor('report_0') as monitor,cost('manual_report'):run(monitor)
    elif a.watch:
        require(a.interval>=10,'Watch interval must be >=10 seconds')
        try:
            while True:status(a.slurm);time.sleep(a.interval)
        except KeyboardInterrupt:pass
    else:status(a.slurm)

if __name__=='__main__':main()

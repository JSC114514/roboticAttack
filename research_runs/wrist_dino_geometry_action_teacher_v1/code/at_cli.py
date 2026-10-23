"""Manual-only Slurm DAG: cache(2) -> train(4) -> select(CPU) -> evaluate(4)."""
import argparse
from datetime import datetime
import fcntl
import getpass
import os
import re
import signal
import subprocess
import time
from at_common import (ROOT,config,cutoff,candidates,prepare,signature,complete,cache_complete,
    read_json,atomic_json,require,Monitor,cost,verify_inputs)

def queue():
    r=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(r.returncode==0,'Slurm unavailable; no submission: '+r.stderr.strip())
    return r.stdout

def command(stage,indices=None,deps=()):
    cmd=['sbatch','--parsable','--job-name=gat_'+signature()[:8]+'_'+stage,
         '--export=ALL,AT_EXPECTED_SIGNATURE='+signature()]
    if indices is not None:cmd.append('--array='+','.join(map(str,indices))+('%2' if stage=='cache' else '%4'))
    if deps:cmd.append('--dependency=afterany:'+':'.join(dict.fromkeys(deps)))
    if stage=='interim':
        # The begin time is interpreted in Asia/Shanghai, set by environment.sh.
        cmd.append('--begin='+datetime.fromisoformat(config()['interim_at']).strftime('%Y-%m-%dT%H:%M:%S'))
    return cmd+[str(ROOT/f'slurm/{stage}.sbatch')]

def stages(one,include_interim=True):
    deps=[]
    # Once evaluation checkpoints are sealed they never change on --resume.
    # Otherwise resumed training could overwrite an already evaluated checkpoint.
    if not complete(ROOT/'reports/SELECT_COMPLETE.json'):
        missing=[i for i in range(2) if not cache_complete(i)]
        if missing:
            require(time.time()<cutoff(),'Training cutoff passed and teachers missing; no GPU chain submitted')
            deps=[one('cache',missing,())]
        train=[i for i,(name,_,_) in enumerate(candidates()) if not complete(ROOT/f'training/{name}/COMPLETE.json')]
        if train and time.time()<cutoff():deps=[one('train',train,deps)]
        deps=[one('select',None,deps)]
    missing=[i for i in range(4) if not complete(ROOT/f'evaluation/worker_{i}_COMPLETE.json')]
    evaluation_deps=deps
    if missing:evaluation_deps=[one('evaluate',missing,deps)]
    if not complete(ROOT/'results/COMPLETE.json'):
        one('report',None,evaluation_deps)
        if include_interim and time.time()<datetime.fromisoformat(config()['interim_at']).timestamp():one('interim',None,())

def submit():
    with cost('prepare_submit'):prepare()
    with (ROOT/'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=queue()
        names=[x.split('|')[2] for x in active.splitlines()]
        if any(n.startswith('gat_') and not n.endswith('_interim') for n in names):
            print('ALREADY_ACTIVE: no duplicate jobs submitted');return
        if complete(ROOT/'results/COMPLETE.json'):
            print('ALREADY_COMPLETE: no resources requested');return
        path=ROOT/'reports/SUBMISSION.json'
        if path.exists():
            old=read_json(path)
            require(old['status']!='AMBIGUOUS_SUBMISSION_RESPONSE','Inspect uncertain Slurm response before retrying')
            atomic_json(old,ROOT/f'reports/SUBMISSIONS/{time.time_ns()}.json')
        meta={'signature':signature(),'status':'SUBMITTING','submitted_unix':time.time(),'jobs':[],
              'max_concurrent_gpus':4,'train_until':config()['train_until'],'expected_closedloop_episodes':640}
        atomic_json(meta,path)
        def one(stage,indices,deps):
            cmd=command(stage,indices,deps);meta['pending_command']=cmd;atomic_json(meta,path)
            try:
                value=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                job=value.stdout.strip().split(';')[0]
                require(value.returncode==0 and re.fullmatch(r'\d+',job),value.stderr or value.stdout)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error));atomic_json(meta,path)
                raise RuntimeError('Inspect Slurm/receipt; no automatic resubmission or cancellation') from error
            meta['jobs'].append({'stage':stage,'job_id':job,'command':cmd});meta.pop('pending_command',None)
            atomic_json(meta,path);print(stage,job,flush=True);return job
        stages(one,include_interim=not any(n.startswith('gat_') and n.endswith('_interim') for n in names))
        meta['status']='SUBMITTED';atomic_json(meta,path)

def status(slurm):
    print('revision',signature()[:12],'max4x4090; train16G/eval12G; cutoff',config()['train_until'])
    for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read_json(p);age=round(time.time()-v['heartbeat_unix'])
        print(p.stem,{k:v.get(k) for k in ('status','phase','completed','total','estimated_remaining_s',
            'candidate','suite','task_id','state_index','condition','loss','optimized_updates','error')},
            'heartbeat_age_s',age,'STALE' if v.get('status')=='RUNNING' and age>90 else '')
    if (ROOT/'results/STATUS.json').exists():
        v=read_json(ROOT/'results/STATUS.json')
        print('RESULTS',{k:v.get(k) for k in ('status','closedloop_episodes','fully_paired_starts','offline_observations','primary')})
    if slurm:
        try:
            print(queue())
            if (ROOT/'reports/SUBMISSION.json').exists():
                ids=[v['job_id'] for v in read_json(ROOT/'reports/SUBMISSION.json')['jobs']]
                if ids:
                    r=subprocess.run(['sacct','-j',','.join(ids),'-P','--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'],capture_output=True,text=True,timeout=20)
                    print(r.stdout or r.stderr)
        except (RuntimeError,subprocess.TimeoutExpired) as error:print('SLURM_UNAVAILABLE',str(error))

def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'),'Manual Slurm allocation required')
    require(os.environ.get('AT_EXPECTED_SIGNATURE')==signature(),'Submitted revision changed')
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Preparation missing')
    index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    require(stage not in ('cache','train','evaluate') or index in range(2 if stage=='cache' else 4),'Invalid worker index')
    key=stage+'_'+str(index) if stage in ('cache','train','evaluate') else stage
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    with (ROOT/f'reports/locks/{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Monitor(key) as monitor,cost(key):
            if stage in ('cache','train','evaluate'):verify_inputs()
            if stage=='train':require(all(cache_complete(i) for i in range(2)),'Teachers incomplete; refuse model load')
            if stage=='cache':
                from at_cache import run
                run(index,monitor)
            elif stage=='train':
                from at_train import run
                run(index,monitor)
            elif stage=='select':
                from at_train import select
                select(monitor)
            elif stage=='evaluate':
                require(complete(ROOT/'reports/SELECT_COMPLETE.json'),'No selected checkpoint; refuse model load')
                from at_evaluate import run
                run(index,monitor)
            else:
                from at_report import run
                run(monitor,final=stage=='report')

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for flag in ('preflight','submit','resume','dry-run','status','watch'):g.add_argument('--'+flag,action='store_true')
    g.add_argument('--worker',choices=('cache','train','select','evaluate','report','interim'))
    p.add_argument('--slurm',action='store_true');p.add_argument('--interval',type=int,default=30);a=p.parse_args()
    if a.preflight:
        with cost('prepare_preflight'):v=prepare()
        print('PREFLIGHT_PASS',signature(),v['fit_frames'],v['verify_frames'],'no jobs submitted')
    elif a.submit or a.resume:submit()
    elif a.dry_run:
        with cost('prepare_dry_run'):prepare()
        def preview(stage,indices,deps):print(' '.join(command(stage,indices,deps)));return stage.upper()+'_JOB'
        stages(preview)
    elif a.worker:worker(a.worker)
    elif a.watch:
        require(a.interval>=10,'Watch interval must be >=10 seconds')
        try:
            while True:status(a.slurm);time.sleep(a.interval)
        except KeyboardInterrupt:pass
    else:status(a.slurm)

if __name__=='__main__':main()

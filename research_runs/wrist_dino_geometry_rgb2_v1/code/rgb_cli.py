"""User-only submission, bounded two-GPU waves, cheap prerequisite failure gates."""
import argparse
import fcntl
import getpass
import os
import re
import signal
import subprocess
import time
from rgb_common import (ROOT,config,require,signature,prepare,read_json,atomic_json,complete,
                        cache_complete,episode_path,Monitor,cost)

def queue():
    result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(result.returncode==0,'Slurm unavailable; no submission: '+result.stderr.strip())
    return result.stdout

def command(stage,indices=None,deps=()):
    cmd=['sbatch','--parsable','--job-name=grgb2_'+signature()[:8]+'_'+stage,
         '--export=ALL,RGB_EXPECTED_SIGNATURE='+signature()]
    if indices is not None:cmd.append('--array='+','.join(map(str,indices))+'%2')
    if deps:cmd.append('--dependency=afterany:'+':'.join(dict.fromkeys(deps)))
    return cmd+[str(ROOT/f'slurm/{stage}.sbatch')]

def stages(one):
    deps=[]
    missing=[i for i in range(2) if not cache_complete(i)]
    if missing:deps=[one('cache',missing,())]
    missing=[i for i,k in enumerate(config()['candidates']) if not complete(ROOT/f'training/{k}/COMPLETE.json')]
    if missing:deps=[one('train',missing,deps)]
    if not complete(ROOT/'evaluation/COMPLETE.json'):deps=[one('evaluate',None,deps)]
    if not complete(ROOT/'results/COMPLETE.json'):one('report',None,deps)

def submit():
    prepare()
    with (ROOT/'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=queue()
        if any(line.split('|')[2].startswith('grgb2_') for line in active.splitlines()):
            print('ALREADY_ACTIVE: no duplicate submission');return
        if complete(ROOT/'results/COMPLETE.json'):
            print('ALREADY_COMPLETE: no resources requested');return
        path=ROOT/'reports/SUBMISSION.json'
        if path.exists():
            old=read_json(path)
            require(old['status']!='AMBIGUOUS_SUBMISSION_RESPONSE','Inspect uncertain response before retrying')
            atomic_json(old,ROOT/f'reports/SUBMISSIONS/{time.time_ns()}.json')
        meta={'signature':signature(),'status':'SUBMITTING','jobs':[],'submitted_unix':time.time(),
              'max_concurrent_gpus':2,'automatic_long_training':False,'new_closedloop_episodes':20}
        atomic_json(meta,path)
        def one(stage,indices,deps):
            cmd=command(stage,indices,deps);meta['pending_command']=cmd;atomic_json(meta,path)
            try:
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                job=result.stdout.strip().split(';')[0]
                require(result.returncode==0 and re.fullmatch(r'\d+',job),result.stderr or result.stdout)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error));atomic_json(meta,path)
                raise RuntimeError('No automatic resubmission/cancellation; inspect receipt and Slurm') from error
            meta['jobs'].append({'stage':stage,'job_id':job,'command':cmd});meta.pop('pending_command',None)
            atomic_json(meta,path);print(stage,job,flush=True);return job
        stages(one);meta['status']='SUBMITTED';atomic_json(meta,path)

def status(slurm):
    print('revision',signature()[:12],'max2x4090; cache12G/train16G/eval12G; no EOT/action loss/long train')
    for k in config()['candidates']:
        print('training',k,'complete',complete(ROOT/f'training/{k}/COMPLETE.json'))
    print('cache_workers_complete',[cache_complete(i) for i in range(2)])
    for path in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read_json(path);age=round(time.time()-v['heartbeat_unix'])
        print(path.stem,{k:v.get(k) for k in ('status','phase','suite','task_id','state_index','condition','completed','total','estimated_remaining_s','loss','q','orthogonal','error')},
              'heartbeat_age_s',age,'STALE' if v.get('status')=='RUNNING' and age>90 else '')
    for path in (ROOT/'results/STATUS.json',ROOT/'reports/SUBMISSION.json'):
        if path.exists():print(path.name,read_json(path))
    if slurm:
        try:
            print(queue())
            path=ROOT/'reports/SUBMISSION.json'
            if path.exists():
                jobs=[j['job_id'] for j in read_json(path)['jobs']]
                if jobs:
                    value=subprocess.run(['sacct','-j',','.join(jobs),'-P','--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'],capture_output=True,text=True,timeout=20)
                    print(value.stdout or value.stderr)
        except (RuntimeError,subprocess.TimeoutExpired) as error:print('SLURM_UNAVAILABLE',str(error))

def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'),'Manual Slurm allocation required')
    require(os.environ.get('RGB_EXPECTED_SIGNATURE')==signature(),'Submitted revision changed')
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Preparation missing')
    index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    require(stage not in ('cache','train') or index in (0,1),'Invalid array task')
    key=stage+'_'+str(index) if stage in ('cache','train') else stage
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    with (ROOT/f'reports/locks/{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Monitor(key) as monitor,cost(key):
            if stage in ('train','evaluate'):
                require(all(cache_complete(i) for i in range(2)),'Cache failed/incomplete; refuse model load')
            if stage=='evaluate':
                require(all(complete(ROOT/f'training/{k}/COMPLETE.json') for k in config()['candidates']),
                        'Training failed/incomplete; refuse model load')
            if stage=='cache':
                from rgb_cache import run
                run(index,monitor)
            elif stage=='train':
                from rgb_train import run
                run(config()['candidates'][index],monitor)
            elif stage=='evaluate':
                from rgb_evaluate import run
                run(monitor)
            else:
                from rgb_report import run
                run(monitor)

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    for flag in ('preflight','submit','dry-run','status','watch'):group.add_argument('--'+flag,action='store_true')
    group.add_argument('--worker',choices=('cache','train','evaluate','report'))
    parser.add_argument('--slurm',action='store_true');parser.add_argument('--interval',type=int,default=30);args=parser.parse_args()
    if args.preflight:
        value=prepare();print('PREFLIGHT_PASS',signature(),value['fit_frames'],value['verify_frames'],'no jobs submitted')
    elif args.submit:submit()
    elif args.dry_run:
        prepare()
        def preview(stage,indices,deps):print(' '.join(command(stage,indices,deps)));return stage.upper()+'_JOB'
        stages(preview)
    elif args.worker:worker(args.worker)
    elif args.watch:
        require(args.interval>=10,'Interval must be >=10 seconds')
        try:
            while True:status(args.slurm);time.sleep(args.interval)
        except KeyboardInterrupt:pass
    else:status(args.slurm)

if __name__=='__main__':main()

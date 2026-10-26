"""Only explicit --submit writes Slurm. Workers never submit or cancel jobs."""
import argparse
import fcntl
import getpass
import os
import re
import resource
import signal
import subprocess
import time
from rr_common import (ROOT,config,signature,prepare,complete,read_json,atomic_json,
    require,verify_inputs,Monitor,cost)


def queue():
    v=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(v.returncode==0,'Slurm unavailable; refusing submission: '+v.stderr.strip())
    return v.stdout


def command(stage,deps=()):
    cmd=['sbatch','--parsable','--job-name=rrp1_'+signature()[:8]+'_'+stage,
         '--export=ALL,RRP_EXPECTED_SIGNATURE='+signature()]
    if stage=='evaluate':cmd.append('--array=0,1%2')
    if deps:cmd.append('--dependency=afterany:'+':'.join(dict.fromkeys(deps)))
    if stage=='interim':cmd.append('--begin='+config()['interim_at'].split('+')[0])
    return cmd+[str(ROOT/f'slurm/{stage}.sbatch')]


def active_jobs(text):
    names=[row.split('|')[2] for row in text.splitlines() if len(row.split('|'))>=3]
    current_interim='rrp1_'+signature()[:8]+'_interim'
    return (any(n.startswith('rrp1_') and n!=current_interim for n in names),current_interim in names)


def stages(one,include_interim=True):
    deps=[]
    if not complete(ROOT/'reports/DIRECTION_COMPLETE.json'):deps=[one('gate',())]
    gate=read_json(ROOT/'reports/DIRECTION_COMPLETE.json') if complete(ROOT/'reports/DIRECTION_COMPLETE.json') else None
    if gate is None or gate['passed']:
        if not complete(ROOT/'training/COMPLETE.json'):deps=[one('train',deps)]
        if not all(complete(ROOT/f'evaluation/WORKER_{i}_COMPLETE.json') for i in (0,1)):
            deps=[one('evaluate',deps)]
    if not complete(ROOT/'results/COMPLETE.json'):one('report',deps)
    if include_interim and not (ROOT/'results/INTERIM_STATUS.json').exists():one('interim',())


def submit():
    prepare()
    with (ROOT/'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=queue()
        main_active,interim_active=active_jobs(active)
        if main_active:
            print('ALREADY_ACTIVE: no duplicate jobs');return
        if complete(ROOT/'results/COMPLETE.json'):
            print('ALREADY_COMPLETE: no jobs requested');return
        path=ROOT/'reports/SUBMISSION.json'
        if path.exists():
            old=read_json(path)
            require(old.get('status')!='AMBIGUOUS_SUBMISSION_RESPONSE','Uncertain previous submission; inspect jobs first')
            atomic_json(old,ROOT/f'reports/SUBMISSIONS/{time.time_ns()}.json')
        receipt={'signature':signature(),'status':'SUBMITTING','jobs':[],
                 'max_concurrent_gpus':2,'automatic_expansion':False}
        atomic_json(receipt,path)
        def one(stage,deps):
            cmd=command(stage,deps);receipt['pending_command']=cmd;atomic_json(receipt,path)
            try:
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                job=result.stdout.strip().split(';')[0]
                require(result.returncode==0 and re.fullmatch(r'\d+',job),result.stderr or result.stdout)
            except BaseException as error:
                receipt.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error));atomic_json(receipt,path)
                raise RuntimeError('No automatic retry/cancellation; inspect Slurm and receipt') from error
            receipt['jobs'].append({'stage':stage,'job_id':job,'command':cmd})
            receipt.pop('pending_command',None);atomic_json(receipt,path);print(stage,job,flush=True);return job
        stages(one,include_interim=not interim_active);receipt['status']='SUBMITTED';atomic_json(receipt,path)


def skipped(stage,index,reason):
    path=ROOT/f'reports/SKIPPED_{stage}_{index}.json'
    atomic_json({'status':'SKIPPED','reason':reason,'signature':signature(),
        'job_id':os.environ.get('SLURM_JOB_ID'),'cuda_initialized':False,'model_loaded':False,
        'note':'short scheduler startup allocation only; no GPU model/work'},path)
    print('SKIPPED',stage,reason)


def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'),'Manual Slurm allocation required')
    require(os.environ.get('RRP_EXPECTED_SIGNATURE')==signature(),'Submitted code revision changed')
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Preparation missing')
    index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    require(stage!='evaluate' or index in (0,1),'Invalid evaluation worker')
    if stage in ('train','evaluate'):
        gate=ROOT/'reports/DIRECTION_COMPLETE.json'
        if not complete(gate):return skipped(stage,index,'direction_stage_failed_or_incomplete')
        if not read_json(gate)['passed']:return skipped(stage,index,'direction_not_supported')
    if stage=='evaluate' and not complete(ROOT/'training/COMPLETE.json'):
        return skipped(stage,index,'training_failed_or_incomplete')
    key=f'{stage}_{index}'
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    with (ROOT/f'reports/locks/{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Monitor(key) as monitor,cost(key):
            verify_inputs()
            if stage=='gate':
                from rr_gate import run
                run(monitor)
            elif stage=='train':
                from rr_train import run
                run(monitor)
            elif stage=='evaluate':
                from rr_evaluate import run
                run(index,monitor)
            else:
                from rr_report import run
                run(monitor,interim=stage=='interim')


def status(slurm=False):
    print('revision',signature()[:12],'max2x4090; gate12G/train16G/eval12G; single2000-stepcandidate')
    for name,path in [('direction',ROOT/'reports/DIRECTION_COMPLETE.json'),('training',ROOT/'training/COMPLETE.json'),
                      ('evaluation0',ROOT/'evaluation/WORKER_0_COMPLETE.json'),('evaluation1',ROOT/'evaluation/WORKER_1_COMPLETE.json')]:
        print(name,complete(path))
    for path in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read_json(path);age=round(time.time()-v['heartbeat_unix'])
        print(path.stem,{k:v.get(k) for k in ('status','phase','completed','total','estimated_remaining_s',
            'relation_error_cm','loss','actual_updates','no_target_count','error')},'heartbeat_age_s',age,
            'STALE' if v.get('status')=='RUNNING' and age>90 else '')
    for path in sorted((ROOT/'reports').glob('SKIPPED_*.json')):print(path.name,read_json(path)['reason'])
    for name in ('STATUS.json','INTERIM_STATUS.json'):
        path=ROOT/'results'/name
        if path.exists():
            v=read_json(path);print(name,{k:v.get(k) for k in ('status','offline_states','episodes','fully_paired_starts')})
    if slurm:
        try:
            print(queue())
            path=ROOT/'reports/SUBMISSION.json'
            if path.exists():
                jobs=[r['job_id'] for r in read_json(path)['jobs']]
                if jobs:
                    v=subprocess.run(['sacct','-j',','.join(jobs),'-P','--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'],capture_output=True,text=True,timeout=20)
                    print(v.stdout or v.stderr)
        except (RuntimeError,subprocess.TimeoutExpired) as error:print('SLURM_UNAVAILABLE',str(error))


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    for flag in ('preflight','submit','dry-run','status','watch'):group.add_argument('--'+flag,action='store_true')
    group.add_argument('--worker',choices=('gate','train','evaluate','report','interim'))
    parser.add_argument('--slurm',action='store_true');parser.add_argument('--interval',type=int,default=30)
    args=parser.parse_args()
    if args.preflight:
        started=time.monotonic();v=prepare()
        atomic_json({'status':'PASS','signature':signature(),'stage':'cpu_preflight','elapsed_s':time.monotonic()-started,
            'process_rss_hwm_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            'cuda_model_loaded':False,'jobs_submitted':False},ROOT/'reports/CPU_PREFLIGHT_COST.json')
        print('PREFLIGHT_PASS',signature(),'fit',v['fit_observations'],'verify',v['verify_observations'],'no jobs submitted')
    elif args.submit:submit()
    elif args.dry_run:
        prepare()
        def preview(stage,deps):print(' '.join(command(stage,deps)));return stage.upper()+'_JOB'
        stages(preview)
    elif args.worker:worker(args.worker)
    elif args.watch:
        require(args.interval>=10,'Watch interval must be >=10 seconds')
        try:
            while True:status(args.slurm);time.sleep(args.interval)
        except KeyboardInterrupt:pass
    else:status(args.slurm)


if __name__=='__main__':main()

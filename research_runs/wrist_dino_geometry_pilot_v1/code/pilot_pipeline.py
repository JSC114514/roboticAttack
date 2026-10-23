"""Explicit manual --submit only. No scheduler calls from preflight or dry-run."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
import time
from pilot_common import (ROOT,atomic_json,complete,config,directories,load_json,preflight,require,signature)

STAGES=('prepare','mechanism','pilot','analyze')
SCRIPTS={stage:ROOT/f'slurm/{index:02d}_{stage}.sbatch' for index,stage in enumerate(STAGES,1)}
NAMES={stage:'geo1_'+stage for stage in STAGES}


def gpu_dependencies(output):
    dependencies=set()
    for line in output.splitlines():
        columns=line.split('|'); require(len(columns)==5,'Cannot parse current Slurm queue')
        job,state,tres,reason,name=columns
        if 'gpu' not in tres.lower(): continue
        require('DependencyNeverSatisfied' not in reason,'Existing GPU job has failed dependency; inspect Slurm first')
        match=re.fullmatch(r'(\d+)(?:_[\d,\[\]%\-]+)?',job.strip())
        require(match,'Cannot safely identify existing GPU job')
        dependencies.add(match.group(1))
    return sorted(dependencies,key=int)


def command_for(stage,sid,previous=None,existing=()):
    command=['sbatch','--parsable','--job-name=geo1_'+sid[:8]+'_'+stage]
    dependencies=[]
    if previous: dependencies.append(previous)
    dependencies.extend(x for x in existing if x not in dependencies)
    if dependencies: command.append('--dependency=afterany:'+':'.join(dependencies))
    return command+[str(SCRIPTS[stage])]


def remaining(prep):
    sid=prep['signature']
    if complete(ROOT/'results/COMPLETE.json',sid): return []
    checks={'prepare':ROOT/'reports/PREPARE_COMPLETE.json','mechanism':ROOT/'reports/MECHANISM_COMPLETE.json',
            'pilot':ROOT/'reports/PILOT_COMPLETE.json'}
    return [stage for stage in STAGES if stage=='analyze' or not complete(checks[stage],sid)]


def submit():
    prep=preflight(); sid=prep['signature']; directories()
    stages=remaining(prep)
    if not stages: print('ALREADY_COMPLETE; no submission'); return
    path=ROOT/'reports/SUBMISSION.json'
    with (ROOT/'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%b|%R|%j'],capture_output=True,text=True,timeout=30)
        require(queue.returncode==0,'Slurm queue unavailable; nothing submitted: '+queue.stderr.strip())
        active={line.split('|')[0].split('_')[0] for line in queue.stdout.splitlines()}
        old=load_json(path) if path.exists() else None
        if old:
            require(old['signature']==sid,'Previous chain used different sources')
            require(old['status'] not in ('SUBMITTING','AMBIGUOUS_SBATCH_RESPONSE'),
                    'Unresolved submission receipt; inspect Slurm/receipt before any manual retry')
            require(not active.intersection(j['job_id'] for j in old['jobs']),'Existing chain is active; use --status')
        require(not any('geo1_'+sid[:8] in line for line in queue.stdout.splitlines()),'Duplicate active geometry pilot')
        deps=gpu_dependencies(queue.stdout) if any(s in ('mechanism','pilot') for s in stages) else []
        if old: atomic_json(old,ROOT/'reports/SUBMISSION_ARCHIVE'/f'{time.time_ns()}.json')
        receipt={'status':'SUBMITTING','signature':sid,'jobs':[],'existing_gpu_dependencies':deps,'resources':config()['resources']}
        atomic_json(receipt,path); previous=None; first_gpu=True
        for stage in stages:
            use_existing=deps if stage in ('mechanism','pilot') and first_gpu else []
            if stage in ('mechanism','pilot'): first_gpu=False
            cmd=command_for(stage,sid,previous,use_existing)
            receipt['pending_command']=cmd; atomic_json(receipt,path)
            try:
                response=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                job=response.stdout.strip().split(';')[0]
                require(response.returncode==0 and re.fullmatch(r'\d+',job),response.stderr.strip() or 'Unknown sbatch response')
            except BaseException as error:
                receipt.update(status='AMBIGUOUS_SBATCH_RESPONSE',error=repr(error)); atomic_json(receipt,path)
                raise RuntimeError('Submission not automatically retried/cancelled; inspect receipt and Slurm') from error
            receipt['jobs'].append({'stage':stage,'job_id':job,'command':cmd}); receipt.pop('pending_command',None)
            atomic_json(receipt,path); previous=job; print(stage,job,flush=True)
        receipt['status']='SUBMITTED'; atomic_json(receipt,path)


def status(slurm=False):
    for name in ('reports/PREFLIGHT.json','reports/SUBMISSION.json','reports/GPU_RGB_INTERFACE.json',
                 'reports/PILOT_GPU_RGB_INTERFACE.json','reports/PROGRESS.json','results/STATUS.json'):
        path=ROOT/name
        if not path.exists(): print(name,'NOT_STARTED'); continue
        value=load_json(path)
        if name.endswith('PREFLIGHT.json'):
            print(name,{k:value[k] for k in ('status','signature')},'GPU checks are separate runtime receipts.')
        else: print(name,value)
    print('completed shards:',{
        'renders':len(list((ROOT/'data/renders').glob('*/COMPLETE.json'))),
        'mechanism':len(list((ROOT/'data/mechanism').glob('*/COMPLETE.json'))),
        'training':len(list((ROOT/'training').glob('*/COMPLETE.json'))),
        'evaluation':len([p for p in (ROOT/'evaluation').glob('*/COMPLETE.json') if p.parent.name!='baselines'])})
    failures=sorted((ROOT/'reports/FAILURES').glob('*.json'))
    if failures: print('latest_failure',load_json(failures[-1]))
    if slurm and (ROOT/'reports/SUBMISSION.json').exists():
        jobs=','.join(j['job_id'] for j in load_json(ROOT/'reports/SUBMISSION.json')['jobs'])
        if jobs:
            for command in [
                ['squeue','-h','-j',jobs,'-o','%i|%T|%M|%R|%j'],
                ['sacct','-j',jobs,'--format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode','-P']]:
                response=subprocess.run(command,capture_output=True,text=True,timeout=30)
                print(response.stdout or response.stderr)
            print('Slurm sampled MaxRSS is separate from process RSS high-water mark.')


def main():
    parser=argparse.ArgumentParser(); modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ('preflight','dry-run','submit','status'): modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--slurm',action='store_true'); args=parser.parse_args()
    if args.preflight:
        prep=preflight(); print('CPU_PREFLIGHT_PASS',prep['signature'],'40 states / 40 episodes / 16 candidates; GPU and scheduler untouched')
    elif args.dry_run:
        sid=signature(); print('DRY_RUN_ONLY; no scheduler query/submission')
        previous=None
        for stage in STAGES:
            print(command_for(stage,sid,previous,['EXISTING_GPU_JOBS'] if stage=='mechanism' else [])); previous=stage.upper()+'_JOB'
        print(config()['resources'])
    elif args.submit: submit()
    else: status(args.slurm)


if __name__=='__main__': main()

"""Only --submit, explicitly run by the user, invokes sbatch. No cancellations."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
import time
from atlas_common import ROOT,atomic_json,config,freeze_id,load_json,preflight,receipt_valid,require

SCRIPTS={'capture':ROOT/'slurm/01_capture.sbatch','analyze':ROOT/'slurm/02_analyze.sbatch'}


def directories():
    for name in ['logs','results','data/attempts','readouts','reports/COSTS','reports/FAILURES','reports/locks','reports/SUBMISSION_ARCHIVE']:
        (ROOT/name).mkdir(parents=True,exist_ok=True)


def gpu_dependencies(output):
    result=set()
    for line in output.splitlines():
        fields=line.split('|'); require(len(fields)==5,'Cannot parse Slurm queue')
        job,state,tres,reason,name=fields
        if 'gpu' not in tres.lower(): continue
        require('DependencyNeverSatisfied' not in reason,'An existing GPU chain has failed dependencies; inspect it first')
        m=re.fullmatch(r'(\d+)(?:_[\d,\[\]%\-]+)?',job.strip())
        require(m,'Unknown Slurm job identifier'); result.add(m.group(1))
    return sorted(result,key=int)


def command_for(stage,signature,previous=None,existing=()):
    cmd=['sbatch','--parsable','--job-name=atlas1_'+signature[:8]+'_'+stage]
    # afterany lets the cheap CPU worker report a failed/incomplete capture, avoiding orphan dependencies.
    if previous: cmd.append('--dependency=afterany:'+previous)
    elif stage=='capture' and existing: cmd.append('--dependency=afterany:'+':'.join(existing))
    return cmd+[str(SCRIPTS[stage])]


def submit():
    prep=preflight(); signature=prep['signature']; directories()
    capture_done=all(receipt_valid(ROOT/f'data/task_{t:02d}/COMPLETE.json',signature) for t in range(10))
    path=ROOT/'reports/SUBMISSION.json'
    with (ROOT/'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%b|%R|%j'],
                             capture_output=True,text=True,timeout=30)
        require(queue.returncode==0,'Queue unavailable; nothing submitted: '+queue.stderr.strip())
        active={line.split('|')[0].split('_')[0] for line in queue.stdout.splitlines()}
        old=load_json(path) if path.exists() else None
        if old:
            require(old['signature']==signature,'Prior submission used different sources')
            require(old['status'] not in ('SUBMITTING','AMBIGUOUS_SBATCH_RESPONSE'),'Unresolved prior submission; inspect receipt')
            require(not active.intersection(j['job_id'] for j in old['jobs']),'This chain already has active jobs; use --status')
        require(not any(('atlas1_'+signature[:8]) in line for line in queue.stdout.splitlines()),'Duplicate active atlas chain')
        result=ROOT/'results/STATUS.json'
        if result.exists():
            r=load_json(result)
            from atlas_common import sha256_file
            require(r['signature']==signature,'Existing results used different sources')
            require(all(sha256_file(ROOT/'results'/p)==h for p,h in r['artifacts'].items()),'Result artifact changed')
            if r['signature']==signature and r['status']=='COMPLETE':
                print('ALREADY_COMPLETE; no submission'); return
            if r['signature']==signature and r['status']=='CORE_COMPLETE_PARTS_PENDING' and not (ROOT/'parts/reviewed/STATUS.json').exists():
                print('CORE_COMPLETE_PARTS_PENDING; no new work to submit. Review parts/review.html and import annotations.'); return
        stages=([] if capture_done else ['capture'])+['analyze']; deps=gpu_dependencies(queue.stdout) if not capture_done else []
        if old: atomic_json(old,ROOT/'reports/SUBMISSION_ARCHIVE'/f'{time.time_ns()}.json')
        receipt={'status':'SUBMITTING','signature':signature,'jobs':[],'resources':config()['resources'],'existing_gpu_dependencies':deps}
        atomic_json(receipt,path); previous=None
        for stage in stages:
            cmd=command_for(stage,signature,previous,deps); receipt['pending_command']=cmd; atomic_json(receipt,path)
            try:
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                job=r.stdout.strip().split(';')[0]
                require(r.returncode==0 and re.fullmatch(r'\d+',job),r.stderr.strip() or 'Invalid sbatch response')
            except BaseException as error:
                receipt.update(status='AMBIGUOUS_SBATCH_RESPONSE',error=repr(error)); atomic_json(receipt,path)
                raise RuntimeError('No automatic retry/cancellation; inspect SUBMISSION.json and Slurm before retrying') from error
            receipt['jobs'].append({'stage':stage,'job_id':job,'command':cmd}); previous=job
            receipt.pop('pending_command',None); atomic_json(receipt,path); print(stage,job,flush=True)
        receipt['status']='SUBMITTED'; atomic_json(receipt,path)


def status(slurm=False):
    for name in ['reports/SUBMISSION.json','reports/PROGRESS.json','results/STATUS.json','parts/reviewed/STATUS.json']:
        p=ROOT/name
        value=load_json(p) if p.exists() else {'status':'NOT_STARTED'}
        print(name,{k:v for k,v in value.items() if k not in ('artifacts','masks')})
    print('capture_shards',sum((ROOT/f'data/task_{t:02d}/COMPLETE.json').exists() for t in range(10)),'/10')
    failures=sorted((ROOT/'reports/FAILURES').glob('*.json'))
    if failures: print('latest_failure',load_json(failures[-1]))
    if slurm and (ROOT/'reports/SUBMISSION.json').exists():
        jobs=','.join(j['job_id'] for j in load_json(ROOT/'reports/SUBMISSION.json')['jobs'])
        r=subprocess.run(['sacct','-j',jobs,'--format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode','-P'],
                         capture_output=True,text=True,timeout=30)
        print('Slurm sampled MaxRSS is separate from process RSS HWM.'); print(r.stdout or r.stderr)


def main():
    p=argparse.ArgumentParser(); group=p.add_mutually_exclusive_group(required=True)
    for name in ('preflight','dry-run','submit','status'): group.add_argument('--'+name,action='store_true')
    group.add_argument('--import-parts',metavar='JSON'); p.add_argument('--slurm',action='store_true')
    args=p.parse_args()
    if args.preflight:
        r=preflight(); print('CPU_PREFLIGHT_PASS',r['signature'],'1200 states / 240 episodes; no model/GPU/scheduler action')
    elif args.dry_run:
        sid=freeze_id(); print('DRY_RUN_ONLY; no scheduler query/submission')
        print(command_for('capture',sid,existing=['EXISTING_GPU_JOBS']))
        print(command_for('analyze',sid,previous='CAPTURE_JOB')); print(config()['resources'])
    elif args.submit: submit()
    elif args.status: status(args.slurm)
    else:
        from atlas_parts import import_annotations
        import_annotations(args.import_parts); print('REVIEWED_80 imported; run --submit for CPU-only completion')


if __name__=='__main__': main()

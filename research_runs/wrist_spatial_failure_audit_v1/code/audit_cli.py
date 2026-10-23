"""Manual, deduplicated submissions for one graphics GPU, then a CPU report."""
import argparse
import getpass
import os
import re
import signal
import subprocess
import time
from audit_common import ROOT, atomic, complete, finish, lock, read, require, signature, prepare, verify, Monitor

def queue():
    r=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(r.returncode==0, 'Cannot check duplicate jobs: '+r.stderr); return r.stdout

def command(stage,deps=()):
    cmd=['sbatch','--parsable','--job-name=sfa_'+signature()[:8]+'_'+stage,
         '--export=ALL,SFA_EXPECTED_SIGNATURE='+signature(),'--kill-on-invalid-dep=yes']
    if deps: cmd+=['--dependency=afterany:'+':'.join(deps)]
    return cmd+[str(ROOT/f'slurm/{stage}.sbatch')]

def stages(submit_one):
    deps=[]
    if not complete(ROOT/'reports/REPLAY_COMPLETE.json'): deps=[submit_one('replay',())]
    if not complete(ROOT/'results/COMPLETE.json'): submit_one('report',deps)

def submit():
    prepare()
    with lock('submission'):
        q=queue()
        if any(len(v.split('|'))>=3 and v.split('|')[2].startswith('sfa_') for v in q.splitlines()):
            print('ALREADY_ACTIVE: no duplicate audit jobs submitted'); return
        if complete(ROOT/'results/COMPLETE.json'):
            print('ALREADY_COMPLETE: no resources requested'); return
        p=ROOT/'reports/SUBMISSION.json'
        if p.exists():
            old=read(p); require(old['status']!='AMBIGUOUS_SUBMISSION_RESPONSE','Inspect uncertain submission response before retry')
            atomic(old,ROOT/f'reports/SUBMISSIONS/{time.time_ns()}.json')
        meta={'signature':signature(),'status':'SUBMITTING','jobs':[],'max_concurrent_gpus':1}; atomic(meta,p)
        def one(stage,deps):
            cmd=command(stage,deps); meta['pending_command']=cmd; atomic(meta,p)
            try:
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=30); job=r.stdout.strip().split(';')[0]
                require(r.returncode==0 and re.fullmatch(r'\d+',job),r.stderr or r.stdout)
            except BaseException as e:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(e)); atomic(meta,p); raise
            meta['jobs'].append({'stage':stage,'job_id':job,'command':cmd}); meta.pop('pending_command',None); atomic(meta,p)
            print(stage,job,flush=True); return job
        stages(one); meta['status']='SUBMITTED'; atomic(meta,p)

def status(slurm=False):
    print('One 4090 / 4 CPU / 12G graphics replay; then 2 CPU / 4G report; NO model/training')
    for name in ('data/MANIFEST.json','results/STATUS.json','reports/FIRST_PAIR_TIMING.json'):
        p=ROOT/name
        if p.exists():
            v=read(p)
            print(name, {k:v[k] for k in ('replays','total_steps','status','replays_complete','replays_total','paired_videos_complete','elapsed_s','estimated_remaining_s') if k in v})
    for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read(p); print(p.stem,'heartbeat_age_s',int(time.time()-v['heartbeat_unix']),v)
    if slurm:
        print(queue()); p=ROOT/'reports/SUBMISSION.json'
        if p.exists():
            ids=[x['job_id'] for x in read(p)['jobs']]
            if ids:
                r=subprocess.run(['sacct','-j',','.join(ids),'-P','--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'],capture_output=True,text=True,timeout=20)
                print(r.stdout or r.stderr)

def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'),'Manual Slurm allocation required')
    require(os.environ.get('SFA_EXPECTED_SIGNATURE')==signature(),'Submitted code changed; do not run stale jobs')
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Preparation missing')
    manifest=verify()
    def stop(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    with lock('worker_'+stage), Monitor(stage) as monitor:
        if stage=='replay':
            from audit_replay import run
            run(manifest,monitor)
        else:
            from audit_report import run
            run(manifest,monitor,with_videos=True)

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    for key in ('preflight','dry-run','submit','resume','status','watch','summary-only'):g.add_argument('--'+key,action='store_true')
    g.add_argument('--worker',choices=('replay','report')); p.add_argument('--slurm',action='store_true'); a=p.parse_args()
    if a.worker: worker(a.worker)
    elif a.submit or a.resume: submit()
    elif a.preflight or a.dry_run:
        v=prepare(); print('PREFLIGHT_PASS',signature(),f"{v['replays']} saved-action replays / {v['total_steps']} steps; no jobs submitted")
        if a.dry_run:
            def preview(stage,deps): print(' '.join(command(stage,deps))); return stage.upper()+'_JOB'
            stages(preview)
    elif a.summary_only:
        if complete(ROOT/'results/COMPLETE.json'):
            print('ALREADY_COMPLETE',ROOT/'results/RESULTS.md'); return
        v=prepare()
        with Monitor('summary_local') as monitor:
            from audit_report import run
            run(v,monitor,with_videos=False)
        print(ROOT/'results/RESULTS.md')
    elif a.watch:
        try:
            while True: status(a.slurm); time.sleep(30)
        except KeyboardInterrupt: pass
    else: status(a.slurm)

if __name__=='__main__': main()

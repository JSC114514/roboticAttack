"""Only explicit user invocation submits diagnostic jobs; no expansion runner."""
import argparse
import getpass
import os
import re
import signal
import subprocess
import time
from rr_common import ROOT,config,read,atomic,require,signature,complete,finish,prepare,verify,lock,Monitor

def queue():
    r=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(r.returncode==0,'Slurm queue unavailable; refuse duplicate-blind submission: '+r.stderr);return r.stdout

def command(stage,deps=()):
    cmd=['sbatch','--parsable','--job-name=srr_'+signature()[:8]+'_'+stage,
         '--export=ALL,RR_EXPECTED_SIGNATURE='+signature(),'--kill-on-invalid-dep=yes']
    if deps:cmd+=['--dependency=afterany:'+':'.join(deps)]
    return cmd+[str(ROOT/f'slurm/{stage}.sbatch')]

def stages(one):
    # A worker interrupted after its last episode must not allocate another GPU just to seal receipts.
    if not complete(ROOT/'reports/EVALUATE_COMPLETE.json'):
        v=read(ROOT/'data/MANIFEST.json')
        receipts=[ROOT/'episodes'/c['start']/m/'COMPLETE.json' for c in v['cases'] for m in ['reference_replay',*config()['methods']]]
        if all(complete(p) for p in receipts):finish(ROOT/'reports/EVALUATE_COMPLETE.json',receipts)
    deps=[]
    if not complete(ROOT/'reports/EVALUATE_COMPLETE.json'):deps=[one('evaluate',())]
    if not complete(ROOT/'results/COMPLETE.json'):one('report',deps)

def submit():
    prepare()
    with lock('submission'):
        q=queue()
        if any(len(line.split('|'))>=3 and line.split('|')[2].startswith('srr_') for line in q.splitlines()):
            print('ALREADY_ACTIVE: no duplicate restoration jobs submitted');return
        if complete(ROOT/'results/COMPLETE.json'):print('ALREADY_COMPLETE: no resources requested');return
        p=ROOT/'reports/SUBMISSION.json'
        if p.exists():
            old=read(p);require(old['status']!='AMBIGUOUS_SUBMISSION_RESPONSE','Inspect uncertain sbatch response before retry')
            atomic(old,ROOT/f'reports/SUBMISSIONS/{time.time_ns()}.json')
        meta={'signature':signature(),'status':'SUBMITTING','jobs':[],'max_concurrent_gpus':1,'expansion_submitted':False};atomic(meta,p)
        def one(stage,deps):
            cmd=command(stage,deps);meta['pending_command']=cmd;atomic(meta,p)
            try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error));atomic(meta,p);raise
            if r.returncode!=0:
                meta.update(status='SUBMISSION_REJECTED',error=r.stderr or r.stdout);atomic(meta,p)
                raise RuntimeError(meta['error'])
            job=r.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+',job):
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=r.stdout);atomic(meta,p);raise RuntimeError('Unrecognized submission response')
            meta['jobs'].append({'stage':stage,'job_id':job,'command':cmd});meta.pop('pending_command',None);atomic(meta,p);print(stage,job,flush=True);return job
        stages(one);meta['status']='SUBMITTED';atomic(meta,p)

def status(slurm=False):
    print('Restoration diagnosis: max1x4090, 4CPU/12G; CPU report2CPU/4G. Expansion PREPARED, NOT SUBMITTED.')
    for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read(p);print(p.stem,'heartbeat_age_s',int(time.time()-v['heartbeat_unix']),v)
    p=ROOT/'results/STATUS.json'
    if p.exists():print('RESULTS',read(p))
    if slurm:
        try:
            print(queue());p=ROOT/'reports/SUBMISSION.json'
            if p.exists():
                ids=[x['job_id'] for x in read(p)['jobs']]
                if ids:
                    r=subprocess.run(['sacct','-j',','.join(ids),'-P','--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'],capture_output=True,text=True,timeout=20);print(r.stdout or r.stderr)
        except (RuntimeError,subprocess.TimeoutExpired,OSError) as error:print('SLURM_QUERY_UNAVAILABLE',repr(error),'Local completion receipts/heartbeat shown above.')

def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'),'Manual Slurm allocation required')
    require(os.environ.get('RR_EXPECTED_SIGNATURE')==signature(),'Code changed since submission')
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Preparation unavailable')
    v=verify()
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with lock('worker_'+stage),Monitor(stage) as monitor:
        if stage=='evaluate':
            from rr_run import run
        else:
            from rr_report import run
        run(v,monitor)

def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    for name in ('preflight','dry-run','submit','resume','status','watch'):g.add_argument('--'+name,action='store_true')
    g.add_argument('--worker',choices=('evaluate','report'));parser.add_argument('--slurm',action='store_true');a=parser.parse_args()
    if a.worker:worker(a.worker)
    elif a.submit or a.resume:submit()
    elif a.preflight or a.dry_run:
        v=prepare();print('PREFLIGHT_PASS',signature(),f"18 reference replays ({v['reference_steps']} steps), 54 policy restoration episodes; expansion320 PREPARED_NOT_SUBMITTED")
        if a.dry_run:
            def preview(stage,deps):print(' '.join(command(stage,deps)));return stage.upper()+'_JOB'
            stages(preview)
    elif a.watch:
        try:
            while True:status(a.slurm);time.sleep(30)
        except KeyboardInterrupt:pass
    else:status(a.slurm)

if __name__=='__main__':main()

"""Manual-only submission, strict duplicate checks, stage/branch continuation."""
import argparse
import getpass
import json
import os
import signal
import subprocess
import time
from va_common import ROOT,PROJECT,config,signature,read,atomic,require,complete,lock,Monitor

PREFIX='wasym1_'
STAGES=('prepare','shared','calibrate','nonattack','patch','report')

def done(stage):
    if stage=='report':
        p=PROJECT/'research_videos'/(ROOT.name+'_singleview')/'MANIFEST.json'
        return complete(ROOT/'results/COMPLETE.json') and p.exists() and read(p).get('status')=='COMPLETE' and read(p).get('episodes')==288
    return complete(ROOT/f'reports/{dict(prepare="PREPARE",shared="SHARED",calibrate="CALIBRATION",nonattack="NONATTACK",patch="PATCH")[stage]}_COMPLETE.json')

def queue():
    r=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(r.returncode==0,'Slurm unavailable; refuse duplicate-blind submission: '+r.stderr)
    return [x.split('|',3) for x in r.stdout.splitlines() if x]

def command(stage,deps=(),after='afterok'):
    kind='gpu' if stage in ('shared','nonattack','patch') else 'cpu'
    cmd=['sbatch','--parsable','--job-name='+PREFIX+signature()[:8]+'_'+stage,'--kill-on-invalid-dep=yes',
         '--export=ALL,VA_EXPECTED_SIGNATURE='+signature()]
    if deps:cmd+=['--dependency='+after+':'+':'.join(dict.fromkeys(deps))]
    return cmd+[str(ROOT/'slurm'/(kind+'.sbatch')),stage]

def check_launchers():
    env=dict(os.environ,VA_EXPECTED_SIGNATURE=signature(),CUDA_VISIBLE_DEVICES='0')
    for kind in ('cpu','gpu'):
        r=subprocess.run(['bash',str(ROOT/'slurm'/(kind+'.sbatch')),'__launch_check__'],capture_output=True,text=True,env=env,timeout=30)
        require(r.returncode==0,'Batch launcher failed: '+kind+' '+r.stderr)
        require(json.loads(r.stdout)=={'root':str(ROOT),'signature':signature()},'Wrong batch target')
    return ['cpu','gpu']

def submit(dry=False):
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    check_launchers()
    with lock('submission'):
        if not dry:
            active=[r for r in queue() if r[2].startswith(PREFIX)]
            if active:
                print('ALREADY_ACTIVE; no duplicates submitted.');print(json.dumps(active));return
            if (ROOT/'SUBMISSION.json').exists():require(read(ROOT/'SUBMISSION.json').get('status')!='AMBIGUOUS_SUBMISSION_RESPONSE','Resolve ambiguous previous submission before resubmitting')
        jobs={};counter=0
        def one(stage,deps=(),after='afterok'):
            nonlocal counter
            if done(stage):return []
            cmd=command(stage,deps,after)
            if dry:
                counter+=1;job='DRY'+str(counter);print(' '.join(cmd))
            else:
                state=dict(signature=signature(),jobs=jobs,pending_command=cmd,status='SUBMITTING',submitted_unix=time.time());atomic(state,ROOT/'SUBMISSION.json')
                try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                except BaseException as error:
                    atomic(dict(state,status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error)),ROOT/'SUBMISSION.json');raise
                require(r.returncode==0,'Submit failed; already-submitted IDs retained: '+r.stderr)
                job=r.stdout.strip().split(';')[0]
                if not job.isdigit():
                    atomic(dict(state,status='AMBIGUOUS_SUBMISSION_RESPONSE',error=r.stdout),ROOT/'SUBMISSION.json');raise RuntimeError('Ambiguous Slurm job ID')
                jobs[stage]=job;atomic(dict(signature=signature(),jobs=jobs,status='SUBMITTED',submitted_unix=time.time()),ROOT/'SUBMISSION.json');print(stage,job,flush=True)
            return [job]
        p=one('prepare');s=one('shared',p);c=one('calibrate',s or p)
        a=one('nonattack',c or s or p);b=one('patch',c or s or p)
        one('report',a+b or c or s or p,'afterany')

def worker(stage):
    require(os.environ.get('VA_EXPECTED_SIGNATURE')==signature(),'Worker revision changed')
    if stage=='__launch_check__':print(json.dumps(dict(root=str(ROOT),signature=signature())));return
    require(stage in STAGES,'Unknown stage')
    def terminate(*_):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    with lock('worker_'+stage),Monitor(stage) as monitor:
        if stage in ('shared','nonattack','patch'):
            require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'),'GPU work requires a user-submitted Slurm allocation')
            from va_common import adapters
            _,native=adapters();native.require_gpu('wrist_view_asymmetry')
            import tensorflow as tf
            tf.config.set_visible_devices([],'GPU');native.runtime_module().prepare_graphics_backend()
            from va_stages import shared,screen
            if stage=='shared':shared(monitor)
            else:screen(stage,monitor)
        elif stage=='prepare':
            from va_prepare import run
            run(monitor)
        elif stage=='calibrate':
            from va_calibrate import run
            run(monitor)
        else:
            from va_report import run
            run(monitor,export=True)

def status(slurm=False):
    print(ROOT.name,'revision',signature()[:12])
    for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read(p);print(p.stem,v.get('status','RUNNING'),v.get('phase',''),str(v.get('completed','?'))+'/'+str(v.get('total','?')),
                       'heartbeat_age_s',round(time.time()-v['heartbeat_unix']),'ETA_s',round(v.get('eta_s',0)),v.get('error') or '')
    if (ROOT/'results/STATUS.json').exists():
        v=read(ROOT/'results/STATUS.json');print('Short branches:',v['status'],v['completed'],'/288; missing',len(v['missing']),'invalid',len(v['invalid']))
    if slurm:
        for r in queue():
            if r[2].startswith(PREFIX):print('Slurm:',*r)

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for name in ('submit','resume','status','watch','prepare','report','check','dry-run'):g.add_argument('--'+name,action='store_true')
    g.add_argument('--worker');p.add_argument('--slurm',action='store_true');a=p.parse_args()
    if not a.worker or a.worker not in ('shared','nonattack','patch'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    if a.submit or a.resume:submit()
    elif a.dry_run:submit(True)
    elif a.worker:worker(a.worker)
    elif a.check:print('Batch launchers PASS:',check_launchers());print('CPU-only; no model, simulator or Slurm submission executed.')
    elif a.status:status(a.slurm)
    elif a.watch:
        try:
            while True:status(a.slurm);time.sleep(15)
        except KeyboardInterrupt:pass
    else:
        stage='prepare' if a.prepare else 'report'
        with lock('worker_'+stage),Monitor(stage+'_local') as monitor:
            if a.prepare:
                from va_prepare import run
                run(monitor)
            else:
                from va_report import run
                run(monitor,export=False)

if __name__=='__main__':main()

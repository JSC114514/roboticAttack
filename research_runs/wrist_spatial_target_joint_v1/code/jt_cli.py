"""Manual-only Slurm submission; two GPU lanes, deduplicated staged resume."""
import argparse
import getpass
import os
from pathlib import Path
import signal
import subprocess
import time
from jt_common import ROOT,config,signature,read,atomic,complete,lock,Monitor,require,selected,cache_receipt

def prefix():return 'stj1_'+signature()[:8]+'_'

def queue():
    result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(result.returncode==0,'Cannot read Slurm; refuse duplicate-blind submission: '+result.stderr)
    return [line.split('|',3) for line in result.stdout.splitlines() if line]

def done(stage):
    if stage=='prepare':return complete(ROOT/'reports/PREPARE_COMPLETE.json')
    if stage.startswith('cache'):
        if not (ROOT/'data/MANIFEST.json').exists():return False
        return all(complete(cache_receipt(r['state_id'])) for r in selected()[int(stage[-1])::2])
    if stage=='targets':return complete(ROOT/'reports/TARGETS_COMPLETE.json')
    if stage.startswith('train'):return complete(ROOT/'training'/config()['methods'][int(stage[-1])]/'COMPLETE.json')
    if stage=='select':return complete(ROOT/'reports/SELECT_COMPLETE.json')
    if stage.startswith('evaluate'):return complete(ROOT/f'reports/EVALUATE_{stage[-1]}_COMPLETE.json')
    return complete(ROOT/'results/COMPLETE.json')

def command(stage,deps,after='afterok'):
    kind='cache' if stage.startswith('cache') else 'train' if stage.startswith('train') else 'evaluate' if stage.startswith('evaluate') else 'cpu'
    cmd=['sbatch','--parsable','--job-name='+prefix()+stage,'--kill-on-invalid-dep=yes',
         '--export=ALL,JT_EXPECTED_SIGNATURE='+signature()]
    if deps:cmd+=['--dependency='+after+':'+':'.join(deps)]
    return cmd+[str(ROOT/'slurm'/(kind+'.sbatch')),stage]

def submit(dry=False):
    with lock('submission'):
        live=[] if dry else queue()
        stale=[r for r in live if r[2].startswith('stj1_') and not r[2].startswith(prefix())]
        require(not stale,'Other revision of this experiment is active; do not duplicate or auto-cancel it: '+str(stale))
        current=[r for r in live if r[2].startswith(prefix())]
        if current:
            print('This pipeline is already active; no duplicate jobs submitted. Resume after it drains:')
            for row in current:print(*row)
            return
        active={r[2]:r[0] for r in live};submitted={};counter=0
        if complete(ROOT/'reports/TARGETS_COMPLETE.json') and not read(ROOT/'data/TARGET_STATS.json')['proceed']:
            print('STOPPED_NO_APPROACH_OBSTRUCTION: no training/evaluation will be submitted.');return
        def one(stage,deps=(),after='afterok'):
            nonlocal counter
            if prefix()+stage in active:return [active[prefix()+stage]]
            if done(stage):return []
            cmd=command(stage,list(dict.fromkeys(deps)),after)
            if dry:
                counter+=1;job='DRY'+str(counter);print(' '.join(cmd))
            else:
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                require(result.returncode==0,'Submission failed; already submitted IDs retained: '+result.stderr)
                job=result.stdout.strip().split(';')[0];require(job.isdigit(),'Unexpected Slurm job ID: '+result.stdout)
                submitted[stage]=job
                # Persist every ID immediately, including when a later submission fails.
                atomic({'signature':signature(),'jobs':submitted,'submitted_unix':time.time(),'automatic_submission':False},ROOT/'SUBMISSION.json')
                print(stage,job,flush=True)
            return [job]
        prepare=one('prepare')
        caches=one('cache0',prepare)+one('cache1',prepare)
        targets=one('targets',caches or prepare)
        lane0=one('train0',targets);lane1=one('train1',targets)
        lane0=one('train2',lane0 or targets);lane1=one('train3',lane1 or targets)
        select=one('select',lane0+lane1 or targets)
        evaluation=one('evaluate0',select)+one('evaluate1',select)
        one('report',evaluation or select,'afterany')

def worker(stage):
    expected=os.environ.get('JT_EXPECTED_SIGNATURE');require(expected==signature(),'Worker revision differs from submitted revision')
    if stage.startswith(('cache','train','evaluate')):
        require(os.environ.get('SLURM_JOB_ID'),'GPU stages require a user-submitted allocation')
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    def terminate(_signum,_frame):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    with lock(stage),Monitor(stage) as monitor:
        if stage=='prepare':
            from jt_prepare import run
            run(monitor)
        elif stage.startswith('cache'):
            from jt_cache import run
            run(int(stage[-1]),monitor)
        elif stage=='targets':
            from jt_prepare import finalize
            finalize(monitor)
            if not read(ROOT/'data/TARGET_STATS.json')['proceed']:
                from jt_report import run
                run(monitor,export=False)
                # This scientific stop cancels dependent queued jobs through afterok/kill-on-invalid-dep.
                # No scancel command, no GPU trial or automatic retry is issued.
                raise SystemExit(3)
        elif stage.startswith('train'):
            from jt_train import run
            run(config()['methods'][int(stage[-1])],monitor)
        elif stage=='select':
            from jt_train import select
            select(monitor)
        elif stage.startswith('evaluate'):
            from jt_evaluate import run
            run(int(stage[-1]),monitor)
        elif stage=='report':
            from jt_report import run
            run(monitor)
        else:raise ValueError(stage)

def status(slurm=False):
    print('Run:',ROOT.name,'revision:',signature()[:12])
    stats=ROOT/'data/TARGET_STATS.json'
    if stats.exists():
        v=read(stats);print('Targets:',v['status'],'fit/verify:',v['fit_frames'],v['verify_frames'],'mean harm:',v['mean_fit_harm'])
    for path in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read(path);age=time.time()-v['heartbeat_unix']
        print(path.stem,v.get('status','RUNNING'),v.get('phase',''),f"{v.get('completed','?')}/{v.get('total','?')}",
              'heartbeat_age_s',round(age),'ETA_s',round(v.get('estimated_remaining_s',0)),v.get('error') or '')
    result=ROOT/'results/STATUS.json'
    if result.exists():
        v=read(result);print('Results:',v['status'],'episodes',v['native_new_episodes'],'/',v['expected_new_episodes'])
        for row in v['summary']:print(row['condition'],'CASR',row['CASR'],'paired',row['clean_success_pairs'],'robot_SR',row['robot_task_SR'])
    if slurm:
        for row in queue():
            if row[2].startswith('stj1_'):print('Slurm:',*row)

def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for name in ('submit','resume','status','watch','prepare','dry-run','check'):group.add_argument('--'+name,action='store_true')
    group.add_argument('--worker');parser.add_argument('--slurm',action='store_true');args=parser.parse_args()
    # Keep local CPU preparation/statistics within the promised two CPU workers.
    if not args.worker or not args.worker.startswith(('cache','train','evaluate')):
        affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,affinity[:2])
    if args.submit or args.resume:submit()
    elif args.dry_run:submit(True)
    elif args.status:status(args.slurm)
    elif args.watch:
        try:
            while True:status(args.slurm);time.sleep(15)
        except KeyboardInterrupt:pass
    elif args.prepare:
        with lock('prepare'),Monitor('prepare_local') as monitor:
            from jt_prepare import run
            run(monitor)
    elif args.check:
        cfg=config();require(cfg['updates']==4000 and cfg['adaptive_budget']==3000 and cfg['probe_updates']*4==20,'Budget protocol changed')
        require(cfg['checkpoint']=='libero_goal' and cfg['patch_xy']==[26,137] and not cfg['eot'],'Fixed protocol changed')
        require(cfg['rules']==['near','far','left','right'],'Rule tie order changed')
        print('CPU protocol check PASS; signature',signature());print('No model, simulator or Slurm job launched.')
    else:worker(args.worker)

if __name__=='__main__':main()

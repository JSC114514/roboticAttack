"""Evaluation only. Slurm mutations occur exclusively after explicit --submit/--resume."""
import argparse
import getpass
import json
import os
import signal
import subprocess
import time
from se_common import ROOT,config,signature,read,atomic,complete,finish,lock,Monitor,require,verify

PREFIX='srse1_'
RELATED_PREFIXES=(PREFIX,'srrf1_','srf_con_')


def prefix():return PREFIX+signature()[:8]+'_'


def queue():
    result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(result.returncode==0, 'Cannot read Slurm; refuse duplicate-blind submission: '+result.stderr)
    return [line.split('|',3) for line in result.stdout.splitlines() if line]


def done(stage):
    if stage=='prepare':return complete(ROOT/'reports/PREPARE_COMPLETE.json')
    if stage in ('evaluate0','evaluate1'):return complete(ROOT/f'reports/EVALUATE_{stage[-1]}_COMPLETE.json')
    require(stage=='report','Unknown stage')
    return complete(ROOT/'results/COMPLETE.json')


def seal_completed_workers():
    if not complete(ROOT/'reports/PREPARE_COMPLETE.json'):return
    value=read(ROOT/'data/MANIFEST.json')
    for index in (0,1):
        receipt=ROOT/f'reports/EVALUATE_{index}_COMPLETE.json'
        if complete(receipt):continue
        episodes=[ROOT/'episodes'/case['start']/method/'COMPLETE.json'
                  for case in value['cases'][index::2] for method in config()['methods']]
        if episodes and all(complete(p) for p in episodes):
            finish(receipt,episodes,new_closedloop_episodes=len(episodes),sealed_on_cpu_without_allocation=True)


def command(stage,deps=(),after='afterok'):
    kind='evaluate' if stage in ('evaluate0','evaluate1') else 'cpu'
    cmd=['sbatch','--parsable','--job-name='+prefix()+stage,'--kill-on-invalid-dep=yes',
         '--export=ALL,SE_EXPECTED_SIGNATURE='+signature()]
    if deps:cmd+=['--dependency='+after+':'+':'.join(deps)]
    return cmd+[str(ROOT/'slurm'/(kind+'.sbatch')),stage]


def check_launchers():
    expected=signature();checked=[]
    environment=dict(os.environ,SE_EXPECTED_SIGNATURE=expected,CUDA_VISIBLE_DEVICES='0')
    for kind in ('evaluate','cpu'):
        result=subprocess.run(['bash',str(ROOT/'slurm'/(kind+'.sbatch')),'__launch_check__'],env=environment,capture_output=True,text=True,timeout=30)
        require(result.returncode==0, 'Batch launcher failed: '+kind+': '+result.stderr)
        require(json.loads(result.stdout)=={'root':str(ROOT),'signature':expected}, 'Batch launcher points to another run')
        checked.append(kind)
    return checked


def submit(dry=False):
    check_launchers()
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):verify()
    with lock('submission'):
        live=[] if dry else queue()
        # Protect this revision and the old relation/shuffle pipelines from duplicate evaluation.
        conflict=[r for r in live if r[2].startswith(RELATED_PREFIXES)]
        if conflict:
            print('Existing relation/shuffle pipeline is active; no duplicate jobs submitted.')
            for row in conflict:print(*row)
            return
        if not dry:seal_completed_workers()
        submitted={};counter=0
        def one(stage,deps=(),after='afterok'):
            nonlocal counter
            if done(stage):return []
            cmd=command(stage,list(dict.fromkeys(deps)),after)
            if dry:
                counter+=1;job='DRY'+str(counter);print(' '.join(cmd))
            else:
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                require(result.returncode==0, 'Submission failed; prior IDs retained: '+result.stderr)
                job=result.stdout.strip().split(';')[0];require(job.isdigit(), 'Invalid Slurm ID: '+result.stdout)
                submitted[stage]=job
                atomic({'signature':signature(),'jobs':submitted,'submitted_unix':time.time(),'initiated_by':'user_manual_command'},ROOT/'SUBMISSION.json')
                print(stage,job,flush=True)
            return [job]
        prepare=one('prepare')
        evaluation=one('evaluate0',prepare)+one('evaluate1',prepare)
        # afterany enables explicit partial reporting when an evaluator fails.
        one('report',evaluation or prepare,'afterany')


def worker(stage):
    require(os.environ.get('SE_EXPECTED_SIGNATURE')==signature(), 'Worker revision differs from submitted code')
    if stage=='__launch_check__':
        print(json.dumps({'root':str(ROOT),'signature':signature()}));return
    require(stage in ('prepare','evaluate0','evaluate1','report'), 'Unknown worker stage')
    def terminate(_signum,_frame):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    with lock('worker_'+stage),Monitor(stage) as monitor:
        if stage.startswith('evaluate'):
            require(os.environ.get('SLURM_JOB_ID'), 'GPU evaluation needs a user-submitted allocation')
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        if stage=='prepare':
            from se_prepare import run
            run(monitor)
        elif stage.startswith('evaluate'):
            from se_evaluate import run
            run(int(stage[-1]),monitor)
        else:
            from se_report import run
            run(monitor)


def status(slurm=False):
    print('Run:',ROOT.name,'revision:',signature()[:12])
    if (ROOT/'data/MANIFEST.json').exists():
        value=read(ROOT/'data/MANIFEST.json');print('Verified PNG assets:',len(value['assets']),'new episodes:',value['maximum_new_episodes'],'training: none')
    for path in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read(path)
        print(path.stem,v.get('status','RUNNING'),v.get('phase',''),f"{v.get('completed','?')}/{v.get('total','?')}",
              'heartbeat_age_s',round(time.time()-v['heartbeat_unix']),'ETA_s',round(v.get('estimated_remaining_s',0)),v.get('error') or '')
    if (ROOT/'results/STATUS.json').exists():
        v=read(ROOT/'results/STATUS.json');print('Results:',v['status'],v['native_new_episodes'],'/',v['expected_new_episodes'])
        for row in v['summary']:print(row['condition'],'CASR',row['CASR'],'paired',row['clean_success_pairs'],'robot_SR',row['robot_task_SR'])
    if slurm:
        for row in queue():
            if row[2].startswith(RELATED_PREFIXES):print('Slurm:',*row)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for name in ('submit','resume','status','watch','prepare','report','dry-run','check'):group.add_argument('--'+name,action='store_true')
    group.add_argument('--worker');parser.add_argument('--slurm',action='store_true');args=parser.parse_args()
    if not args.worker or not args.worker.startswith('evaluate'):
        affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,affinity[:2])
    if args.submit or args.resume:submit()
    elif args.dry_run:submit(True)
    elif args.status:status(args.slurm)
    elif args.watch:
        try:
            while True:status(args.slurm);time.sleep(15)
        except KeyboardInterrupt:pass
    elif args.prepare or args.report:
        name='prepare' if args.prepare else 'report'
        with lock('worker_'+name),Monitor(name+'_local') as monitor:
            if args.prepare:
                from se_prepare import run
            else:
                from se_report import run
            run(monitor)
    elif args.check:
        cfg=config()
        require(cfg['checkpoint']=='libero_goal' and cfg['patch_xy']==[26,137] and cfg['patch_shape']==[3,50,50] and not cfg['eot'], 'Fixed protocol changed')
        require(cfg['updates']==4000 and cfg['max_concurrent_gpus']==2 and len(cfg['assets'])==4, 'Scope changed')
        print('Actual batch launcher check PASS:',', '.join(check_launchers()))
        print('CPU protocol PASS; signature',signature(),'; no model, simulator or Slurm job launched.')
    else:worker(args.worker)


if __name__=='__main__':main()

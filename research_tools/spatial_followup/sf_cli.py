"""Explicit manual Slurm submission, deduplication, resume and progress."""
import argparse
import getpass
import os
import re
import signal
import subprocess
import time
from sf_common import ROOT, PROJECT, config, read, atomic, require, signature, complete, finish, prepare, verify, lock, Monitor, candidates, external_complete


def queue():
    result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(result.returncode==0,'Queue unavailable; refusing duplicate-blind submission: '+result.stderr)
    return result.stdout


def prefix():return 'srf_'+config()['scope'][:3]+'_'+signature()[:8]+'_'


def command(stage,deps=(),after='afterok'):
    cmd=['sbatch','--parsable','--job-name='+prefix()+stage,'--export=ALL,FOLLOWUP_EXPECTED_SIGNATURE='+signature(),
         '--kill-on-invalid-dep=yes']
    if deps:cmd+=['--dependency='+after+':'+':'.join(deps)]
    return cmd+[str(ROOT/'slurm'/f'{stage}.sbatch')]


def stages(one,base_dependency=()):
    if config()['scope']=='control':
        training=[]
        for index,seed in enumerate(config()['seeds']):
            names=[name for name,_,s in candidates() if s==seed]
            if not all(complete(ROOT/'training'/n/'COMPLETE.json') for n in names):
                training.append(one('train'+str(index),base_dependency))
        if not complete(ROOT/'reports/SELECT_COMPLETE.json'):
            selection=[one('select',training or base_dependency)]
        else:selection=[]
    else:selection=[]
    evaluations=[]
    for index in range(2):
        receipt=ROOT/f'reports/EVALUATE_{index}_COMPLETE.json'
        # If interrupted after its last episode, seal on CPU instead of allocating
        # a GPU solely to load a model and mark the stage complete.
        if not complete(receipt):
            cases=read(ROOT/'data/MANIFEST.json')['cases'][index::2]
            if config()['scope']=='expansion':
                names=read(ROOT/'data/MANIFEST.json')['condition_order']
            else:names=[name for name,_,_ in candidates()]
            artifacts=[];ready=True
            for case in cases:
                root=ROOT/'episodes'/case['start'];skip=root/'SKIPPED_CLEAN_FAILURE.json'
                if complete(skip):artifacts.append(skip)
                elif all(complete(root/name/'COMPLETE.json') for name in names):
                    artifacts.extend(root/name/'COMPLETE.json' for name in names)
                else:ready=False;break
            if ready:finish(receipt,artifacts)
        if not complete(receipt):evaluations.append(one('evaluate'+str(index),selection or base_dependency))
    if not complete(ROOT/'results/COMPLETE.json'):
        # CPU report can explain partial outputs after a GPU failure; no retry GPU.
        one('report',evaluations,after='afterany')


def submit():
    prepare()
    with lock('submission'):
        current=queue()
        if any(len(p:=line.split('|'))>=3 and p[2].startswith('srf_'+config()['scope'][:3]+'_') for line in current.splitlines()):
            print('ALREADY_ACTIVE: no duplicate jobs requested');return
        if complete(ROOT/'results/COMPLETE.json'):
            print('ALREADY_COMPLETE: no resources requested');return
        dependency=[]
        if config()['scope']=='control':
            base=PROJECT/'research_runs/wrist_spatial_relation_expand_v1'
            if not external_complete(base,base/'results/COMPLETE.json'):
                submission=base/'reports/SUBMISSION.json'
                require(submission.exists(),'Submit expansion first; controls wait for its completed evidence')
                saved=read(submission);require(saved['status']=='SUBMITTED','Expansion submission incomplete/ambiguous')
                jobs=[r['job_id'] for r in saved['jobs'] if r['stage']=='report']
                require(len(jobs)==1,'Expansion report dependency unavailable');dependency=jobs
        path=ROOT/'reports/SUBMISSION.json'
        if path.exists():
            old=read(path);require(old['status']!='AMBIGUOUS_SUBMISSION_RESPONSE','Inspect uncertain sbatch response before retry')
            atomic(old,ROOT/'reports/SUBMISSIONS'/f'{time.time_ns()}.json')
        meta={'signature':signature(),'status':'SUBMITTING','jobs':[],'max_concurrent_gpus':2};atomic(meta,path)
        def one(stage,deps=(),after='afterok'):
            cmd=command(stage,deps,after);meta['pending_command']=cmd;atomic(meta,path)
            try:result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error));atomic(meta,path);raise
            if result.returncode:
                meta.update(status='SUBMISSION_REJECTED',error=result.stderr or result.stdout);atomic(meta,path);raise RuntimeError(meta['error'])
            job=result.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+',job):
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=result.stdout);atomic(meta,path);raise RuntimeError('Unknown sbatch response')
            meta['jobs'].append({'stage':stage,'job_id':job,'command':cmd});meta.pop('pending_command',None);atomic(meta,path)
            print(stage,job,flush=True);return job
        stages(one,dependency);meta['status']='SUBMITTED';atomic(meta,path)


def status(slurm=False):
    print(config()['scope'],'max2 GPUs; evaluation 1x4090/4CPU/12G each; training 1x4090/4CPU/16G; CPU2/4G')
    for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        value=read(p);print(p.stem,'heartbeat_age_s',int(time.time()-value['heartbeat_unix']),value)
    p=ROOT/'results/STATUS.json'
    if p.exists():print('RESULTS',read(p))
    print('central videos:',PROJECT/'research_videos'/ROOT.name/'INDEX.html')
    if slurm:
        try:
            print(queue());p=ROOT/'reports/SUBMISSION.json'
            if p.exists():
                ids=[r['job_id'] for r in read(p)['jobs']]
                if ids:
                    result=subprocess.run(['sacct','-j',','.join(ids),'-P','--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'],capture_output=True,text=True,timeout=20)
                    print(result.stdout or result.stderr)
        except (OSError,RuntimeError,subprocess.TimeoutExpired) as error:print('SLURM_QUERY_UNAVAILABLE',repr(error))


def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'),'Manual Slurm allocation required')
    require(os.environ.get('FOLLOWUP_EXPECTED_SIGNATURE')==signature(),'Code changed since submission')
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Preparation unavailable');verify()
    if config()['scope']=='control':
        # afterok report dependency includes CPU gallery export. On resume also
        # require successful final expansion before allocating scientific work.
        base=PROJECT/'research_runs/wrist_spatial_relation_expand_v1'
        require(external_complete(base,base/'results/COMPLETE.json'),'Expansion incomplete; no control work started')
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with lock('worker_'+stage),Monitor(stage) as monitor:
        if stage.startswith('train'):
            from sf_train import run
            run(int(stage[-1]),monitor)
        elif stage=='select':
            from sf_train import select
            select(monitor)
        elif stage.startswith('evaluate'):
            from sf_evaluate import run
            run(int(stage[-1]),monitor)
        else:
            from sf_report import run
            run(monitor)
            require(complete(ROOT/'results/COMPLETE.json'),'Partial report written; failed/incomplete episodes require explicit resume')


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    for name in ('preflight','dry-run','submit','resume','status','watch'):group.add_argument('--'+name,action='store_true')
    group.add_argument('--worker',choices=('train0','train1','select','evaluate0','evaluate1','report'))
    parser.add_argument('--slurm',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.submit or args.resume:submit()
    elif args.preflight or args.dry_run:
        value=prepare();print('PREFLIGHT_PASS',signature(),'scope',config()['scope'],'starts',len(value['cases']))
        if config()['scope']=='control':
            d=value['target_details'];eligible=[r for r in d if r['eligible']]
            print('matched targets:',len(d),'eligible',len(eligible),'alpha_zero',sum(r['alpha']==0 for r in eligible))
        if args.dry_run:
            def preview(stage,deps=(),after='afterok'):
                print(' '.join(command(stage,deps,after)));return stage.upper()+'_JOB'
            stages(preview,('EXPANSION_REPORT_JOB',) if config()['scope']=='control' else ())
    elif args.watch:
        try:
            while True:status(args.slurm);time.sleep(30)
        except KeyboardInterrupt:pass
    else:status(args.slurm)


if __name__=='__main__':main()

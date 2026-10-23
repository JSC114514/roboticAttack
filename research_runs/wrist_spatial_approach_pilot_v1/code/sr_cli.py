"""Manual submissions only; failed prerequisites release/avoid GPU allocations."""
import argparse,fcntl,getpass,os,re,signal,subprocess,time
from sr_common import (ROOT,config,prepare,signature,complete,cache_complete,read_json,atomic_json,
    require,Monitor,cost,verify_inputs,candidates)

def queue():
    r=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(r.returncode==0,'Slurm unavailable: '+r.stderr);return r.stdout
def command(stage,indices=None,deps=(),afterany=False):
    c=['sbatch','--parsable','--job-name=sap_'+signature()[:8]+'_'+stage,
        '--export=ALL,SR_EXPECTED_SIGNATURE='+signature(),'--kill-on-invalid-dep=yes']
    if indices is not None:c.append('--array='+','.join(map(str,indices))+'%2')
    if deps:c.append('--dependency='+('afterany' if afterany else 'afterok')+':'+':'.join(deps))
    return c+[str(ROOT/f'slurm/{stage}.sbatch')]
def stages(one):
    deps=[]
    if not complete(ROOT/'reports/SELECT_COMPLETE.json'):
        missing=[i for i in range(2) if not cache_complete(i)]
        if missing:deps=[one('cache',missing,())]
        if not complete(ROOT/'reports/COVERAGE_COMPLETE.json'):deps=[one('coverage',None,deps)]
        missing=[i for i,(name,_,_) in enumerate(candidates()) if not complete(ROOT/f'training/{name}/COMPLETE.json')]
        if missing:deps=[one('train',missing,deps)]
        deps=[one('select',None,deps)]
    missing=[i for i in range(2) if not complete(ROOT/f'evaluation/worker_{i}_COMPLETE.json')]
    if missing:deps=[one('evaluate',missing,deps)]
    if not complete(ROOT/'results/COMPLETE.json'):one('report',None,deps,True)
def submit():
    with cost('prepare_submit'):prepare()
    with (ROOT/'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=queue()
        if any(line.split('|')[2].startswith('sap_') for line in q.splitlines()):print('ALREADY_ACTIVE: no duplicate submissions, including other revisions');return
        if complete(ROOT/'results/COMPLETE.json'):print('ALREADY_COMPLETE: no resources requested');return
        p=ROOT/'reports/SUBMISSION.json'
        if p.exists():
            v=read_json(p);require(v['status']!='AMBIGUOUS_SUBMISSION_RESPONSE','Inspect uncertain sbatch response before retry')
            atomic_json(v,ROOT/f'reports/SUBMISSIONS/{time.time_ns()}.json')
        meta={'signature':signature(),'status':'SUBMITTING','jobs':[],'max_concurrent_gpus':2};atomic_json(meta,p)
        def one(stage,indices=None,deps=(),afterany=False):
            cmd=command(stage,indices,deps,afterany);meta['pending_command']=cmd;atomic_json(meta,p)
            try:
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);job=r.stdout.strip().split(';')[0]
                require(r.returncode==0 and re.fullmatch(r'\d+',job),r.stderr or r.stdout)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error));atomic_json(meta,p);raise
            meta['jobs'].append({'stage':stage,'job_id':job,'command':cmd});meta.pop('pending_command',None);atomic_json(meta,p);print(stage,job,flush=True);return job
        stages(one);meta['status']='SUBMITTED';atomic_json(meta,p)
def status(slurm=False):
    print('revision',signature()[:12],'mixed four-suite data; frozen Goal; max2x4090; train16G/eval12G')
    for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read_json(p);age=int(time.time()-v['heartbeat_unix'])
        print(p.stem,{k:v.get(k) for k in ('status','phase','completed','total','estimated_remaining_s','loss','optimized_updates','candidate','error')},'heartbeat_age_s',age)
    for name in ('COVERAGE','STATUS'):
        p=ROOT/f'results/{name}.json'
        if p.exists():print(name,read_json(p))
    if slurm:
        print(queue());p=ROOT/'reports/SUBMISSION.json'
        if p.exists():
            ids=[r['job_id'] for r in read_json(p)['jobs']]
            if ids:
                r=subprocess.run(['sacct','-j',','.join(ids),'-P','--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'],capture_output=True,text=True,timeout=20);print(r.stdout or r.stderr)
def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'),'User-submitted Slurm allocation required')
    require(os.environ.get('SR_EXPECTED_SIGNATURE')==signature(),'Submitted revision changed')
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Preparation unavailable')
    i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));n=4 if stage=='train' else 2
    require(stage not in ('cache','train','evaluate') or 0<=i<n,'Invalid array index')
    key=f'{stage}_{i}' if stage in ('cache','train','evaluate') else stage
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    with (ROOT/f'reports/locks/{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Monitor(key) as monitor,cost(key):
            if stage!='report':verify_inputs()
            if stage=='cache':
                from sr_cache import run
                run(i,monitor)
            elif stage=='coverage':
                from sr_report import coverage
                coverage(monitor)
            elif stage=='train':
                require(complete(ROOT/'reports/COVERAGE_COMPLETE.json'),'Mixed-suite coverage unavailable; refuse GPU model load')
                from sr_train import run
                run(i,monitor)
            elif stage=='select':
                require(all(complete(ROOT/f'training/{name}/COMPLETE.json') for name,_,_ in candidates()),'Training incomplete')
                from sr_train import select
                select(monitor)
            elif stage=='evaluate':
                require(complete(ROOT/'reports/SELECT_COMPLETE.json'),'No PNG selection; refuse model load')
                from sr_evaluate import run
                run(i,monitor)
            else:
                from sr_report import run
                run(monitor,final=True)
def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for s in ('preflight','submit','resume','dry-run','status','watch'):g.add_argument('--'+s,action='store_true')
    g.add_argument('--worker',choices=('cache','coverage','train','select','evaluate','report'));p.add_argument('--slurm',action='store_true');p.add_argument('--interval',type=int,default=30);a=p.parse_args()
    if a.worker:worker(a.worker)
    elif a.submit or a.resume:submit()
    elif a.preflight or a.dry_run:
        with cost('prepare_local'):v=prepare()
        print('PREFLIGHT_PASS',signature(),'96 source trajectories; at most240 fit/240 verify; no jobs submitted')
        if a.dry_run:
            def preview(stage,indices=None,deps=(),afterany=False):print(' '.join(command(stage,indices,deps,afterany)));return stage.upper()+'_JOB'
            stages(preview)
    elif a.watch:
        require(a.interval>=10,'interval must be at least10 seconds')
        try:
            while True:status(a.slurm);time.sleep(a.interval)
        except KeyboardInterrupt:pass
    else:status(a.slurm)
if __name__=='__main__':main()

"""Manual two-wave submission, bounded arrays, provenance and read-only monitoring."""
from __future__ import annotations
import argparse,fcntl,getpass,json,os,re,subprocess,time
from common import ROOT,cfg,signature,read_json,atomic_json,complete,require
from data import prepare

PILOT=('individual','projection','full')
NIGHT=('full_s42','full_s43','projection_s43')

def queue():
    r=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=30)
    require(r.returncode==0,'Slurm queue unavailable: '+r.stderr.strip());return r.stdout

def path_for(phase):return ROOT/f'reports/SUBMISSION_{phase}.json'

def receipt(stage,index=None):
    if stage=='pilot_train':return ROOT/f'training/pilot/{PILOT[index]}/COMPLETE.json'
    if stage=='night_train':return ROOT/f'training/night/{NIGHT[index]}/COMPLETE.json'
    if stage=='night_eval':return ROOT/f'evaluation/night/{cfg()["suites"][index]}_COMPLETE.json'
    return ROOT/{'oracle':'evaluation/pilot/ORACLE_COMPLETE.json','pilot_check':'evaluation/pilot/CHECK_COMPLETE.json',
                 'pilot_report':'results/pilot/COMPLETE.json','night_report':'results/night/COMPLETE.json'}[stage]

def command(stage,indices=None,dependencies=(),dependency_kind='afterany'):
    cmd=['sbatch','--parsable','--job-name=gm1_'+signature()[:8]+'_'+stage,
         '--export=ALL,GEOM_EXPECTED_SIGNATURE='+signature()]
    if indices is not None:cmd.append('--array='+','.join(str(i) for i in indices)+'%'+str(min(len(indices),4)))
    if dependencies:cmd.append('--dependency='+dependency_kind+':'+':'.join(dict.fromkeys(dependencies)))
    return cmd+[str(ROOT/f'slurm/{stage}.sbatch')]

def stages(phase,submit_one,pilot_dependency=None):
    """CPU reports use afterany so failed experiments still yield an explicit report."""
    if phase=='pilot':
        deps=[];missing=[i for i in range(3) if not complete(receipt('pilot_train',i))]
        if missing:deps.append(submit_one('pilot_train',missing,()))
        if not complete(receipt('oracle')):deps.append(submit_one('oracle',None,()))
        if not complete(receipt('pilot_check')):deps.append(submit_one('pilot_check',None,deps))
        if not complete(receipt('pilot_report')):submit_one('pilot_report',None,deps)
    else:
        deps=[];missing=[i for i in range(3) if not complete(receipt('night_train',i))]
        if missing:
            # An overnight wave may be queued now but starts only after the pilot report passes.
            deps.append(submit_one('night_train',missing,[pilot_dependency] if pilot_dependency else [],'afterok'))
        missing_eval=[i for i in range(4) if not complete(receipt('night_eval',i))]
        if missing_eval:deps.append(submit_one('night_eval',missing_eval,deps))
        if not complete(receipt('night_report')):submit_one('night_report',None,deps)

def submit(phase):
    prepare();(ROOT/'reports/locks').mkdir(parents=True,exist_ok=True)
    with (ROOT/'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);active=queue();prefix='gm1_'+signature()[:8]+'_'
        for line in active.splitlines():
            name=line.split('|')[2]
            require(not name.startswith('gm1_') or name.startswith(prefix),'An older geometry-match revision is active; inspect status')
        if any(line.split('|')[2].startswith(prefix+phase+'_') or
               (phase=='pilot' and line.split('|')[2]==prefix+'oracle') for line in active.splitlines()):
            print('ALREADY_ACTIVE: '+phase+'; nothing submitted');return
        pilot_dependency=None
        if phase=='night' and not complete(receipt('pilot_report')):
            require(path_for('pilot').exists(),'Submit --submit-pilot first; night method is already fixed')
            prior=read_json(path_for('pilot'));require(prior['signature']==signature(),'Pilot source revision mismatch')
            report_jobs=[j['job_id'] for j in prior['jobs'] if j['stage']=='pilot_report']
            require(len(report_jobs)==1,'Pilot report job unavailable')
            pilot_dependency=report_jobs[0]
            require(any(line.split('|')[0].split('_')[0]==pilot_dependency for line in active.splitlines()),
                    'Pilot report did not pass and is no longer active; inspect its failure before night submission')
        path=path_for(phase)
        if path.exists():atomic_json(read_json(path),ROOT/f'reports/SUBMISSIONS/{phase}_{time.time_ns()}.json')
        meta={'status':'SUBMITTING','signature':signature(),'phase':phase,'jobs':[],
              'max_concurrent_gpus':4,'primary_candidate':'full_s42','submitted_unix':time.time()}
        atomic_json(meta,path)
        def one(stage,indices,deps,kind='afterany'):
            cmd=command(stage,indices,deps,kind);meta['pending_command']=cmd;atomic_json(meta,path)
            try:result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=repr(error));atomic_json(meta,path)
                raise RuntimeError('Submission response unavailable; no automatic retry/cancellation') from error
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE',error=result.stderr.strip());atomic_json(meta,path)
                raise RuntimeError('No automatic retry/cancellation; inspect Slurm and submission receipt')
            meta['jobs'].append({'stage':stage,'job_id':job,'command':cmd});meta.pop('pending_command',None)
            atomic_json(meta,path);print(stage,job,flush=True);return job
        stages(phase,one,pilot_dependency)
        meta['status']='SUBMITTED' if meta['jobs'] else 'ALREADY_COMPLETE';atomic_json(meta,path)

def status(slurm=False):
    print('source',signature()[:12],'GPU=4090 max_concurrency=4 primary=full_s42')
    for phase in ('pilot','night'):
        p=path_for(phase)
        if p.exists():
            value=read_json(p);print('submission',phase,value.get('status'),'matches_source',value.get('signature')==signature(),
                                     [(j['stage'],j['job_id']) for j in value.get('jobs',[])])
        for name in ('STATUS.json','COMPLETE.json'):
            p=ROOT/f'results/{phase}/{name}'
            if p.exists():print(str(p.relative_to(ROOT)),read_json(p))
    for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read_json(p);age=time.time()-v['heartbeat_unix']
        print(p.stem,{k:v.get(k) for k in ('status','phase','completed','total','estimated_remaining_s','loss','q','orthogonal','error')},
              'heartbeat_age_s',round(age),'STALE_HEARTBEAT' if v.get('status')=='RUNNING' and age>90 else '')
    if slurm:
        print(queue())
        ids=[]
        for phase in ('pilot','night'):
            if path_for(phase).exists():ids += [j['job_id'] for j in read_json(path_for(phase)).get('jobs',[])]
        if ids:
            r=subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode','-P'],
                             capture_output=True,text=True,timeout=30);print(r.stdout or r.stderr)

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for flag in ('preflight','submit-pilot','submit-night','status','watch'):g.add_argument('--'+flag,action='store_true')
    g.add_argument('--dry-run',choices=('pilot','night','all'))
    p.add_argument('--slurm',action='store_true');p.add_argument('--interval',type=int,default=30);a=p.parse_args()
    if a.preflight:print(prepare())
    elif a.submit_pilot:submit('pilot')
    elif a.submit_night:submit('night')
    elif a.dry_run:
        def preview(stage,indices,deps,kind='afterany'):
            print(' '.join(command(stage,indices,deps,kind)));return stage.upper()+'_JOB'
        for phase in ('pilot','night') if a.dry_run=='all' else (a.dry_run,):
            stages(phase,preview,'PILOT_REPORT_JOB' if phase=='night' else None)
    elif a.watch:
        require(a.interval>=10,'Watch interval must be >=10 seconds')
        try:
            while True:status(a.slurm);time.sleep(a.interval)
        except KeyboardInterrupt:pass
    else:status(a.slurm)

if __name__=='__main__':main()

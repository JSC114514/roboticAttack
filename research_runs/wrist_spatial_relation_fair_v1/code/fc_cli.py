"""Fair-control training and paired evaluation; manual --submit/--resume only."""
import argparse
import getpass
import json
import os
import signal
import subprocess
import time
from fc_common import ROOT,config,signature,read,atomic,complete,finish,lock,Monitor,require,verify,candidates,cases,condition_names,episode_receipt

PREFIX='srfair1_'
RELATED_PREFIXES=(PREFIX,)


def prefix():return PREFIX+signature()[:8]+'_'


def queue():
    result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'],capture_output=True,text=True,timeout=20)
    require(result.returncode==0, 'Cannot read Slurm; refuse duplicate-blind submission: '+result.stderr)
    return [line.split('|',3) for line in result.stdout.splitlines() if line]


def done(stage):
    if stage=='prepare':return complete(ROOT/'reports/PREPARE_COMPLETE.json')
    if stage.startswith('train'):return complete(ROOT/'training'/candidates()[int(stage[-1])][0]/'COMPLETE.json')
    if stage=='select':return complete(ROOT/'reports/FAIRNESS_COMPLETE.json')
    if stage.startswith('evaluate_'):
        _,cohort,index=stage.split('_');return complete(ROOT/f'reports/EVALUATE_{cohort}_{index}_COMPLETE.json')
    if stage=='report_original40':return complete(ROOT/'reports/ORIGINAL40_REPORT_COMPLETE.json')
    require(stage=='report','Unknown stage: '+stage)
    return complete(ROOT/'results/COMPLETE.json')


def seal_completed_workers():
    if not complete(ROOT/'reports/PREPARE_COMPLETE.json'):return
    for cohort in ('original40','expanded40'):
        for index in (0,1):
            stage=f'evaluate_{cohort}_{index}'
            if done(stage):continue
            receipts=[episode_receipt(c,n) for c in cases(cohort)[index::2] for n in condition_names()]
            if receipts and all(complete(p) for p in receipts):
                finish(ROOT/f'reports/EVALUATE_{cohort}_{index}_COMPLETE.json',receipts,sealed_on_cpu=True)


def command(stage,deps=(),after='afterok'):
    kind='train' if stage.startswith('train') else 'evaluate' if stage.startswith('evaluate_') else 'cpu'
    cmd=['sbatch','--parsable','--job-name='+prefix()+stage,'--kill-on-invalid-dep=yes','--export=ALL,FC_EXPECTED_SIGNATURE='+signature()]
    if deps:cmd+=['--dependency='+after+':'+':'.join(deps)]
    return cmd+[str(ROOT/'slurm'/(kind+'.sbatch')),stage]


def check_launchers():
    checked=[];expected=signature();environment=dict(os.environ,FC_EXPECTED_SIGNATURE=expected,CUDA_VISIBLE_DEVICES='0')
    for kind in ('train','evaluate','cpu'):
        result=subprocess.run(['bash',str(ROOT/'slurm'/(kind+'.sbatch')),'__launch_check__'],env=environment,capture_output=True,text=True,timeout=30)
        require(result.returncode==0,'Batch launcher failed: '+kind+': '+result.stderr)
        require(json.loads(result.stdout)=={'root':str(ROOT),'signature':expected},'Batch points to another run')
        checked.append(kind)
    return checked


def submit(dry=False):
    check_launchers()
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):verify()
    with lock('submission'):
        current=[] if dry else [r for r in queue() if r[2].startswith(PREFIX)]
        if current:
            print('ALREADY_ACTIVE: no duplicate jobs submitted.')
            for row in current:print(*row)
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
                state={'signature':signature(),'jobs':submitted,'pending_command':cmd,'status':'SUBMITTING','submitted_unix':time.time()}
                atomic(state,ROOT/'SUBMISSION.json')
                try:result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                except BaseException as error:
                    atomic({**state,'status':'AMBIGUOUS_SUBMISSION_RESPONSE','error':repr(error)},ROOT/'SUBMISSION.json');raise
                require(result.returncode==0,'Submission failed; prior IDs retained: '+result.stderr)
                job=result.stdout.strip().split(';')[0]
                if not job.isdigit():
                    atomic({**state,'status':'AMBIGUOUS_SUBMISSION_RESPONSE','error':result.stdout},ROOT/'SUBMISSION.json')
                    raise RuntimeError('Ambiguous Slurm response; inspect queue before retry')
                submitted[stage]=job;atomic({'signature':signature(),'jobs':submitted,'status':'SUBMITTED','submitted_unix':time.time()},ROOT/'SUBMISSION.json')
                print(stage,job,flush=True)
            return [job]
        uncertain=ROOT/'SUBMISSION.json'
        if not dry and uncertain.exists():require(read(uncertain).get('status')!='AMBIGUOUS_SUBMISSION_RESPONSE','Previous submit response ambiguous; inspect queue before retry')
        prepared=one('prepare')
        a=one('train0',prepared);b=one('train1',prepared)
        a=one('train2',a or prepared);b=one('train3',b or prepared)
        selected=one('select',a+b or prepared)
        original=one('evaluate_original40_0',selected)+one('evaluate_original40_1',selected)
        early_report=one('report_original40',original or selected,'afterany')
        expanded=one('evaluate_expanded40_0',original or selected)+one('evaluate_expanded40_1',original or selected)
        one('report',expanded+early_report or original or selected,'afterany')


def select(monitor):
    verify();assets={}
    for name,method,seed in candidates():
        receipt=ROOT/'training'/name/'checkpoint_004000.json'
        require(complete(receipt) and complete(receipt.parent/'COMPLETE.json'),'Final training incomplete: '+name)
        saved=read(receipt);require(saved['step']==saved['optimized']==4000 and saved['no_target']==0,'Wrong update budget')
        logs=[ROOT/p for p in read(receipt.parent/'COMPLETE.json')['artifacts'] if '/ATTEMPT_' in p and p.endswith('.jsonl')]
        actual={}
        for path in logs:
            for line in path.read_text().splitlines():
                row=json.loads(line);actual[row['step']]=row['state_id']
        expected=read(ROOT/'data/SAMPLE_SCHEDULE.json')[str(seed)]
        require([actual.get(i+1) for i in range(4000)]==expected,'Training sample sequence differs')
        assets[name]={'path':str(receipt.with_suffix('.png')),'method':method,'seed':seed,'updates':4000}
    atomic({'assets':assets,'reference':'clean','expert_reference':False,'shared_pool':217,'native_goal_checkpoint':True,
            'same_initial_PNG_and_order':True,'same_Adam_schedule':True,'same_no_EOT':True,'dual_branch_RGB_gradient':True},ROOT/'data/FAIRNESS.json')
    finish(ROOT/'reports/FAIRNESS_COMPLETE.json',[ROOT/'data/FAIRNESS.json',*[ROOT/'training'/n/'COMPLETE.json' for n,_,_ in candidates()]])
    monitor.update(phase='fairness_checks_passed',completed=4,total=4)


def worker(stage):
    require(os.environ.get('FC_EXPECTED_SIGNATURE')==signature(),'Worker revision differs')
    if stage=='__launch_check__':print(json.dumps({'root':str(ROOT),'signature':signature()}));return
    def terminate(_sig,_frame):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    with lock('worker_'+stage),Monitor(stage) as monitor:
        if stage.startswith(('train','evaluate_')):
            require(os.environ.get('SLURM_JOB_ID'),'GPU work requires a user-submitted allocation')
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        if stage=='prepare':
            from fc_prepare import run
            run(monitor)
        elif stage.startswith('train'):
            from fc_train import run
            run(int(stage[-1]),monitor)
        elif stage=='select':select(monitor)
        elif stage.startswith('evaluate_'):
            from fc_evaluate import run
            _,cohort,index=stage.split('_');run(int(index),cohort,monitor)
        elif stage in ('report','report_original40'):
            from fc_report import run
            run(monitor,export=stage=='report')
        else:raise ValueError(stage)


def status(slurm=False):
    print('Run:',ROOT.name,'revision',signature()[:12])
    if (ROOT/'data/MANIFEST.json').exists():
        v=read(ROOT/'data/MANIFEST.json');print('Shared pool:',v['training_observations'],'reused episodes:',v['reused_episodes'],'planned new:',v['new_episodes'])
    for path in sorted((ROOT/'reports/PROGRESS').glob('*.json')):
        v=read(path);print(path.stem,v.get('status','RUNNING'),v.get('phase',''),f"{v.get('completed','?')}/{v.get('total','?')}",
            'heartbeat_age_s',round(time.time()-v['heartbeat_unix']),'ETA_s',round(v.get('estimated_remaining_s',0)),v.get('error') or '')
    if (ROOT/'results/STATUS.json').exists():
        v=read(ROOT/'results/STATUS.json');print('Results:',v['status'],v['episodes_completed'],'/',v['planned_episodes'])
        for row in v['summary']:
            if row['cohort']!='all80':print(row['cohort'],row['condition'],'CASR',row['CASR'],'paired',row['clean_success_pairs'])
    if slurm:
        for row in queue():
            if row[2].startswith(PREFIX):print('Slurm:',*row)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for name in ('submit','resume','status','watch','prepare','report','check','dry-run'):group.add_argument('--'+name,action='store_true')
    group.add_argument('--worker');parser.add_argument('--slurm',action='store_true');args=parser.parse_args()
    if not args.worker or not args.worker.startswith(('train','evaluate_')):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    if args.submit or args.resume:submit()
    elif args.dry_run:submit(True)
    elif args.status:status(args.slurm)
    elif args.watch:
        try:
            while True:status(args.slurm);time.sleep(15)
        except KeyboardInterrupt:pass
    elif args.check:
        cfg=config();require(cfg['checkpoint']=='libero_goal' and cfg['updates']==4000 and not cfg['eot'] and cfg['max_concurrent_gpus']==2,'Protocol changed')
        print('Actual batch launcher checks PASS:',check_launchers());print('No model, simulator or Slurm submission executed.')
    elif args.prepare or args.report:
        stage='prepare' if args.prepare else 'report'
        with lock('worker_'+stage),Monitor(stage+'_local') as monitor:
            if args.prepare:
                from fc_prepare import run
                run(monitor)
            else:
                from fc_report import run
                run(monitor,export=False)
    else:worker(args.worker)


if __name__=='__main__':main()



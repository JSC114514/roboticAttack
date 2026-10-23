"""Manual Slurm DAG. Read-only status and CPU checks never launch GPU work."""
import argparse
import ast
import getpass
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from jr_common import (ROOT, PYTHON, config, signature, read, atomic, complete, verify,
                       lock, require, Monitor, episode_receipt)

PREFIX = 'jointret1_'


def queue():
    p = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%j|%R'],
                       capture_output=True, text=True, timeout=30)
    require(p.returncode == 0, 'Cannot inspect queue; refusing duplicate-blind submission: ' + p.stderr)
    return [v.split('|', 3) for v in p.stdout.splitlines() if v]


def plans(workers, done_clean=False, done_eval=False, done_report=False, done_video=False):
    require(1 <= workers <= 4, 'Use 1–4 evaluation workers')
    stages = []
    if not done_clean: stages.append(dict(name='clean', launcher='clean.sbatch', after='afterok', parents=[]))
    if not done_eval: stages.append(dict(name='evaluate', launcher='evaluate.sbatch', after='afterok',
                                       parents=['clean'] if not done_clean else [], array=f'0-{workers-1}%{workers}'))
    if not done_report: stages.append(dict(name='report', launcher='cpu.sbatch', after='afterany',
                                         parents=[v['name'] for v in stages if v['name'] in ('clean','evaluate')]))
    if not done_video: stages.append(dict(name='videos', launcher='cpu.sbatch', after='afterany',
                                         parents=[v['name'] for v in stages if v['name']=='report']))
    return stages


def command(item, workers, dependencies=()):
    cmd = ['sbatch', '--parsable', '--job-name='+PREFIX+signature()[:8]+'_'+item['name'],
           '--export=ALL,JR_EXPECTED_SIGNATURE='+signature()]
    if dependencies: cmd += ['--dependency='+item['after']+':'+':'.join(dependencies), '--kill-on-invalid-dep=yes']
    if item.get('array'): cmd += ['--array='+item['array']]
    return cmd + [str(ROOT/'slurm'/item['launcher']), item['name'], str(workers)]


def check():
    parsed = []
    for folder in ('code', 'tests'):
        for path in sorted((ROOT / folder).glob('*.py')):
            ast.parse(path.read_text(), filename=str(path)); parsed.append(path.name)
    launchers = list((ROOT/'slurm').glob('*.sbatch')) + [ROOT/'slurm/environment.sh', ROOT/'run.sh']
    for path in launchers:
        p = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
        require(p.returncode == 0, 'Shell syntax error: '+str(path)+p.stderr)
    env = dict(os.environ, JR_EXPECTED_SIGNATURE=signature(), CUDA_VISIBLE_DEVICES='0')
    for name in ('clean', 'evaluate', 'cpu'):
        p = subprocess.run(['bash', str(ROOT/'slurm'/(name+'.sbatch')), '__launch_check__', '4'],
                           env=env, capture_output=True, text=True, timeout=30)
        require(p.returncode == 0, 'Launcher failure: '+name+': '+p.stderr)
        require(json.loads(p.stdout) == {'root': str(ROOT), 'signature': signature()}, 'Wrong launcher destination')
    p = subprocess.run([PYTHON, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(ROOT/'tests')],
                       capture_output=True, text=True, timeout=180)
    print(p.stdout, end=''); require(p.returncode == 0, 'CPU contract tests failed: '+p.stderr)
    proof=dict(status='PASS', parsed=len(parsed), shell_files=len(launchers), tests=p.stdout.strip(),
               GPU_simulation_tested=False, signature=signature(), checked_unix=time.time())
    atomic(proof,ROOT/'reports/CPU_CHECKS.json'); print(json.dumps(proof))


def status(slurm=False):
    model=Path(config()['model_dir']); cache=model/'.cache/huggingface/download'
    data = dict(signature=signature(), model_download_complete=(model/'JOINT_DOWNLOAD.json').exists(),
                completed_weight_shards=len(list(model.glob('model-*-of-*.safetensors'))), expected_weight_shards=4,
                download_partial_cache_bytes=sum(p.stat().st_size for p in cache.glob('*.incomplete')),
                prepared=False, completed=0, expected=1040, completed_clean=0, confirmation='prepared only; not submitted')
    if (ROOT/'data/MANIFEST.json').exists():
        m = read(ROOT/'data/MANIFEST.json'); data['prepared'] = m['signature'] == signature()
        if data['prepared']:
            for c in m['cases']:
                for r in m['conditions']:
                    if complete(episode_receipt(c, r['name']), artifacts=False):
                        data['completed'] += 1; data['completed_clean'] += r['name']=='clean'
    data['progress'] = [read(p) for p in sorted((ROOT/'reports/PROGRESS').glob('*.json'))]
    if (ROOT/'results/STATUS.json').exists(): data['results'] = read(ROOT/'results/STATUS.json')
    if slurm:
        data['queue'] = [r for r in queue() if r[2].startswith(PREFIX)]
        # Query accounting only for this run's recorded IDs; no historical scan.
        p = ROOT/'SUBMISSION.json'
        ids = list(read(p).get('jobs', {}).values()) if p.exists() else []
        if ids:
            result = subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode','-P'],
                                    capture_output=True, text=True, timeout=30)
            data['accounting'] = result.stdout if result.returncode==0 else result.stderr
    print(json.dumps(data, indent=2, ensure_ascii=False)); return data


def submit(workers, dry=False):
    check()
    if not dry:
        with Monitor('prepare') as monitor:
            from jr_prepare import run
            run(monitor)
        m = verify()
    else: m = read(ROOT/'data/ASSETS.json') if (ROOT/'data/ASSETS.json').exists() else None
    with lock('submission'):
        if not dry:
            current = [r for r in queue() if r[2].startswith(PREFIX)]
            if current:
                print('ALREADY_ACTIVE: no duplicate jobs submitted.'); print(json.dumps(current)); return
            uncertain = ROOT/'SUBMISSION.json'
            if uncertain.exists(): require(read(uncertain).get('status') != 'AMBIGUOUS_SUBMISSION_RESPONSE',
                                          'Ambiguous previous sbatch response: inspect SUBMISSION.json and queue before resuming')
        clean_done = bool(m) and all(complete(episode_receipt(c,'clean')) for c in m['cases'])
        eval_done = bool(m) and all(complete(episode_receipt(c,r['name'])) for c in m['cases'] for r in m['conditions'])
        report_done = complete(ROOT/'results/COMPLETE.json') if (ROOT/'results/COMPLETE.json').exists() else False
        video_status = ROOT/'reports/VIDEO_STATUS.json'
        video_done = eval_done and video_status.exists() and read(video_status)['status']=='COMPLETE'
        jobs = {}; serial = 0
        for item in plans(workers, clean_done, eval_done, report_done, video_done):
            cmd = command(item, workers, [jobs[v] for v in item['parents'] if v in jobs])
            if dry:
                serial += 1; jobs[item['name']] = str(serial); print(shlex.join(cmd)); continue
            state = dict(signature=signature(), jobs=dict(jobs), workers=workers, status='SUBMITTING',
                         pending_command=cmd, submitted_unix=time.time())
            atomic(state, ROOT/'SUBMISSION.json')
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except BaseException as error:
                atomic({**state, 'status':'AMBIGUOUS_SUBMISSION_RESPONSE', 'error':repr(error)}, ROOT/'SUBMISSION.json'); raise
            if p.returncode != 0:
                atomic({**state, 'status':'SUBMISSION_FAILED', 'error':p.stderr}, ROOT/'SUBMISSION.json')
                raise RuntimeError('sbatch failed; earlier IDs retained: '+p.stderr)
            job = p.stdout.strip().split(';')[0]
            if not job.isdigit():
                atomic({**state,'status':'AMBIGUOUS_SUBMISSION_RESPONSE','error':p.stdout}, ROOT/'SUBMISSION.json')
                raise RuntimeError('Unclear sbatch response; inspect queue before retry')
            jobs[item['name']] = job
            atomic({**state, 'jobs':dict(jobs), 'status':'SUBMITTED', 'pending_command':None}, ROOT/'SUBMISSION.json')
            print(item['name'], job, flush=True)
        if not jobs: print('ALL_COMPLETE: no jobs needed')


def worker(stage, workers):
    require(os.environ.get('JR_EXPECTED_SIGNATURE') == signature(), 'Submitted code revision differs')
    if stage == '__launch_check__':
        print(json.dumps(dict(root=str(ROOT), signature=signature()))); return
    def terminate(signum, frame): raise RuntimeError('Scheduler termination signal '+str(signum)+'; completed episodes preserved')
    signal.signal(signal.SIGTERM, terminate)
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    with lock(stage+'_'+str(index)), Monitor(stage+'_'+str(index)) as monitor:
        if stage in ('clean','evaluate'):
            from jr_rollout import run
            run(stage, index, workers, monitor)
        elif stage == 'report':
            from jr_report import run
            run(monitor)
        elif stage == 'videos':
            from jr_report import videos
            videos(monitor)
        else: raise ValueError(stage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    for name in ('download','assets','prepare','check','submit','resume','dry-run','status','watch','report','export-videos'):
        group.add_argument('--'+name, action='store_true')
    group.add_argument('--worker', choices=('clean','evaluate','report','videos','__launch_check__'))
    p.add_argument('--workers', type=int, choices=range(1,5), default=4)
    p.add_argument('--slurm', action='store_true'); args = p.parse_args()
    if args.worker: worker(args.worker, args.workers)
    elif args.download:
        from download_model import download
        download()
    elif args.check: check()
    elif args.submit or args.resume or args.dry_run: submit(args.workers, args.dry_run)
    elif args.status: status(args.slurm)
    elif args.watch:
        try:
            while True: status(args.slurm); time.sleep(15)
        except KeyboardInterrupt: pass
    else:
        stage = 'assets' if args.assets else 'prepare' if args.prepare else 'report' if args.report else 'videos'
        with lock(stage), Monitor(stage) as monitor:
            if stage in ('prepare','assets'):
                from jr_prepare import run, assets
                (assets if stage=='assets' else run)(monitor)
            else:
                from jr_report import run, videos
                (run if stage=='report' else videos)(monitor)


if __name__ == '__main__': main()

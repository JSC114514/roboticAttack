"""Manual submission, bounded resume and monitoring. No implicit training jobs."""
import argparse
import fcntl
import getpass
import os
import re
import signal
import subprocess
import time
from protocol import (ROOT, config, native, require, signature, prepare, read_json, atomic_json,
                      complete, worker_complete, Monitor, cost)

def queue():
    result = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%j|%R'],
                            capture_output=True, text=True, timeout=20)
    require(result.returncode == 0, 'Slurm unavailable; no jobs submitted: ' + result.stderr.strip())
    return result.stdout

def commands(missing, dependency=None):
    prefix = 'gc1_' + signature()[:8] + '_'
    export = '--export=ALL,GC_EXPECTED_SIGNATURE=' + signature()
    result = []
    if missing:
        result.append(('rollout', ['sbatch', '--parsable', '--job-name=' + prefix + 'rollout', export,
            '--array=' + ','.join(map(str, missing)) + '%3', str(ROOT / 'slurm/rollout.sbatch')]))
    report = ['sbatch', '--parsable', '--job-name=' + prefix + 'report', export]
    if missing:
        report.append('--dependency=afterany:' + (dependency or '<ROLLOUT_JOB_ID>'))
    result.append(('report', report + [str(ROOT / 'slurm/report.sbatch')]))
    return result

def submit():
    prepare()
    (ROOT / 'reports/locks').mkdir(parents=True, exist_ok=True)
    with (ROOT / 'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = queue()
        if any(line.split('|')[2].startswith('gc1_') for line in active.splitlines()):
            print('ALREADY_ACTIVE: no duplicate submitted')
            return
        if complete(ROOT / 'results/COMPLETE.json'):
            print('ALREADY_COMPLETE: no resources requested')
            return
        path = ROOT / 'reports/SUBMISSION.json'
        if path.exists():
            prior = read_json(path)
            require(prior['status'] != 'AMBIGUOUS_SUBMISSION_RESPONSE',
                    'Previous response ambiguous; inspect Slurm before explicitly resolving receipt. No automatic retry.')
            atomic_json(prior, ROOT / f'reports/SUBMISSIONS/{time.time_ns()}.json')
        meta = {'signature': signature(), 'status': 'SUBMITTING', 'jobs': [], 'submitted_unix': time.time(),
                'max_concurrent_gpus': 3, 'new_training': False}
        atomic_json(meta, path)
        missing = [i for i in range(3) if not worker_complete(i)]
        rollout_job = None
        for stage, command in commands(missing):
            command = [arg.replace('<ROLLOUT_JOB_ID>', rollout_job or '') for arg in command]
            meta['pending_command'] = command
            atomic_json(meta, path)
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=30)
                job = response.stdout.strip().split(';')[0]
                require(response.returncode == 0 and re.fullmatch(r'\d+', job), response.stderr or response.stdout)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE', error=repr(error))
                atomic_json(meta, path)
                raise RuntimeError('Submission response uncertain; inspect receipt/Slurm; no retry or cancellation') from error
            meta['jobs'].append({'stage': stage, 'job_id': job, 'command': command})
            meta.pop('pending_command', None)
            atomic_json(meta, path)
            if stage == 'rollout':
                rollout_job = job
            print(stage, job, flush=True)
        meta['status'] = 'SUBMITTED'
        atomic_json(meta, path)

def status(slurm):
    print('revision', signature()[:12], '3x4090 maximum; 12G/4CPU per GPU; 60 native episodes; no training')
    completed = sum(complete(ROOT / f'evaluation/episodes/t{t:02d}_s{s}_{c}/COMPLETE.json')
                    for t in config()['tasks'] for s in config()['states'] for c in config()['conditions'])
    print('completed episodes', completed, '/60')
    for path in sorted((ROOT / 'reports/PROGRESS').glob('*.json')):
        value = read_json(path)
        age = round(time.time() - value['heartbeat_unix'])
        print(path.stem, {k: value.get(k) for k in ('status', 'phase', 'task_id', 'state_index', 'condition',
              'completed', 'total', 'estimated_remaining_s', 'error')}, 'heartbeat_age_s', age,
              'STALE' if value.get('status') == 'RUNNING' and age > 90 else '')
    for path in (ROOT / 'results/STATUS.json', ROOT / 'reports/SUBMISSION.json'):
        if path.exists():
            print(path.name, read_json(path))
    if slurm:
        try:
            print(queue())
            path = ROOT / 'reports/SUBMISSION.json'
            if path.exists():
                jobs = [j['job_id'] for j in read_json(path)['jobs']]
                if jobs:
                    result = subprocess.run(['sacct', '-j', ','.join(jobs), '-P',
                        '--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'], capture_output=True, text=True, timeout=20)
                    print(result.stdout or result.stderr)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print('SLURM_UNAVAILABLE:', str(error))

def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'), 'Workers require a manual Slurm allocation')
    require(os.environ.get('GC_EXPECTED_SIGNATURE') == signature(), 'Submitted source revision changed')
    require(complete(ROOT / 'reports/PREPARE_COMPLETE.json'), 'Preparation missing or stale')
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    require(stage == 'report' or 0 <= index < 3, 'Invalid array index')
    key = f'rollout_{index}' if stage == 'rollout' else 'report'
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    (ROOT / 'reports/locks').mkdir(parents=True, exist_ok=True)
    with (ROOT / f'reports/locks/{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Monitor(key) as monitor, cost(key):
            if stage == 'rollout':
                from rollout import run_worker
                run_worker(index, monitor)
            else:
                from closed_report import run_report
                run_report(monitor)

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for option in ('preflight', 'submit', 'dry-run', 'status', 'watch'):
        group.add_argument('--' + option, action='store_true')
    group.add_argument('--worker', choices=('rollout', 'report'))
    parser.add_argument('--slurm', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if args.preflight:
        manifest = prepare()
        print('PREFLIGHT_PASS', signature(), 'episodes', manifest['episodes'], 'no jobs submitted')
    elif args.submit:
        submit()
    elif args.dry_run:
        prepare()
        for stage, command in commands([i for i in range(3) if not worker_complete(i)]):
            print(stage, ' '.join(command))
    elif args.worker:
        worker(args.worker)
    elif args.watch:
        require(args.interval >= 10, 'Watch interval must be >=10 seconds')
        try:
            while True:
                status(args.slurm)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        status(args.slurm)

if __name__ == '__main__':
    main()

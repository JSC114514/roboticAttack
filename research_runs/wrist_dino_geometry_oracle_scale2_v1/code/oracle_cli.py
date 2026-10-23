"""Manual one-GPU submission, deduplication, episode resume and status."""
import argparse
import fcntl
import getpass
import os
import re
import signal
import subprocess
import time
from oracle_common import (ROOT, config, signature, prepare, read_json, atomic_json, complete,
                           require, rollout_complete, episode_path, Monitor, cost)

def queue():
    result = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%j|%R'],
                            capture_output=True, text=True, timeout=20)
    require(result.returncode == 0, 'Slurm unavailable; no submission: ' + result.stderr.strip())
    return result.stdout

def commands(needs_rollout):
    prefix = 'gos2_' + signature()[:8] + '_'
    export = '--export=ALL,ORACLE_EXPECTED_SIGNATURE=' + signature()
    result = []
    if needs_rollout:
        result.append(('rollout', ['sbatch', '--parsable', '--job-name=' + prefix + 'rollout', export,
                                  str(ROOT / 'slurm/rollout.sbatch')]))
    command = ['sbatch', '--parsable', '--job-name=' + prefix + 'report', export]
    if needs_rollout:
        command.append('--dependency=afterany:<ROLLOUT_JOB_ID>')
    result.append(('report', command + [str(ROOT / 'slurm/report.sbatch')]))
    return result

def submit():
    prepare()
    with (ROOT / 'reports/locks/submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = queue()
        if any(line.split('|')[2].startswith('gos2_') for line in active.splitlines()):
            print('ALREADY_ACTIVE: nothing submitted')
            return
        if complete(ROOT / 'results/COMPLETE.json'):
            print('ALREADY_COMPLETE: no resources requested')
            return
        path = ROOT / 'reports/SUBMISSION.json'
        if path.exists():
            prior = read_json(path)
            require(prior['status'] != 'AMBIGUOUS_SUBMISSION_RESPONSE', 'Inspect ambiguous previous response before retrying')
            atomic_json(prior, ROOT / f'reports/SUBMISSIONS/{time.time_ns()}.json')
        meta = {'signature': signature(), 'status': 'SUBMITTING', 'jobs': [], 'submitted_unix': time.time(),
                'max_concurrent_gpus': 1, 'new_training': False, 'new_episodes': 20,
                'reused_clean_episodes': 10, 'scope': 'internal_DINO_mechanism'}
        atomic_json(meta, path)
        rollout_id = None
        for stage, command in commands(not rollout_complete()):
            command = [arg.replace('<ROLLOUT_JOB_ID>', rollout_id or '') for arg in command]
            meta['pending_command'] = command
            atomic_json(meta, path)
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=30)
                job = response.stdout.strip().split(';')[0]
                require(response.returncode == 0 and re.fullmatch(r'\d+', job), response.stderr or response.stdout)
            except BaseException as error:
                meta.update(status='AMBIGUOUS_SUBMISSION_RESPONSE', error=repr(error))
                atomic_json(meta, path)
                raise RuntimeError('Submission uncertain; no automatic retry or cancellation') from error
            meta['jobs'].append({'stage': stage, 'job_id': job, 'command': command})
            meta.pop('pending_command', None)
            atomic_json(meta, path)
            if stage == 'rollout':
                rollout_id = job
            print(stage, job, flush=True)
        meta['status'] = 'SUBMITTED'
        atomic_json(meta, path)

def status(slurm):
    print('revision', signature()[:12], '1x4090/4CPU/12G; scale2: 20 new + 10 reused clean; no training')
    done = sum(complete(episode_path(t, s, c)) for t in config()['tasks']
               for s in config()['states'] for c in config()['conditions'])
    print('completed episodes', done, '/30')
    for path in sorted((ROOT / 'reports/PROGRESS').glob('*.json')):
        value = read_json(path)
        age = round(time.time() - value['heartbeat_unix'])
        print(path.stem, {k: value.get(k) for k in ('status', 'phase', 'task_id', 'state_index', 'condition',
            'completed', 'total', 'estimated_remaining_s', 'capture_elapsed_s', 'error')}, 'heartbeat_age_s', age,
            'STALE' if value.get('status') == 'RUNNING' and age > 90 else '')
    for path in (ROOT / 'reports/SUBMISSION.json', ROOT / 'results/STATUS.json'):
        if path.exists():
            print(path.name, read_json(path))
    if slurm:
        try:
            print(queue())
            path = ROOT / 'reports/SUBMISSION.json'
            if path.exists():
                ids = [j['job_id'] for j in read_json(path)['jobs']]
                if ids:
                    result = subprocess.run(['sacct', '-j', ','.join(ids), '-P',
                        '--format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode'], capture_output=True, text=True, timeout=20)
                    print(result.stdout or result.stderr)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print('SLURM_UNAVAILABLE:', str(error))

def worker(stage):
    require(os.environ.get('SLURM_JOB_ID'), 'Worker requires a manually submitted Slurm allocation')
    require(os.environ.get('ORACLE_EXPECTED_SIGNATURE') == signature(), 'Submitted revision changed')
    require(complete(ROOT / 'reports/PREPARE_COMPLETE.json'), 'Preparation missing or stale')
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    with (ROOT / f'reports/locks/{stage}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Monitor(stage) as monitor, cost(stage):
            if stage == 'rollout':
                from oracle_rollout import run
            else:
                from oracle_report import run
            run(monitor)

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('preflight', 'submit', 'dry-run', 'status', 'watch'):
        group.add_argument('--' + flag, action='store_true')
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
        for stage, command in commands(not rollout_complete()):
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

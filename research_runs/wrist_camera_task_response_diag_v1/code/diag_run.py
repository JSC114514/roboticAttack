# 发布注释：调度入口只在用户显式--submit/--resume时调用sbatch；--dry-run仅打印命令。
# 运行来源与发布哈希分别记录在SOURCE_MANIFEST.json；历史运行目录不改动。
"""Manual-only diagnostic Slurm DAG: one prepare, three model units, CPU report."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'
PREFIX = 'wcamdiag1_'


def _read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def submission_lock():
    folder = ROOT/'reports'; folder.mkdir(exist_ok=True, parents=True)
    with (folder/'submission.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def queue():
    p = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%j|%R'], capture_output=True, text=True, timeout=30)
    if p.returncode: raise RuntimeError('Cannot inspect existing jobs: '+p.stderr)
    return [line.split('|', 3) for line in p.stdout.splitlines() if line]


def blocking_jobs(rows):
    """A stale pending CPU report never blocks a fresh manual GPU resume."""
    active, reports = [], []
    for row in rows:
        if not row[2].startswith(PREFIX): continue
        (reports if row[1] == 'PENDING' and row[2].endswith('_report') else active).append(row)
    return active, reports


def build_plan(workers=2):
    from diag_common import signature, complete, config
    if not 1 <= workers <= 2: raise ValueError('Use one or two GPU workers')
    stages = []
    if not complete(ROOT/'reports/PREPARE_COMPLETE.json'):
        stages.append(dict(name='prepare', units=[dict(command='prepare')], parents=[], gpu=True,
                           memory_gb=12, cpus=4, concurrency=1, after='afterok'))
    units = [dict(command='evaluate', model=model) for model in config()['models']
             if not complete(ROOT/'reports'/f'EVALUATE_{model}_COMPLETE.json')]
    if units:
        stages.append(dict(name='evaluate', units=units, parents=[s['name'] for s in stages], gpu=True,
                           memory_gb=12, cpus=4, concurrency=workers, after='afterok'))
    stages.append(dict(name='report', units=[dict(command='report')], parents=[s['name'] for s in stages], gpu=False,
                       memory_gb=4, cpus=2, concurrency=1, after='afterany'))
    return dict(signature=signature(), workers=workers, stages=stages)


def slurm_command(item, plan_path, signature, dependencies=()):
    command = ['sbatch', '--parsable', '--job-name='+PREFIX+signature[:8]+'_'+item['name'],
               '--cpus-per-task='+str(item['cpus']), '--mem='+str(item['memory_gb'])+'G',
               '--time='+('02:00:00' if item['gpu'] else '00:30:00'),
               '--output='+str(ROOT/'logs/%x_%A_%a.out'), '--error='+str(ROOT/'logs/%x_%A_%a.err'),
               '--export=ALL,DIAG_EXPECTED_SIGNATURE='+signature]
    command += ['--partition=rtx4090', '--gres=gpu:4090:1'] if item['gpu'] else ['--partition=cpu']
    if len(item['units']) > 1: command.append(f'--array=0-{len(item["units"])-1}%{item["concurrency"]}')
    if dependencies:
        command += ['--dependency='+item['after']+':'+':'.join(dependencies), '--kill-on-invalid-dep=yes']
    command += ['--wrap', shlex.join(['bash', str(ROOT/'run.sh'), '--worker', item['name'], '--plan', str(plan_path)])]
    return command


def check():
    from diag_common import preflight, signature, atomic
    parsed = []
    for folder in ('code', 'tests'):
        for path in sorted((ROOT/folder).glob('*.py')):
            ast.parse(path.read_text(), filename=str(path)); parsed.append(path.name)
    p = subprocess.run(['bash', '-n', str(ROOT/'run.sh')], capture_output=True, text=True)
    if p.returncode: raise RuntimeError(p.stderr)
    preflight_result = preflight()
    print(json.dumps(dict(preflight=preflight_result), indent=2), flush=True)
    p = subprocess.run([PYTHON, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT/'tests'), '-p', 'test_*.py'],
                       capture_output=True, text=True, timeout=180)
    print(p.stdout, end=''); print(p.stderr, end='')
    if p.returncode: raise RuntimeError('CPU tests failed')
    atomic(dict(status='PASS', signature=signature(), parsed=parsed, GPU_tested=False,
                preflight=preflight_result, checked_unix=time.time()), ROOT/'reports/CPU_CHECKS.json')


def submit(workers=2, dry=False):
    from diag_common import signature, atomic
    check()
    with submission_lock():
        previous = ROOT/'reports/SUBMISSION.json'
        prior = _read(previous) if previous.exists() else {}
        active, reports = blocking_jobs(queue()) if not dry else ([], [])
        if active:
            print(json.dumps(dict(status='ALREADY_ACTIVE', jobs=active), indent=2)); return
        if prior.get('status') == 'AMBIGUOUS_SUBMISSION_RESPONSE' and not dry:
            raise RuntimeError('Inspect recorded ambiguous sbatch response and queue before resubmission')
        plan = build_plan(workers)
        encoded = json.dumps(plan, sort_keys=True).encode()
        plan_path = ROOT/'reports'/('PLAN_'+hashlib.sha256(encoded).hexdigest()[:12]+'.json')
        if not dry:
            atomic(plan, plan_path); (ROOT/'logs').mkdir(exist_ok=True)
        jobs = {}
        for item in plan['stages']:
            command = slurm_command(item, plan_path, plan['signature'], [jobs[p] for p in item['parents']])
            if dry:
                jobs[item['name']] = str(len(jobs)+1); print(shlex.join(command)); continue
            state = dict(status='SUBMITTING', jobs=jobs.copy(), pending_command=command, plan=str(plan_path),
                         signature=plan['signature'], prior_pending_cpu_reports=reports)
            atomic(state, previous)
            try:
                p = subprocess.run(command, capture_output=True, text=True, timeout=30)
            except BaseException as error:
                atomic({**state, 'status':'AMBIGUOUS_SUBMISSION_RESPONSE', 'error':repr(error)}, previous); raise
            if p.returncode:
                atomic({**state, 'status':'SUBMISSION_FAILED', 'error':p.stderr}, previous)
                raise RuntimeError('Submission failed; earlier job IDs retained: '+p.stderr)
            job = p.stdout.strip().split(';')[0]
            if not job.isdigit():
                atomic({**state, 'status':'AMBIGUOUS_SUBMISSION_RESPONSE', 'error':p.stdout}, previous)
                raise RuntimeError('Unclear sbatch response; inspect before retry')
            jobs[item['name']] = job
            atomic({**state, 'status':'SUBMITTED', 'jobs':jobs.copy(), 'pending_command':None}, previous)
            print(item['name'], job, flush=True)


def execute(command, model=None):
    from diag_common import Monitor
    with Monitor(command+('_'+model if model else '')) as monitor:
        if command == 'prepare':
            from diag_prepare import prepare
            return prepare(monitor)
        if command == 'evaluate':
            from diag_evaluate import evaluate
            return evaluate(model, monitor)
        from diag_report import report
        return report(monitor)


def worker(name, path):
    from diag_common import signature, require
    plan = _read(path)
    # CPU reports must still expose failed/incomplete work after a source change.
    if name != 'report':
        require(signature() == plan['signature'] == os.environ.get('DIAG_EXPECTED_SIGNATURE'),
                'Submitted source/config revision changed; make a fresh manual submission')
    item = next(x for x in plan['stages'] if x['name'] == name)
    unit = item['units'][int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))]
    execute(unit['command'], unit.get('model'))


def status(slurm=False):
    from diag_common import signature
    state = dict(signature=signature(), automatic_submission=False)
    for name, path in [('submission', ROOT/'reports/SUBMISSION.json'), ('summary', ROOT/'reports/SUMMARY.json')]:
        if path.exists():
            value = _read(path)
            state[name] = {k: value[k] for k in ('status', 'signature', 'jobs', 'counts', 'plan') if k in value}
    state['progress'] = [_read(p) for p in sorted((ROOT/'reports/PROGRESS').glob('*.json'))]
    if slurm:
        state['queue'] = [row for row in queue() if row[2].startswith(PREFIX)]
        ids = list(state.get('submission', {}).get('jobs', {}).values())
        if ids:
            p = subprocess.run(['sacct', '-j', ','.join(ids), '--format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode', '-P'], capture_output=True, text=True, timeout=30)
            state['accounting'] = p.stdout if not p.returncode else p.stderr
    print(json.dumps(state, indent=2)); return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('submit', 'resume', 'status', 'watch', 'dry-run', 'check', 'report'):
        group.add_argument('--'+flag, action='store_true')
    group.add_argument('--worker', choices=('prepare', 'evaluate', 'report'))
    parser.add_argument('--slurm', action='store_true'); parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.worker:
        if not args.plan: parser.error('--worker requires --plan')
        worker(args.worker, args.plan)
    elif args.check: check()
    elif args.submit or args.resume or args.dry_run: submit(args.workers, args.dry_run)
    elif args.status: status(args.slurm)
    elif args.watch:
        try:
            while True: status(args.slurm); time.sleep(30)
        except KeyboardInterrupt: pass
    elif args.report: execute('report')


if __name__ == '__main__': main()

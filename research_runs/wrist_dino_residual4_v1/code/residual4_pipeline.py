"""Default CPU preflight; Slurm is touched only by explicit user --submit."""
from __future__ import annotations

import argparse
import ast
import getpass
import json
import os
import re
import subprocess
import time

from residual4_common import (ROOT, PYTHON, atomic_json, budget, cases, completed_episode,
    episode_folder, exclusive, load_json, operational_hashes, preflight, require, sha256, signature, specs)

STAGES = [('prepare', '01_prepare_cpu.sbatch'), ('validate', '02_validate_4090.sbatch'),
          ('analyze', '03_analyze_cpu.sbatch')]


def prepare():
    try:
        report = preflight()
        previous_path = ROOT / 'reports/SUBMISSION.json'
        if previous_path.exists():
            require(load_json(previous_path)['signature'] == report['signature'],
                    'Submitted sources changed. Preserve existing run and create an independent version')
        for path in sorted([*ROOT.glob('code/*.py'), *ROOT.glob('tests/*.py')]):
            ast.parse(path.read_text(), filename=str(path))
        for path in sorted([ROOT / 'run.sh', *ROOT.glob('slurm/*')]):
            response = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True, timeout=20)
            require(response.returncode == 0, 'Shell syntax failed: ' + str(path) + ': ' + response.stderr)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        response = subprocess.run([str(PYTHON), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(ROOT / 'tests')],
                                  capture_output=True, text=True, env=environment, timeout=180)
        report['cpu_tests'] = {'returncode': response.returncode, 'output': response.stdout, 'stderr': response.stderr}
        report['status'] = 'CPU_PREFLIGHT_PASS' if response.returncode == 0 else 'CPU_TESTS_FAILED'
        atomic_json(report, ROOT / 'reports/CPU_PREFLIGHT.json')
        require(response.returncode == 0, 'CPU tests failed:\n' + response.stdout + response.stderr)
        atomic_json(report, ROOT / 'reports/PREPARATION.json')
        print('CPU_PREFLIGHT_PASS ' + json.dumps({'signature': report['signature'], 'budget': budget(),
            'tests': response.stdout.strip(), 'models_loaded': False, 'slurm_submitted': False}), flush=True)
        return report
    except Exception as error:
        import traceback
        atomic_json({'status': 'FAILED', 'error': repr(error), 'traceback': traceback.format_exc(),
                     'failed_unix': time.time(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                     'models_loaded': False}, ROOT / 'reports/PREPARE_FAILED.json')
        raise


def queue():
    response = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%b|%j'],
                              capture_output=True, text=True, timeout=40)
    require(response.returncode == 0, 'Scheduler unavailable; nothing submitted: ' + response.stderr.strip())
    rows = []
    for line in response.stdout.splitlines():
        fields = line.split('|', 3)
        require(len(fields) == 4, 'Unexpected scheduler output')
        rows.append({'job_id': fields[0].split('_')[0], 'state': fields[1], 'gres': fields[2], 'name': fields[3]})
    return rows


def pending_shards(sig):
    return [case['shard'] for case in cases() if any(
        completed_episode(episode_folder(case, spec), sig) is None for spec in specs(case))]


def verify_complete_result(status, sig):
    require(status['signature'] == sig, 'Completed source signature changed')
    require(all(sha256(ROOT / 'results' / p) == h for p, h in status['assets'].items()), 'Result checksum mismatch')
    require(all(sha256(ROOT / p) == h for p, h in status['episode_assets'].items()), 'Episode checksum mismatch')


def submit():
    # This function is never called by prepare/status/analysis/workers or tests
    # without subprocess/queue mocks. The assistant does not call it.
    with exclusive(ROOT / 'reports/.submission.lock'):
        report = prepare()
        result_path = ROOT / 'results/STATUS.json'
        if result_path.exists() and load_json(result_path)['status'] == 'COMPLETE':
            verify_complete_result(load_json(result_path), report['signature'])
            print('ALREADY_COMPLETE: no submission')
            return
        path = ROOT / 'reports/SUBMISSION.json'
        previous = load_json(path) if path.exists() else None
        if previous:
            require(previous['signature'] == report['signature'], 'Submitted source changed')
            require(previous['status'] not in ['SUBMITTING', 'SUBMISSION_UNCERTAIN'],
                    'Prior submission is ambiguous; inspect recorded job name and scheduler before retry')
        current = queue()
        active = {r['job_id'] for r in current}
        if previous:
            require(not (active & {r['job_id'] for r in previous['jobs']}), 'This run already has active/pending jobs')
        require(not any('gpu' in r['gres'].lower() for r in current), 'Another own GPU batch is active/pending')
        remaining = pending_shards(report['signature'])
        (ROOT / 'logs').mkdir(exist_ok=True)
        token = str(time.time_ns())
        archive = ROOT / 'reports/submissions' / (token + '.json')
        receipt = {'status': 'SUBMITTING', 'signature': report['signature'], 'jobs': [], 'token': token,
                   'created_unix': time.time(), 'budget': budget(), 'operational_hashes': operational_hashes(),
                   'remaining_shards': remaining}
        def save():
            atomic_json(receipt, archive)
            atomic_json(receipt, path)
        save()
        dependency = None
        for stage, filename in STAGES:
            if stage == 'validate' and not remaining:
                receipt.setdefault('skipped_stages', []).append(stage)
                save()
                continue
            name = 'dr4_' + token[-8:] + '_' + stage
            command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--job-name=' + name]
            if dependency:
                command.append('--dependency=afterok:' + dependency)
            if stage == 'validate':
                command.append('--array=' + ','.join(map(str, remaining)) + '%4')
            command.append(str(ROOT / 'slurm' / filename))
            receipt['pending_command'] = command
            save()
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=40)
            except (subprocess.TimeoutExpired, OSError) as error:
                receipt.update(status='SUBMISSION_UNCERTAIN', failed_stage=stage, error=repr(error),
                               scheduler_lookup_job_name=name)
                save()
                raise RuntimeError('Ambiguous submission; no automatic retry: ' + str(path)) from error
            job_id = response.stdout.strip().split(';')[0]
            if response.returncode or not re.fullmatch(r'\d+', job_id):
                uncertain = not response.returncode or bool(re.fullmatch(r'\d+', job_id)) or any(
                    word in (response.stderr + response.stdout).lower()
                    for word in ['socket', 'timeout', 'connection', 'communication'])
                receipt.update(status='SUBMISSION_UNCERTAIN' if uncertain else 'SUBMISSION_FAILED',
                               failed_stage=stage, error=response.stderr or response.stdout,
                               scheduler_lookup_job_name=name)
                save()
                raise RuntimeError('Submission stopped; previous recorded jobs preserved: ' + receipt['error'])
            receipt['jobs'].append({'stage': stage, 'job_id': job_id, 'command': command})
            receipt.pop('pending_command', None)
            dependency = job_id
            save()
            print(stage + ': ' + job_id, flush=True)
        receipt.update(status='SUBMITTED')
        save()
        print('Manual submission recorded: 4 episodes, max 4 RTX4090, no training.', flush=True)


def status():
    output = {'scope': 'Local receipts only; use squeue/sacct for live scheduler status.',
              'budget': budget(), 'workers': [], 'submission': 'NOT_SUBMITTED'}
    path = ROOT / 'reports/SUBMISSION.json'
    if path.exists():
        receipt = load_json(path)
        output['submission'] = {k: receipt.get(k) for k in ['status', 'created_unix', 'remaining_shards']}
        output['submission']['jobs'] = [{k: r[k] for k in ['stage', 'job_id']} for r in receipt['jobs']]
    for case in cases():
        path = ROOT / 'episodes' / case['id'] / 'STATUS.json'
        if path.exists():
            value = load_json(path)
            output['workers'].append({k: value.get(k) for k in ['case_id', 'status', 'phase',
                'episodes_complete', 'episodes_expected', 'episode', 'decision_block', 'heartbeat_unix', 'error']})
    for label, relative in [('cpu_preflight', 'reports/CPU_PREFLIGHT.json'), ('results', 'results/STATUS.json')]:
        path = ROOT / relative
        if path.exists():
            value = load_json(path)
            output[label] = {k: value[k] for k in ['status', 'signature', 'episodes', 'decision', 'eligible_cases'] if k in value}
    for action, success_path in [('PREPARE', ROOT / 'reports/PREPARATION.json'), ('ANALYZE', ROOT / 'results/STATUS.json')]:
        path = ROOT / 'reports' / (action + '_FAILED.json')
        if path.exists():
            failure = load_json(path)
            output[action.lower() + '_failure'] = {'error': failure['error'].replace('\\n', '\n').splitlines()[0][:240],
                'path': str(path.relative_to(ROOT)),
                'resolved_by_later_success': success_path.exists() and success_path.stat().st_mtime_ns > path.stat().st_mtime_ns}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    for name in ['submit', 'preflight', 'status', 'analyze']:
        group.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    if args.submit:
        submit()
    elif args.status:
        status()
    elif args.analyze:
        from residual4_analyze import analyze
        result = analyze()
        require(result['status'] == 'COMPLETE', 'Run incomplete; inspect results/STATUS.json')
    else:
        prepare()


if __name__ == '__main__':
    main()

"""CPU preparation/status by default; submission requires explicit --submit."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import time
from pathlib import Path
from pilot_common import (ROOT, PYTHON, atomic_json, budget, exclusive, load_json, operational_hashes,
                          preflight, prepared, require, sha256, signature, shards, specs, key)

STAGES = [('prepare', '01_prepare_cpu.sbatch'), ('calibration', '02_calibration_4090.sbatch'),
          ('select', '03_select_cpu.sbatch'), ('main', '04_main_4090.sbatch'), ('analyze', '05_analyze_cpu.sbatch')]


def prepare():
    report = preflight()
    import ast
    for path in sorted([*ROOT.glob('code/*.py'), *ROOT.glob('tests/*.py')]):
        ast.parse(path.read_text(), filename=str(path))
    for path in sorted([ROOT / 'run.sh', *ROOT.glob('slurm/*')]):
        result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True, timeout=20)
        require(result.returncode == 0, 'Shell syntax failed: ' + str(path) + ': ' + result.stderr)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    command = [str(PYTHON), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(ROOT / 'tests')]
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=120)
    report['cpu_tests'] = {'returncode': result.returncode, 'output': result.stdout, 'stderr': result.stderr}
    require(result.returncode == 0, 'CPU tests failed:\n' + result.stdout + result.stderr)
    atomic_json(report, ROOT / 'reports/PREPARATION.json')
    atomic_json(report, ROOT / 'reports/CPU_PREFLIGHT.json')
    print('CPU_PREFLIGHT_PASS ' + json.dumps({'signature': report['signature'], 'budget': budget(),
                                           'tests': result.stdout.strip(), 'model_loaded': False, 'slurm_submitted': False}), flush=True)
    return report


def queue():
    response = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%b|%j'],
                              capture_output=True, text=True, timeout=40)
    require(response.returncode == 0, 'Scheduler unavailable; no jobs submitted: ' + response.stderr.strip())
    result = []
    for line in response.stdout.splitlines():
        fields = line.split('|', 3)
        require(len(fields) == 4, 'Unexpected scheduler output')
        result.append({'job_id': fields[0].split('_')[0], 'state': fields[1], 'gres': fields[2], 'name': fields[3]})
    return result


def pending_shards(stage, sig):
    from pilot_run import completed_episode, verify_screen
    selection_path = ROOT / 'reports/SELECTION.json'
    selection_hash = sha256(selection_path) if stage == 'main' and selection_path.exists() else 'none'
    pending = []
    for index, (suite, task) in enumerate(shards(stage)):
        folder = ROOT / 'episodes' / stage / suite / ('task_' + str(task).zfill(2))
        complete = all(completed_episode(folder / key(spec), sig, selection_hash) is not None
                       for spec in specs(stage, task))
        if complete and stage == 'calibration':
            if (folder / 'SCREEN_STATUS.json').exists():
                verify_screen(folder)
            else:
                complete = False
        if not complete:
            pending.append(index)
    return pending


def submit():
    # Only this explicit entry point calls sbatch. No subprocess command is assembled by a shell.
    report = prepare()
    (ROOT / 'logs').mkdir(exist_ok=True)
    with exclusive(ROOT / 'reports/.submission.lock'):
        final = ROOT / 'results/STATUS.json'
        if final.exists():
            status = load_json(final)
            require(status['signature'] == report['signature'], 'Completed run signature changed')
            if status['status'] == 'COMPLETE':
                require(all(sha256(ROOT / 'results' / path) == h for path, h in status['assets'].items()), 'Completed result assets changed')
                print('ALREADY_COMPLETE: no submission', flush=True)
                return
        path = ROOT / 'reports/SUBMISSION.json'
        previous = load_json(path) if path.exists() else None
        if previous:
            require(previous['signature'] == report['signature'], 'Previously submitted source changed; preserve this run')
            require(previous['status'] not in ['SUBMITTING', 'SUBMISSION_UNCERTAIN'],
                    'Previous submission may have succeeded without a recorded job ID; inspect scheduler/receipt before retry')
        current = queue()
        active = {row['job_id'] for row in current}
        if previous:
            require(not (active & {item['job_id'] for item in previous['jobs']}), 'This run already has active or pending jobs')
        require(not any('gpu' in row['gres'].lower() for row in current),
                'Another own GPU batch is active/pending; refusing an overlapping GPU allocation')
        remaining_by_stage = {stage: pending_shards(stage, report['signature']) for stage in ['calibration', 'main']}
        token = str(time.time_ns())
        archive = ROOT / 'reports/submissions' / (token + '.json')
        receipt = {'status': 'SUBMITTING', 'signature': report['signature'], 'jobs': [],
                   'created_unix': time.time(), 'token': token, 'max_gpu_concurrent': 4,
                   'budget': budget(), 'operational_hashes': operational_hashes()}

        def save():
            atomic_json(receipt, archive)
            atomic_json(receipt, path)

        save()
        dependency = None
        for stage, filename in STAGES:
            remaining = None
            if stage in ['calibration', 'main']:
                remaining = remaining_by_stage[stage]
                if not remaining:
                    receipt.setdefault('skipped_stages', []).append({'stage': stage, 'reason': 'all shards verified complete'})
                    save()
                    print(stage + ': already complete; no GPU job submitted', flush=True)
                    continue
            name = 'dcm_' + token[-8:] + '_' + stage
            command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--job-name=' + name]
            if dependency:
                command.append('--dependency=afterok:' + dependency)
            if remaining is not None:
                command.append('--array=' + ','.join(str(i) for i in remaining) + '%4')
            command.append(str(ROOT / 'slurm' / filename))
            receipt['pending_command'] = command
            save()
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=40)
            except (subprocess.TimeoutExpired, OSError) as error:
                receipt.update(status='SUBMISSION_UNCERTAIN', failed_stage=stage, error=repr(error),
                               scheduler_lookup_job_name=name)
                save()
                raise RuntimeError('Submission state uncertain; no automatic retry. Inspect ' + str(path)) from error
            job_id = response.stdout.strip().split(';')[0]
            if response.returncode or not re.fullmatch(r'\d+', job_id):
                uncertain = (not response.returncode or bool(re.fullmatch(r'\d+', job_id)) or
                             any(word in (response.stderr + response.stdout).lower() for word in ['socket', 'timeout', 'connection', 'communication']))
                receipt.update(status='SUBMISSION_UNCERTAIN' if uncertain else 'SUBMISSION_FAILED',
                               failed_stage=stage, error=response.stderr or response.stdout,
                               scheduler_lookup_job_name=name)
                save()
                raise RuntimeError('Submission failed; earlier recorded jobs were preserved: ' + receipt['error'])
            receipt['jobs'].append({'stage': stage, 'job_id': job_id, 'command': command})
            receipt.pop('pending_command', None)
            dependency = job_id
            save()
            print(stage + ': ' + job_id, flush=True)
        receipt['status'] = 'SUBMITTED'
        save()
        print('SUBMITTED: dependency pipeline; max 4 RTX4090; no training. reports/SUBMISSION.json', flush=True)


def status():
    reports = []
    for path in sorted(ROOT.glob('episodes/*/*/task_*/STATUS.json')):
        row = load_json(path)
        reports.append({'path': str(path.relative_to(ROOT)), **{name: row.get(name) for name in [
            'status', 'phase', 'suite', 'task_id', 'episodes_complete', 'episodes_expected', 'episode',
            'decision_block', 'heartbeat_unix', 'error']}})
    output = {'workers': reports, 'submission': load_json(ROOT / 'reports/SUBMISSION.json')
              if (ROOT / 'reports/SUBMISSION.json').exists() else 'NOT_SUBMITTED'}
    for name, path in [('selection', ROOT / 'reports/SELECTION.json'), ('timing', ROOT / 'reports/TIMING.json'),
                       ('results', ROOT / 'results/STATUS.json')]:
        if path.exists():
            value = load_json(path)
            output[name] = value if name == 'timing' else {k: value[k] for k in ['status', 'signature', 'episodes'] if k in value}
    for action in ['SELECT', 'ANALYZE']:
        path = ROOT / 'reports' / (action + '_FAILED.json')
        if path.exists():
            success = ROOT / ('reports/SELECTION.json' if action == 'SELECT' else 'results/STATUS.json')
            output[action.lower() + '_failed'] = {**load_json(path),
                'resolved_by_later_success': success.exists() and success.stat().st_mtime_ns > path.stat().st_mtime_ns}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--submit', action='store_true')
    actions.add_argument('--preflight', action='store_true')
    actions.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.submit:
        submit()
    elif args.status:
        status()
    else:
        prepare()


if __name__ == '__main__':
    main()

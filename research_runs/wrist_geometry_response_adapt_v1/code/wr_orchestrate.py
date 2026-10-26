"""Manual-only, resumable Slurm DAG for the wrist adaptation experiment."""
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
PREFIX = 'wradapt1_'


def profiles_for(stage):
    from wr_common import profiles
    return [p['name'] for p in profiles(stage)]


def _read(path):
    return json.loads(Path(path).read_text())


def _atomic(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n'); tmp.replace(path)


@contextmanager
def _lock():
    folder = ROOT/'reports'; folder.mkdir(exist_ok=True, parents=True)
    with (folder/'submission.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def queue():
    p = subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%j|%R'], capture_output=True,text=True,timeout=30)
    if p.returncode:
        raise RuntimeError('Cannot check existing jobs: '+p.stderr)
    return [line.split('|',3) for line in p.stdout.splitlines() if line]


def resume_jobs(rows, stage, previous):
    """A tracked, pending CPU-only report cannot duplicate model/simulator work."""
    report_id = str(previous.get('jobs', {}).get('report', ''))
    report_name = PREFIX + previous.get('signature', '')[:8] + '_' + stage + '_report'
    blocking, pending_reports = [], []
    for row in rows:
        if not row[2].startswith(PREFIX):
            continue
        if (previous.get('stage') == stage and row[0] == report_id
                and row[1] == 'PENDING' and row[2] == report_name):
            pending_reports.append(row)
        else:
            blocking.append(row)
    return blocking, pending_reports


def episode_path(stage, candidate, case, profile):
    return ROOT/'evaluation'/stage/candidate/f'{case["suite"]}_t{case["task_id"]:02d}_i{case["state_index"]:02d}'/profile/'COMPLETE.json'


def build_plan(stage, workers):
    from wr_common import signature, complete, candidates, cases
    maximum = 2 if stage == 'pilot' else 4
    if not 1 <= workers <= maximum:
        raise ValueError(f'{stage} supports 1–{maximum} GPU workers')
    configs = candidates(stage)
    all_cases = cases(stage)
    profiles = profiles_for(stage)
    stages = []
    def add(name, units, parents, gpu=False, memory=4, cpus=2, after='afterok'):
        if units:
            stages.append(dict(name=name, units=units, parents=parents, gpu=gpu, memory_gb=memory,
                               cpus=cpus, concurrency=workers if gpu else 1, after=after))
    expert_done = complete(ROOT/'reports'/f'EXPERT_{stage}_COMPLETE.json')
    add('prepare-expert', [dict(command='prepare-expert')] if not expert_done else [], [])
    add('prepare-camera', [dict(command='prepare-camera',shard=s,shards=workers) for s in range(workers)
        if not complete(ROOT/'reports'/f'CAMERA_{stage}_{workers}_{s}_COMPLETE.json')],
        ['prepare-expert'], True, 12, 4)
    def eval_units(names):
        return [dict(command='evaluate',candidate=name,shard=s,shards=workers)
                for name in names for s in range(workers)
                if (any(not complete(episode_path(stage,name,c,p))
                       for i,c in enumerate(all_cases) if i % workers == s for p in profiles)
                    or (s==0 and not complete(ROOT/f'evaluation/{stage}/{name}/OFFLINE_COMPLETE.json')))]
    add('base-eval', eval_units(['base']), ['prepare-camera','prepare-expert'], True,12,4)
    add('calibrate', [dict(command='calibrate')] if not complete(ROOT/'reports'/f'CALIBRATE_{stage}_COMPLETE.json') else [],
        ['base-eval','prepare-camera','prepare-expert'],True,12,4)
    add('train', [dict(command='train',candidate=c['name']) for c in configs
                  if not complete(ROOT/'training'/stage/c['name']/'COMPLETE.json')],
        ['calibrate','base-eval','prepare-camera','prepare-expert'],True,16,4)
    add('evaluate', eval_units([c['name'] for c in configs]), ['train','base-eval','prepare-camera','prepare-expert'],True,12,4)
    # Report also runs after failed dependencies to expose incomplete results.
    add('report',[dict(command='report')], [s['name'] for s in stages], after='afterany')
    existing = {s['name'] for s in stages}
    for item in stages:
        item['parents'] = [n for n in item['parents'] if n in existing]
    return dict(stage=stage,workers=workers,signature=signature(),stages=stages)


def slurm_command(item, plan_path, stage, signature, dependencies=()):
    count = len(item['units'])
    job_name = PREFIX + signature[:8] + '_' + stage + '_' + item['name']
    command = ['sbatch','--parsable','--job-name='+job_name,'--cpus-per-task='+str(item['cpus']),
               '--mem='+str(item['memory_gb'])+'G','--time='+('12:00:00' if item['name']=='train' else '08:00:00'),
               '--output='+str(ROOT/'logs'/'%x_%A_%a.out'), '--error='+str(ROOT/'logs'/'%x_%A_%a.err'),
               '--export=ALL,WR_EXPECTED_SIGNATURE='+signature]
    if item['gpu']:
        command += ['--partition=rtx4090','--gres=gpu:4090:1']
    else:
        command += ['--partition=cpu']
    if count > 1:
        command += [f'--array=0-{count-1}%{item["concurrency"]}']
    if dependencies:
        command += ['--dependency='+item['after']+':'+':'.join(dependencies),'--kill-on-invalid-dep=yes']
    invocation = ['bash', str(ROOT/'run.sh'),'--worker',item['name'],'--plan',str(plan_path)]
    command += ['--wrap', shlex.join(invocation)]
    return command


def check():
    parsed = []
    for folder in ('code','tests'):
        for path in sorted((ROOT/folder).glob('*.py')):
            ast.parse(path.read_text(),filename=str(path)); parsed.append(path.name)
    shell = subprocess.run(['bash','-n',str(ROOT/'run.sh')], capture_output=True,text=True)
    if shell.returncode: raise RuntimeError(shell.stderr)
    p = subprocess.run([PYTHON,'-B',str(ROOT/'code/wr_main.py'),'check'],capture_output=True,text=True)
    print(p.stdout,end='')
    if p.returncode: raise RuntimeError(p.stderr or 'Core static checks failed')
    p = subprocess.run([PYTHON,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_*.py'],capture_output=True,text=True,timeout=180)
    print(p.stdout,end=''); print(p.stderr,end='')
    if p.returncode: raise RuntimeError('CPU tests failed')
    from wr_common import signature
    _atomic(dict(status='PASS',signature=signature(),parsed=parsed,GPU_tested=False,checked_unix=time.time()),ROOT/'reports/CPU_CHECKS.json')


def submit(stage, workers, dry=False):
    from wr_common import signature
    check()
    with _lock():
        previous = ROOT/'reports'/f'SUBMISSION_{stage}.json'
        prior = _read(previous) if previous.exists() else {}
        active, pending_reports = resume_jobs(queue(), stage, prior) if not dry else ([], [])
        if active:
            print(json.dumps(dict(status='ALREADY_ACTIVE',jobs=active),ensure_ascii=False,indent=2)); return
        if prior.get('status') == 'AMBIGUOUS_SUBMISSION_RESPONSE':
            raise RuntimeError('Inspect the recorded ambiguous sbatch response and squeue before resubmission')
        if pending_reports:
            print(json.dumps(dict(status='PENDING_CPU_REPORT_DOES_NOT_BLOCK_RESUME',jobs=pending_reports),ensure_ascii=False),flush=True)
        plan = build_plan(stage, workers)
        encoded = json.dumps(plan,sort_keys=True).encode()
        plan_path = ROOT/'reports'/('PLAN_'+stage+'_'+hashlib.sha256(encoded).hexdigest()[:12]+'.json')
        if not dry:
            _atomic(plan,plan_path); (ROOT/'logs').mkdir(exist_ok=True)
        jobs = {}
        for item in plan['stages']:
            command = slurm_command(item,plan_path,stage,signature(),[jobs[p] for p in item['parents'] if p in jobs])
            if dry:
                jobs[item['name']] = str(len(jobs)+1); print(shlex.join(command)); continue
            state = dict(status='SUBMITTING',jobs=jobs.copy(),pending_command=command,plan=str(plan_path),signature=signature(),stage=stage,
                         prior_pending_cpu_reports=pending_reports)
            _atomic(state,previous)
            try:
                p = subprocess.run(command,capture_output=True,text=True,timeout=30)
            except BaseException as error:
                _atomic({**state,'status':'AMBIGUOUS_SUBMISSION_RESPONSE','error':repr(error)},previous); raise
            if p.returncode:
                _atomic({**state,'status':'SUBMISSION_FAILED','error':p.stderr},previous)
                raise RuntimeError('Submission failed; earlier job IDs retained: '+p.stderr)
            job = p.stdout.strip().split(';')[0]
            if not job.isdigit():
                _atomic({**state,'status':'AMBIGUOUS_SUBMISSION_RESPONSE','error':p.stdout},previous)
                raise RuntimeError('Unclear sbatch response; do not blindly retry')
            jobs[item['name']] = job
            _atomic({**state,'status':'SUBMITTED','jobs':jobs.copy(),'pending_command':None},previous)
            print(item['name'],job,flush=True)


def worker(name, path):
    from wr_common import signature
    plan = _read(path)
    if signature() != plan['signature'] or os.environ.get('WR_EXPECTED_SIGNATURE') != signature():
        raise RuntimeError('Submitted source/config revision differs; use a new manual submission')
    item = next(x for x in plan['stages'] if x['name']==name)
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    unit = item['units'][index]
    command = [PYTHON,'-B',str(ROOT/'code/wr_main.py'),unit['command'],'--stage',plan['stage'],'--resume']
    for key in ('candidate','shard','shards'):
        if key in unit: command += ['--'+key,str(unit[key])]
    p = subprocess.run(command)
    if p.returncode: raise SystemExit(p.returncode)


def status(stage, slurm=False):
    from wr_common import signature, complete
    state = dict(stage=stage,signature=signature(),gpu_jobs_submitted_automatically=False)
    state['progress'] = [_read(p) for p in sorted((ROOT/'reports/PROGRESS').glob('*.json')) if '_'+stage+'_' in p.stem]
    results = ROOT/'results'/stage/'SUMMARY.json'
    if results.exists():
        value=_read(results)
        state['results']=dict(status=value['status'],selection=value.get('selection'),
            candidates={k:{field:v[field] for field in ('complete','completed','expected',
                'clean_success_rate','perturbed_success_rate')} for k,v in value['candidates'].items()})
    submission = ROOT/'reports'/f'SUBMISSION_{stage}.json'
    if submission.exists(): state['submission'] = _read(submission)
    if slurm:
        state['queue'] = [r for r in queue() if r[2].startswith(PREFIX)]
        ids = list(state.get('submission',{}).get('jobs',{}).values())
        if ids:
            p = subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode','-P'],capture_output=True,text=True,timeout=30)
            state['accounting'] = p.stdout if not p.returncode else p.stderr
    print(json.dumps(state,ensure_ascii=False,indent=2)); return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    for flag in ('check','prepare','submit','resume','dry-run','status','watch','report'):
        group.add_argument('--'+flag,action='store_true')
    group.add_argument('--worker',choices=('prepare-expert','prepare-camera','base-eval','calibrate','train','evaluate','report'))
    p.add_argument('--stage',choices=('pilot','main'),default='pilot'); p.add_argument('--workers',type=int)
    p.add_argument('--slurm',action='store_true'); p.add_argument('--plan',type=Path)
    args = p.parse_args(); workers = args.workers or (2 if args.stage=='pilot' else 4)
    if args.worker: worker(args.worker,args.plan)
    elif args.check: check()
    elif args.submit or args.resume or args.dry_run: submit(args.stage,workers,args.dry_run)
    elif args.status: status(args.stage,args.slurm)
    elif args.watch:
        try:
            while True: status(args.stage,args.slurm); time.sleep(30)
        except KeyboardInterrupt: pass
    else:
        command = 'prepare-expert' if args.prepare else 'report'
        raise SystemExit(subprocess.call([PYTHON,'-B',str(ROOT/'code/wr_main.py'),command,'--stage',args.stage,'--resume']))


if __name__ == '__main__': main()

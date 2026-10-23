"""Independent follow-up artifacts; historical adapters are read-only."""
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import resource
import sys
import threading
import time
import traceback

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parents[1]
ROOT = Path(os.environ['FOLLOWUP_ROOT']).resolve()
SOURCE = PROJECT / 'research_runs/wrist_spatial_approach_pilot_v1'
RESCUE = PROJECT / 'research_runs/wrist_spatial_relation_rescue_v1'
AUDIT = PROJECT / 'research_runs/wrist_spatial_failure_audit_v1'
for folder in (SOURCE, AUDIT, RESCUE): sys.path.insert(1, str(folder / 'code'))
import sr_common as source

read, sha, atomic, require = source.read_json, source.sha, source.atomic_json, source.require
native = source.native


def config(): return read(ROOT / 'config.json')


def signature_for(run_root):
    require(source.signature() == '67d5f01434fb00f41b14218b426ec1563dc19850603dd9b09e280d2455b312a2', 'Source revision changed')
    files = [run_root / 'config.json', run_root / 'README.md', run_root / 'run.sh']
    files += [p for d in ('code', 'slurm', 'tests') for p in (run_root / d).glob('*') if p.is_file()]
    files += list(TOOLS.glob('*.py')) + [PROJECT / 'research_tools/export_video_gallery.py']
    import rr_common
    require(rr_common.signature() == '8eea68de3b7bc73b96acc2e4c3fcab6cf70e59b0b29c62a659d54623d741a9d1', 'Rescue revision changed')
    return native.digest({'files': {str(p.relative_to(PROJECT)): sha(p) for p in sorted(files)},
                          'source': source.signature(), 'rescue': rr_common.signature()})


@lru_cache(None)
def signature(): return signature_for(ROOT)


def external_complete(root, path):
    if not path.exists():return False
    value=read(path)
    require(value['signature']==signature_for(root) and value['status']=='COMPLETE','External revision differs: '+str(path))
    require(all((root/k).is_file() and sha(root/k)==v for k,v in value['artifacts'].items()),'External artifacts changed: '+str(path))
    return True


def complete(path):
    path = Path(path)
    if not path.exists(): return False
    value = read(path)
    require(value['signature'] == signature() and value['status'] == 'COMPLETE', 'Stale completion: ' + str(path))
    require(all((ROOT / k).is_file() and sha(ROOT / k) == v for k, v in value['artifacts'].items()), 'Changed artifacts: ' + str(path))
    return True


def finish(path, files, **extra):
    atomic({'status': 'COMPLETE', 'signature': signature(), 'completed_unix': time.time(),
            'artifacts': {str(Path(p).relative_to(ROOT)): sha(p) for p in files}, **extra}, path)


@contextmanager
def lock(name):
    p = ROOT / 'reports/locks' / (name + '.lock'); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def resources():
    result = {'process_rss_hwm_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0,
              'child_rss_hwm_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024}
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        if torch.cuda.is_initialized():
            result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          device=torch.cuda.get_device_name(0))
    return result


class Monitor:
    def __init__(self, stage):
        self.stage = stage; self.data = {}; self.started = time.monotonic()
        self.guard = threading.Lock(); self.stop = threading.Event()
    def update(self, **values):
        with self.guard: self.data.update(values)
    def write(self):
        with self.guard: data = dict(self.data)
        done = data.get('completed', 0) - data.get('resume_completed', 0)
        elapsed = time.monotonic() - self.started
        if done > 0 and data.get('total'):
            data.update(throughput_per_s=done / elapsed, estimated_remaining_s=(data['total']-data['completed'])*elapsed/done)
        atomic({**data, 'stage': self.stage, 'heartbeat_unix': time.time(), 'elapsed_s': elapsed,
                'job_id': os.environ.get('SLURM_JOB_ID'), **resources()}, ROOT / 'reports/PROGRESS' / (self.stage+'.json'))
    def __enter__(self):
        def pulse():
            while not self.stop.wait(30): self.write()
        self.thread = threading.Thread(target=pulse, daemon=True); self.thread.start(); self.write(); return self
    def __exit__(self, typ, error, tb):
        self.stop.set(); self.thread.join(5)
        value = {'stage': self.stage, 'status': 'FAILED' if typ else 'COMPLETE', 'error': repr(error) if typ else None,
                 'traceback': ''.join(traceback.format_exception(typ, error, tb)) if typ else None,
                 'elapsed_s': time.monotonic()-self.started, 'signature': signature(), **resources()}
        self.update(**value); self.write()
        atomic(value, ROOT / 'reports/COSTS' / f'{self.stage}_{time.time_ns()}.json')


def candidates():
    return [(f'{m}_s{s}', m, s) for s in config()['seeds'] for m in config()['methods']]


def verify():
    value = read(ROOT / 'data/MANIFEST.json')
    require(value['signature'] == signature(), 'Prepared revision changed')
    for p, expected in value['inputs'].items(): require(sha(p) == expected, 'Changed input: '+p)
    for p, expected in value['model_file_stats'].items():
        stat = Path(p).stat(); require([stat.st_size, stat.st_mtime_ns] == expected, 'Changed checkpoint: '+p)
    return value


def prepare():
    if complete(ROOT / 'reports/PREPARE_COMPLETE.json'): return verify()
    started=time.monotonic()
    import torch
    import numpy as np
    from sf_math_targets import donor_map, matched_targets
    import rr_common
    require(rr_common.complete(RESCUE / 'results/COMPLETE.json'), 'Completed rescue evidence changed')
    source_manifest = source.verify_inputs()
    prepared = RESCUE / 'data/EXPANSION_PREPARED.json'; spec = read(prepared)
    require(len(spec['starts']) == 40 and len(spec['conditions']) == 8, 'Expansion scope changed')
    for folder in ('data', 'reports/locks', 'reports/PROGRESS', 'reports/COSTS', 'reports/GATES', 'logs', 'results', 'episodes', 'comparisons'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    inputs = {str(prepared): sha(prepared), str(RESCUE / 'results/COMPLETE.json'): sha(RESCUE / 'results/COMPLETE.json')}
    inputs.update(source_manifest['inputs'])
    for p in spec['conditions'].values():
        if p: inputs[p] = sha(p)
    for name, path in spec['conditions'].items(): require(not path or sha(path) == spec['condition_sha256'][name], 'PNG changed')
    for start in spec['starts']:
        require(start['suite'] == 'libero_goal' and start['state_index'] in (2,3,4,5), 'Unexpected evaluation start')
        states = torch.load(start['initial_file'], map_location='cpu', weights_only=False)
        require(len(states) > start['state_index'], 'Missing initial state')
    require(not set((2,3,4,5)) & set(source.config()['fit_states'] + source.config()['verify_states']), 'Evaluation overlaps fitting trajectories')
    cases = [{**v, 'start': f"{v['suite']}_t{v['task_id']:02d}_s{v['state_index']:02d}"} for v in spec['starts']]
    value = {'signature': signature(), 'cases': cases, 'conditions': spec['conditions'],
             'condition_order': spec['condition_order'], 'inputs': inputs, 'model_file_stats': source_manifest['model_file_stats'],
             'checkpoint': source_manifest['checkpoint'], 'claim': 'Goal-only development expansion; native PNG; not historical-unseen confirmation',
             'scope': config()['scope'], 'maximum_episodes': 320 if config()['scope'] == 'expansion' else 160}
    if config()['scope'] == 'control':
        rows = list(source.records('fit')); donors = donor_map(rows, config()['shuffle_seed'])
        # Cache original controller commands explicitly; no approximate action Jacobian.
        stat_path = Path(source_manifest['checkpoint']) / 'dataset_statistics.json'
        require(stat_path.is_file(), 'Native action normalization file unavailable')
        stats = read(stat_path)[native.runtime_config()['unnorm_keys']['libero_goal']]['action']
        require(stats.get('mask', [True]*7)[:3] == [True]*3, 'Unexpected translation normalization mask')
        low = np.asarray(stats['q01'], float)[:3]; high = np.asarray(stats['q99'], float)[:3]
        targets = {}; details = []
        for row in rows:
            donor = donors[row['state_id']]; clean = row['teacher']['clean'].numpy()[..., :3]
            command = np.clip(.5*(clean+1)*(high-low+1e-8)+low, -1, 1)*.05
            a,b,alpha = matched_targets(command, row['teacher']['targets']['radial_reverse'].numpy(),
                                        row['teacher']['unit'], donor['teacher']['unit'])
            targets[row['state_id']] = {'matched_relation': a.tolist(), 'shuffled_relation': b.tolist()}
            details.append({'state_id': row['state_id'], 'donor_state_id': donor['state_id'], 'alpha': alpha,
                            'eligible': not row['teacher']['no_target'], 'angle_degrees': float(np.degrees(np.arccos(np.clip(np.dot(row['teacher']['unit'],donor['teacher']['unit']),-1,1)))),
                            'original_no_target_reason': row['teacher']['no_target_reason']})
        # Invalid-direction rows remain in the manifest, but retain the same old skip rule.
        atomic({'signature': signature(), 'targets': targets, 'details': details,
                'units': 'native_controller_goal_delta_m_not_actual_motion'}, ROOT / 'data/TARGETS.json')
        inputs[str(stat_path)] = sha(stat_path)
        for e in source.episodes():
            if e['cohort'] == 'fit':
                p = source.cache_receipt(e['episode_id']); require(source.complete(p), 'Missing fit cache')
                inputs[str(p)] = sha(p); inputs[str(p.with_suffix('.pt'))] = sha(p.with_suffix('.pt'))
        for seed in config()['seeds']:
            src = SOURCE / f'data/initial_s{seed}.png'; inputs[str(src)] = sha(src)
            import shutil
            shutil.copy2(src, ROOT / f'data/initial_s{seed}.png')
        value.update(target_details=details, train_frames=len(rows), eligible=sum(not r['teacher']['no_target'] for r in rows))
    atomic(value, ROOT / 'data/MANIFEST.json')
    files = [ROOT / 'data/MANIFEST.json']
    if config()['scope'] == 'control': files += [ROOT / 'data/TARGETS.json', *[ROOT / f'data/initial_s{s}.png' for s in config()['seeds']]]
    finish(ROOT / 'reports/PREPARE_COMPLETE.json', files)
    atomic({'stage':'CPU_prepare','status':'COMPLETE','elapsed_s':time.monotonic()-started,
            'signature':signature(),**resources()},ROOT/'reports/COSTS'/f'prepare_{time.time_ns()}.json')
    return verify()


def records(cohort):
    import torch
    targets = read(ROOT / 'data/TARGETS.json')['targets']
    for row in source.records(cohort):
        if cohort == 'fit':
            row['teacher']['targets'] = {k: torch.tensor(v, dtype=torch.float32) for k,v in targets[row['state_id']].items()}
        yield row


def training_module():
    """Private module instance reuses the audited optimizer, writes only NEW root."""
    from sf_train import make_engine
    module = source.module('followup_training_private', SOURCE / 'code/sr_train.py')
    for name, value in dict(ROOT=ROOT, config=config, candidates=candidates, records=records, complete=complete,
                            finish=finish, signature=signature, make_engine=make_engine).items():
        setattr(module, name, value)
    return module

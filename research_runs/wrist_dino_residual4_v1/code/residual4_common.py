"""Four new episodes; pinned imports from the completed 77-episode experiment."""
from __future__ import annotations
import importlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT.parent / 'wrist_dino_functional77_v1'


def frozen_module(name):
    if not name.startswith('functional77_'):
        raise RuntimeError('Not a frozen77 module: ' + name)
    directory = str(FROZEN_ROOT / 'code')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != (FROZEN_ROOT / 'code').resolve():
        raise RuntimeError('Frozen import path mismatch: ' + name)
    return module


_native = frozen_module('functional77_common')
for _name in ['require', 'load_json', 'sha256', 'digest', 'atomic_json', 'atomic_torch',
              'exclusive', 'parent_module', 'completed_episode', 'library_versions', 'PYTHON']:
    globals()[_name] = getattr(_native, _name)


@lru_cache(maxsize=1)
def cfg():
    value = load_json(ROOT / 'config.json')
    require(value['new_episode_count'] == value['max_gpu_concurrent'] == 4, 'Four-episode budget changed')
    require(value['component'] == 'residual_full' and value['amplitude_policy'] == 'unscaled_original_complement',
            'Residual condition changed')
    for key in ['finite_difference_relative_steps', 'finite_difference_error_limit',
                'norm_match_relative_tolerance', 'native_action_atol', 'snapshot_blocks', 'random_seed']:
        require(value[key] == _native.cfg()[key], 'Shared Probe/runtime setting changed: ' + key)
    return value


@lru_cache(maxsize=1)
def cases():
    rows = load_json(ROOT / 'CASES.json')['cases']
    require([r['id'] for r in rows] == ['c03', 'c04', 'c05', 'c09'] and
            [r['shard'] for r in rows] == list(range(4)), 'Frozen case selection changed')
    require(len({r['state_key'] for r in rows}) == 2, 'There are two independent physical initial states')
    return rows


def specs(case):
    return [{'case_id': case['id'], 'direction': 'inject', 'component': 'residual_full'}]


def episode_key(spec):
    require(spec['direction'] == 'inject' and spec['component'] == 'residual_full', 'Unplanned condition')
    return 'inject__residual_full'


def episode_folder(case, spec):
    return ROOT / 'episodes' / case['id'] / episode_key(spec)


def budget():
    require(sum(len(specs(c)) for c in cases()) == 4, 'More than four new episodes requested')
    return {'total_new': 4, 'independent_initial_states': 2, 'reused_comparison_records': 20,
            'case_workers': 4, 'max_gpu_concurrent': 4, 'new_training': 0,
            'automatic_additional_episodes': False}


def source_hashes():
    paths = [ROOT / name for name in ['config.json', 'CASES.json', 'PLAN.md', 'SOURCE_PROVENANCE.json']]
    paths += [*ROOT.glob('code/*.py'), *ROOT.glob('tests/*.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def operational_hashes():
    return {str(p.relative_to(ROOT)): sha256(p)
            for p in sorted([ROOT / 'run.sh', *ROOT.glob('slurm/*')]) if p.is_file()}


@lru_cache(maxsize=1)
def signature():
    require(_native.signature() == cfg()['source_parent_signature'], 'Frozen77 source/runtime changed')
    return digest({'sources': source_hashes(), 'frozen77_signature': _native.signature(),
                   'libraries': library_versions()})


def preflight():
    parent = _native.prepared()
    require(_native.signature() == cfg()['source_parent_signature'], 'Frozen77 source changed')
    status = load_json(FROZEN_ROOT / 'results/STATUS.json')
    require(status['status'] == 'COMPLETE' and status['episodes'] == 77 and status['quality_pass'],
            'Completed77 evidence is incomplete')
    require(all(sha256(FROZEN_ROOT / 'results' / p) == h for p, h in status['assets'].items()),
            'Completed77 result checksums changed')
    audit = load_json(FROZEN_ROOT / 'results/HISTORICAL_AUDIT.json')
    require(len(audit) == 20 and all(r['old_success'] == r['fresh_success'] and
            r['old_steps'] == r['fresh_steps'] and r['aligned_predicted_action_max_error'] == 0 for r in audit),
            'Prerequisite control comparison no longer supports historical reuse')
    inputs = {**load_json(ROOT / 'SOURCE_PROVENANCE.json')['frozen_source_and_result_files'],
              **load_json(ROOT / 'CASES.json')['frozen_inputs']}
    for patch, path in parent['patch_paths'].items():
        inputs[path] = sha256(path)
    require(all(sha256(p) == h for p, h in inputs.items()), 'Frozen source/reference checksum changed')
    for case in cases():
        for name, ref in case['references'].items():
            receipt = load_json(ref['receipt'])
            row = receipt['row']
            require(receipt['status'] == 'COMPLETE' and receipt['signature'] == ref['signature'] and
                    row['success'] == ref['expected_success'], 'Reference outcome changed: ' + name)
            require(row['rollout_seed'] == case['rollout_seed'] and row['state_index'] == case['state_index'] and
                    row['suite'] == case['suite'] and row['task_id'] == case['task_id'], 'Reference pairing mismatch')
    rates, overhead = [], []
    for case in cases():
        reference = load_json(case['references']['functional_injection']['receipt'])['row']
        rates.append(reference['elapsed_s'] / reference['decision_blocks'])
        folder = FROZEN_ROOT / 'episodes' / case['id']
        worker = load_json(folder / 'STATUS.json')
        total = sum(load_json(p)['row']['elapsed_s'] for p in folder.glob('*/COMPLETE.json'))
        overhead.append(max(0., worker['elapsed_s'] - total))
    maximum_blocks = 65  # unchanged LIBERO Long limit 520 / chunk size 8
    return {'status': 'CPU_PREFLIGHT_PASS', 'signature': signature(), 'budget': budget(),
            'sources': source_hashes(), 'operational_hashes': operational_hashes(),
            'frozen_inputs': inputs, 'patch_paths': parent['patch_paths'], 'library_versions': library_versions(),
            'models_loaded': False, 'cuda_initialized': False, 'slurm_submitted': False,
            'runtime_validation': 'PENDING_USER_SUBMISSION', 'created_unix': time.time(),
            'timing_estimate': {'reference_s_per_block_max': max(rates),
                'reference_worker_overhead_s_max': max(overhead),
                'estimated_longest_worker_s_with_30pct_margin': 1.3 * (maximum_blocks * max(rates) + max(overhead)),
                'four_gpu_budget_minutes_excluding_queue': cfg()['time_budget_minutes_excluding_queue']}}


def prepared():
    path = ROOT / 'reports/PREPARATION.json'
    require(load_json(ROOT / 'reports/CPU_PREFLIGHT.json')['status'] == 'CPU_PREFLIGHT_PASS', 'Latest CPU preflight failed')
    failure = ROOT / 'reports/PREPARE_FAILED.json'
    require(not failure.exists() or failure.stat().st_mtime_ns <= path.stat().st_mtime_ns, 'Unresolved preparation failure')
    value = load_json(path)
    require(value['signature'] == signature() and value['library_versions'] == library_versions(), 'Source/runtime changed')
    require(all(sha256(p) == h for p, h in value['frozen_inputs'].items()), 'Frozen reference changed')
    _native.prepared()
    return value

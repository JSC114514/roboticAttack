"""CPU provenance checks; no simulator imports or policy construction."""
import json
import math
from pathlib import Path
import struct
from jr_common import ROOT, PROJECT, config, read, sha, atomic, finish, complete, require, signature
from jr_assets import inventory, confirmation_conditions
from jr_loading import code_compatibility


def validate_statistics(stats, suites):
    for suite in suites:
        key = suite + '_no_noops'; require(key in stats, 'Missing joint normalization: ' + key)
        for kind, width in [('action', 7), ('proprio', 8)]:
            v = stats[key][kind]
            for n in ('q01', 'q99'):
                require(len(v[n]) == width and all(math.isfinite(float(x)) for x in v[n]), 'Invalid ' + key + '/' + kind + '/' + n)
            require(all(float(a) <= float(b) for a, b in zip(v['q01'], v['q99'])), 'Reversed normalization interval')
        require(stats[key]['action'].get('mask', [True] * 7) == [True] * 6 + [False], 'Native gripper normalization mask changed')


def validate_safetensor(path):
    # Structural verification without torch/model allocation or a second 15GB read.
    with path.open('rb') as f:
        b = f.read(8); require(len(b) == 8, 'Truncated safetensor header: ' + str(path))
        length = struct.unpack('<Q', b)[0]
        require(0 < length < 32 * 1024 * 1024, 'Invalid safetensor header length')
        header = json.loads(f.read(length))
    end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
    require(8 + length + end == path.stat().st_size, 'Truncated safetensor weights: ' + str(path))


def model_proof():
    c = config(); p = Path(c['model_dir'])
    require(p.is_dir(), 'Joint model missing. Run: bash ' + str(ROOT / 'run.sh') + ' --download')
    required = ['config.json', 'dataset_statistics.json', 'model.safetensors.index.json',
                'action_head--300000_checkpoint.pt', 'proprio_projector--300000_checkpoint.pt',
                'modeling_prismatic.py', 'configuration_prismatic.py', 'preprocessor_config.json',
                'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'processing_prismatic.py',
                'tokenizer.model', 'special_tokens_map.json', 'added_tokens.json', 'generation_config.json']
    for n in required: require((p / n).is_file(), 'Joint download incomplete: ' + n)
    shards = sorted(set(read(p / 'model.safetensors.index.json')['weight_map'].values()))
    require(len(shards) == 4, 'Unexpected joint shard index')
    metadata = p / '.cache/huggingface/download/config.json.metadata'
    proof = p / 'JOINT_DOWNLOAD.json'
    verified_hashes = {}
    if proof.exists():
        v = read(proof)
        require(v['status'] == 'COMPLETE' and v['revision'] == c['revision'] and v['repo_id'] == c['repo_id'], 'Download provenance differs')
        for name, item in v['files'].items():
            verified_hashes[name] = sha(p/name)
            require((p/name).stat().st_size == item['bytes'] and verified_hashes[name] == item['sha256'], 'Downloaded model hash changed: '+name)
    else:
        require(metadata.is_file() and metadata.read_text().splitlines()[0] == c['revision'], 'Pinned HF download metadata missing')
    # Each file must belong to the pinned snapshot, including tokenizer/code files.
    for name in required + shards:
        meta = p / '.cache/huggingface/download' / (name + '.metadata')
        require(meta.is_file() and meta.read_text().splitlines()[0] == c['revision'], 'Component snapshot differs: '+name)
        etag = meta.read_text().splitlines()[1]
        if len(etag) == 64:
            if name not in verified_hashes: verified_hashes[name] = sha(p/name)
            require(verified_hashes[name] == etag, 'LFS component checksum differs: '+name)
    for n in shards:
        q = p / n; require(q.is_file(), 'Missing weight shard: ' + n); validate_safetensor(q)
    validate_statistics(read(p / 'dataset_statistics.json'), c['suites'])
    # Use the audited native class implementation, with no auto-sync disk writes.
    old = read(PROJECT / c['runtime_run'] / 'config.json')['evaluation']
    official = Path(old['paths']['openvla_oft_root'])
    code_checks = code_compatibility(p, official)
    require(read(p / 'config.json')['auto_map'] == {
        'AutoConfig': 'configuration_prismatic.OpenVLAConfig',
        'AutoModelForVision2Seq': 'modeling_prismatic.OpenVLAForActionPrediction'}, 'Joint auto_map differs')
    files = [p / n for n in required + shards]
    return dict(repo_id=c['repo_id'], revision=c['revision'], model_dir=str(p), head_step=300000,
                code_checks=code_checks, normalization_source=str(p / 'dataset_statistics.json'),
                model_file_stats={str(q): [q.stat().st_size, q.stat().st_mtime_ns] for q in files},
                small_hashes={str(q): sha(q) for q in files if q.suffix not in ('.pt', '.safetensors')},
                download_receipt=str(proof) if proof.exists() else str(metadata))


def assets(monitor):
    import torch
    c = config(); rows = inventory(); old = read(PROJECT / c['initial_manifest'])
    inputs = {str(PROJECT / c['initial_manifest']): sha(PROJECT / c['initial_manifest']),
              str(PROJECT / c['fair_manifest']): sha(PROJECT / c['fair_manifest'])}
    for r in rows:
        for n in ('path', 'receipt', 'initial_source'):
            if r.get(n): inputs[r[n]] = sha(r[n])
    cases = []; confirmation = []
    for suite_index, suite in enumerate(c['suites']):
        for task in range(10):
            entry = old['initial_states'][f'{suite}/{task}']; p = Path(entry['path'])
            states = torch.load(p, map_location='cpu', weights_only=False)
            require(len(states) > max(c['screen_states'] + c['confirmation_states']), 'Initial states missing: ' + str(p))
            inputs[str(p)] = sha(p)
            for state in c['screen_states'] + c['confirmation_states']:
                case = dict(suite=suite, suite_index=suite_index, task_id=task, state_index=state,
                            initial_file=str(p), task_name=entry['task_name'],
                            rollout_seed=c['seed_base'] + 10000 * suite_index + 100 * task + state,
                            start=f'{suite}_t{task:02d}_s{state:02d}')
                (cases if state in c['screen_states'] else confirmation).append(case)
    require(len(cases) == 40 and len(rows) == 26, 'Screen episode budget differs')
    core = confirmation_conditions(rows); require(len(core) == 14, 'Confirmation budget differs')
    value = dict(signature=signature(), conditions=rows, cases=cases, inputs=inputs,
                 total_episodes=1040, learned_patch_count=22, data_regenerated=False)
    atomic(value, ROOT / 'data/ASSETS.json')
    atomic(dict(signature=signature(), cases=confirmation, conditions=core, total_episodes=560,
                status='PREPARED_ONLY_NOT_SUBMITTED'), ROOT / 'data/CONFIRMATION_PREPARED.json')
    monitor.update(phase='assets_verified', completed=26, total=26)
    return value


def run(monitor):
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    if complete(ROOT / 'reports/PREPARE_COMPLETE.json'):
        from jr_common import verify
        return verify()
    value = assets(monitor)
    monitor.update(phase='joint_model_component_checks', completed=0, total=1)
    proof = model_proof(); value['inputs'].update(proof['small_hashes'])
    receipt = Path(proof['download_receipt']); value['inputs'][str(receipt)] = sha(receipt)
    # Retain only the relevant historical Goal records for model-to-model pairing.
    historical = PROJECT / 'research_runs/wrist_spatial_relation_fair_v1/results/STATUS.json'
    value['historical_goal_manifest'] = str(PROJECT / cpath())
    value['historical_goal_results'] = str(historical)
    value['inputs'][str(historical)] = sha(historical)
    value.update(model=proof, checkpoint=proof['model_dir'], model_file_stats=proof['model_file_stats'])
    atomic(proof, ROOT / 'reports/MODEL_PROVENANCE.json')
    atomic(value, ROOT / 'data/MANIFEST.json')
    finish(ROOT / 'reports/PREPARE_COMPLETE.json', [ROOT / 'data/MANIFEST.json', ROOT / 'data/ASSETS.json',
           ROOT / 'data/CONFIRMATION_PREPARED.json', ROOT / 'reports/MODEL_PROVENANCE.json'])
    monitor.update(phase='READY', completed=1, total=1)
    return value


def cpath(): return config()['fair_manifest']

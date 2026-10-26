"""Pin existing state11 clean trajectories; derive onsets without attack responses."""
import copy
from pathlib import Path
from dp_common import (ROOT, PROJECT, SCREEN, SCREEN_SIGNATURE, config, read, sha, atomic,
                       require, signature, complete, finish, verify, source_files)
from dp_math import conditions, contact_onset


def run(monitor):
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):
        return verify()
    from PIL import Image
    c = config()
    parent = read(SCREEN/'data/MANIFEST.json')
    proof = read(SCREEN/'reports/PREPARE_COMPLETE.json')
    require(parent['signature'] == proof['signature'] == SCREEN_SIGNATURE, 'Clean source revision differs')
    require(proof['status'] == 'COMPLETE', 'Clean source not prepared')
    require(proof['artifacts']['data/MANIFEST.json'] == sha(SCREEN/'data/MANIFEST.json'), 'Clean manifest changed')
    require(parent['checkpoint'] == c['model_dir'] and parent['model']['revision'] == c['revision'], 'Joint model differs')
    require(c['state_index'] == 11 and c['durations'] == [8, 32] and c['chunk'] == 8, 'Fixed design changed')
    inputs = {str(SCREEN/'data/MANIFEST.json'): sha(SCREEN/'data/MANIFEST.json')}
    inputs.update(parent['model']['small_hashes'])
    inputs[parent['model']['download_receipt']] = sha(parent['model']['download_receipt'])
    cases = []
    required = ('START.npz', 'START.json', 'FIRST_ACTION.npy', 'INTERFACE_CHECK.json',
                'EPISODE.json', 'TRACE.parquet', 'STEPS.parquet')
    for template in parent['cases']:
        if template['state_index'] != 11 or template['task_id'] not in c['tasks'][template['suite']]:
            continue
        case = copy.deepcopy(template)
        files = source_files(case)
        require(files is not None and all(n in files for n in required), 'Incomplete clean source: '+case['start'])
        meta = read(files['EPISODE.json'])
        for key in ('suite', 'task_id', 'state_index', 'rollout_seed', 'start', 'task_name'):
            require(meta[key] == case[key], 'Clean identity mismatch: '+key)
        require(meta['checkpoint'] == c['model_dir'] and meta['revision'] == c['revision'] and
                meta['condition'] == 'clean' and meta['num_open_loop_steps'] == c['chunk'] and
                meta['max_steps'] == c['max_steps'][case['suite']], 'Clean deployment/horizon differs')
        require(case['rollout_seed'] == c['seed_base']+10000*case['suite_index']+100*case['task_id']+11,
                'Clean seed differs')
        gate = read(files['INTERFACE_CHECK.json'])
        require(gate['status'] == 'PASS' and gate['passive_sampler_native_reset_identical'], 'Clean reset gate absent')
        onset, eligibility = contact_onset(meta['first_target_contact_step'], c['chunk'])
        case.update(onset=onset, eligibility=eligibility, clean_success=meta['success'],
                    clean_executed_steps=meta['executed_steps'], first_clean_contact_step=meta['first_target_contact_step'],
                    reference_files={n: str(files[n]) for n in required})
        require(onset is None or onset < meta['executed_steps'], 'Onset after clean termination')
        for n in required:
            inputs[str(files[n])] = sha(files[n])
        for path in (files['SOURCE_COMPLETE.json'], Path(case['initial_file'])):
            inputs[str(path)] = sha(path)
        cases.append(case)
    require(len(cases) == 16 and len(conditions(c)) == 8, 'Fixed task/condition coverage differs')
    p = PROJECT/c['patch']['path']
    require(p.is_file() and sha(p) == c['patch']['sha256'], 'Fixed historical PNG differs')
    with Image.open(p) as image:
        require(image.size == (50, 50) and image.mode == 'RGB', 'PNG geometry/mode differs')
    inputs[str(p)] = sha(p)
    for p, expected in parent['model_file_stats'].items():
        s = Path(p).stat()
        require([s.st_size, s.st_mtime_ns] == expected, 'Model file changed: '+p)
    value = dict(signature=signature(), cases=cases, conditions=conditions(c), inputs=inputs,
                 checkpoint=c['model_dir'], model=parent['model'], model_file_stats=parent['model_file_stats'],
                 expected=128, runnable=sum(x['eligibility']=='ELIGIBLE' for x in cases)*8,
                 source_signature=SCREEN_SIGNATURE, training=False, source_root=str(SCREEN))
    atomic(value, ROOT/'data/MANIFEST.json')
    atomic(value, ROOT/'data/ASSETS.json')
    atomic([dict(start=x['start'], suite=x['suite'], task_id=x['task_id'], onset=x['onset'],
                 first_clean_contact_step=x['first_clean_contact_step'], eligibility=x['eligibility']) for x in cases],
           ROOT/'data/SAMPLING.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json', [ROOT/'data'/n for n in ('MANIFEST.json','ASSETS.json','SAMPLING.json')])
    monitor.update(phase='READY', completed=16, total=16, runnable=value['runnable'], training=False)
    return value

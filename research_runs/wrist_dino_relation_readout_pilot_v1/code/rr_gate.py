"""Fixed Discovery-only directional test; never selects a new target."""
import numpy as np
import torch
from rr_common import (ROOT, PILOT, Pool, target, config, complete, finish, atomic_json,
                       make_engine, native, require)
from rr_math import Readout, gate_decision


def run(monitor):
    destination = ROOT/'reports/DIRECTION_COMPLETE.json'
    if complete(destination): return
    pool = Pool('fit')
    rows = [r for r in pool if r['frame_rank'] == config()['gate_frame_rank']]
    require(len(rows) == 12 and len({r['episode_id'] for r in rows}) == 12, 'Gate selection changed')
    native.require_gpu('relation_direction')
    engine = make_engine()
    decoder = Readout(PILOT/'readouts/b22.npz')
    engine.verify_patch_path(rows[0])
    results = []
    for i, row in enumerate(rows):
        path = ROOT/f'data/directions/{row["state_id"]}.json'
        receipt = path.with_name(path.stem+'_COMPLETE.json')
        if complete(receipt):
            from rr_common import read_json
            results.append(read_json(path)); continue
        item = {k:row[k] for k in ('state_id','episode_id','task_id','state_index','frame_rank')}
        delta = target(row)
        if delta is None:
            item.update(applicable=False, reason='no_geometry_target')
        else:
            action, clean = engine.forward(row['observation'],row['language'])
            identity, untouched = engine.forward(row['observation'],row['language'],{'dino':clean['dino']})
            require(np.array_equal(action,identity), 'Zero replacement changed native action')
            clean_feature = torch.from_numpy(clean['dino']).double()
            clean_readout = decoder(clean_feature)
            truth_relative=np.asarray(row['geometry']['target_xyz'])-np.asarray(row['geometry']['eef_xyz'])
            rendered_relation={sign:(decoder(torch.as_tensor(row['target_features']['translation_'+sign]))-clean_readout)[0].tolist()
                               for sign in ('plus','minus')}
            plus, _ = engine.forward(row['observation'],row['language'],{'dino':row['target_features']['translation_plus']})
            minus, _ = engine.forward(row['observation'],row['language'],{'dino':row['target_features']['translation_minus']})
            reference = plus[0,:3]-minus[0,:3]
            norm = float(np.linalg.norm(reference))
            direction = decoder.intervention(delta)
            directions = [('relation',direction)] + [
                (f'random_{j}', decoder.random_null(direction.norm(),config()['seed']+1000*i+j))
                for j in range(config()['random_directions'])]
            interventions = {}
            for name, vector in directions:
                outputs = []
                for sign in (1,-1):
                    donor = clean_feature + sign*vector
                    changed, captures = engine.forward(row['observation'],row['language'],{'dino':donor})
                    require(np.array_equal(captures['siglip'],clean['siglip']), 'DINO-only control changed SigLIP')
                    actual = torch.from_numpy(captures['dino']).double()
                    movement = decoder(actual)-clean_readout
                    continuous = decoder.pool(actual,rounded=False)-decoder.pool(clean_feature,rounded=False)
                    actual_norm = float((actual-clean_feature).norm())
                    outputs.append({'sign':sign,'actions':changed.tolist(),
                        'readout_delta_m':movement[0].tolist(),
                        'continuous_readout_delta_m':(continuous@decoder.weight)[0].tolist(),
                        'actual_feature_norm':actual_norm,'requested_feature_norm':float(vector.norm()),
                        'feature_norm_ratio':actual_norm/float(vector.norm()),
                        'relation_target_error_m':float((movement-torch.as_tensor(sign*delta)).norm()) if name=='relation' else None,
                        'null_readout_leak_m':float(movement.norm()) if name!='relation' else None})
                response=np.asarray(outputs[0]['actions'])[0,:3]-np.asarray(outputs[1]['actions'])[0,:3]
                interventions[name]={'score':float(response@reference/norm) if norm>1e-12 else None,
                                     'positive_negative_first_delta':response.tolist(),'outputs':outputs}
            item.update(applicable=norm>1e-12, reason=None if norm>1e-12 else 'zero_rendered_reference_action',
                actual_delta_m=delta.tolist(),clean_actions=action.tolist(),
                clean_readout_m=clean_readout[0].tolist(),rendered_plus_actions=plus.tolist(),
                clean_relative_truth_m=truth_relative.tolist(),
                clean_readout_error_cm=float(np.linalg.norm(clean_readout[0].numpy()-truth_relative)*100),
                rendered_relation_deltas_m=rendered_relation,
                rendered_plus_relation_error_cm=float(np.linalg.norm(np.asarray(rendered_relation['plus'])-delta)*100),
                rendered_minus_actions=minus.tolist(),rendered_reference_first_norm=norm,
                relation_score=interventions['relation']['score'],
                random_score=float(np.mean([interventions[f'random_{j}']['score'] for j in range(4)])) if norm>1e-12 else None,
                interventions=interventions,
                full_chunk_retained=True,spatial_only=True,cls_register_unmodified=True)
        atomic_json(item,path); finish(receipt,[path]);results.append(item)
        monitor.update(phase='relation_direction',completed=i+1,total=12,state_id=row['state_id'])
    decision=gate_decision(results)
    output=ROOT/'results/DIRECTION.json'
    atomic_json({'fixed_samples':12,'seed':42,'random_directions':4,'rows':results,**decision,
        'meaning':'exploratory_budget_gate_not_significance_or_unique_causal_semantics'},output)
    finish(destination,[output],**decision,**native.resources())
    monitor.update(phase='direction_pass' if decision['passed'] else 'direction_not_supported',**decision)

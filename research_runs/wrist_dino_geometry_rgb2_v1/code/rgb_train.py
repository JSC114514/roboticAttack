"""Same two feature losses and a single shared RGB patch, no action objective."""
import json
import random
import time
import numpy as np
import torch
from rgb_common import (ROOT, MATCH, config, native, module, require, signature, read_json,
                        atomic_torch, complete, finish, make_engine)
from rgb_cache import records

signed_projection = module('rgb_loss_reference', MATCH / 'code/engine.py').signed_projection

def objective(patched, minus, plus, kind):
    result = signed_projection(patched, minus, plus)
    if result is None: return None
    projection, q, orthogonal, energy = result
    require(kind in ('projection','full'), 'Unknown feature loss')
    loss = projection if kind == 'projection' else projection + orthogonal
    return loss, {'loss': float(loss.detach()), 'q': float(q.detach()),
                  'projection_loss': float(projection.detach()), 'orthogonal': float(orthogonal.detach()),
                  'target_energy': energy}

def update(engine, patch, optimizer, row, kind, initial):
    optimizer.zero_grad(set_to_none=True)
    before = patch.detach().clone()
    feature, _ = engine.spatial_patch(row['observation'], patch)
    target = row['target_features']
    result = objective(feature, target['translation_minus'], target['translation_plus'], kind)
    if result is None:
        return {'optimized': False, 'no_target': True, 'gradient_l2': 0., 'loss': None, 'q': None,
                'projection_loss': None, 'orthogonal': None, 'pixel_step_l2': 0.}
    loss, values = result
    require(torch.isfinite(loss), 'Nonfinite feature loss')
    loss.backward()
    require(patch.grad is not None and torch.isfinite(patch.grad).all(), 'Invalid patch gradient')
    gradient = float(patch.grad.norm())
    optimizer.step()
    with torch.no_grad(): patch.clamp_(0,1)
    require(all(p.grad is None for p in engine.wrapper.vla.parameters()), 'Victim accumulated gradients')
    return {**values, 'optimized': True, 'no_target': False, 'gradient_l2': gradient,
            'pixel_step_l2': float((patch.detach()-before).norm()),
            'pixel_from_initial_l2': float((patch.detach()-initial).norm())}

def run(kind, monitor):
    native.require_gpu('rgb2_train')
    folder = ROOT / 'training' / kind
    if complete(folder / 'COMPLETE.json'): return
    rows = list(records('fit'))
    require(len(rows) == 240, 'Fit pool changed')
    engine = make_engine()
    engine.verify_patch_path(rows[0])
    initial = native.load_png(ROOT / 'data/initial.png', engine.wrapper.device)
    patch = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([patch], lr=config()['learning_rate'])
    start = optimized = no_target = 0
    last = folder / 'LAST.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        require(saved['signature'] == signature() and saved['kind'] == kind, 'Resume identity mismatch')
        with torch.no_grad(): patch.copy_(saved['patch'].to(patch.device))
        optimizer.load_state_dict(saved['optimizer'])
        start, optimized, no_target = saved['visit'], saved['optimized'], saved['no_target']
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f'ATTEMPT_{time.time_ns()}.jsonl'
    monitor.update(phase='shared_'+kind, completed=start*240, resume_completed=start*240, total=4800)
    with log.open('x') as stream:
        for visit in range(start, config()['visits']):
            order = list(rows)
            random.Random(config()['seed']+visit).shuffle(order)
            for n, row in enumerate(order):
                values = update(engine, patch, optimizer, row, kind, initial)
                optimized += int(values['optimized']); no_target += int(values['no_target'])
                step = visit*240+n+1
                stream.write(json.dumps({'step': step, 'state_id': row['state_id'], **values})+'\n')
                if step % 25 == 0: monitor.update(phase='shared_'+kind, completed=step, total=4800, **values)
            atomic_torch({'signature': signature(), 'kind': kind, 'visit': visit+1,
                          'patch': patch.detach().cpu(), 'optimizer': optimizer.state_dict(),
                          'optimized': optimized, 'no_target': no_target}, last)
    png = folder / 'final.png'
    native.save_png(patch, png)
    finish(folder / 'SHARED_COMPLETE.json', [png, last, *folder.glob('ATTEMPT_*.jsonl')],
           updates=4800, optimized=optimized, no_target=no_target, frames=240, visits=20)
    # Diagnostic individual patches share neither optimizer nor pixels. Each gets
    # 100 updates vs 20 visits/image for the shared patch; budgets are explicit.
    ids = read_json(ROOT / 'data/MANIFEST.json')['individual_ids']
    monitor.update(phase='individual_'+kind,completed=0,resume_completed=0,total=len(ids)*100)
    artifacts = [folder / 'SHARED_COMPLETE.json']
    for n, row in enumerate(r for r in rows if r['state_id'] in ids):
        part = folder / 'individual' / row['state_id']
        receipt = part / 'COMPLETE.json'
        if not complete(receipt):
            current = torch.nn.Parameter(initial.clone())
            opt = torch.optim.Adam([current], lr=config()['learning_rate'])
            count_optimized = count_no_target = 0
            for step in range(config()['individual_updates']):
                values = update(engine, current, opt, row, kind, initial)
                count_optimized += int(values['optimized']); count_no_target += int(values['no_target'])
                if step % 25 == 0: monitor.update(phase='individual_'+kind, state_id=row['state_id'],
                    completed=n*100+step, total=len(ids)*100, **values)
            image = part / 'final.png'
            native.save_png(current, image)
            finish(receipt, [image], state_id=row['state_id'], updates=100, cohort='fit_only',
                   optimized=count_optimized,no_target=count_no_target)
        artifacts.append(receipt)
    finish(folder / 'COMPLETE.json', artifacts, kind=kind, seed=42, shared_updates=4800,
           individual_updates=1200, **native.resources())

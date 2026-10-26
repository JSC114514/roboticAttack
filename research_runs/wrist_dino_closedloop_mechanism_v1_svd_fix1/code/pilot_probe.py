"""Read-only native OFT forwards, endpoint VJPs and same-state interventions."""
from __future__ import annotations

import time
import numpy as np
import torch
from pilot_common import cfg, digest, parent_module, require
from pilot_math import (GROUPS, action_stats, apply_delta, basis, cosine, feature_stats,
                        finite_difference_result, footprint, group_readouts, matched_change,
                        norm, project, random_basis, random_mask, spatial_mask)


def seeded(*values):
    return int(digest([cfg()['random_seed'], *values])[:15], 16)


class Probe:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.engine = parent_module('engine').OFTEngine(wrapper)
        self.paste = parent_module('deployment').paste_png

    def environment_actions(self, action):
        return np.asarray(self.wrapper.vla._unnormalize_actions(
            action.detach().float().cpu().numpy()[0], self.wrapper.gen_cfg.unnorm_key), dtype=np.float32)

    def pair(self, observation, instruction, rgb):
        clean_record = {'observation': observation, 'task_name': instruction}
        patched_observation = self.paste(observation, rgb)
        patched_record = {'observation': patched_observation, 'task_name': instruction}
        with torch.no_grad():
            clean_inputs, proprio = self.engine.base(clean_record)
            patched_inputs, patched_proprio = self.engine.base(patched_record)
            require(np.array_equal(proprio, patched_proprio), 'Proprio changed during visual intervention')
            clean_action, clean, _ = self.engine.forward(clean_inputs, proprio)
            patched_action, patched, _ = self.engine.forward(patched_inputs, proprio)
        require(torch.equal(clean_inputs['input_ids'], patched_inputs['input_ids']), 'Instruction changed')
        support = footprint(clean_inputs['pixel_values'], patched_inputs['pixel_values'])
        return {'clean_inputs': clean_inputs, 'patched_inputs': patched_inputs, 'proprio': proprio,
                'clean_action': clean_action.detach(), 'patched_action': patched_action.detach(),
                'clean_dino': clean['dino'].detach(), 'patched_dino': patched['dino'].detach(),
                'clean_siglip': clean['siglip'].detach(), 'patched_siglip': patched['siglip'].detach(),
                'support': support, 'observation': observation, 'patched_observation': patched_observation,
                'instruction': instruction}

    def intervention(self, pair, replacement, endpoint='patched'):
        with torch.no_grad():
            action, captures, _ = self.engine.forward(pair[endpoint + '_inputs'], pair['proprio'], replacement=replacement)
        require(torch.equal(captures['siglip'], pair[endpoint + '_siglip']), 'DINO hook altered SigLIP output')
        require(torch.equal(captures['dino'], replacement.to(captures['dino'].dtype)), 'Replacement did not reach consumed DINO tokens')
        return action.detach()

    def validate(self, pair):
        errors = {}
        for endpoint in ['clean', 'patched']:
            native = np.asarray(self.wrapper.predict_action(pair[endpoint + '_observation'] if endpoint == 'patched'
                                                          else pair['observation'], pair['instruction']), dtype=np.float32).reshape(8, 7)
            engine_action = self.environment_actions(pair[endpoint + '_action'])
            error = float(np.max(np.abs(native - engine_action)))
            require(error <= cfg()['native_action_atol'], 'Native/engine action mismatch: ' + str(error))
            same = self.intervention(pair, pair[endpoint + '_dino'], endpoint)
            identity_error = float((same - pair[endpoint + '_action']).abs().max())
            require(identity_error <= cfg()['native_action_atol'], 'Identity replacement mismatch')
            errors[endpoint + '_native_max_error'] = error
            errors[endpoint + '_identity_max_error'] = identity_error
        require(self.engine.dino.num_prefix_tokens == 5 and pair['clean_dino'].shape[:2] == (1, 256),
                'CLS/register exclusion contract changed')
        errors.update(status='NATIVE_AND_IDENTITY_PASS', consumed_dense_tokens=256, prefix_tokens_excluded=5)
        return errors

    def functional(self, pair, finite=False, on_phase=None):
        readouts, fallbacks = group_readouts(pair['clean_action'], pair['patched_action'])
        vectors = {name: [] for name in GROUPS}
        checks, gradients, vjp_seconds = [], {}, 0.
        for endpoint in ['clean', 'patched']:
            if on_phase:
                on_phase('endpoint_vjp_' + endpoint)
            vjp_started = time.monotonic()
            with torch.enable_grad():
                action, captures, _ = self.engine.forward(pair[endpoint + '_inputs'], pair['proprio'], leaf=True)
                for index, (name, readout) in enumerate(readouts.items()):
                    loss = (action * readout).sum()
                    gradient = torch.autograd.grad(loss, captures['dino'], retain_graph=index < 2)[0].detach().float()
                    require(torch.isfinite(gradient).all(), 'Nonfinite endpoint VJP')
                    vectors[name].append(gradient)
                    gradients[endpoint + '_' + name] = gradient.cpu()
            del action, captures, loss
            torch.cuda.synchronize() if pair['clean_dino'].is_cuda else None
            vjp_seconds += time.monotonic() - vjp_started
            if finite:
                base = pair[endpoint + '_dino']
                for name, readout in readouts.items():
                    gradient = vectors[name][-1]
                    for relative in cfg()['finite_difference_relative_steps']:
                        if norm(gradient) < 1e-12:
                            checks.append({'endpoint': endpoint, 'group': name, 'relative_step': relative,
                                           'relative_error': None, 'sign_agrees': False, 'nonzero': False,
                                           'predicted_derivative': 0., 'measured_derivative': 0.})
                            continue
                        epsilon = max(norm(base) * relative, 1e-4)
                        direction = gradient / gradient.norm()
                        plus, minus = apply_delta(base, direction, epsilon), apply_delta(base, direction, -epsilon)
                        plus_action = self.intervention(pair, plus, endpoint)
                        minus_action = self.intervention(pair, minus, endpoint)
                        predicted = float((gradient * (plus.float() - minus.float())).sum()) / (2 * epsilon)
                        measured = float(((plus_action - minus_action) * readout).sum()) / (2 * epsilon)
                        checks.append({'endpoint': endpoint, 'group': name, 'relative_step': relative,
                                       'actual_central_feature_l2': norm(plus.float() - minus.float()),
                                       **finite_difference_result(predicted, measured)})
        rows = {name: basis(values) for name, values in vectors.items()}
        rows['all'] = basis([value for values in vectors.values() for value in values])
        summary = {'rank_' + name: len(value) for name, value in rows.items()}
        summary['endpoint_vjp_elapsed_s'] = vjp_seconds
        for name, values in vectors.items():
            summary[name + '_clean_vjp_l2'] = norm(values[0])
            summary[name + '_patched_vjp_l2'] = norm(values[1])
            summary[name + '_endpoint_vjp_cosine'] = cosine(*values)
            summary[name + '_readout_zero_gap_fallback'] = fallbacks[name]
        return rows, checks, summary, gradients

    def replacement(self, pair, family, group, strength, seed, rows=None, random_control=False, orthogonal=False):
        clean, patched = pair['clean_dino'], pair['patched_dino']
        delta = clean.float() - patched.float()
        stats = {'family': family, 'group': group, 'strength': strength, 'random_control': random_control,
                 'control_seed': seed, 'orthogonal_complement': orthogonal}
        if family == 'spatial':
            mask = spatial_mask(group, pair['support'])
            selected = delta * mask[None, :, None]
            target = apply_delta(patched, selected, strength)
            stats.update(selected_token_count=int(mask.sum()), control_rank=int(mask.sum()) * delta.shape[-1])
            if random_control:
                control_mask = random_mask(int(mask.sum()), seed, delta.device)
                change = delta * control_mask[None, :, None]
                replacement, matching = matched_change(patched, change, norm(target.float() - patched.float()),
                                                       cfg()['norm_match_relative_tolerance'])
                stats.update(matching, control_token_count=int(control_mask.sum()),
                             control_mask_overlap=int((mask & control_mask).sum()))
            else:
                replacement = target
        else:
            require(family == 'functional' and rows is not None, 'Functional basis missing')
            selected = project(delta, rows)
            if orthogonal:
                selected = delta - selected
            target = apply_delta(patched, selected, strength)
            stats.update(control_rank=len(rows), sensitive_component_l2=norm(project(delta, rows)),
                         orthogonal_component_l2=norm(delta - project(delta, rows)),
                         basis_orthogonality_max_error=0. if len(rows) == 0 else
                             float((rows @ rows.T - torch.eye(len(rows), device=rows.device)).abs().max()))
            if random_control:
                control = random_basis(len(rows), delta.numel(), seed, delta.device)
                change = project(delta, control)
                replacement, matching = matched_change(patched, change, norm(target.float() - patched.float()),
                                                       cfg()['norm_match_relative_tolerance'])
                stats.update(matching, natural_random_projection_l2=norm(change))
            else:
                replacement = target
        stats.update(feature_stats(clean, patched, replacement))
        require(torch.isfinite(replacement).all(), 'Nonfinite intervention feature')
        return replacement, stats

    def screen(self, capture, rgb, context, on_phase=None):
        pair = self.pair(capture['observation'], capture['task_name'], rgb)
        validation = self.validate(pair)
        require(np.max(np.abs(self.environment_actions(pair['patched_action']) - capture['formal_action_chunk']))
                <= cfg()['native_action_atol'], 'Captured action does not reproduce at same observation')
        rows, checks, gradient_summary, gradients = self.functional(pair, finite=True, on_phase=on_phase)
        records = []

        def record(name, replacement, stats, endpoint='patched'):
            action = self.intervention(pair, replacement, endpoint)
            records.append({**context, 'candidate': name, 'endpoint': endpoint, **stats,
                            **action_stats(pair['clean_action'], pair['patched_action'], action),
                            'normalized_action': action.cpu().tolist(),
                            'environment_action': self.environment_actions(action).tolist()})

        delta = pair['clean_dino'].float() - pair['patched_dino'].float()
        for strength in cfg()['strengths']:
            replacement = apply_delta(pair['patched_dino'], delta, strength)
            record('whole_restore', replacement, {'family': 'whole', 'group': 'all', 'strength': strength,
                   'random_control': False, **feature_stats(pair['clean_dino'], pair['patched_dino'], replacement)})
        record('whole_transplant', pair['patched_dino'], {'family': 'whole', 'group': 'all', 'strength': 1.,
                'random_control': False}, endpoint='clean')
        for family, groups in [('spatial', cfg()['spatial_candidates']), ('functional', cfg()['functional_candidates'])]:
            for group in groups:
                for strength, random_control in [(1., False), (1., True), (.5, False)]:
                    seed = seeded(context, family, group)
                    replacement, stats = self.replacement(pair, family, group, strength, seed,
                                                         rows=rows.get(group), random_control=random_control)
                    record(family + ':' + group, replacement, stats)
        replacement, stats = self.replacement(pair, 'functional', 'all', 1., seeded(context, 'orthogonal'),
                                             rows=rows['all'], orthogonal=True)
        record('functional_orthogonal', replacement, stats)

        # Real RGB controls use exactly the native preprocessing, never the STE.
        for name in ['left_half', 'top_half', 'half_alpha']:
            observation = {**capture['observation'], 'wrist_image': capture['observation']['wrist_image'].copy()}
            x, y = 26, 137
            region = observation['wrist_image'][y:y+50, x:x+50]
            if name == 'left_half':
                region[:, :25] = rgb[:, :25]
            elif name == 'top_half':
                region[:25] = rgb[:25]
            else:
                region[:] = np.rint(.5 * region.astype(np.float32) + .5 * rgb).astype(np.uint8)
            with torch.no_grad():
                inputs, proprio = self.engine.base({'observation': observation, 'task_name': capture['task_name']})
                action, feature, _ = self.engine.forward(inputs, proprio)
            records.append({**context, 'candidate': 'real_rgb:' + name, 'family': 'real_rgb', 'group': name,
                            'strength': .5, 'random_control': False,
                            'dino_gap_from_clean_l2': norm(feature['dino'].float() - pair['clean_dino'].float()),
                            'siglip_gap_from_clean_l2': norm(feature['siglip'].float() - pair['clean_siglip'].float()),
                            **action_stats(pair['clean_action'], pair['patched_action'], action),
                            'normalized_action': action.cpu().tolist(),
                            'environment_action': self.environment_actions(action).tolist()})
        bundle = {'context': context, 'observation': capture['observation'], 'task_name': capture['task_name'],
                  'clean_dino': pair['clean_dino'].cpu(), 'patched_dino': pair['patched_dino'].cpu(),
                  'clean_siglip': pair['clean_siglip'].cpu(), 'patched_siglip': pair['patched_siglip'].cpu(),
                  'clean_action': pair['clean_action'].cpu(), 'patched_action': pair['patched_action'].cpu(),
                  'basis': {name: value.cpu() for name, value in rows.items()}, 'endpoint_vjps': gradients,
                  'support': pair['support'].cpu(), 'finite_differences': checks, 'validation': validation,
                  'gradient_summary': gradient_summary}
        return records, checks, gradient_summary, validation, bundle

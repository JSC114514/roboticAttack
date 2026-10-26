"""The only new intervention: unscaled orthogonal complement at original amplitude."""
from residual4_common import frozen_module, require

_math = frozen_module('functional77_math')
norm, project, apply_delta = _math.norm, _math.project, _math.apply_delta
directional_action_stats, group_readouts = _math.directional_action_stats, _math.group_readouts


def component_replacement(clean, patched, rows, component, direction, seed, tolerance=.03):
    import torch
    require(component == 'residual_full' and direction == 'inject', 'Only one new condition is authorized')
    delta = patched.detach().float() - clean.detach().float()
    selected = project(delta, rows)
    residual = delta - selected
    replacement = apply_delta(clean, residual)  # Deliberately no matched_change / gain adjustment.
    actual = replacement.float() - clean.float()
    target_norm, actual_norm = norm(residual), norm(actual)
    cast_error = abs(actual_norm - target_norm) / max(target_norm, 1e-12)
    stats = {'component': component, 'direction': direction, 'recipient_endpoint': 'clean',
        'amplitude_policy': 'unscaled_original_complement', 'norm_gain': 1., 'control_seed': seed,
        'functional_rank': len(rows), 'component_rank': delta.numel() - len(rows),
        'feature_dimension': delta.numel(), 'dino_delta_l2': norm(delta),
        'sensitive_component_l2': norm(selected), 'orthogonal_component_l2': target_norm,
        'natural_component_l2': target_norm, 'requested_change_l2': target_norm,
        'target_change_l2': target_norm, 'actual_change_l2': actual_norm,
        'sensitive_fraction_of_delta': norm(selected) / max(norm(delta), 1e-12),
        'actual_change_fraction_of_delta': actual_norm / max(norm(delta), 1e-12),
        'norm_match_ok': cast_error <= tolerance, 'norm_match_relative_error': cast_error,
        'norm_reference': 'unscaled_FP32_residual_not_small_functional_change',
        'control_degenerate': target_norm <= 1e-12,
        'actual_change_cosine_with_intended_direction': _math.cosine(actual, residual),
        'actual_change_functional_projection_l2': norm(project(actual, rows)),
        'actual_change_outside_functional_l2': norm(actual - project(actual, rows)),
        'basis_orthogonality_max_error': 0. if len(rows) == 0 else
            float((rows @ rows.T - torch.eye(len(rows), device=rows.device)).abs().max())}
    require(torch.isfinite(replacement).all(), 'Nonfinite residual replacement')
    require(stats['norm_match_ok'], 'Original residual amplitude lost during cast')
    return replacement, stats

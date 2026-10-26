"""Feature-space operations with explicit post-cast intervention magnitudes."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from pilot_common import require

GROUPS = {'translation': slice(0, 3), 'rotation': slice(3, 6), 'gripper': slice(6, 7)}


def norm(value):
    return float(value.detach().float().norm())


def cosine(first, second):
    a, b = first.detach().float().flatten(), second.detach().float().flatten()
    denominator = float(a.norm() * b.norm())
    return None if denominator < 1e-12 else float(torch.dot(a, b) / denominator)


def basis(vectors, relative_cutoff=1e-5):
    """Direct CPU FP64 SVD of <=6 VJPs, avoiding FP32 Gram whitening.

    Clean/patch endpoint VJPs may be almost parallel. Forming M M^T and
    dividing by its small eigenvalues amplified FP32 roundoff in the old
    implementation. Keep the same squared-singular-value rank threshold;
    return FP32 rows on the original device and retain the orthogonality gate.
    Only this small numerical decomposition runs on CPU; policy execution
    and the simulated rollouts still require an allocated Slurm GPU.
    """
    require(len(vectors) > 0 and 0 < relative_cutoff < 1, 'Invalid sensitivity basis request')
    device, dimension = vectors[0].device, vectors[0].numel()
    matrix = torch.stack([v.detach().to(device='cpu', dtype=torch.float64).flatten() for v in vectors])
    require(torch.isfinite(matrix).all(), 'Nonfinite sensitivity VJP')
    # Normalize first so a large physical readout does not suppress another one.
    lengths = matrix.norm(dim=1)
    matrix = matrix[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    if len(matrix) == 0:
        return torch.empty((0, dimension), dtype=torch.float32, device=device)
    _left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    squared = singular.square()
    keep = squared > max(float(squared.max()) * relative_cutoff, 1e-7)
    result = right[keep].to(dtype=torch.float32).contiguous()
    require(torch.allclose(result @ result.T, torch.eye(len(result), device=result.device), atol=2e-3),
            'Sensitivity basis lost orthogonality')
    return result.to(device=device)


def random_basis(rank, dimension, seed, device):
    require(0 <= rank <= dimension, 'Invalid control rank')
    generator = torch.Generator(device=device).manual_seed(int(seed))
    if rank == 0:
        return torch.empty((0, dimension), device=device)
    matrix = torch.randn((dimension, rank), generator=generator, device=device, dtype=torch.float32)
    return torch.linalg.qr(matrix, mode='reduced').Q.T.contiguous()


def project(delta, rows):
    require(rows.ndim == 2 and rows.shape[1] == delta.numel(), 'Basis/feature mismatch')
    vector = delta.detach().float().flatten()
    return (rows.T @ (rows @ vector)).reshape(delta.shape)


def apply_delta(base, change, strength=1.):
    return (base.detach().float() + float(strength) * change.detach().float()).to(base.dtype)


def matched_change(base, direction, target_norm, tolerance=.03):
    """Match the actual BF16 change, not the requested FP32 change."""
    length = norm(direction)
    if target_norm <= 1e-12:
        return base.detach().clone(), {'norm_match_ok': True, 'norm_gain': 0., 'norm_match_relative_error': 0.,
                                     'target_change_l2': 0., 'actual_change_l2': 0., 'control_degenerate': True}
    require(length > 1e-12, 'Nonzero target but zero random control direction')
    gain = float(target_norm) / length
    lo, hi = 0., gain * 2
    best, best_gain, best_error = None, None, float('inf')
    for _ in range(24):
        current = apply_delta(base, direction, gain)
        actual = norm(current.float() - base.float())
        error = abs(actual - target_norm)
        if error < best_error:
            best, best_gain, best_error = current, gain, error
        if error <= tolerance * target_norm:
            break
        if actual < target_norm:
            lo = gain
            if gain >= hi * .999:
                hi *= 2
        else:
            hi = gain
        gain = (lo + hi) / 2
    return best, {'norm_match_ok': best_error <= tolerance * target_norm, 'norm_gain': best_gain,
                  'norm_match_relative_error': best_error / target_norm, 'target_change_l2': target_norm,
                  'actual_change_l2': norm(best.float() - base.float()), 'control_degenerate': False}


def footprint(clean_pixels, patched_pixels):
    """Exact support in the consumed, native-preprocessed Wrist DINO RGB tensor.

    Includes the real JPEG/crop/resize effects. This is input-patch support,
    not a claim that self-attention confines a token's information to this area.
    """
    require(clean_pixels.shape == patched_pixels.shape and clean_pixels.shape[1:] == (12, 224, 224),
            'Unexpected two-view native processor layout')
    require(torch.equal(clean_pixels[:, :6], patched_pixels[:, :6]), 'Front image unexpectedly changed')
    support = (clean_pixels[:, 6:9] != patched_pixels[:, 6:9]).any(dim=1).float()[0]
    return F.max_pool2d(support[None, None], 14, stride=14).flatten().bool()


def spatial_mask(name, support):
    require(support.shape == (256,), 'Expected consumed dense-token grid 16x16')
    if name == 'footprint':
        return support.clone()
    require(name.startswith('outside_'), 'Unknown spatial mask')
    y, x = torch.meshgrid(torch.arange(16, device=support.device), torch.arange(16, device=support.device), indexing='ij')
    quadrant = name.removeprefix('outside_')
    require(quadrant in ['top_left', 'top_right', 'bottom_left', 'bottom_right'], 'Unknown quadrant')
    chosen = (y < 8 if quadrant.startswith('top') else y >= 8) & (x < 8 if quadrant.endswith('left') else x >= 8)
    return chosen.flatten() & ~support


def random_mask(count, seed, device):
    require(0 <= count <= 256, 'Invalid token count')
    generator = torch.Generator(device=device).manual_seed(int(seed))
    mask = torch.zeros(256, device=device, dtype=torch.bool)
    mask[torch.randperm(256, generator=generator, device=device)[:count]] = True
    return mask


def feature_stats(clean, patched, replacement):
    delta = patched.float() - clean.float()
    change = replacement.float() - patched.float()
    token_delta = delta.norm(dim=-1).flatten()
    return {'dino_delta_l2': norm(delta), 'actual_change_l2': norm(change),
            'change_fraction_of_delta': norm(change) / max(norm(delta), 1e-12),
            'change_cosine_with_clean_direction': cosine(change, -delta),
            'dino_delta_token_q25': float(torch.quantile(token_delta, .25)),
            'dino_delta_token_median': float(token_delta.median()),
            'dino_delta_token_q75': float(torch.quantile(token_delta, .75)),
            'remaining_dino_gap_l2': norm(replacement.float() - clean.float())}


def action_stats(clean, patched, intervened):
    gap = patched.float() - clean.float()
    difference = intervened.float() - patched.float()
    result = {'action_gap_l2': norm(gap), 'action_change_l2': norm(difference),
              'remaining_action_gap_l2': norm(intervened.float() - clean.float()),
              'action_gap_closing_fraction': None if norm(gap) < 1e-8 else
                  1 - norm(intervened.float() - clean.float()) / norm(gap),
              'action_change_cosine_toward_clean': cosine(difference, -gap)}
    for group, indices in GROUPS.items():
        result[group + '_gap_l2'] = norm(gap[..., indices])
        result[group + '_change_l2'] = norm(difference[..., indices])
        result[group + '_remaining_gap_l2'] = norm((intervened - clean)[..., indices])
    return result


def group_readouts(clean_action, patched_action):
    """One fixed readout per action group, along its observed same-state gap.

    These are six endpoint VJPs, not a full action Jacobian and not semantic
    labels for individual feature channels.
    """
    result, fallbacks = {}, {}
    for name, indices in GROUPS.items():
        values = torch.zeros_like(clean_action, dtype=torch.float32)
        values[..., indices] = (patched_action - clean_action)[..., indices].detach().float()
        fallback = norm(values) < 1e-8
        if fallback:
            values[0, 0, indices.start] = 1.
        result[name] = values / values.norm()
        fallbacks[name] = fallback
    return result, fallbacks


def finite_difference_result(predicted, measured):
    denominator = max(abs(predicted), abs(measured), 1e-8)
    return {'predicted_derivative': predicted, 'measured_derivative': measured,
            'relative_error': abs(predicted - measured) / denominator,
            'sign_agrees': bool(predicted * measured > 0),
            'nonzero': bool(abs(predicted) > 1e-8 and abs(measured) > 1e-8)}


def intervention_window(kind, c):
    if kind in ['native']:
        return (0, 0)
    if kind.startswith('whole_'):
        return (0, math.inf)
    return tuple(c['late_window'] if kind.endswith('late') else c['early_window'])

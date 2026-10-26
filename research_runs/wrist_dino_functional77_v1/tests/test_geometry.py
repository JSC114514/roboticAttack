import pytest
import torch

from functional77_math import (apply_delta, basis, component_replacement, directional_action_stats,
    matched_change, norm, project, random_basis)


@pytest.mark.parametrize('seed', [1, 17, 39])
def test_near_parallel_vjp_span_matches_independent_qr(seed):
    gen = torch.Generator().manual_seed(seed)
    core = torch.linalg.qr(torch.randn(1024, 3, generator=gen)).Q.T
    noise = torch.randn(3, 1024, generator=gen)
    noise /= noise.norm(dim=1, keepdim=True)
    matrix = torch.cat([core, core + .01 * noise])
    rows = basis(list(matrix))
    reference = torch.linalg.qr(matrix.double().T).Q
    delta = torch.linspace(-1, 1, 1024)
    assert len(rows) == 6
    assert (rows.double() @ rows.double().T - torch.eye(6)).abs().max() < 2e-6
    assert torch.allclose(project(delta, rows).double(), reference @ (reference.T @ delta.double()), atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize('epsilon,expected', [(.001, 1), (.02, 2)])
def test_squared_singular_cutoff(epsilon, expected):
    assert len(basis([torch.tensor([1., 0.]), torch.tensor([1., epsilon])])) == expected


def test_real_dimension_bf16_svd_and_no_cuda():
    gen = torch.Generator().manual_seed(21)
    core = torch.linalg.qr(torch.randn(256 * 1024, 3, generator=gen)).Q.T
    noise = torch.randn_like(core)
    noise /= noise.norm(dim=1, keepdim=True)
    rows = basis(list(torch.cat([core, core + .01 * noise]).bfloat16()))
    assert rows.shape == (6, 256 * 1024) and rows.dtype == torch.float32
    assert (rows.double() @ rows.double().T - torch.eye(6)).abs().max() < 2e-6
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('direction', ['restore', 'inject'])
@pytest.mark.parametrize('component', ['functional', 'random', 'orthogonal'])
def test_all_controls_match_recipient_specific_actual_bf16_norm(direction, component):
    gen = torch.Generator().manual_seed(83)
    clean = torch.randn(1, 256, 8, generator=gen).bfloat16()
    patched = (clean.float() + .4 * torch.randn(clean.shape, generator=gen)).bfloat16()
    rows = random_basis(6, clean.numel(), 27, 'cpu')
    global_before = torch.random.get_rng_state().clone()
    replacement, info = component_replacement(clean, patched, rows, component, direction, 42)
    sensitive = project(patched.float() - clean.float(), rows)
    base, sign = (patched, -1) if direction == 'restore' else (clean, 1)
    expected = norm(apply_delta(base, sign * sensitive).float() - base.float())
    actual = norm(replacement.float() - base.float())
    assert replacement.dtype == torch.bfloat16
    assert abs(actual - expected) <= expected * .03
    assert info['norm_match_ok']
    assert info['component_rank'] == (clean.numel() - 6 if component == 'orthogonal' else 6)
    assert torch.equal(torch.random.get_rng_state(), global_before), 'Control generator must not change global RNG'
    if component == 'functional':
        assert torch.equal(replacement, apply_delta(base, sensitive, sign))


def test_float_orthogonal_direction_and_dose_are_both_matched():
    clean = torch.zeros(1, 1, 4)
    patched = torch.tensor([[[3., 4., 12., 0.]]])
    rows = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    replacement, info = component_replacement(clean, patched, rows, 'orthogonal', 'inject', 11)
    assert torch.allclose(replacement, torch.tensor([[[0., 0., 5., 0.]]]))
    assert info['actual_change_l2'] == 5 and info['orthogonal_component_l2'] == 12
    assert info['actual_change_functional_projection_l2'] == 0


@pytest.mark.parametrize('component', ['functional', 'random', 'orthogonal'])
def test_zero_functional_rank_is_explicit_identity(component):
    clean, patched = torch.zeros(1, 1, 4), torch.ones(1, 1, 4)
    replacement, info = component_replacement(clean, patched, torch.empty(0, 4), component, 'restore', 0)
    assert torch.equal(replacement, patched)
    assert info['control_degenerate'] and info['actual_change_l2'] == 0


def test_zero_control_cannot_impersonate_nonzero_target():
    with pytest.raises(RuntimeError, match='zero random control direction'):
        matched_change(torch.ones(4).bfloat16(), torch.zeros(4), 1.)
    with pytest.raises(RuntimeError, match='Nonfinite'):
        basis([torch.tensor([float('nan')])])


def test_injection_action_reference_is_clean_recipient():
    clean, patched = torch.zeros(1, 8, 7), torch.ones(1, 8, 7)
    injected = clean + .25 * (patched - clean)
    result = directional_action_stats(clean, patched, injected, 'inject')
    assert result['action_fraction_toward_donor'] == pytest.approx(.25)
    assert result['recipient_action_change_l2'] == pytest.approx(norm(injected))

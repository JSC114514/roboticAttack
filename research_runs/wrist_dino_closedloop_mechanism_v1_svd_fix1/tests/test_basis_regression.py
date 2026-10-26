"""Regression for the actual preparation failure; CPU synthetic data only."""
import json
from types import SimpleNamespace

import pytest
import torch

import pilot_common as common
import pilot_math as math
import pilot_pipeline as pipeline


def near_parallel(seed, dimension=1024):
    generator = torch.Generator().manual_seed(seed)
    core = torch.linalg.qr(torch.randn(dimension, 3, generator=generator), mode='reduced').Q.T
    noise = torch.randn(3, dimension, generator=generator)
    noise = noise / noise.norm(dim=1, keepdim=True)
    return torch.cat([core, core + .01 * noise])


@pytest.mark.parametrize('seed', range(8))
def test_near_parallel_endpoint_vectors_are_orthogonal_and_keep_their_span(seed):
    matrix = near_parallel(seed)
    result = math.basis(list(matrix))
    assert result.shape == (6, 1024)
    assert (result @ result.T - torch.eye(6)).abs().max() < 2e-6
    # Independent QR reference, not another invocation of the SVD algorithm.
    reference = torch.linalg.qr(matrix.double().T, mode='reduced').Q
    delta = torch.linspace(-1, 1, 1024, dtype=torch.float64)
    expected = reference @ (reference.T @ delta)
    actual = math.project(delta.float(), result).double()
    assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize('epsilon,expected_rank', [(.001, 1), (.02, 2)])
def test_original_squared_singular_value_cutoff_is_preserved(epsilon, expected_rank):
    first = torch.tensor([1., 0., 0., 0.])
    second = torch.tensor([1., epsilon, 0., 0.])
    rows = math.basis([first, second])
    assert len(rows) == expected_rank
    assert math.norm(math.project(torch.tensor([0., 0., 1., 0.]), rows)) == 0


def test_bfloat16_real_feature_dimension_keeps_dtype_device_rank_and_orthogonality():
    matrix = near_parallel(17, dimension=256 * 1024).bfloat16()
    result = math.basis(list(matrix))
    assert result.dtype == torch.float32 and result.device == matrix.device
    assert result.shape == (6, 256 * 1024)
    # A 262144-term FP32 inner product has its own reduction error. Check the
    # actual returned directions in FP64, and separately retain the runtime's
    # original FP32 0.002 gate; do not confuse dot-product error with bad rows.
    accurate_gram = result.double() @ result.double().T
    assert (accurate_gram - torch.eye(6, dtype=torch.float64)).abs().max() < 2e-6
    assert (result @ result.T - torch.eye(6)).abs().max() < 2e-3
    assert not torch.cuda.is_initialized()


def test_duplicate_zero_and_nonfinite_directions_are_handled_explicitly():
    row = torch.tensor([1., 2., 3., 4.])
    result = math.basis([row, row * 2, torch.zeros_like(row)])
    assert len(result) == 1
    assert math.basis([torch.zeros_like(row), torch.zeros_like(row)]).shape == (0, 4)
    with pytest.raises(RuntimeError, match='Nonfinite'):
        math.basis([torch.tensor([1., float('nan')])])


def test_cpu_failure_is_recorded_visible_and_blocks_stale_preparation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'preflight', lambda: {'signature': 'fixed'})
    (tmp_path / 'run.sh').write_text('#!/usr/bin/env bash\n')
    def fake_run(command, **kwargs):
        if command[0] == 'bash':
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        return SimpleNamespace(returncode=1, stdout='synthetic failure', stderr='test traceback')
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    common.atomic_json({'status': 'CPU_PREFLIGHT_PASS', 'signature': 'fixed'}, tmp_path / 'reports/PREPARATION.json')
    with pytest.raises(RuntimeError, match='CPU tests failed'):
        pipeline.prepare()
    assert common.load_json(tmp_path / 'reports/CPU_PREFLIGHT.json')['status'] == 'CPU_TESTS_FAILED'
    assert common.load_json(tmp_path / 'reports/PREPARE_FAILED.json')['status'] == 'FAILED'
    pipeline.status()
    output = json.loads(capsys.readouterr().out)
    assert output['cpu_preflight']['status'] == 'CPU_TESTS_FAILED'
    assert not output['prepare_failed']['resolved_by_later_success']
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='Latest CPU preflight failed'):
        common.prepared()

"""Unit-patch fixture conformance; never a pretrained-model qualification."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from particle_sliders_ltx25_dummy_preview import binding as product

spec = importlib.util.spec_from_file_location('ltx_fixture_packing', Path(__file__).resolve().parents[1] / 'tools/ltx25_fixture_packing.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_roundtrip_preserves_temporal_order_and_vjp():
    layout = module.FixtureUnitPatch(3, 2, 2)
    packed = torch.arange(12 * 128, dtype=torch.float32).reshape(1, 12, 128).requires_grad_()
    field = layout.unpack(packed)
    assert field.shape == (1, 128, 3, 2, 2)
    assert torch.equal(layout.pack(field), packed)
    assert torch.equal(field[0, :, 2, 1, 1], packed[0, 11])
    weights = torch.arange(field.numel(), dtype=field.dtype).reshape_as(field)
    actual = torch.autograd.grad((field * weights).sum(), packed)[0]
    assert torch.equal(actual, layout.pack(weights))


def test_real_dummy_adapter_output_matches_without_changing_ownership():
    adapter = product.make_native({'dummy': True, 'geometry': [3, 2, 2]})
    generator = torch.Generator().manual_seed(7)
    row = product.VideoRow('person still', 'person smiling',
        torch.randn(1, 12, 128, generator=generator), torch.randn(1, 2, 8, generator=generator))
    try:
        packed = adapter._pack(row.neutral, row, 1.)
        with adapter.network.scaled(0.):
            sample = adapter.backend.forward_velocity(packed, scale=0.).sample
        actual = module.FixtureUnitPatch(3, 2, 2).unpack(sample)
        expected = adapter._field(packed)
        assert torch.equal(actual, expected)
        parameters = tuple(adapter.network.parameters())
        grad_actual = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
        grad_expected = torch.autograd.grad(expected.square().sum(), parameters)
        for left, right in zip(grad_actual, grad_expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        adapter.ledger.check(adapter.backend.network)
    finally:
        adapter.close()


@pytest.mark.parametrize('shape', [(1, 11, 128), (2, 12, 128), (1, 12, 127), (1, 128, 3, 2, 2)])
def test_refuses_wrong_packed_shape(shape):
    with pytest.raises(ValueError, match='geometry differs'):
        module.FixtureUnitPatch(3, 2, 2).unpack(torch.zeros(shape))


def test_rejects_invalid_geometry_values_and_nonfinite_data():
    for values in [(True, 2, 2), (0, 2, 2), (3, 2, 2, 4)]:
        with pytest.raises(ValueError):
            module.FixtureUnitPatch(*values)
    with pytest.raises(ValueError, match='finite floating-point'):
        module.FixtureUnitPatch(1, 1, 1).unpack(torch.full((1, 1, 128), float('nan')))


def test_rejects_non_cpu_fixture_without_using_gpu():
    with pytest.raises(ValueError, match='CPU tensors only'):
        module.FixtureUnitPatch(1, 1, 1).unpack(torch.empty(1, 1, 128, device='meta'))


def test_source_gate_runs_before_packing(monkeypatch):
    def reject():
        raise ValueError('fixture source closure mismatch')
    monkeypatch.setattr(module, 'admit', reject)
    with pytest.raises(ValueError, match='source closure mismatch'):
        module.FixtureUnitPatch(1, 1, 1).unpack(torch.zeros(1, 1, 128))

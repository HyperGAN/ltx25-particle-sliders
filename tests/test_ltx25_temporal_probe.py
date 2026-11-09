"""The probe characterizes the dummy critic without admitting native execution."""
import importlib.util
from pathlib import Path

import pytest
import torch
from particle_sliders_ltx25_dummy_preview.binding import VideoFieldCritic

spec = importlib.util.spec_from_file_location('ltx_temporal_probe', Path(__file__).resolve().parents[1] / 'tools/ltx25_temporal_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        critic = VideoFieldCritic(3, 2, (3, 2, 2), width=4).double()
        response = torch.randn(2, 3, 3, 2, 2, dtype=torch.float64)
        condition = torch.randn(2, 2, dtype=torch.float64)
    return critic, response, condition


def test_frame_order_blindness_and_rng_modes_preserved():
    critic, response, condition = fixture()
    critic.train()
    critic.graph['field0'].eval()
    modes = [m.training for m in critic.modules()]
    rng = torch.get_rng_state().clone()
    result = module.probe_dummy_temporal_order(critic, response, condition, [2, 0, 1])
    assert result == {'scores_equal': True, 'temporal_qualification': False, 'scope': 'dummy CPU fixture only'}
    assert [m.training for m in critic.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('permutation', [[0, 1, 2], [0, 0, 1], [True, 0, 2], [0, 1]])
def test_refuses_identity_or_invalid_permutation(permutation):
    critic, response, condition = fixture()
    with pytest.raises(ValueError, match='permutation'):
        module.probe_dummy_temporal_order(critic, response, condition, permutation)


def test_refuses_hooks_before_running_and_preserves_state():
    critic, response, condition = fixture()
    before = {k: v.clone() for k, v in critic.state_dict().items()}
    handle = critic.register_forward_hook(lambda *args: pytest.fail('hook ran'))
    try:
        with pytest.raises(ValueError, match='hooks'):
            module.probe_dummy_temporal_order(critic, response, condition, [2, 1, 0])
    finally:
        handle.remove()
    assert all(torch.equal(critic.state_dict()[k], v) for k, v in before.items())


def test_mutated_critic_is_rejected_and_rolled_back(monkeypatch):
    critic, response, condition = fixture()
    before = {k: v.clone() for k, v in critic.state_dict().items()}
    original = critic.graph.forward
    def mutate(*args, **kwargs):
        critic.graph['field0'].weight.add_(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(critic.graph, 'forward', mutate)
    with pytest.raises(ValueError, match='state changed'):
        module.probe_dummy_temporal_order(critic, response, condition, [2, 1, 0])
    assert all(torch.equal(critic.state_dict()[k], v) for k, v in before.items())


def test_refuses_non_cpu_critic_without_touching_gpu():
    critic, response, condition = fixture()
    critic.to('meta')
    with pytest.raises(ValueError, match='CPU only'):
        module.probe_dummy_temporal_order(critic, response, condition, [2, 1, 0])


def test_refuses_untracked_buffers():
    critic, response, condition = fixture()
    critic.register_buffer('scratch', torch.tensor(7), persistent=False)
    with pytest.raises(ValueError, match='buffers'):
        module.probe_dummy_temporal_order(critic, response, condition, [2, 1, 0])
    assert critic.scratch.item() == 7


def test_float32_roundoff_does_not_imply_temporal_sensitivity():
    critic, response, condition = fixture()
    result = module.probe_dummy_temporal_order(critic.float(), response.float(), condition.float(), [2, 0, 1])
    assert result['scores_equal'] is True
    assert result['temporal_qualification'] is False


def test_mutating_critic_cannot_change_caller_inputs(monkeypatch):
    critic, response, condition = fixture()
    old_response, old_condition = response.clone(), condition.clone()
    state = {k: v.clone() for k, v in critic.state_dict().items()}
    original = critic.forward
    def mutate(response, condition):
        response.add_(3)
        condition.add_(2)
        return original(response, condition)
    monkeypatch.setattr(critic, 'forward', mutate)
    with pytest.raises(ValueError, match='mutated diagnostic input'):
        module.probe_dummy_temporal_order(critic, response, condition, [2, 1, 0])
    assert torch.equal(response, old_response)
    assert torch.equal(condition, old_condition)
    assert all(torch.equal(critic.state_dict()[k], value) for k, value in state.items())


def test_source_admission_failure_precedes_critic_execution(monkeypatch):
    critic, response, condition = fixture()
    def reject():
        raise ValueError('source closure mismatch')
    monkeypatch.setattr(module, 'admit', reject)
    monkeypatch.setattr(critic, 'forward', lambda *args: pytest.fail('critic executed'))
    with pytest.raises(ValueError, match='source closure mismatch'):
        module.probe_dummy_temporal_order(critic, response, condition, [2, 1, 0])

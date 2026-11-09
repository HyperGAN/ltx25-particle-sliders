"""Bounded CPU diagnostic for the existing dummy critic, not native admission."""
from copy import deepcopy

import torch
from particle_sliders_ltx25_dummy_preview.binding import VideoFieldCritic, admit, native_sources


def probe_dummy_temporal_order(critic, response, condition, permutation):
    """Compare a CPU fixture against the same field with its temporal axis reordered.

    Distinguishing one pair would not prove temporal qualification or model quality.
    Only the existing exact dummy critic is accepted; no native model is loaded.
    """
    admit()
    native_sources()
    if type(critic) is not VideoFieldCritic:
        raise ValueError('only the exact dummy VideoFieldCritic is admitted')
    if (response.device.type != 'cpu' or condition.device.type != 'cpu' or
            any(parameter.device.type != 'cpu' for parameter in critic.parameters())):
        raise ValueError('this diagnostic is CPU only')
    if any(True for _ in critic.buffers()):
        raise ValueError('critic buffers are not admitted in the bounded diagnostic')
    if response.ndim != 5 or response.shape[2] < 2:
        raise ValueError('a non-singleton five-dimensional temporal field is required')
    if not torch.isfinite(response).all() or not torch.isfinite(condition).all():
        raise ValueError('finite fixture inputs are required')
    if (not isinstance(permutation, (list, tuple)) or
            any(type(i) is not int for i in permutation) or
            sorted(permutation) != list(range(response.shape[2])) or
            list(permutation) == list(range(response.shape[2]))):
        raise ValueError('a non-identity permutation of the native temporal axis is required')
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in critic.modules()):
        raise ValueError('critic hooks are not admitted in the bounded diagnostic')
    state = deepcopy(critic.state_dict())
    modes = [(m, m.training) for m in critic.modules()]
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            critic.eval()
            original = response.detach().clone()
            reordered = response.detach()[:, :, list(permutation), :, :].clone()
            frozen_condition = condition.detach().clone()
            original_expected = original.clone()
            reordered_expected = reordered.clone()
            condition_expected = frozen_condition.clone()
            before = critic(original, frozen_condition).detach().clone()
            if not torch.equal(original, original_expected) or not torch.equal(frozen_condition, condition_expected):
                raise ValueError('critic mutated diagnostic input')
            after = critic(reordered, frozen_condition).detach().clone()
            if not torch.equal(reordered, reordered_expected) or not torch.equal(frozen_condition, condition_expected):
                raise ValueError('critic mutated diagnostic input')
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                raise ValueError('finite critic scores are required')
            current = critic.state_dict()
            if current.keys() != state.keys() or any(not torch.equal(current[k], state[k]) for k in state):
                raise ValueError('critic state changed during diagnostic')
            tolerance = 16 * torch.finfo(before.dtype).eps
            return {'scores_equal': torch.allclose(before, after, atol=tolerance, rtol=tolerance),
                    'temporal_qualification': False,
                    'scope': 'dummy CPU fixture only'}
    finally:
        critic.load_state_dict(state, strict=True)
        for module, mode in modes:
            module.training = mode

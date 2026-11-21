"""Opt-in activation-dependent conditioning for an exact open LTX 2.5 RAW session."""
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
from math import fsum, isfinite
from types import ModuleType
from zipfile import ZipFile

import torch

from .scheduled_render import DISTILLED_SIGMAS, _render_with_conditions, _runtime_owner

CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_MEMBER = "particle_sliders/adaptive_steering.py"
CORE_MEMBER_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"


@lru_cache(maxsize=1)
def _load_shared(source):
    module = ModuleType("ps_ltx25_gate_core")
    exec(compile(source, CORE_MEMBER, "exec"), module.__dict__)
    return module


def gate_core(core_wheel):
    """Load the exact standalone #269 equations alongside the archived RAW owner."""
    with ZipFile(core_wheel) as wheel:
        source = wheel.read(CORE_MEMBER)
    if sha256(source).hexdigest() != CORE_MEMBER_SHA256:
        raise ValueError("LTX 2.5 requires the exact reviewed core #269 adaptive steering source")
    return _load_shared(source)


def coupled_latent_readout(video, audio):
    """Concatenate the FP32 token means of native packed video and audio inputs."""
    if any(latent.ndim != 3 or latent.shape[0] != 1 or not all(latent.shape)
           or not latent.is_floating_point() for latent in (video, audio)) or video.device != audio.device:
        raise ValueError("LTX 2.5 adaptive readout needs one coupled conditional video/audio sample")
    return torch.cat((video.mean(dim=1, dtype=torch.float32), audio.mean(dim=1, dtype=torch.float32)), dim=-1)[0]


def _check_interval(pipe, position, timestep):
    scheduler = pipe.scheduler
    expected_index = None if position == 0 else position
    if (pipe.do_classifier_free_guidance or pipe.do_spatio_temporal_guidance
            or pipe.do_modality_isolation_guidance):
        raise ValueError("LTX 2.5 adaptive strength requires the native distilled single conditional path")
    if (scheduler.begin_index != 0 or scheduler.step_index != expected_index
            or len(scheduler.timesteps) != len(DISTILLED_SIGMAS)
            or timestep.shape != (1,) or not torch.equal(timestep, scheduler.timesteps[position:position + 1])):
        raise RuntimeError("LTX 2.5 adaptive steering lost native solver interval alignment")


@torch.no_grad()
def render(session, *, prompt, core_wheel, method, parameter, evidence_ref, strength,
           direction=None, gate_seed=None, clabs=None, **request):
    """Gate the existing joint connector conditioning at a fixed RAW strength.

    Parameter and direction identify the caller's fitted probe and active
    rollout calibration. A random control takes the matched firing rate and
    no direction. CLABS takes explicit calibrated controller fields.
    """
    runtime = _runtime_owner(session)
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("LTX 2.5 strength must be finite and in [0, 1]")
    if strength == 0:
        return session.render(prompt=prompt, strength=0, **request)
    if not isinstance(evidence_ref, str) or not evidence_ref.strip():
        raise ValueError("LTX 2.5 adaptive strength requires a fit/calibration evidence reference")
    if type(parameter) not in (int, float) or not isfinite(parameter):
        raise ValueError("LTX 2.5 requires an explicit finite gate parameter")
    core = gate_core(core_wheel)
    config = core.CLABSParameters(**clabs) if clabs is not None else None
    gate = core.ActivationGate(method, parameter, seed=gate_seed, clabs=config)
    if method == "random":
        if direction is not None:
            raise ValueError("LTX 2.5 matched random control does not use a direction")
    else:
        if not isinstance(direction, torch.Tensor) or direction.ndim != 1 or not direction.is_floating_point():
            raise ValueError("LTX 2.5 adaptive strength requires a fitted joint latent direction")
        direction = direction.detach().to(device=runtime.device, dtype=torch.float32)
        norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("LTX 2.5 direction must be finite and nonzero")
        direction = direction / norm
    trace = []

    def select(position, kwargs):
        _check_interval(runtime.pipe, position, kwargs["timestep"])
        readout = coupled_latent_readout(kwargs["hidden_states"], kwargs["audio_hidden_states"])
        score = None if direction is None else core.activation_projection(readout, direction, strength=strength).item()
        decision = gate.step(score)
        effective = strength if decision.fired else 0.0
        trace.append({"interval": position, "timestep": kwargs["timestep"].item(), "score": score,
                      "fired": decision.fired, "probability": decision.probability, "effective_strength": effective})
        return effective

    result, sigmas = _render_with_conditions(
        session, runtime, prompt=prompt, strength=0, candidates=(0.0, strength), select=select,
        request=request,
    )
    applied = [entry["effective_strength"] for entry in trace]
    dose = fsum(value * abs(left - right) for value, left, right in zip(applied, sigmas[:-1], sigmas[1:], strict=True))
    schedule = {"sigmas": sigmas, "effective_strengths": applied, "sigma_integrated_dose": dose,
                "gate": {"method": method, "parameter": parameter, "seed": gate_seed, "evidence_ref": evidence_ref,
                         "core_revision": CORE_REVISION, "core_source_sha256": CORE_MEMBER_SHA256,
                         "readout": "packed_video_token_mean_then_packed_audio_token_mean_fp32",
                         "direction": None if direction is None else direction.tolist(),
                         "clabs": None if config is None else asdict(config), "nominal_strength": strength,
                         "usage": asdict(gate.usage), "decisions": trace}}
    return {**result, "strength": strength, "strength_schedule": schedule}

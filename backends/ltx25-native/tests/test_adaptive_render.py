from contextlib import contextmanager
from os import environ
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from ps_ltx25_native.adaptive_render import CORE_MEMBER, gate_core, render
from ps_ltx25_native.native import NativeRuntime
from ps_ltx25_native.render import DISTILLED_SIGMAS, observation
from ps_ltx25_native.scheduled_render import render as replay


@pytest.fixture
def core_wheel():
    path = environ.get("LTX25_STEERING_CORE_WHEEL")
    if not path:
        pytest.skip("set LTX25_STEERING_CORE_WHEEL to the exact core #269 wheel")
    return path


@pytest.fixture
def runtime_session():
    active = 0.0

    @contextmanager
    def scaled(value):
        nonlocal active
        previous = active
        active = value
        try:
            yield
        finally:
            active = previous

    class Connectors(torch.nn.Module):
        def forward(self, features, mask, *, padding_side):
            assert padding_side == "left"
            return features + active, features - active, mask

    class Denoiser(torch.nn.Module):
        is_cache_enabled = False

        def forward(self, *, hidden_states, audio_hidden_states, timestep, encoder_hidden_states,
                    audio_encoder_hidden_states, encoder_attention_mask, audio_encoder_attention_mask):
            assert torch.equal(encoder_attention_mask, audio_encoder_attention_mask)
            return (torch.ones_like(hidden_states) * encoder_hidden_states.mean(),
                    torch.ones_like(audio_hidden_states) * audio_encoder_hidden_states.mean())

    def encode_prompt(*, prompt_embeds, prompt_attention_mask, **request):
        assert request["do_classifier_free_guidance"] is False
        return prompt_embeds, prompt_attention_mask, None, None

    runtime = NativeRuntime.__new__(NativeRuntime)
    runtime.identity = {"fixture": "coupled native FlowMatchEuler"}
    runtime.device = runtime.dit_device = torch.device("cpu")
    runtime.transformer = Denoiser()
    runtime.connectors = Connectors()
    runtime.tokenize = lambda prompt, count: {"attention_mask": torch.ones((1, 1), dtype=torch.bool)}
    runtime.features = lambda tokens: torch.zeros(1, 1, 1)
    runtime.pipe = SimpleNamespace(
        scheduler=FlowMatchEulerDiscreteScheduler(), tokenizer=SimpleNamespace(padding_side="left"),
        encode_prompt=encode_prompt, do_classifier_free_guidance=False,
        do_spatio_temporal_guidance=False, do_modality_isolation_guidance=False,
    )

    def close():
        runtime.close()

    def original(*, prompt, strength, repeat_denoiser=False):
        tokens = runtime.tokenize(prompt, 1)
        with observation(runtime), scaled(strength):
            video_context, audio_context, mask = runtime.connectors(
                runtime.features(tokens), tokens["attention_mask"], padding_side="left",
            )
            video_scheduler = runtime.pipe.scheduler
            audio_scheduler = FlowMatchEulerDiscreteScheduler()
            for scheduler in (video_scheduler, audio_scheduler):
                scheduler.set_timesteps(sigmas=DISTILLED_SIGMAS)
                scheduler.set_begin_index(0)
            video, audio = torch.zeros(1, 3, 1), torch.zeros(1, 5, 1)
            for timestep in video_scheduler.timesteps:
                args = dict(hidden_states=video, audio_hidden_states=audio, timestep=timestep[None],
                            encoder_hidden_states=video_context, audio_encoder_hidden_states=audio_context,
                            encoder_attention_mask=mask, audio_encoder_attention_mask=mask)
                video_prediction, audio_prediction = runtime.transformer(**args)
                if repeat_denoiser:
                    runtime.transformer(**args)
                video = video_scheduler.step(video_prediction, timestep, video).prev_sample
                audio = audio_scheduler.step(audio_prediction, timestep, audio).prev_sample
        return {"frames": video, "audio": audio, "strength": strength}

    session = SimpleNamespace(
        selection="raw", _open=lambda: None, render=original,
        manifest={"specification": {"configuration": {"geometry": {"text_tokens": 1}}}},
        binding=SimpleNamespace(close=close, render=original, identity={"model": runtime.identity},
                                network=SimpleNamespace(scaled=scaled)),
    )
    yield runtime, session
    assert active == 0
    assert not runtime.transformer._forward_pre_hooks
    assert runtime.pipe.scheduler.step_index is None


def test_adaptive_live_audio_feedback_and_exact_coupled_replay(core_wheel, runtime_session):
    _, session = runtime_session
    result = render(session, prompt="room", core_wheel=core_wheel, method="cfp", parameter=0,
                    direction=torch.tensor([0., 1.]), strength=.5, evidence_ref="fixture")
    schedule = result["strength_schedule"]
    assert schedule["effective_strengths"] == [.5] + [0.] * 7
    trace = schedule["gate"]["decisions"]
    assert trace[0]["score"] == 0
    assert all(row["score"] > 0 and not row["fired"] for row in trace[1:])
    assert schedule["gate"]["usage"] == {"fired": 1, "eligible": 8}
    assert schedule["sigma_integrated_dose"] == .5 * (schedule["sigmas"][0] - schedule["sigmas"][1])
    frozen = replay(session, prompt="room", strength=.5, step_gains=[1.] + [0.] * 7)
    for key in ("frames", "audio"):
        assert torch.equal(result[key], frozen[key])


@pytest.mark.parametrize("rate", [0., 1.])
def test_random_endpoints_match_ordinary_off_and_constant(rate, core_wheel, runtime_session):
    _, session = runtime_session
    result = render(session, prompt="room", core_wheel=core_wheel, method="random", parameter=rate,
                    gate_seed=3, strength=.5, evidence_ref="fixture")
    ordinary = session.render(prompt="room", strength=rate * .5)
    assert result["strength_schedule"]["effective_strengths"] == [rate * .5] * 8
    for key in ("frames", "audio"):
        assert torch.equal(result[key], ordinary[key])


def test_wrong_cfg_path_and_repeat_fail_and_restore(core_wheel, runtime_session):
    runtime, session = runtime_session
    options = dict(prompt="room", core_wheel=core_wheel, method="random", parameter=1,
                   gate_seed=1, strength=.5, evidence_ref="fixture")
    original_scheduler = runtime.pipe.scheduler
    runtime.pipe.do_classifier_free_guidance = True
    with pytest.raises(ValueError, match="single conditional"):
        render(session, **options)
    runtime.pipe.do_classifier_free_guidance = False
    with pytest.raises(RuntimeError, match="solver interval alignment"):
        render(session, repeat_denoiser=True, **options)
    assert runtime.pipe.scheduler is original_scheduler
    assert not runtime.transformer._forward_pre_hooks
    assert render(session, **options)["strength_schedule"]["gate"]["usage"]["eligible"] == 8


def test_clabs_applies_shared_feedback_with_request_local_state(core_wheel, runtime_session):
    _, session = runtime_session
    options = dict(prompt="room", core_wheel=core_wheel, method="clabs", parameter=.4,
                   direction=torch.tensor([0., 1.]), gate_seed=5, strength=.5, evidence_ref="fixture",
                   clabs=dict(mean=0., std=1., epsilon=1.e-6, setpoint=.2, proportional_gain=.5,
                              integral_gain=.2, beta=.3))
    result = render(session, **options)
    gate = result["strength_schedule"]["gate"]
    core = gate_core(core_wheel)
    expected = core.ActivationGate("clabs", options["parameter"], seed=options["gate_seed"],
                                   clabs=core.CLABSParameters(**options["clabs"]))
    for row in gate["decisions"]:
        decision = expected.step(row["score"])
        assert (row["probability"], row["fired"]) == (decision.probability, decision.fired)
    assert render(session, **options)["strength_schedule"] == result["strength_schedule"]


def test_zero_strength_uses_original_path_without_opening_probe(runtime_session):
    _, session = runtime_session
    result = render(session, prompt="room", core_wheel=None, method=None, parameter=None,
                    strength=0, evidence_ref=None)
    ordinary = session.render(prompt="room", strength=0)
    assert "strength_schedule" not in result
    for key in ("frames", "audio"):
        assert torch.equal(result[key], ordinary[key])


def test_modified_shared_steering_source_is_rejected(tmp_path):
    wheel_path = tmp_path / "substitute.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(CORE_MEMBER, "raise AssertionError('unreviewed source ran')\n")
    with pytest.raises(ValueError, match="exact reviewed core"):
        gate_core(wheel_path)

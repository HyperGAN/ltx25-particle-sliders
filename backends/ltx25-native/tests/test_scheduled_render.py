from contextlib import contextmanager
from types import SimpleNamespace

import torch

from ps_ltx25_native.native import NativeRuntime
from ps_ltx25_native.render import DISTILLED_SIGMAS
from ps_ltx25_native.scheduled_render import render


def test_conditions_match_each_native_audio_and_video_interval():
    active = 0.0
    conditioned = []
    observed = []

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
            conditioned.append(active)
            return features + active, features - active, mask

    class Denoiser(torch.nn.Module):
        is_cache_enabled = False

        def forward(self, *, encoder_hidden_states, audio_encoder_hidden_states,
                    encoder_attention_mask, audio_encoder_attention_mask):
            assert torch.equal(encoder_attention_mask, audio_encoder_attention_mask)
            observed.append((float(encoder_hidden_states.item()), float(audio_encoder_hidden_states.item())))

    def encode_prompt(*, prompt_embeds, prompt_attention_mask, **request):
        assert request["do_classifier_free_guidance"] is False
        return prompt_embeds, prompt_attention_mask, None, None

    runtime = NativeRuntime.__new__(NativeRuntime)
    runtime.identity = {"fixture": "conditioning dispatch"}
    runtime.device = runtime.dit_device = torch.device("cpu")
    runtime.transformer = Denoiser()
    runtime.connectors = Connectors()
    runtime.tokenize = lambda prompt, count: {"attention_mask": torch.ones((1, 1), dtype=torch.bool)}
    runtime.features = lambda tokens: torch.ones(1, 1, 1)
    scheduler = SimpleNamespace(sigmas=torch.tensor([*DISTILLED_SIGMAS, 0.0]))
    runtime.pipe = SimpleNamespace(scheduler=scheduler, tokenizer=SimpleNamespace(padding_side="left"),
                                   encode_prompt=encode_prompt)

    def close():
        runtime.close()

    def original(*, prompt, strength):
        tokens = runtime.tokenize(prompt, 1)
        with scaled(strength):
            video, audio, mask = runtime.connectors(runtime.features(tokens), tokens["attention_mask"],
                                                    padding_side="left")
            for _ in DISTILLED_SIGMAS:
                runtime.transformer(encoder_hidden_states=video, audio_encoder_hidden_states=audio,
                                    encoder_attention_mask=mask, audio_encoder_attention_mask=mask)
        return {"frames": "fixture", "audio": "fixture"}

    session = SimpleNamespace(
        selection="raw", _open=lambda: None, render=original,
        manifest={"specification": {"configuration": {"geometry": {"text_tokens": 1}}}},
        binding=SimpleNamespace(close=close, render=original, identity={"model": runtime.identity},
                                network=SimpleNamespace(scaled=scaled)),
    )
    original(prompt="room", strength=0.5)
    baseline = list(observed)
    observed.clear()
    render(session, prompt="room", strength=0.5, step_gains=[1.0] * 8)
    assert observed == baseline

    observed.clear()
    conditioned.clear()
    result = render(session, prompt="room", strength=0.5, step_gains=[1.5] * 4 + [0.5] * 4)
    assert observed == [(1.75, 0.25)] * 4 + [(1.25, 0.75)] * 4
    assert conditioned == [0.75, 0.25, 0.5]
    assert result["strength_schedule"]["effective_strengths"] == [0.75] * 4 + [0.25] * 4
    assert len(result["strength_schedule"]["sigmas"]) == 9
    assert runtime.pipe.scheduler is scheduler and active == 0.0
    assert not runtime.transformer._forward_pre_hooks

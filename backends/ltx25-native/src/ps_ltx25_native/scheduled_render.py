"""Supply the matching adapted audiovisual conditioning to each native interval."""

from sys import float_info

import torch

from .native import NativeRuntime
from .render import DISTILLED_SIGMAS, observation


def _strengths(strength, step_gains):
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("LTX 2.5 nominal strength must be in [0, 1]")
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != len(DISTILLED_SIGMAS):
        raise ValueError("LTX 2.5 requires one gain per distilled denoising interval")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("LTX 2.5 step gains must be finite and nonnegative")
    values = [strength * gain for gain in step_gains]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("LTX 2.5 effective strengths must be in [0, 1]")
    return values


def _condition_bank(runtime, adapter, prompt, token_count, strengths):
    pipe = runtime.pipe
    tokens = runtime.tokenize(prompt, token_count)
    padding = getattr(pipe.tokenizer, "padding_side", "left")
    bank = {}
    with observation(runtime):
        for strength in dict.fromkeys(strengths):
            with adapter.scaled(strength):
                embeds, mask, _, _ = pipe.encode_prompt(
                    prompt=None, prompt_embeds=runtime.features(tokens),
                    prompt_attention_mask=tokens["attention_mask"],
                    do_classifier_free_guidance=False, num_videos_per_prompt=1,
                    max_sequence_length=token_count, device=runtime.device,
                )
                bank[strength] = runtime.connectors(embeds, mask, padding_side=padding)
    return bank


@torch.no_grad()
def render(session, *, prompt, step_gains, strength=1.0, **request):
    """Use original Gemma/connector outputs at each step of the original AV sampler."""
    session._open()
    if session.selection != "raw" or session.binding.render is None:
        raise ValueError("LTX 2.5 scheduling requires an open RAW render session")
    owners = [cell.cell_contents for cell in session.binding.close.__closure__ or ()]
    runtimes = [owner for owner in owners if isinstance(owner, NativeRuntime)]
    if len(runtimes) != 1 or runtimes[0].identity != session.binding.identity["model"]:
        raise ValueError("expected the original LTX 2.5 native factory owner")
    runtime = runtimes[0]
    if runtime.transformer.is_cache_enabled:
        raise ValueError("LTX 2.5 scheduled conditioning requires the uncached transformer profile")
    strengths = _strengths(strength, step_gains)
    config = session.manifest["specification"]["configuration"]
    bank = _condition_bank(runtime, session.binding.network, prompt, config["geometry"]["text_tokens"], strengths)
    sigmas = None
    position = 0

    def substitute(_model, args, kwargs):
        nonlocal position, sigmas
        if position >= len(strengths):
            raise RuntimeError("LTX 2.5 made extra native denoiser calls")
        if sigmas is None:
            sigmas = runtime.pipe.scheduler.sigmas.detach().cpu().tolist()
            if len(sigmas) != len(strengths) + 1:
                raise RuntimeError("LTX 2.5 actual scheduler interval count differs")
        video, audio, mask = bank[strengths[position]]
        conditions = {
            "encoder_hidden_states": video, "audio_encoder_hidden_states": audio,
            "encoder_attention_mask": mask, "audio_encoder_attention_mask": mask,
        }
        for name, value in conditions.items():
            original = kwargs[name]
            if (value.shape, value.dtype, value.device) != (original.shape, original.dtype, original.device):
                raise RuntimeError("LTX 2.5 scheduled conditioning differs in native coordinates: " + name)
        position += 1
        return args, {**kwargs, **conditions}

    hook = runtime.transformer.register_forward_pre_hook(substitute, with_kwargs=True)
    try:
        result = session.render(prompt=prompt, strength=strength, **request)
    finally:
        hook.remove()
    if position != len(strengths):
        raise RuntimeError("LTX 2.5 finished before applying each conditioning")
    return {**result, "strength_schedule": {"sigmas": sigmas, "effective_strengths": strengths}}

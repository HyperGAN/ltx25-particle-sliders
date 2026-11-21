"""Original distilled native audiovisual sampler, not a custom denoising loop."""
from contextlib import contextmanager
from copy import deepcopy
import math
import random
import torch
from .contracts import geometry

DISTILLED_SIGMAS = (1., .99375, .9875, .98125, .975, .909375, .725, .421875)


@contextmanager
def observation(runtime):
    """Restore scheduler/flags/RNG even after failed render/cache preparation."""
    pipe = runtime.pipe
    scheduler = pipe.scheduler
    old = {key: deepcopy(value) for key, value in pipe.__dict__.items() if key.startswith("_") and
           key in {"_guidance_scale", "_stg_scale", "_modality_scale", "_guidance_rescale", "_audio_guidance_scale",
                   "_audio_stg_scale", "_audio_modality_scale", "_audio_guidance_rescale", "_attention_kwargs",
                   "_interrupt", "_current_timestep", "_num_timesteps"}}
    keys = set(old)
    tracked = keys | {"_guidance_scale", "_stg_scale", "_modality_scale", "_guidance_rescale", "_audio_guidance_scale",
                     "_audio_stg_scale", "_audio_modality_scale", "_audio_guidance_rescale", "_attention_kwargs",
                     "_interrupt", "_current_timestep", "_num_timesteps"}
    modules = {id(module): module for name in ("transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder")
               for owner in [getattr(pipe, name, None) or getattr(runtime, name, None)] if owner is not None
               for module in owner.modules()}
    modes = [(module, module.training) for module in modules.values()]
    registered = [(module, dict(module._buffers), set(module._non_persistent_buffers_set)) for module in modules.values()]
    buffers = {id(value): (value, value.detach().clone()) for _, old_buffers, _ in registered
               for value in old_buffers.values() if value is not None}
    def restore_buffer(value, saved):
        with torch.no_grad():
            if (value.shape, value.dtype, value.device) == (saved.shape, saved.dtype, saved.device):
                value.copy_(saved)
            else:
                value.data = saved.clone()
    def restore_registration(module, original, nonpersistent):
        module._buffers.clear()
        module._buffers.update(original)
        module._non_persistent_buffers_set.clear()
        module._non_persistent_buffers_set.update(nonpersistent)
    def restore_flag(key):
        if key in keys:
            setattr(pipe, key, old[key])
        else:
            pipe.__dict__.pop(key, None)
    py = random.getstate()
    devices = sorted({d.index for d in (runtime.device, runtime.dit_device) if d.type == "cuda"})
    primary = None
    with torch.random.fork_rng(devices=devices):
        try:
            pipe.scheduler = deepcopy(scheduler)
            yield
        except BaseException as exc:
            primary = exc
            raise
        finally:
            from particle_sliders.restoration import _restore_all
            _restore_all([lambda: setattr(pipe, "scheduler", scheduler),
                *(lambda key=key: restore_flag(key) for key in tracked),
                *(lambda value=value, saved=saved: restore_buffer(value, saved) for value, saved in buffers.values()),
                *(lambda module=module, original=original, nonpersistent=nonpersistent:
                    restore_registration(module, original, nonpersistent) for module, original, nonpersistent in registered),
                *(lambda module=module, training=training: setattr(module, "training", training) for module, training in modes),
                lambda: random.setstate(py)], primary=primary)


def validate_render(g, *, prompt, strength, seed, output_type="pil"):
    if geometry(g) != g or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("exact frozen geometry and nonempty native prompt required")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("finite native adapter strength in[0,1] required")
    if type(seed) is not int or not 0 <= seed < 2**63 or output_type not in ("pil", "latent"):
        raise ValueError("explicit native seed and PIL-video or full latent output required")


def pipeline_arguments(g, *, seed, output_type="pil"):
    return dict(height=g["height"], width=g["width"], num_frames=g["frames"], frame_rate=g["fps"],
        sigmas=list(DISTILLED_SIGMAS), guidance_scale=1., stg_scale=0., modality_scale=1., guidance_rescale=.7,
        audio_guidance_scale=1., audio_stg_scale=0., audio_modality_scale=1., audio_guidance_rescale=.7,
        spatio_temporal_guidance_blocks=None, enable_prompt_enhancement=False, use_cross_timestep=True,
        output_type=output_type, max_sequence_length=g["text_tokens"], return_dict=True)


@torch.no_grad()
def render_video(runtime, adapter, g, *, prompt, strength=1., seed=7, output_type="pil"):
    validate_render(g, prompt=prompt, strength=strength, seed=seed, output_type=output_type)
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    if list(DISTILLED_SIGMA_VALUES) != list(DISTILLED_SIGMAS):
        raise ValueError("native distilled sigma Source differs")
    tokens = runtime.tokenize(prompt, g["text_tokens"])
    with observation(runtime), adapter.scaled(strength):
        embeds = runtime.features(tokens)
        if runtime.pipe._execution_device != runtime.device:
            raise ValueError("native pipeline execution placement differs from explicit text/latent device")
        generator = torch.Generator(device=runtime.device).manual_seed(seed)
        with runtime.native_pipeline_placement():
            result = runtime.pipe(prompt=None, prompt_embeds=embeds, prompt_attention_mask=tokens["attention_mask"],
                generator=generator, **pipeline_arguments(g, seed=seed, output_type=output_type))
    return {"frames": result.frames, "audio": result.audio,
            "sample_rate": int(runtime.pipe.vocoder.config.output_sampling_rate), "fps": g["fps"],
            "native_joint_AV": True, "audio_invariant": False, "seed": seed, "strength": strength}


def export_video(result, path):
    """Export native frames AND native joint audio; never label a tensor video."""
    from diffusers.utils import encode_video
    from pathlib import Path
    if Path(path).exists():
        raise FileExistsError("finished outputs are write-once")
    frames = result["frames"][0]
    # Exact official public encoder accepts uint8 T,H,W,C and C,samples audio.
    import numpy as np
    frames = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames]))
    audio = result["audio"][0].detach().float().cpu()
    encode_video(frames, fps=result["fps"], audio=audio, audio_sample_rate=result["sample_rate"], output_path=str(path))

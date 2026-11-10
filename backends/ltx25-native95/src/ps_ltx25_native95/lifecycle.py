"""Shared PG095 game/FULL/RAW lifecycle; actual model logic remains native."""
from copy import deepcopy
from pathlib import Path
import torch
from particle_sliders.lifecycle import Binding, Capabilities, UnsupportedOperation, native_cleanup
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg095 import ConditionalPG095, engine_options
from particle_sliders.restoration import _restore_all
from .contracts import (BACKEND_NAME, PROTOCOL, SOURCE_PROFILE, INITIALIZATION_POLICY, WEIGHT_POLICY, PG_VERSION, PG_SOURCE_PROFILE,
                        digest, geometry, response_shape, rate_overrides, validate_options, validate_core)
from .initialization import Construction, construction_for, initialize_owned, validate_receipt
from .adapter import TextConnectorLoRA
from .native import NativeRuntime, NativeVideoResponse
from .critic import VideoCritic
from .cache import CachedResponses
from .render import observation


def capabilities(options):
    return ResponseCapabilities("adapter_gan", "video", "native-video-velocity", "continuous_field", response_shape(options),
        PG_VERSION, PG_SOURCE_PROFILE, penalty_input_domain="auto", base_precisions=("float32", "bfloat16"),
        qualification="New full5D native Source only; tiny CPU/pretrained/GPU/quality checks have NOT run")


INFERENCE_CAPABILITIES = Capabilities("adapter_gan", render=True, render_outputs=("pil", "latent"),
    base_precisions=("float32", "bfloat16"), operations=("response", "render"),
    qualification="Native joint audiovisual pipeline Source; no executed native/pretrained/quality evidence yet")


def preflight(options, request, staged):
    if set(staged) - {"runtime", "construction"} or request["native"]:
        raise UnsupportedOperation("training requires explicit local native runtime and authenticated full frozen cache")
    construction = staged.get("construction")
    if not isinstance(construction, Construction) or construction.options_sha256 != digest(options):
        raise ValueError("open_training binds actual fresh/resume intent before allocation/optimizers")
    runtime = staged.get("runtime")
    if runtime is not None and (str(runtime.device) != options.get("text_device", "cpu") or
            str(runtime.dit_device) != options.get("dit_device", "cpu") or
            runtime.dtype != getattr(torch, options.get("base_precision", "bfloat16"))):
        raise ValueError("borrowed runtime placement/precision differs")


def runtime_from(options):
    return NativeRuntime(options["model_dir"], model_lock_sha256=options["model_lock_sha256"],
        text_device=options.get("text_device", "cpu"), dit_device=options.get("dit_device", "cpu"),
        precision=options.get("base_precision", "bfloat16"))


def prepare(options, *, runtime=None, construction=None):
    validate_core()
    validate_options(options)
    preflight(options, {"native": {}}, {"runtime": runtime, "construction": construction})
    owns = runtime is None
    runtime = runtime_from(options) if owns else runtime
    restores, adapter, backend, closed = [native_cleanup(module) for module in
        (runtime.transformer, runtime.text_encoder, runtime.connectors)], None, None, False
    def close():
        nonlocal closed
        if closed:
            return
        _restore_all([*([] if adapter is None else [adapter.close]), *reversed(restores), *([runtime.close] if owns else [])])
        restores.clear()
        if owns and backend is not None:
            backend.model = backend.transformer = None
        closed = True
    try:
        shape = response_shape(options)
        rows = CachedResponses(options["cache_dir"], runtime, options["cache_index_sha256"], geometry(options))
        devices = sorted({device.index for device in (runtime.device, runtime.dit_device) if device.type == "cuda"})
        with torch.random.fork_rng(devices=devices):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get("seed", 7), devices)
            adapter = TextConnectorLoRA(runtime.text_encoder, runtime.connectors)
            critic = VideoCritic(shape, rows.condition_width, options.get("critic_width", 8)).to(runtime.device)
            receipt = initialize_owned(adapter, critic, construction, options)  # BEFORE PG optimizers.
            adapter.attach()
            backend = NativeVideoResponse(runtime, adapter, shape)
            backend.fresh_initialization = receipt.as_dict()
        game = ConditionalPG095(backend, rows, critic, particles=None, response_domain="continuous_field",
            batch_size=options.get("batch", 2), total_steps=options.get("steps", 32), seed=options.get("seed", 7),
            sigma=options.get("sigma", 1.), response_scale=options.get("response_scale", 1.),
            recipe_overrides=rate_overrides(options), **engine_options(dict(options, pg_version=PG_VERSION, pg_source_profile=PG_SOURCE_PROFILE)))
        configuration = {"geometry": geometry(options), "rank": 4, "alpha": 4., "weight_policy": WEIGHT_POLICY,
            "source_profile": SOURCE_PROFILE, "initialization_policy": INITIALIZATION_POLICY, "initialization": receipt.as_dict()}
        spec = NATIVE_BACKEND.specification(backend, model_identity=runtime.identity, protocol=PROTOCOL, configuration=configuration)
        return Binding(game, close, inference=spec, before_update=lambda request: bool(rows.identity()),
                       observation=lambda: observation(runtime))
    except BaseException as primary:
        _restore_all([close], primary=primary)
        raise


def validate_inference(options, spec):
    if set(options) - {"model_dir", "model_lock_sha256", "text_device", "dit_device", "base_precision"}:
        raise ValueError("RAW inference accepts local model and placement only; no game/cache/initialization")
    if (not isinstance(options.get("model_dir"), str) or not Path(options["model_dir"]).is_dir() or
            any(options.get(key, "cpu") not in ("cpu", "cuda:0", "cuda:1") for key in ("text_device", "dit_device")) or
            options.get("base_precision", "bfloat16") not in ("float32", "bfloat16")):
        raise ValueError("explicit supported staged model/precision/placement required")
    c = spec.configuration
    if (spec.identity["protocol"] != PROTOCOL or set(c) != {"geometry", "rank", "alpha", "weight_policy", "source_profile",
            "initialization_policy", "initialization"} or (c["rank"], c["alpha"], c["weight_policy"], c["source_profile"],
            c["initialization_policy"]) != (4, 4., WEIGHT_POLICY, SOURCE_PROFILE, INITIALIZATION_POLICY) or geometry(c["geometry"]) != c["geometry"]):
        raise ValueError("only exact NEW native-video RAW cohort; historical UNI/dummy packets stay separate")
    validate_receipt(c["initialization"], c["initialization"].get("cohort_id"))


def preflight_inference(options, request, spec):
    if request["operation"] == "render":
        from .render import validate_render
        native = request["native"]
        if set(native) - {"prompt", "images"} or native.get("images") is not None:
            raise ValueError("this exact native text-to-video profile does not accept reference images")
        validate_render(spec.configuration["geometry"], prompt=native.get("prompt"), strength=request["strength"],
                        seed=0, output_type=request["output_type"])
    elif set(request["native"]) != {"condition", "latent", "timestep"}:
        raise ValueError("native response requires complete recorded condition/latent/time together")


def prepare_inference(options, spec):
    validate_core()
    validate_inference(options, spec)
    runtime, adapter, backend = runtime_from(options), None, None
    def close():
        _restore_all([*([] if adapter is None else [adapter.close]), runtime.close])
        if backend is not None:
            backend.model = backend.transformer = None
    try:
        devices = sorted({d.index for d in (runtime.device, runtime.dit_device) if d.type == "cuda"})
        with torch.random.fork_rng(devices=devices):
            adapter = TextConnectorLoRA(runtime.text_encoder, runtime.connectors).attach()
        backend = NativeVideoResponse(runtime, adapter, response_shape(spec.configuration["geometry"]))
        backend.fresh_initialization = spec.configuration["initialization"]  # Receipt ONLY; initializer NOT called.
        identity = native_identity(backend, model_identity=runtime.identity, protocol=PROTOCOL)
        def response(*, strength, seed, condition, latent, timestep):
            with adapter.scaled(strength):
                return backend._velocity(condition, latent, torch.as_tensor(timestep)).detach().clone()
        def render(*, strength, seed, prompt, images=None, output_type="pil"):
            if images is not None:
                raise ValueError("native video references are unsupported")
            from .render import render_video
            return render_video(runtime, adapter, spec.configuration["geometry"], prompt=prompt,
                                strength=strength, seed=seed, output_type=output_type)
        return InferenceBinding(adapter, identity, close, response=response, render=render, observation=lambda: observation(runtime))
    except BaseException as primary:
        _restore_all([close], primary=primary)
        raise


NATIVE_BACKEND = InferenceBackend(BACKEND_NAME, INFERENCE_CAPABILITIES, prepare_inference, validate_inference, preflight=preflight_inference)
BACKEND = ResponseBackend(BACKEND_NAME, capabilities({}), prepare, validate_options, configure_capabilities=capabilities, preflight=preflight)


def open_training(config, *, resume=None, **staged):
    from particle_sliders.lifecycle import read_packet
    validate_options(config.options)
    packet = deepcopy(resume) if isinstance(resume, dict) else read_packet(resume) if resume is not None else None
    construction = construction_for(config, packet)
    session = BACKEND.open(config, operation="resume" if packet is not None else "train", construction=construction, **staged)
    try:
        if packet is not None:
            session.restore(packet)
        return session
    except BaseException as primary:
        _restore_all([session.close], primary=primary)
        raise

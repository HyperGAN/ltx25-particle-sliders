"""Actual local Gemma4/LTX2 components, complete packing and AV call contract."""
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import torch
from particle_sliders.pg09_backends import FrozenResponseAdapter
from .contracts import PROTOCOL, SOURCE_PROFILE, file_hash, verify_model


def map_tensors(value, device, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype if value.is_floating_point() and dtype is not None else value.dtype)
    if isinstance(value, dict):
        return {key: map_tensors(child, device, dtype) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(child, device, dtype) for child in value)
    return value


def check_sources():
    import diffusers
    import transformers
    roots = {"diffusers": Path(diffusers.__file__).parent, "transformers": Path(transformers.__file__).parent}
    pins = json.loads(Path(__file__).with_name("native-sources.json").read_text())
    for name, expected in pins.items():
        package, relative = name.split("/", 1)
        if file_hash(roots[package] / relative) != expected:
            raise ValueError("native operative dependency Source differs: " + name)
    return pins


class NativeRuntime:
    """Explicit two-device placement, ONE DiT; no device map/offload/quantization.

    Gemma/connectors/VAEs/vocoder and game trainables use text_device. The
    frozen DiT uses dit_device. Actual differentiable tensor copies bridge
    them; no placement helper changes the native token/position ordering.
    """
    def __init__(self, directory, *, model_lock_sha256, text_device="cpu", dit_device="cpu", precision="bfloat16"):
        from .component_loader import LOAD_PROFILE, load_pipeline
        if text_device not in ("cpu", "cuda:0", "cuda:1") or dit_device not in ("cpu", "cuda:0", "cuda:1"):
            raise ValueError("explicit supported native placements required")
        if precision not in ("float32", "bfloat16"):
            raise ValueError("FP32/BF16 only")
        self.device, self.dit_device, self.dtype = torch.device(text_device), torch.device(dit_device), getattr(torch, precision)
        self.identity = verify_model(directory, model_lock_sha256)
        self.pipe = None
        try:
            pins = check_sources()
            # Same complete original components and optional None choices;
            # move each full component before loading the next, never auto-map.
            self.pipe = load_pipeline(directory, dtype=self.dtype,
                text_device=self.device, dit_device=self.dit_device)
            self.transformer, self.text_encoder, self.connectors = self.pipe.transformer, self.pipe.text_encoder, self.pipe.connectors
            if any(getattr(module, "gradient_checkpointing", False) for owner in
                   (self.transformer, self.text_encoder, self.connectors) for module in owner.modules()):
                raise ValueError("gradient checkpointing is not enabled in this Source profile")
            if (self.pipe.vae_spatial_compression_ratio, self.pipe.vae_temporal_compression_ratio,
                    self.transformer.config.patch_size, self.transformer.config.patch_size_t,
                    self.transformer.config.in_channels, self.transformer.config.out_channels) != (32, 8, 1, 1, 128, 128):
                raise ValueError("actual native packing/config differs from bound model")
            self.identity.update({"source_profile": SOURCE_PROFILE, "native_sources": pins,
                "precision": precision, "text_device": str(self.device), "dit_device": str(self.dit_device),
                "loading_profile": LOAD_PROFILE,
                "versions": {key: version(key) for key in ("torch", "diffusers", "transformers", "particlegan")},
                "product_sources": {path.name: file_hash(path) for path in
                    [Path(__file__).with_name(name) for name in ("native.py", "adapter.py", "render.py", "cache.py", "contracts.py",
                                                               "critic.py", "initialization.py", "lifecycle.py", "component_loader.py")]},
                "numerics": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}})
        except BaseException:
            self.close()
            raise

    def tokenize(self, prompt, tokens):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("nonempty native prompt required")
        tokenizer = self.pipe.tokenizer
        old_side, old_pad = tokenizer.padding_side, tokenizer.pad_token
        try:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            full = tokenizer(prompt.strip(), add_special_tokens=True, truncation=False)["input_ids"]
            if len(full) > tokens:
                raise ValueError("prompt exceeds declared native sequence; no silent truncation")
            result = tokenizer(prompt.strip(), padding="max_length", max_length=tokens,
                truncation=False, add_special_tokens=True, return_tensors="pt")
            return {"input_ids": result.input_ids.to(self.device), "attention_mask": result.attention_mask.to(self.device)}
        finally:
            tokenizer.padding_side, tokenizer.pad_token = old_side, old_pad

    def features(self, tokens):
        result = self.text_encoder(**map_tensors(tokens, self.device), output_hidden_states=True, use_cache=False)
        # EXACT native LTX pipeline stack ALL Gemma layers, no hidden-layer
        # selection, pooling or feature sketch. Mask/padded rows stay explicit.
        return torch.stack(result.hidden_states, dim=-1).flatten(2, 3)

    def encode(self, tokens, *, features=None):
        features = self.features(tokens) if features is None else features
        return self.connectors(features, tokens["attention_mask"].to(self.device), padding_side="left")

    @contextmanager
    def native_pipeline_placement(self):
        """Placement-only forward bridge; official scheduler/AV algorithm intact."""
        original, marker = self.transformer.forward, object()
        saved = self.transformer.__dict__.get("forward", marker)
        def forward(*args, **kwargs):
            # Coordinates/masks/timesteps retain their ORIGINAL native dtype.
            result = original(*map_tensors(args, self.dit_device),
                              **map_tensors(kwargs, self.dit_device))
            return map_tensors(result, self.device)
        self.transformer.forward = forward
        try:
            yield
        finally:
            if saved is marker:
                del self.transformer.forward
            else:
                self.transformer.forward = saved

    def call(self, **native):
        native = map_tensors(native, self.dit_device)
        for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states", "audio_encoder_hidden_states"):
            native[key] = native[key].to(dtype=self.dtype)
        result = self.transformer(**native)
        return tuple(value.to(self.device) for value in result)

    def close(self):
        self.pipe = self.transformer = self.text_encoder = self.connectors = None


def held_positive_features(positive, neutral, plus_tokens, neutral_tokens):
    """Historical non-concept token-ID hold, BEFORE connector registers/reorder.

    Native tokenizer left-padding is retained. Only valid token positions are
    held; padding never becomes a concept. As in original apply_unused_hold,
    repeated shared IDs resolve to the last matching neutral position.
    """
    plus_ids, neutral_ids = plus_tokens["input_ids"][0], neutral_tokens["input_ids"][0]
    plus_mask, neutral_mask = plus_tokens["attention_mask"][0].bool(), neutral_tokens["attention_mask"][0].bool()
    concept = set(plus_ids[plus_mask].tolist()) - set(neutral_ids[neutral_mask].tolist())
    if not concept:
        raise ValueError("positive prompt has no concept token absent from neutral")
    indices = {int(token): index for index, token in enumerate(neutral_ids.tolist()) if bool(neutral_mask[index])}
    result = positive.clone()
    for index, token in enumerate(plus_ids.tolist()):
        if bool(plus_mask[index]) and token not in concept and token in indices:
            result[:, index] = neutral[:, indices[token]]
    return result


class NativeVideoResponse(FrozenResponseAdapter):
    family = SOURCE_PROFILE
    time_coordinates = "original native scaled scheduler timestep, no redivision; raw video velocity"

    def __init__(self, runtime, network, shape):
        self.runtime, self.transformer, self.shape = runtime, runtime.transformer, tuple(shape)
        # Public core fingerprints/RAW identities must see ALL actual frozen
        # native roots, including adapted TE/connectors, not just the DiT.
        self.model = torch.nn.ModuleDict({"text_encoder": runtime.text_encoder, "connectors": runtime.connectors,
                                          "transformer": runtime.transformer})
        super().__init__(network, model_identity=runtime.identity)

    def _velocity(self, condition, latent, timestep):
        c = condition
        if tuple(latent.shape) != self.shape or not torch.isfinite(latent).all():
            raise ValueError("actual full5D native latent required")
        video = c.get("video_context")
        if video is None:
            video, _, mask = self.runtime.encode(c["tokens"])
            if not torch.equal(mask, c["native"]["encoder_attention_mask"].to(mask.device)):
                raise ValueError("actual student video connector mask differs from frozen native mask")
        native = deepcopy(c["native"])
        native["hidden_states"] = self.runtime.pipe._pack_latents(latent, 1, 1)
        native["encoder_hidden_states"] = video
        native["timestep"] = timestep.reshape(1)
        native["sigma"] = timestep.reshape(1)
        native["return_dict"] = False
        result = self.runtime.call(**native)[0]
        _, _, t, h, w = self.shape
        result = self.runtime.pipe._unpack_latents(result, t, h, w, 1, 1).float()
        if result.shape != latent.shape or not torch.isfinite(result).all():
            raise ValueError("native full video velocity differs")
        return result

    def metadata(self):
        return dict(super().metadata(), protocol=PROTOCOL, source_profile=SOURCE_PROFILE,
            response_geometry=list(self.shape), audio_training="same frozen-neutral audio latents/conditioning/masks per row",
            finished_sampling="official jointly coupled audiovisual sampler; no invariant-audio claim",
            rank=4, alpha=4., target_names=list(self.network.target_names), particle_bank=None,
            fresh_initialization=getattr(self, "fresh_initialization", None))

"""Exact native cohort; no old UNI/dummy artifacts are reinterpreted."""
import hashlib
import json
import math
from pathlib import Path

BACKEND_NAME = "ltx25_native_video95"
CORE_REVISION = "a3767b8f1b70be7427811c2ca96c14824db46d3c"
CORE_VERSION = "0.14.1+convnd.1"
PG_VERSION = "0.9.5+convnd.1"
PG_SOURCE_PROFILE = "pg095-convnd-per-offset-v1"
PG_SOURCE_MANIFEST_SHA256 = "0542a3d40ed8d7eaae0c73bdcab52672920b6194b0a5d4cc2df95a89663ec9b2"
PG_SOURCES_SHA256 = "873bd86171eb1cc68bdb74570219590b7fd06d2cf54f70dd6a8d1aa075f9186e"
MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
MODEL_REVISION = "4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3"
PROTOCOL = "ltx25-native-full-video-velocity-frozen-neutral-audio-pg095-convnd-v2"
SOURCE_PROFILE = "ltx25-native-gemma4-te4-video-connectors-r4-a4-convnd-first-row-recovery-v4"
INITIALIZATION_POLICY = "ltx25-fresh-owned-QR-host-kaiming-zero-up-v1"
WEIGHT_POLICY = "raw-current-no-model-weight-EMA"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def geometry(options):
    value = {key: options.get(key, default) for key, default in
             (("height", 512), ("width", 768), ("frames", 17), ("text_tokens", 128), ("fps", 24.))}
    for key in ("height", "width"):
        if type(value[key]) is not int or value[key] < 64 or value[key] % 32:
            raise ValueError("native canvas must be a declared multiple of32; no resize/projection")
    if type(value["frames"]) is not int or value["frames"] < 9 or (value["frames"] - 1) % 8:
        raise ValueError("native temporal grid is8k+1 with at least two latent frames")
    if type(value["text_tokens"]) is not int or not 128 <= value["text_tokens"] <= 1024 or value["text_tokens"] % 128:
        raise ValueError("complete tokenizer sequence must fit an explicit128-register multiple, never truncate")
    if isinstance(value["fps"], bool) or not isinstance(value["fps"], (float, int)) or not math.isfinite(value["fps"]) or value["fps"] <= 0:
        raise ValueError("finite positive native frame rate required")
    if int(value["fps"]) != value["fps"]:
        raise ValueError("official encode_video uses integer fps; no silent media-rate rounding")
    return value


def response_shape(options):
    g = geometry(options)
    return (1, 128, (g["frames"] - 1) // 8 + 1, g["height"] // 32, g["width"] // 32)


def rate_overrides(options):
    result = {}
    for key in ("lr", "d_lr_mult", "prior_lr_mult"):
        if key in options:
            value = options[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("common rate overrides must be finite positive literal scalars")
            result[key] = value
    return result


def validate_options(options):
    allowed = {"model_dir", "model_lock_sha256", "text_device", "dit_device", "base_precision", "cache_dir",
               "cache_index_sha256", "height", "width", "frames", "text_tokens", "fps", "batch", "steps", "seed",
               "critic_width", "ema_decay", "lr", "d_lr_mult", "prior_lr_mult", "sigma", "response_scale"}
    if set(options) - allowed:
        raise ValueError("unknown native options; offload/device-map/checkpointing/quantization unsupported")
    if "ema_decay" not in options or options["ema_decay"] is not None:
        raise ValueError("explicit ema_decay=None required; no adapter, critic or bank weight EMA")
    geometry(options)
    rate_overrides(options)
    for key in ("model_dir", "cache_dir"):
        if not isinstance(options.get(key), str) or not Path(options[key]).is_dir():
            raise ValueError("explicit local staged model/cache directory required")
    for key in ("model_lock_sha256", "cache_index_sha256"):
        if not isinstance(options.get(key), str) or len(options[key]) != 64:
            raise ValueError("exact model/cache content authority required")
    for key in ("text_device", "dit_device"):
        if options.get(key, "cpu") not in ("cpu", "cuda:0", "cuda:1"):
            raise ValueError("explicit CPU or visible CUDA placement required")
    if options.get("base_precision", "bfloat16") not in ("float32", "bfloat16"):
        raise ValueError("only full FP32 or BF16 base, FP32 owned trainables")
    for key, default in (("batch", 2), ("steps", 32), ("critic_width", 8)):
        if type(options.get(key, default)) is not int or options.get(key, default) < 1:
            raise ValueError("positive declared integer " + key + " required")


def validate_core():
    from importlib.metadata import distribution, version
    pins = json.loads(Path(__file__).with_name("current_core.json").read_text())
    if version("particle-sliders-core") != CORE_VERSION or version("particlegan") != PG_VERSION:
        raise ValueError("this new native path requires the exact unreleased core0141+convnd/PG095+convnd cohort")
    root = Path(distribution("particle-sliders-core").locate_file("particle_sliders"))
    for name, expected in pins["checked_api_files"].items():
        if file_hash(root / name) != expected:
            raise ValueError("shared native API Source differs: " + name)


def verify_model(directory, expected):
    root = Path(directory).resolve(strict=True)
    lock = root / "native-model-lock.json"
    if file_hash(lock) != expected:
        raise ValueError("local native model lock differs")
    value = json.loads(lock.read_text())
    if (value.get("model_id"), value.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("requires this exact LTX2.5 source snapshot")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Root staging must bind every native runtime file")
    required = {"model_index.json", "transformer/config.json", "connectors/config.json", "text_encoder/config.json",
                "vae/config.json", "audio_vae/config.json", "vocoder/config.json", "scheduler/scheduler_config.json"}
    for name, expected_file in files.items():
        relative = Path(name)
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise ValueError("staged model paths must remain inside one immutable snapshot")
        if path.stat().st_size != expected_file["bytes"] or file_hash(path) != expected_file["sha256"]:
            raise ValueError("staged native file differs: " + name)
    if not required <= files.keys():
        raise ValueError("native runtime configs missing from model authority")
    captures = Path(__file__).with_name("model-configs")
    for name in required:
        if json.loads((root / name).read_text()) != json.loads((captures / name).read_text()):
            raise ValueError("actual runtime config differs from captured primary source: " + name)
    # Weight and tokenizer files are required as well, not just config JSONs.
    for folder in ("transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder"):
        if not any(name.startswith(folder + "/") and name.endswith(".safetensors") for name in files):
            raise ValueError("missing declared staged weights for " + folder)
    if not any(name.startswith("tokenizer/") for name in files):
        raise ValueError("native tokenizer files must be staged and content-bound")
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "model_lock_sha256": expected, "files": files}

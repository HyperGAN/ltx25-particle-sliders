"""Original three-decoder tail, using the durable PREBRANCH AV artifact only."""
from pathlib import Path
from types import SimpleNamespace
import ast
import gc
import importlib
import json

from .admission import DECODER
from .api import GEOMETRY, runtime_policy
from .io import atomic_json, fsync_file
from .sources import ROOT, pinned, read_json, record, validate_factory, BOUNDARY_NATIVE_SOURCE


def _decoder_files(options, phase):
    from ps_ltx25_native95.contracts import MODEL_ID, MODEL_REVISION
    root = Path(options["model_dir"]).absolute()
    lock = record(root / "native-model-lock.json")
    if lock["sha256"] != options["model_lock_sha256"] or lock["sha256"] != phase["model_identity"]["model_lock_sha256"]:
        raise ValueError("same original model lock as completed sampler required")
    value = read_json(lock)
    if ((value["model_id"], value["revision"]) != (MODEL_ID, MODEL_REVISION)
            or value["files"] != phase["model_identity"]["files"]):
        raise ValueError("same complete model snapshot authority required")
    # Only decoder payloads are consumed in this phase, never Gemma/DiT weights.
    for folder in ("vae", "audio_vae", "vocoder"):
        if not any(x.startswith(folder + "/") and x.endswith(".safetensors") for x in value["files"]):
            raise ValueError("whole original decoder weights required")
        for name, expected in value["files"].items():
            if name.startswith(folder + "/"):
                relative = Path(name)
                path = (root / relative).resolve(strict=True)
                if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve(strict=True)):
                    raise ValueError("decoder paths must stay within the immutable model root")
                pin = record(path)
                if pin["bytes"] != expected["bytes"] or pin["sha256"] != expected["sha256"]:
                    raise ValueError("original decoder file differs: " + name)
        captured = ROOT.parent / "ps_ltx25_native95/model-configs" / folder / "config.json"
        if json.loads((root / folder / "config.json").read_text()) != json.loads(captured.read_text()):
            raise ValueError("captured original decoder config required")
    return root


def _tail_source(native_module):
    source = record(native_module.__file__)
    if {k:source[k] for k in ("bytes", "sha256")} != BOUNDARY_NATIVE_SOURCE:
        raise ValueError("exact original pipeline Source required")
    tree = ast.parse(Path(native_module.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LTX2Pipeline")
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    start = next(i for i, n in enumerate(call.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "output_type == 'latent'")
    path = ROOT / "original_decode_tail.py"
    code = path.read_text()
    body = ast.parse(code).body[0].body
    if [ast.dump(n, include_attributes=False) for n in body] != [ast.dump(n, include_attributes=False) for n in call.body[start:]]:
        raise ValueError("only exact original decoder-tail statements/order admitted")
    return path, code


def decode_saved_boundary(phase_pin, *, admission, options, output):
    check, torch = runtime_policy(admission, DECODER)
    from ps_ltx25_native95.native import check_sources
    from ps_ltx25_native95.contracts import validate_core
    from ps_ltx25_native95.render import observation, export_video
    from diffusers import LTX2Pipeline
    from diffusers.models.autoencoders import AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    from .typed_boundary import validate_prebranch_payload
    validate_core()
    check_sources()
    phase = read_json(phase_pin)
    if (phase["captured_factory"] != validate_factory()
            or phase["producer_source"]["sha256"] != record(ROOT / "api.py")["sha256"]):
        raise ValueError("exact declared public producer/factory Source required")
    if (phase["schema"] != "LTX25-owned-public-latent-phase-v1"
            or phase["sampling_output_type"] != "latent" or phase["future_decoder_output_type"] != "pil"
            or phase["complete_native_joint_sampling"] is not True or phase["native_owner_closed"] is not True
            or phase["complete_finished_media"] is not False or phase["native_calls"] != 1 or phase["decoder_calls"] != 0
            or phase["returned_latents_used_as_decode_input"] is not False
            or phase["request"]["geometry"] != GEOMETRY):
        raise ValueError("genuine complete original latent phase required; no returned-latent retagging")
    if admission.plan["physical_GPU_UUIDs"] != phase["execution_plan"]["physical_GPU_UUIDs"][:1]:
        raise ValueError("same physical sampler CUDA0 required; no saved RNG/device remap")
    native = read_json(phase["native_receipt"])
    if (native["artifact"] != phase["artifact"] or native["complete_native_joint_sampling"] is not True
            or native["decoder_calls_started"] != 0 or native["sampling_output_type"] != "latent"
            or native["future_decoder_output_type"] != "pil" or native["complete_finished_media_claimed"] is not False
            or native["actual_runtime_identity"] != phase["model_identity"]):
        raise ValueError("exact original saved boundary/phase identity required")
    directory = _decoder_files(options, phase)
    native_module = importlib.import_module(LTX2Pipeline.__module__)
    tail_path, code = _tail_source(native_module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # A known or unknown prior tail attempt is not automatic permission to replay.
    atomic_json(Path(phase_pin["path"]).with_name("decode-pil-claim.json"),
                dict(phase=phase_pin, owner_receipt=admission.owner_receipt, output=str(output.absolute()), automatic_retry=False))
    check()
    payload = torch.load(pinned(phase["artifact"]), map_location="cpu", weights_only=True)
    typed = validate_prebranch_payload(payload, torch=torch)
    placed = {}
    pipe = runtime = module = result = None
    try:
        for name, cls in (("vae", AutoencoderKLLTX2Video), ("audio_vae", AutoencoderKLLTX2Audio), ("vocoder", LTX2VocoderWithBWE)):
            check()
            module = cls.from_pretrained(str(directory / name), local_files_only=True, low_cpu_mem_usage=True,
                                         device_map=None, use_safetensors=True, torch_dtype=torch.bfloat16)
            module.requires_grad_(False).eval().to(device="cuda:0", dtype=torch.bfloat16)
            placed[name] = module
            module = None
            gc.collect()
            check()
        pipe = LTX2Pipeline(scheduler=None, text_encoder=None, tokenizer=None, connectors=None, transformer=None,
                            vae=placed["vae"], audio_vae=placed["audio_vae"], vocoder=placed["vocoder"],
                            prompt_enhancer=None, processor=None, duration_head=None)
        if (any(getattr(pipe, k) is not None for k in ("scheduler", "text_encoder", "tokenizer", "connectors", "transformer"))
                or pipe.vae.config.timestep_conditioning is not False
                or (pipe.vae_spatial_compression_ratio, pipe.vae_temporal_compression_ratio) != (32, 8)):
            raise ValueError("original three-decoder-only ownership/geometry required")
        runtime = SimpleNamespace(pipe=pipe, device=torch.device("cuda:0"), dit_device=torch.device("cuda:0"))
        namespace = dict(vars(native_module))
        exec(compile(code, str(tail_path), "exec"), namespace)
        decode = namespace["decode_original_tail"]
        check()
        with torch.no_grad(), observation(runtime):
            torch.set_rng_state(typed["CPU_RNG"])
            torch.cuda.set_rng_state(typed["CUDA_RNG"]["0"], 0)
            generator = torch.Generator(device="cuda:0")
            generator.set_state(typed["generator_state"])
            decoded = decode(pipe, typed["video"].to("cuda:0"), typed["audio"].to("cuda:0"), generator,
                             SimpleNamespace(dtype=torch.bfloat16), torch.device("cuda:0"),
                             typed["decode_timestep"], typed["decode_noise_scale"], 1, "pil", True)
        result = dict(frames=decoded.frames, audio=decoded.audio, sample_rate=int(pipe.vocoder.config.output_sampling_rate),
                      fps=24., native_joint_AV=True, audio_invariant=False, **{k:phase["request"][k] for k in ("seed", "strength")})
        if (len(result["frames"][0]) != 17 or any(f.size != (768, 512) for f in result["frames"][0])
                or type(result["audio"]) is not torch.Tensor or not result["audio"].numel()
                or not torch.isfinite(result["audio"]).all() or result["sample_rate"] != 48000):
            raise ValueError("complete original17-frame/native48k AV required")
        check()
        path = output / "finished.mp4"
        export_video(result, path)
        fsync_file(path)
        receipt = dict(schema="LTX25-owned-original-decoder-tail-v1", phase=phase_pin,
                       complete_native_receipt=phase["native_receipt"], complete_native_artifact=phase["artifact"],
                       output=record(path), request=phase["request"], model_identity=phase["model_identity"],
                       owner_receipt=admission.owner_receipt, execution_plan=admission.plan,
                       native_sampler_calls=0, original_decoder_tail_calls=1, new_teacher_or_training_calls=0,
                       sampling_output_type="latent", consumer_output_type="pil", tail_source=record(tail_path),
                       saved_artifact_retagged=False, returned_denormalized_latents_used=False,
                       whole_AV_catastrophic_status="unmeasured", effect_score=None, current_eligibility=None)
    finally:
        decoded = result = runtime = pipe = module = None
        placed.clear()
        gc.collect()
    if record(phase["artifact"]["path"]) != phase["artifact"]:
        raise RuntimeError("original saved boundary changed during decode")
    check()
    receipt["decoder_owner_closed"] = True
    atomic_json(output / "decoder-result.json", receipt)
    return receipt

"""Complete original component loading with explicit sequential placement.

This changes allocation ownership/order only. It is not CPU offload, a device
map, quantization, a truncated model, or a native algorithm implementation.
"""
import gc
from pathlib import Path

LOAD_PROFILE = "ltx25-original-components-sequential-CPU-load-explicit-place-v1"
# Largest full component first; each is already placed before the next load.
MODEL_COMPONENTS = ("transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder")


def component_classes():
    from diffusers import LTX2Pipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
    from diffusers.models.condition_embedders import LTX2TextConnectors
    from diffusers.models.transformers import LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    from transformers import Gemma4UnifiedForConditionalGeneration, GemmaTokenizer
    return dict(pipeline=LTX2Pipeline, scheduler=FlowMatchEulerDiscreteScheduler,
        vae=AutoencoderKLLTX2Video, audio_vae=AutoencoderKLLTX2Audio,
        text_encoder=Gemma4UnifiedForConditionalGeneration, tokenizer=GemmaTokenizer,
        connectors=LTX2TextConnectors, transformer=LTX2VideoTransformer3DModel,
        vocoder=LTX2VocoderWithBWE)


def load_pipeline(directory, *, dtype, text_device, dit_device):
    """One complete CPU component at a time, then its final admitted device.

    Caller has already authenticated every model file and operative dependency
    Source. No automatic placement or offload hook is installed. Components
    retain the same original classes, BF16/FP32 request and pretrained files.
    """
    root = Path(directory)
    classes = component_classes()
    placed = {}
    module = None
    try:
        for name in MODEL_COMPONENTS:
            options = dict(local_files_only=True, low_cpu_mem_usage=True, device_map=None,
                           use_safetensors=True)
            # Same dtype spelling chosen by pinned pipeline_loading_utils for
            # Transformers>=4.56 versus Diffusers ModelMixin components.
            options["dtype" if name == "text_encoder" else "torch_dtype"] = dtype
            module = classes[name].from_pretrained(str(root / name), **options)
            target = dit_device if name == "transformer" else text_device
            module.requires_grad_(False).eval().to(device=target, dtype=dtype)
            placed[name] = module
            module = None
            gc.collect()  # Release completed loader temporaries before the next full component.
        placed["scheduler"] = classes["scheduler"].from_pretrained(str(root / "scheduler"), local_files_only=True)
        placed["tokenizer"] = classes["tokenizer"].from_pretrained(str(root / "tokenizer"), local_files_only=True)
        pipe = classes["pipeline"](**placed, prompt_enhancer=None, processor=None, duration_head=None)
        if any(getattr(pipe, name) is not owner for name, owner in placed.items()):
            raise ValueError("native pipeline must own exactly the complete loaded component objects")
        return pipe
    except BaseException:
        module = None
        placed.clear()
        gc.collect()
        raise

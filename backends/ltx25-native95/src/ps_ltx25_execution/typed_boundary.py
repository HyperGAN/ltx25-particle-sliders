"""Read-only typed boundary adapter; callable only after external owner admission.

No imports or operations occur on Torch here. The actual decoder supplies its
already admitted module. This function does not load/read/mutate payloads or
cast/denormalize tensors; the original pinned decoder tail owns all math.
"""


def require(value,message):
    if not value:raise RuntimeError(message)


def validate_prebranch_payload(payload,*,torch):
    expected={'schema','video_latents_unpacked_before_original_dtype_cast_and_denormalize',
        'audio_latents_already_original_denormalized_and_unpacked','generator_state','CPU_RNG','CUDA_RNG',
        'prompt_dtype','decode_timestep','decode_noise_scale','output_type'}
    require(type(payload)is dict and set(payload)==expected
        and payload['schema']=='LTX25-complete-native-AV-before-original-decode-v1',
        'Exact original saved prebranch payload schema, not a returned latent tensor')
    video=payload['video_latents_unpacked_before_original_dtype_cast_and_denormalize']
    audio=payload['audio_latents_already_original_denormalized_and_unpacked']
    require(type(video)is torch.Tensor and tuple(video.shape)==(1,128,3,16,24)
        and video.dtype is torch.float32 and video.device.type=='cpu'
        and type(audio)is torch.Tensor and tuple(audio.shape)==(1,8,18,16)
        and audio.dtype is torch.float32 and audio.device.type=='cpu'
        and torch.isfinite(video).all()and torch.isfinite(audio).all(),
        'Complete original finite CPU FP32 boundary domains, no crop or repeated unpack')
    require(payload['output_type']=='latent'and payload['prompt_dtype']=='torch.bfloat16'
        and payload['decode_timestep']==0. and payload['decode_noise_scale']is None,
        'Truthful original sampling latent plus frozen PIL consumer options, never retag payload as PIL')
    require(type(payload['CUDA_RNG'])is dict and set(payload['CUDA_RNG'])=={'0','1'}
        and all(type(x)is torch.Tensor and x.dtype is torch.uint8 and x.device.type=='cpu'
            for x in(payload['generator_state'],payload['CPU_RNG'],*payload['CUDA_RNG'].values())),
        'Exact retained per-call/CPU/physicalCUDA0+1 state, no RNG remap')
    return dict(video=video,audio=audio,saved_sampling_output_type='latent',consumer_output_type='pil',
        generator_state=payload['generator_state'],CPU_RNG=payload['CPU_RNG'],CUDA_RNG=payload['CUDA_RNG'],
        decode_timestep=payload['decode_timestep'],decode_noise_scale=payload['decode_noise_scale'],
        saved_artifact_retagged=False,returned_denormalized_video_used=False)

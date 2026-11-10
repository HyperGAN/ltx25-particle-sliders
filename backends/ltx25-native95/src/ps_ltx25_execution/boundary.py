"""Execution-owned durable AV snapshot; original native decode statements exact.

The Source-pinned original __call__ receives ONE observer call before its decode
branch. No native operation/cast/order/scheduler or model Source is replaced.
This is Source instrumentation, not a numerical-equivalence/fit proof.
"""
from contextlib import contextmanager
import ast
import hashlib
import json
from pathlib import Path
import sys
import textwrap


def record(path):
    path = Path(path).absolute()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def require(value, message):
    if not value:
        raise RuntimeError(message)


@contextmanager
def capture_before_native_decode(runtime, directory, expected_source, check, *, after_capture=None):
    """One owned pipeline object only; restore its class on success/error."""
    import torch
    from ps_ltx25_native95.first_rows import save_new_tensor
    from ps_ltx25_native95.cache import atomic_json
    from ps_ltx25_native95.render import DISTILLED_SIGMAS
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    pipe = runtime.pipe
    original_class = type(pipe)
    require(original_class.__name__ == 'LTX2Pipeline', 'exact original native pipeline, no prior wrapper')
    native_module = sys.modules[original_class.__module__]
    actual_source = record(native_module.__file__)
    require(all(actual_source[k] == expected_source[k] for k in ('bytes', 'sha256')),
            'exact Source-pinned original LTX2 pipeline body')
    raw = Path(native_module.__file__).read_text()
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LTX2Pipeline')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
    start = min([function.lineno] + [n.lineno for n in function.decorator_list])
    original = textwrap.dedent('\n'.join(raw.splitlines()[start - 1:function.end_lineno])) + '\n'
    marker = '    if output_type == "latent":\n'
    inserted = ('    _LTX25_DURABLE_AV_BEFORE_DECODE(self, latents, audio_latents, generator, '
                'prompt_embeds.dtype, decode_timestep, decode_noise_scale, output_type)\n\n')
    require(original.count(marker) == 1, 'one original complete-latent/decode boundary')
    generated = original.replace(marker, inserted + marker)
    require(generated.replace(inserted, '') == original, 'only the declared observer statement is inserted')
    from .sources import BOUNDARY_NATIVE_SOURCE, BOUNDARY_GENERATED_SOURCE
    require({k:actual_source[k]for k in ('bytes','sha256')}==BOUNDARY_NATIVE_SOURCE
            and len(generated.encode())==BOUNDARY_GENERATED_SOURCE['bytes']
            and hashlib.sha256(generated.encode()).hexdigest()==BOUNDARY_GENERATED_SOURCE['sha256'],
            'exact insert-only captured native boundary function')
    generated_path = directory / 'actual-native-call-with-durable-observer.py'
    with generated_path.open('x') as stream:
        stream.write(generated)
        stream.flush()
        __import__('os').fsync(stream.fileno())
    calls = []
    def save(self, video, audio, generator, prompt_dtype, decode_timestep, decode_noise_scale, output_type):
        check()
        require(self is pipe and not calls and output_type == 'latent', 'one owned native latent-return boundary')
        require(isinstance(video, torch.Tensor) and tuple(video.shape) == (1, 128, 3, 16, 24)
                and isinstance(audio, torch.Tensor) and audio.ndim == 4 and audio.numel() > 0
                and torch.isfinite(video).all() and torch.isfinite(audio).all(),
                'complete original native video/audio latents, no cropped substitute')
        require(self._interrupt is False and len(self.scheduler.timesteps) == len(DISTILLED_SIGMAS)
                and float(self._current_timestep) == float(self.scheduler.timesteps[-1]),
                'all original eight native joint sampler steps completed before decode')
        require(isinstance(generator, torch.Generator), 'actual original per-call generator')
        saved = save_new_tensor(directory / 'complete-native-AV-before-decode.pt', dict(
            schema='LTX25-complete-native-AV-before-original-decode-v1',
            video_latents_unpacked_before_original_dtype_cast_and_denormalize=video,
            audio_latents_already_original_denormalized_and_unpacked=audio,
            generator_state=generator.get_state(), CPU_RNG=torch.get_rng_state(),
            CUDA_RNG={str(d): torch.cuda.get_rng_state(d) for d in (0, 1)},
            prompt_dtype=str(prompt_dtype), decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale, output_type=output_type))
        metadata = dict(schema='LTX25-complete-native-AV-before-decode-receipt-v1',
            complete_native_joint_sampling=True, decoder_calls_started=0, artifact=saved,
            video_shape=list(video.shape), video_dtype=str(video.dtype),
            audio_shape=list(audio.shape), audio_dtype=str(audio.dtype),
            actual_original_native_source=actual_source, declared_original_native_source=expected_source,
            actual_execution_observer_source=record(__file__), generated_native_call_source=record(generated_path),
            declared_insert_only_inverse_exact=True, numerical_equivalence_proof=False,
            actual_runtime_identity=runtime.identity, no_native_forward_or_sampling_replay=True,
            sampling_output_type=output_type,future_decoder_output_type='pil',
            original_latent_return_branch_continues_unchanged=True,complete_finished_media_claimed=False)
        atomic_json(directory / 'complete-native-AV-before-decode.json', metadata)
        calls.append(metadata)
        check()
        if after_capture is not None:
            after_capture(metadata)  # placement only AFTER exact complete latents are durable
    namespace = dict(vars(native_module))
    namespace['__name__'] = 'LTX25_execution_owned_durable_native_decode'
    namespace['_LTX25_DURABLE_AV_BEFORE_DECODE'] = save
    exec(compile(generated, str(generated_path), 'exec'), namespace)
    owned_class = type('LTX25OwnedDurableDecodePipeline', (original_class,),
                       dict(__call__=namespace['__call__'], __module__=__name__))
    pipe.__class__ = owned_class
    try:
        yield calls
        require(len(calls) == 1, 'complete joint latents must be durable before any successful native decode')
    finally:
        pipe.__class__ = original_class

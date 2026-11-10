# LTX2.5 native owner API and matched32 results

This separate distribution stages the captured full native LTX2.5 path and a
new owner-injected execution interface. It preserves the twelve Python files
and ten JSON files in `ps_ltx25_native95` byte for byte from source commit
`edcf9c3b121191f95e2b0153bf5e7797afb6ca1a`. The historical root UNI implementation,
dummy backend and their commands are separate and unchanged.

The new `ps_ltx25_execution` package does not allocate devices, reserve memory,
download weights or create an execution owner. Its `ltx25-native95` console
deliberately refuses heavy commands without an owner. A runner calls
`ps_ltx25_execution.cli.main(argv, admission=owner)` or the Python functions
below. `check` and `--help` require no framework imports. A receipt string or
configuration flag alone does not admit execution.

The local offline wheel build, owned-target installation, four new software
checks and actual installed console owner-refusal check **passed** on2026-11-10.
The installed wheel preserved all12 factory Python and10 factory JSON bodies.
These checks imported no framework, loaded no model and used no GPU. They do
not establish public dependency availability or full native execution through
this installed wheel. The captured native previews and the completed
[matched RAW32 exact tie](LEARNING-RATES.md) have their own provenance. No weights
or ordinary Generate feature are released here.

The demonstrated software-only packaging commands, after supplying Python3.12
with pip26.2.2 and setuptools84.2.1, are:

```bash
python3.12 -m pip wheel --no-deps --no-index --no-build-isolation --no-cache-dir \
  --wheel-dir ./dist ./backends/ltx25-native95
python3.12 -m pip install --no-deps --no-index --no-compile \
  ./dist/particle_sliders_ltx25_native95-0.1.0a2+convnd.1.owner1-py3-none-any.whl
```

Use an environment and output directories you own. These commands install this
software only; the exact runtime prerequisites below must already be supplied
for heavy work. No complete public runtime resolver command is claimed.

## Exact dependencies and assets

The required profiles are core `0.14.1+convnd.1` at
checked API baseline `a3767b8f1b70be7427811c2ca96c14824db46d3c` (prepared
metadata-only derivative `a43208105be80234b2d331ef80b5583cffefc4d0`) and ParticleGAN
`0.9.5+convnd.1` at `37f1b23281c551a196f9e508be691a6cf724c858`,
profile `pg095-convnd-per-offset-v1`. They are explicitly separate, unreleased
profiles. Stock core/PG or a newer release is not a fallback. Exact public
artifact availability has not been established by this proposal. The
[separate dependency packaging proposal](../../dependency-packaging/ltx25-convnd/README.md)
pins the exact PG local version in core metadata while retaining all checked
Python and manifest bytes. Do not advertise an install command until those
actual artifacts are available and their packaging has been verified.

The runtime requires Torch `2.9.1+cu128`, the declared Diffusers source commit
and Transformers `5.15.0`; the factory verifies exact operative source hashes,
not merely version labels. The frozen-storage helper also verifies its pinned
Torch checkpoint/autograd implementation. See the packaged `current_core.json`,
`native-sources.json` and execution `storage-runtime-sources.json`.

Stage `Lightricks/LTX-2.5-Diffusers` revision
`4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3` in your own immutable directory.
Supply the exact `native-model-lock.json` SHA and all original files listed by
that lock. The original factory validates whole files and captured configs;
there is no download or replacement-model path. Decoder-only execution consumes
only the three original decoder components under that same snapshot authority.

## An owner supplied by your runner

Use `OwnedAdmission(plan, owner_receipt, check_live)`. `plan` has exactly:

```python
{
    "profile": STORAGE,  # or DECODER for the separate tail
    "CUDA_VISIBLE_DEVICES": runner_visible_devices,
    "physical_GPU_UUIDs": runner_physical_gpu_uuids,
}
```

`owner_receipt` is the actual runner's `{path, bytes, sha256}` FileRecord.
`check_live()` must affirm `True` and enforce actual ownership. The runner must
establish the exact environment, authorize framework import, maintain caps,
live host/GPU floors, device leases, cancellation/known-failure checks and
independent lifetime monitoring, then perform true teardown. The library does
not issue or verify a private coordinator grant, create CURRENT entries, or
replace that lifecycle. Establish deterministic algorithms, FP32 default dtype
and disabled TF32 before entering this interface; it refuses a different
captured numerical policy instead of silently changing global settings.

`STORAGE` uses two visible devices: text encoder/connectors and PG on logical
CUDA0, complete DiT arithmetic on CUDA1, original DiT blocks16–31 stored on CPU
between calls, and inactive original video VAE/audio VAE/vocoder on CPU. All
original48 DiT blocks remain; BF16 weights and Parameter objects are retained.
Original nonreentrant checkpointing spans each complete training step and
backward, then restores exact attributes before FULL/RAW commits. CPU storage
is not CPU model arithmetic, quantization, a device map or a smaller model.

`DECODER` uses one visible device, whose physical UUID must equal the sampler's
original CUDA0. The saved generator/RNG is not remapped. The caller's memory
plan must cover the actual phase; there is no fit guarantee. Captured factory
identity and actual storage/checkpoint placement are recorded separately.

The original backend registrations retain their original objects and names
for RAW identity. They are low-level interfaces, not owner-enforcing proxies.
Use this owned API; generic direct dispatch into the original inference backend
does not manufacture admission. The wrapper leaves the original code files and
strict core factory/capability/network comparisons intact.

## Cache reuse, fresh training and exact FULL resume

```python
def train_missing(owner, config, new_output, *, full=None, stop_after=32):
    from ps_ltx25_execution import train
    return train(config, new_output, admission=owner,
                 resume=full, stop_after=stop_after)
```

`config` is the original `SessionConfig`: backend `ltx25_native_video95`, an
explicit cohort, local model/cache roots and exact lock/index SHA values.
The captured profile uses full768×512 video,17 frames/24fps,128 text registers,
rank4/alpha4, batch2, seed7 and horizon128. Explicitly use `ema_decay=None`.
The actual rate tuples tested at32 are G0.012/D-mult1.5 and G0.001/D-mult18,
both actual D0.018000000000000002. There is no particle bank;
`prior_lr_mult=2.5` is unused (nominal0.03 versus0.0025). The completed one-case
comparison tied exactly: each ON mean0.5 minus shared OFF mean0.4117647058823529
is gain0.08823529411764708. Both recipes remain provisional; no unique rate,
automatic override or default is selected. See [the complete scope and portable
values](LEARNING-RATES.md).

The factory binds fresh/restore intent before optimizer construction. Fresh
initialization applies its scoped policy once, before optimizers, preserving
pretrained weights and zero-UP leaves. Resume uses the authentic typed FULL
without an initializer, teachers or replayed committed updates. Entire original
manifest, game, optimizers, stream/RNG and checkpoint checksum equality precede
checkpoint execution. The wrapper writes a genuine initialized FULL0/RAW0 or
the exact restored entry, then FULL/RAW/COMMIT after every returned complete
step. `stop_after` changes the stopping budget, never the saved horizon. Save
callbacks run after the checkpoint scope has exited. A failed/unknown step
cannot be automatically retried by that wrapper. If cleanup itself fails, the
owner retains cleanup handles/error history and refuses steps or commits. A
later `close()` retries only incomplete cleanup roles; it clears handles and
marks closed only when cleanup succeeds. This does not retry any ML operation.

For an existing measured cohort, reuse its completed cache and exact session
paths. Relocating this package does not migrate a FULL's configuration, paths,
runtime identity or source locks. The measured two initial rows/four teachers
are distinct from the generic cache builder's full eight-step trajectories.
`cache(options, pairs, new_output, admission=owner, seeds=(7,19))` calls that
original generic builder for explicitly new work; it does not recreate the
private first-row lineage or imply the measured two-row recipe. An interrupted
or completed cache is not automatically rebuilt. Training and RAW inference
never call the cache builder.

## Split native sampling and original decoding

```python
def sample_once(owner, raw, local_options, new_output, prompt):
    from ps_ltx25_execution import render_latents
    return render_latents(raw, admission=owner, options=local_options,
                          prompt=prompt, seed=7, strength=1.0, output=new_output)

def decode_once(owner, phase_file_record, model_options, new_output):
    from ps_ltx25_execution import decode_saved_boundary
    return decode_saved_boundary(phase_file_record, admission=owner,
                                 options=model_options, output=new_output)
```

Sampling `local_options` contains exactly the original local model/lock and
text0/DiT1/BF16 declaration. Strict original RAW loading happens first. The
original jointly coupled eight-step AV sampler uses its unchanged sigmas,
CFG1/STG0/modality scale1, prompt, seed, geometry and generator. Public
`output_type="latent"` is read only after the last joint step. A run-owned
insert-only observer durably saves the **prebranch FP32 video** and already
denormalized/unpacked audio plus CPU/CUDA/per-call RNG, then closes resting block
wrappers. The original latent branch returns normally. Its denormalized video
return is discarded; it is not the decoder artifact. No text-encoder CPU copy,
VAE decode or MP4 export happens in the sampling phase.

Only a complete `latent-phase.json` manifest admits the decoder API. A durable
boundary without that completion manifest remains partial work for the runner
to reconcile explicitly. The decoder authenticates the artifact, original
model snapshot/three decoder files, physical CUDA0 and complete typed domains.
It loads only the original video VAE, audio VAE and vocoder, executes the exact
original PIL tail and native MP4 export, with no Gemma, connectors, DiT, adapter,
teacher, training or repeated sampler call. It writes a new claim before the
tail; an existing claim, including an unknown outcome, refuses automatic replay.
Original source files/artifact bytes remain unchanged. Output is the full native
17-frame video with48k joint audio; audio is not claimed invariant under ON.

The frozen visual warm-sunset metric is provisional, not whole-AV catastrophic
coverage. Missing/uncertain eligibility remains unmeasured. No PASS, effect
claim, LR promotion, public default, weights or ordinary Generate follows from
these wrappers or from publishing a preview.

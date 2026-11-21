# Native LTX 2.5

This package runs the full LTX 2.5 video response with Particle Sliders
core 0.15.2 and the released ParticleGAN 0.11.1. Training and inference use
RAW weights. The critic computes native Conv3d video and Conv1d context
features. It stores each convolution's kernel in the two-axis layout that
stock per-offset DualNorm accepts. Each native offset remains a separate
channel matrix. The forward pass views the same weights in their native
shape.

Use a fresh cohort with this package. Its optimizer and critic checkpoint
shapes are part of its source identity. The original checkout retains the
loader and dependencies for earlier FULL and RAW artifacts.
[Original native checkout](https://github.com/HyperGAN/ltx25-particle-sliders/blob/ffc51b2df3c0644c6cd763fda8cb43f55c59436b/backends/ltx25-native95/README.md)

## Install

Use an owned Python 3.11–3.14 environment and install Torch 2.9.1 for
your hardware first. The published execution profile uses Torch
`2.9.1+cu128` and checks the captured Diffusers, Transformers and Torch
source files at runtime.

```sh
python -m pip install ./backends/ltx25-native
ltx25-native check
```

Stage the exact LTX 2.5 Diffusers model revision
`4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3` and supply a local
`native-model-lock.json` with the SHA-256 and size of each required file.
The library validates the snapshot, authenticated teacher cache, and native
packing before training. The console can check source identity without
loading the model. The execution API requires a runner-supplied
`OwnedAdmission(plan, owner_receipt, check_live)`.

## Train or resume

Use backend `ltx25_native_video` and explicitly set `ema_decay=None`
in your `SessionConfig`. Provide the model and cache directories and their
lock hashes. The owned training profile uses two GPUs, a 768 by 512 pixel
video at 24 fps with 17 frames, and the original rank-4 text and video
connector adapters. The runner owns device leases, memory limits, live
checks, termination, and resource cleanup.

```python
from ps_ltx25_execution import train

result = train(config, output_directory, admission=owner,
               resume=full_checkpoint, stop_after=32)
```

Pass `resume=None` for fresh work. Scoped initialization runs before
optimizers and leaves the frozen model and zero-UP adapter weights intact.
Resume checks the exact FULL packet, optimizer state, and RNG. The API
writes FULL and RAW at each successfully returned update. Keep the target
training horizon in the configuration. `stop_after` limits this invocation.

The original matched RAW32 results describe their captured cohort. See the
[measured rates and scope](https://github.com/HyperGAN/ltx25-particle-sliders/blob/ffc51b2df3c0644c6cd763fda8cb43f55c59436b/backends/ltx25-native95/LEARNING-RATES.md).

## Render and decode

`render_latents` loads an exact RAW export and runs the native joint audio
and video sampler. It saves the pre-decode tensors and writes a completed
latent-phase manifest. `decode_saved_boundary` consumes that manifest
and loads only the original video VAE, audio VAE, and vocoder on the
original sampling device. Both APIs use the runner's actual admission.
The ownerless console refuses model-loading commands.

For an open strict RAW render session, call
`ps_ltx25_native.scheduled_render.render(session, prompt=..., step_gains=...)`.
Use one gain per native distilled interval. The adapter is in the Gemma text
encoder and connectors, so the add-on builds the native audio and video
conditioning for each distinct strength and selects it at the denoiser call.
The result includes `strength_schedule` with the actual sigmas and applied
strengths. A CPU dispatch fixture passed.

The archived Warm Sunset RAW3 seed-7 clip gained +0.00293 with early-heavy
strength, +0.00166 with constant, and +0.00029 with late-heavy at nominal
strength 2/3. The original native route reproduced the constant audio and video
latents and finished MP4 exactly. All three arms passed the whole-video and
stereo integrity checks. This is a single-seed result. The held-out
comparison is pending.
[Seed-7 evidence](docs/warm-sunset-raw3-seed7-scheduled-strength.json)

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

### Activation-dependent strength

`ps_ltx25_native.adaptive_render.render` can gate the saved RAW edit on
the same single-conditional distilled sampler. It reads one live packed
video/audio pair before each denoiser call. The probe input is the
concatenation of the FP32 token means of the video and audio latents, in
that order. A decision selects either the OFF conditioning or the fixed
nominal-strength conditioning for **both** modalities. It uses the paper's
allocation equation to gate the existing conditioning edit. The pipeline's
coupled latents and solver stay in their native path.
[Right Token, Right Time](https://arxiv.org/html/2611.18274v1#S3)

Build a wheel from the exact shared-core commit
`b1f780decc516808cfd3019d5e5b674457161f86` and pass its path as `core_wheel`.
The adaptive reader verifies the bytes of the shared gate module before
using it. The strict RAW owner continues to use the installed core and
runtime that match the artifact's source receipt. Keep historical packages
and checkpoints in their captured cohorts.
[Shared implementation](https://github.com/HyperGAN/particle-sliders/pull/269)

```python
from ps_ltx25_native.adaptive_render import render

result = render(
    session, prompt=prompt, seed=render_seed, core_wheel=core_wheel,
    method="cfp", parameter=calibrated_threshold, direction=fitted_direction,
    strength=qualified_nominal_strength, evidence_ref=calibration_artifact_id,
)
trace = result["strength_schedule"]
```

Fit the direction on matched native activation examples. Use
`gate_core(core_wheel).calibrate_gate_budget` on separate trajectories with
the saved edit active. The calibration callback can pool the actual fired
and eligible counts as `GateUsage`. CLABS takes a calibrated bias,
`gate_seed`, and explicit `clabs` fields for its projection statistics and
controller gains. The matched random arm takes the same target rate and a
`gate_seed`. The receipt provides the applied strengths, firing counts and
sigma-integrated dose so the fixed schedule can replay the choices.
Report realized rate and dose, compare the matching RAW checkpoint against
OFF and constant controls on disjoint finished videos, and check whole-video
and stereo integrity separately. The CPU fixture uses the native FlowMatch
Euler scheduler and a small coupled denoiser. Set `LTX25_STEERING_CORE_WHEEL`
to the same core wheel to run its adaptive tests.

### Measured Warm Sunset CFP

The historical Warm Sunset RAW3 run used Contrastive Feature Projection
(CFP) on the eight-step joint LTX 2.5 sampler. A fitted joint probe reads
the pre-edit packed video and audio token means. At each native interval,
the shared gate compares its signed projection against a calibrated
threshold and selects either OFF or the existing nominal-strength connector
conditioning at 2/3 for both streams. The threshold comes from bisection
over active calibration trajectories with the adapter strength fixed.
The fit, calibration, and evaluation seeds are disjoint. The selected
threshold was `-0.12069271435029807`, with 9 of 16 calibration intervals
firing for a target of 8 and a tolerance of one interval. The gate adds
no optimizer updates or network layers to the saved RAW model. Use this
procedure with a fitted probe and active calibration for each supported
RAW slider and sampler.
[Paper equations and budget calibration](https://arxiv.org/html/2611.18274v1#S3) ·
[Pinned shared gate](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/packages/particle-sliders-core/src/particle_sliders/adaptive_steering.py)

On the two held-out cases, CFP fired on the last three of eight intervals.
The archived constant requested `0.606250007947286` on every interval.
At the native FP32 connector multiply it resolved to
`0.606249988079071`, one representable step below the nearest constant
`0.6062500476837158`. The absolute difference in integrated coefficient
was 3.79 × 10⁻⁸ on the shared audio and video grid. The table reports the
original renders. The nearest FP32 control remains unmeasured.
Independent whole-video and stereo audio checks passed for OFF, CFP, and
the archived constant on both seeds. [Pinned connector multiplier](https://github.com/HyperGAN/ltx25-particle-sliders/blob/c99e0af0a873d7d2c751c446fe53a9fbdae193fb/backends/ltx25-native/src/ps_ltx25_native/adapter.py).

| Seed | CFP minus OFF | Archived constant minus OFF | CFP minus archived constant |
| --- | ---: | ---: | ---: |
| 7 | −0.00018261 | +0.00069049 | −0.00087310 |
| 2307 | +0.00046803 | −0.00059195 | +0.00105998 |
| Equal-case mean | +0.00014271 | +0.00004927 | +0.00009344 |

The small positive mean covers this two-case panel, with a loss on seed 7.
The measured runtime used the historical `ltx25_native_video95` RAW packet
and ParticleGAN 0.10.1. The gate and scheduled-render source used in the
measurement match the released package's gate and scheduler source bytes.
The current package admits its newer RAW cohort. Full native regeneration
of the historical media requires the captured RAW3 runtime, model files,
prompt inputs, and fitted probe. The public record provides the bracket,
recorded gate projections and intensities for arithmetic replay, and the
original checkpoint and source hashes for identification.
[Two-case CFP evidence](docs/warm-sunset-raw3-cfp-two-case.json) ·
[Pinned LTX gate](https://github.com/HyperGAN/ltx25-particle-sliders/blob/a35d0f54321016b04aa80f14a26a2da639438187/backends/ltx25-native/src/ps_ltx25_native/adaptive_render.py)

From `backends/ltx25-native`, use the wheel named by
`LTX25_STEERING_CORE_WHEEL` to replay the published decisions and scores
on CPU.

```python
import json
import os
from math import fsum, isclose, ulp
from pathlib import Path
from statistics import fmean
from sys import float_info

from ps_ltx25_native.adaptive_render import gate_core

record = json.loads(Path("docs/warm-sunset-raw3-cfp-two-case.json").read_text())
core = gate_core(Path(os.environ["LTX25_STEERING_CORE_WHEEL"]))
calibration = record["calibration"]
rollouts = {row["threshold"]: row["fired"] for row in calibration["active_rollout_bracket"]}
answer = core.calibrate_gate_budget(
    lambda threshold: core.GateUsage(rollouts[threshold], calibration["eligible_intervals"]),
    target_rate=calibration["target_firing_rate"],
    bounds=tuple(rollouts)[:2],
    tolerance=calibration["rate_tolerance"],
    max_iterations=float_info.mant_dig,
)
assert answer.parameter == calibration["resolved_threshold"]
assert answer.usage.fired == calibration["calibration_fired"]

sigmas = record["actual_sigmas"]
widths = [left - right for left, right in zip(sigmas[:-1], sigmas[1:], strict=True)]
contrasts = []
for case in record["matched_cases"]:
    gate = core.ActivationGate("cfp", answer.parameter)
    mask = [gate.step(score).fired for score in case["cfp_pre_edit_projections"]]
    assert mask == case["cfp_firing_mask"]
    dose = fsum(record["nominal_strength"] * width
                for width, fired in zip(widths, mask, strict=True) if fired)
    fixed = fsum(case["matched_constant_strength"] * width for width in widths)
    assert dose == case["sigma_dose"]
    assert isclose(fixed, dose, rel_tol=0.0, abs_tol=ulp(dose))
    intensity = case["intensity"]
    contrast = {
        "adaptive_ON_minus_OFF": intensity["cfp"] - intensity["off"],
        "constant_ON_minus_OFF": intensity["matched_constant"] - intensity["off"],
        "adaptive_minus_matched_constant": intensity["cfp"] - intensity["matched_constant"],
    }
    assert contrast == case["contrasts"]
    contrasts.append(contrast)
assert {key: fmean(row[key] for row in contrasts) for key in contrasts[0]} == record["mean"]
```

### Measured Warm Sunset CLABS

The shared calibration picked a bias at 7 firings in 16 intervals, within
one firing of the paper's 50% test budget. On two held-out Warm Sunset RAW3
seeds, CLABS averaged +0.00004836 over slider OFF. Its constant control
averaged +0.00009977 over OFF, a mean advantage of +0.00005141. Both
CLABS constants used the nearest representable FP32 scalar on the native
grid. The largest absolute difference in integrated coefficient was
1.82 × 10⁻⁸. CLABS led on seed 7, and the constant led on seed 2307.
Each clip passed independent video and stereo integrity checks. The
measured CFP result above remains the stronger adaptive comparison for
this panel.
[CLABS calibration, firing decisions, and per-case outcomes](docs/warm-sunset-raw3-clabs-two-case.json)

### Qualified fixed schedules

Early-heavy strength led on both tested Warm Sunset RAW3 cases. At nominal
strength 2/3, the mean finished-video intensity gain over matched OFF was
+0.00197 early-heavy, +0.00110 constant, and +0.00034 late-heavy. The three
profiles applied the same strength integrated across the native sigmas.
On held-out seed 2307, early-heavy gained +0.00101 versus +0.00053 for
constant. Both cases passed whole-video and stereo integrity checks. The
earlier seed-19 attempt remains unresolved. These results apply to this RAW3
slider and the tested eight-step sampler.
[Two-case evidence](docs/warm-sunset-raw3-two-case-scheduled-strength.json) ·
[Seed-7 evidence](docs/warm-sunset-raw3-seed7-scheduled-strength.json)

### Replay the measured Warm Sunset schedule

Run from the `backends/ltx25-native` directory with the captured Warm
Sunset RAW3 session. The two-case evidence records the RAW checkpoint
hash and the actual scheduler grid. To replay the early-heavy arm for
seed 2307, supply `archived_prompt` from `request.prompt` in the original
latent-phase receipt. The measured clip was produced by passing that
latent boundary to the guarded original decoder.

```python
import json
from pathlib import Path

from ps_ltx25_native.scheduled_render import render

evidence = json.loads(Path("docs/warm-sunset-raw3-two-case-scheduled-strength.json").read_text())
case = evidence["heldout"]
nominal = evidence["nominal_strength"]
strengths = case["arms"]["early"]["effective_strengths"]
step_gains = [strength / nominal for strength in strengths]
geometry = session.manifest["specification"]["configuration"]["geometry"]
if geometry != evidence["geometry"]:
    raise ValueError("Use the measured Warm Sunset sampler geometry")

result = render(session, prompt=archived_prompt, seed=case["seed"],
                strength=nominal, step_gains=step_gains, output_type="latent")
trace = result["strength_schedule"]
if trace["sigmas"] != evidence["actual_sigmas"] or trace["effective_strengths"] != strengths:
    raise RuntimeError("Replay left the measured Warm Sunset schedule")
```

The nominal strength is 2/3. The first six gains are 1.5, the seventh is
1.2422680615086459, and the last is 0.5. Both modalities use the recorded
strengths and the same Euler interval widths. The early and constant
dose sums differ by one floating-point ULP on this grid.
[Per-arm strengths and outcomes](docs/warm-sunset-raw3-two-case-scheduled-strength.json)

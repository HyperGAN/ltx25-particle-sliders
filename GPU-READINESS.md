# LTX-2.5 native readiness: blocked, metadata inspection only

This preparation is on a separate readiness branch based on the migration candidate
product head `53b97f596b4ae0954a1f96267949d99f209e2364`. Its shared core remains
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. No runtime binding, core pin, historical
cohort, native guard or training objective changes here. Both Claude actual-diff
and RETURN review are required before this preparation can support GPU execution
or native landing. The former repository publication hold was later superseded
only for the explicitly authorized public source snapshot; this readiness note
does not authorize GPU execution, native qualification or a package release.

## First useful blockers and observed local evidence

The current modern factory explicitly accepts only `dummy=True` and model identity
`dummy-ltx25-api-fixture-v2`. It has no native model factory or live admission path.
This is a real blocker regardless of available GPU hardware. No native GPU
training or media command can be supplied responsibly from the present state.

A bounded directory-name inspection on 2026-11-01 found no `models--Lightricks--LTX*`
cache entries at the inspected main and legacy Hugging Face hub roots. This is
absence in those inspected locations only, not proof that no other machine has
weights. No model snapshot was provided, so revision, shard completeness, tokenizer,
connector, scheduler and VAE provenance remain unknown. No weight payload was
opened, hashed, deserialized or downloaded. Cache and model identities must be
rechecked on the root-owned execution host; caches on different hosts are not
interchangeable evidence.

The inspected isolated CPU environment has Torch 2.14.1+cpu, Transformers 5.22.1,
PEFT 0.21.2, HNDL 0.10.0 and HyperGAN 2.0.0; Diffusers is not installed there.
Core import origin is the exact a794e3b Git source. Package versions alone do not
qualify the native loader. The existing product/runtime hash manifest remains
mandatory. The core owner's synthetic CUDA qualification does not qualify this
product's native model, video shape, audio path, mask semantics or editing quality.

Two public source prompt YAML files are present: smile and chiaroscuro, with their
corresponding historical configs. Source/blob hashes are in the existing extraction
provenance. These are prompt data, not a native cached-teacher dataset. No native
latent/teacher cache with immutable model, prompt, seed, time, geometry, mask,
dtype, audio-conditioning and source identities has been prepared or verified.

## Missing native bridge and contracts

The intended historical reference is `Lightricks/LTX-2.5-Diffusers`; distilled
`transformer/` and SFT `transformer_full/` are distinct variants of LTX-2.5. Any
first native proposal must use the distilled variant and a new explicit native
cohort. It must never relabel dummy artifacts or silently substitute historical
UNI. Historical embedding UNI remains historical; the modern dummy velocity GAN
objective remains experimental and has no established native concept quality.

1. Stage a complete, immutable model snapshot under authorized local access.
   Record the actual Hugging Face commit, model index, component configs, tokenizer,
   scheduler, shard/LFS manifest and already verified integrity receipts. A path or
   matching shard filename is not integrity/completeness evidence. License and
   gated access must follow the established project terms; this task changes neither.
2. Pin and inspect a native loader source revision. Historical source calls
   `LTX2Pipeline` and, for its separate SFT fallback, `LTX2VideoTransformer3DModel`.
   The native video VAE reference is `AutoencoderKLLTX2Video`; the text encoder is
   LTX-specific `gemma4_unified`, not a substituted generic Gemma. Verify actual
   config/class/API compatibility, component names and tokenizer behavior before
   loading large payloads. Diffusers' current availability is not established here.
3. Define a reviewed staged/borrowed-model factory and pre-hook admission identity.
   Validate source, model, graph, host schema and frozen/trainable ownership before
   attaching anything. The existing modern adapter uses rank 4 / alpha 4, TE last
   four attention layers and video connectors, with no routed bank. A native bridge
   must not quietly switch to historical rank 16, automatic PEFT fallback, broader
   DiT targets or new routing. Frozen base computation must still carry gradients
   from video responses back to adapters; teacher-only paths may use no-grad.
4. Verify the actual native packing/unpacking contract. Source fixtures use
   `[B, T*H*W, C]` packed video and reversibly unpack to `[B,C,T,H,W]`, with 128
   fixture video channels. Real channel count, patch sizes, temporal/spatial
   compression, positions, padding and masks must come from native config/API and
   be proved with round trips. Do not assume unit patches or reinterpret axes.
   Preserve non-singleton time at the shared game/penalty boundary. Native field
   admission does not authorize a flat vector or a connector token field.
5. Verify native text, hold and conditioning semantics. Non-concept holds must
   precede connector padding/register insertion. The historical path left-pads to
   a multiple of 128, normally 1024. Connector output rows are not token-aligned.
   Attention masks and frozen valid-row neutral condition features must use the
   actual supported API. Distilled fixture sigma 0.725 becomes native scaled time
   725; video/audio time shapes and scale multiplier must be read and checked.
6. Define exact dtype and device ownership. Shared trainable leaves and critic
   require FP32 on one declared game device. Historical native loading requests
   BF16 frozen base modules. Any placement across devices must preserve gradients
   and return native responses on the declared game device through differentiable
   transfers. No blanket pipeline `.to()` or undeclared offload/checkpoint/autocast
   changes. Snapshot and restore borrowed flags, modes, buffers, adapter scales,
   hooks, owner references and CPU/CUDA RNG; prove partial-failure cleanup.
7. Freeze neutral audio latents, audio conditioning and masks consistently across
   teacher/student branches. Frozen audio weights do not imply invariant audio
   output through cross-modal attention. No audio or media qualification is claimed;
   audio qualification remains separately owned.

The current HNDL critic executes spatial graphs per frame and averages scores.
It is invariant to frame permutation; it cannot establish temporal order or motion.
A real temporal critic needs a separately reviewed architecture and identity-bound
HNDL source/plan with actual operator support, matched outputs/VJPs and penalty
second derivatives. The present verified `conv` graph is spatial, not a silently
available Conv3d graph. Root must decide the native temporal qualification contract
before any native video-training claim. No substitute graph is invented here.

The working embedding teacher has genuine `[B,T,D]` responses. Shared admission
currently refuses that geometry. Adopting it requires a reviewed token-response
and mask contract: valid-token penalty semantics, masked critic reduction, exact
PG source/domain admission and source/model/shape/checkpoint identity. Pooling a
response, treating token axes as image axes, or flattening time is not acceptable.
The neutral condition readout is not a substitute response-domain admission.

## Safe preflight

Only this deterministic, non-launching command is ready:

```bash
python tools/ltx25_native_preflight.py
# Always exits 2 and reports native_launch_permitted=false.
# Optional: --snapshot <existing-local-snapshot> reads bounded JSON metadata
# and stats referenced shards. It never opens a weight payload or authorizes a launch.
```

Even plausible complete fixture metadata cannot unlock the tool. Each JSON read
is limited to 1 MiB. Absolute/traversing/non-weight shard references are rejected.
Shard presence remains explicitly different from verified weights/completeness.
The tool imports no model package, contacts no service and selects no device.
There is no `--execute` option, training command, card claim or media export.

## Proposed first qualification, after prerequisites and reviews

Root and its existing GPU-visible executor own devices, scheduling and budgets.
Use fixed source prompt rows and seed 7, with a hashed prompt/latent/teacher protocol.
A possible first geometry is 9 frames at 32 by 32 pixels, corresponding to the
existing source fixture proposal; it is not admitted until native config proves
it valid. Reject incompatible geometry instead of silently changing axes/shapes.

Run qualification in this order: OFF must reproduce the independently frozen
native response; verify exact adapter/base gradient ownership and audio-condition
identity; execute one actual shared update and check finiteness; test raw and EMA
numerical responses only if the reviewed native factory truly supports them; then
compare an uninterrupted three-step run with one step plus exact checkpoint/resume.
Use sequential sessions rather than simultaneous full-model copies. Maximum
proposed budget is six total paired game updates across those two runs and twenty
minutes including initialization. These are proposed caps, not authorization to run.

The current dummy API supports numerical raw/EMA and exact resume, but generic
native export is refused. A future native inference specification and data removal
check require their own source/model identity review. Do not call dummy exports
native artifacts or claim decoded video/audio from numerical velocities.

Stop immediately on missing identity/metadata, invalid mask/packing/time, OFF
mismatch, disconnected adapter, any base gradient, nonfinite output/update,
checkpoint mismatch, hook/owner/RNG restoration failure, timeout, OOM or a root
budget/occupancy conflict. Do not retry on another device, weaken guards, lower
precision, change the objective or bypass a failed check automatically. Cleanup
and return evidence to root; only root may schedule another attempt.

## Resource estimates, not measurements

The historical source describes roughly a 12-billion-parameter text encoder and
22-billion-parameter DiT. If those counts match the staged model and frozen weights
are BF16, their weights alone estimate about 24 GB plus 44 GB (decimal), excluding
video/audio VAEs, connectors, vocoder, activations, temporary transfers and runtime
workspace. That is not a fit guarantee or a measured peak. Frozen weights still
need backward activations for adapter gradients. FP32 adapter parameters, gradients
and two optimizer moment arrays alone estimate 16 bytes per trainable parameter;
actual PG optimizer state and native host counts must be inspected separately.

Actual model parameter counts, activation costs, per-device placement and peaks
are unknown. Root must supply a device list and explicit per-device byte caps.
A conservative proposed ceiling is 80 percent of assigned capacity, subject to
root's stricter live-occupancy budget. No card availability, memory measurement or
native throughput was observed or fabricated by this task.

## CPU temporal diagnostic

`tools/ltx25_temporal_probe.py` accepts only the exact existing CPU dummy critic
and a genuine five-dimensional fixture with a non-identity temporal permutation.
It requires the exact installed core and product source closures before scoring.
It compares the original and reordered field while preserving the RNG, module
modes and critic state; it rejects hooks, invalid inputs and state mutation.
Critic calls receive private input copies, and any input mutation is rejected. A
passing diagnostic never authorizes native execution or establishes temporal
quality. It demonstrates that the current framewise-mean critic gives the same
score after frame reordering, and preserves this gap for root's temporal review.
Score equality uses a tolerance of 16 machine epsilons for the score dtype; even
a differing score is diagnostic evidence only, not temporal qualification.

## Reversible fixture packing contract

`tools/ltx25_fixture_packing.py` provides a source-admitted CPU-only reversible
contract for the current singleton-batch, 128-channel, unit-patch dummy fixture.
It validates the declared temporal and spatial sizes, preserves time order and
requires finite floating-point tensors. Tests compare both values and adapter
parameter gradients with the existing dummy `VideoResponse._field` implementation.
This is an independent conformance tool; it is not installed into or called by
the current factory and does not alter its checkpoint/source identity. It is not
evidence that real LTX 2.5 uses unit patches, the same axis mapping or this
channel count. A native packing bridge remains blocked on an immutable model
configuration, verified loader and native round-trip evidence.

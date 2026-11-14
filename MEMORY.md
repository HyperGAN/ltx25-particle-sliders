# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
| --- | ---: | --- | ---: | --- |
| Native training, split placement | 2 | GPU0: 48, GPU1: 80 | 56 | B2;512×768;17frames; text128; BF16; no checkpointing |
| Native sampling, split placement | 2 | GPU0: 48, GPU1: 48 | 56 | B1;512×768;17frames; text128; BF16; all components stay resident |
| Native training, full supported text bound | 2 | GPU0: 64, GPU1: 96 | 64 | B2;same17frames/512×768;text1024;BF16;no checkpointing |
| Native training, colocated GPU | 1 | 128 | 56 | B2;512×768;17frames;text128;BF16;all components on one GPU |


These are **calculated planning budgets, not measured peaks or hard runtime ceilings**.
They apply only to the bounded profiles in the table, for one process/slider at a
time. GiB means 2^30 bytes. Add concurrent jobs' resident allocations; do not sum
mutually exclusive loading, conditioning, training and decoding phases as if all
their models were resident together. Allow the stated host RAM independently of
VRAM. Larger batches, longer inputs, extra references, FP32 bases, different
attention kernels or allocator fragmentation can require more. There is no finite
universal maximum for arbitrary inputs. These calculations start no model jobs.

## Actual native residency

Model `Lightricks/LTX-2.5-Diffusers@4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3`. The original
[contracts](backends/ltx25-native95/src/ps_ltx25_native95/contracts.py),
[component loader](backends/ltx25-native95/src/ps_ltx25_native95/component_loader.py),
[native forward](backends/ltx25-native95/src/ps_ltx25_native95/native.py),
[adapter](backends/ltx25-native95/src/ps_ltx25_native95/adapter.py) and
[model configs](backends/ltx25-native95/src/ps_ltx25_native95/model-configs)
define this full native AV model's video-response training route.

The code loads one complete CPU component at a time, then moves it to its
declared device. Gemma4 + connectors + video/audioVAEs + vocoder remain together
on `text_device`; the ONE full DiT remains on `dit_device`. They are **not**
mutually exclusive phases. No device map, quantization, offload, gradient
checkpointing or reduced model is supported in this Source profile. CPU placement
is accepted but converts that device's VRAM ledger into additional host residency.

## Calculation

* BF16 DiT = 37,975,718,400 tensor bytes = **35.368 GiB**.
  Gemma4=22.277, connectors=5.909, videoVAE=1.352,
  audioVAE=0.099, vocoder=0.241 GiB (unindexed components use
  serialized-file upper bounds). Text-device frozen sum = **29.878 GiB**;
  one-GPU frozen sum=65.245 GiB. Do not add both four- and
  eight-shard versions from the publisher listing: the index describes one DiT.
* Adapter rank4/alpha4 targets the last4 Gemma4 attention layers and8 video
  connector layers only. `sum(4*(in+out))` is bounded by **1,818,624**
  scalars, using max8192 projected query width and even counting a K host in
  K=V layers that actually omit it. This is explicitly an upper bound, not an
  assertion that an absent K is trained. Add30,113 width8 Conv3D/context
  critic scalars. Combined FP32/update allowance <0.138 GiB, no bank.
* Default video field `[1,128,3,16,24]` has1152 video tokens. The48-layer
  DiT has video hidden4096, audio hidden2048, FF expansion, cross-modality
  attention and text context. Bound native audio at128 tokens for this short
 17-frame profile. An uncheckpointed B2 saved-graph envelope
  `2*B*48*((1152*(12*4096+3*16384+4*4096))+
  (128*(12*2048+3*8192+4*2048)))` is 24.938 GiB.
  Round it to28 GiB for default retained response/penalty graphs. Text-side5 GiB
  covers all49 Gemma hidden states, the49×3840→connector projection, last4
  trainable-layer activations and all8 video connector graphs. Add8 GiB/card
  kernel/allocator margin; round to48/80. Text1024 increases these envelopes
  to20/42 GiB, plus10 workspace, rounded to64/96.
* Sampling keeps all frozen components, but no backward graph: text-device
  frozen+8 decode+8 workspace and DiT frozen+4 live+8 workspace round to48/48.
  Colocating all components adds their weights/graphs, not max of the two cards:
  65.245+33 activations+trainables+16 workspace rounds to128.
* Host56: component-sequential loading peak is at most the35.37 GiB full DiT
  CPU object +9.30 GiB largest shard/load copy, plus up to2 GiB cached rows and
 8 GiB runtime. Max-text profile raises host allowance to64 for hidden/context
  staging. All128 cached rows are disk-backed; only active row(s) need resident
  tensor copies. Same-device CPU execution needs the corresponding model/graph
  totals in RAM in addition to this loader budget.

No finite maximum exists across accepted requests: H/W have no upper bound,
frames can be any8k+1≥9 and batch is unbounded positive. The table fixes fps24,
17 frames,512×768, batch1/2 and text128/1024; it is not a claim that longer
videos fit these budgets. Config/index metadata is captured in [inputs](docs/memory/inputs.json).


## Optimizer and estimate policy

Training figures freeze pretrained weights, use RAW current adapters and explicitly
disable weight EMA. They include no full-base gradients or EMA shadow model.
The PG 0.9.5 `bispf_optimized` recipe uses zero-momentum DualNorm, not Adam's two
full-size moment tensors. Its direction blend still retains protected gradients,
clones originals/displacements, and converts up to two protected normals to
FP64. For a conservative planning allowance, the tables reserve **80 bytes per
trainable scalar in total**: 8 for FP32 weights+gradients, 16 for two FP64 normals,
and 56 for protected FP32 gradients, FP32/FP64 direction/projection/polar,
rollback and serialization temporaries. This is an explicit allowance, not a claim
that every temporary is simultaneously allocated or that it is an exact peak.
Small scalar optimizer statistics remain enabled. Recipe changes need a new ledger.
See released [PG 0.9.5 recipes](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/recipes.py),
[DualNorm](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/dualnorm.py)
and [direction blend](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/direction_blend.py).

Activation entries below are architecture-based envelopes. Fused/memory-efficient
SDPA is assumed for the pretrained model; a math attention fallback that materializes
all score matrices can exceed them. No model benchmark or runtime measurement was
performed for this documentation. Machine-readable inputs, arithmetic and all table
rows are in [docs/memory/inputs.json](docs/memory/inputs.json).

## Explicit rounding ledger

| Operation | Calculated GPU subtotal (GiB) → planning budget | Host subtotal (GiB) → planning budget |
| --- | --- | --- |
| Native training, split placement | GPU0: 43.015 → 48; GPU1: 71.368 → 80 | 54.669 → 56 |
| Native sampling, split placement | GPU0: 45.878 → 48; GPU1: 47.368 → 48 | 54.669 → 56 |
| Native training, full supported text bound | GPU0: 60.015 → 64; GPU1: 87.368 → 96 | 60.669 → 64 |
| Native training, colocated GPU | 114.383 → 128 | 54.669 → 56 |

The detailed host terms and per-device arithmetic are included in the machine-readable inputs.

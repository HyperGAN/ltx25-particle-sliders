# Native LTX owner interface and learning-rate results

The separate [native owner API](backends/ltx25-native95/README.md) provides the
full original LTX2.5 route with explicit runner ownership and exact captured RAW
factory bytes. Its offline wheel/owned install/four software checks/installed
console refusal passed. The completed [RAW32 warm-sunset comparison](backends/ltx25-native95/LEARNING-RATES.md)
tied at G0.012 and G0.001; portable values preserve both, with no automatic
default or catastrophic eligibility claim. Exact public ConvNd dependency
artifacts and full native execution through this installed wheel remain pending.
The retained historical UNI/dummy documentation below describes its own scope.

The original README follows unchanged.

# LTX-2.5 particle sliders — experimental public source

This model-specific repository extracts the public LTX-2.5 implementation from
[HyperGAN/particle-sliders at2b4b88e](https://github.com/HyperGAN/particle-sliders/tree/2b4b88e3f4ba62ccdfaecebff288a78cd6ba847b).
It preserves historical embedding UNI and the separately named **dummy-only**
modern GAN preview. This repository publishes the extracted source snapshot on
`main`; it is not the earlier minimal scaffold. **No real pretrained training,
GPU/media, memory or output-quality qualification is established.** No
model weights, downloads, training campaign, workflow or deployment is included.

The native reference is `Lightricks/LTX-2.5-Diffusers`; distilled `transformer/`
and full/SFT `transformer_full/` are variants of LTX-2.5. Shared library code stays
in `particle-sliders-core`: no GAN mathematics/core API is copied or changed.

Use a new environment and install the existing isolated distribution:

```bash
python3.12 -m venv .venv
.venv/bin/pip install ./backends/ltx25-modern
```

Its namespace remains `particle_sliders_ltx25_dummy_preview`; entrypoints remain
`ltx25-dummy-video-gan-preview`. The active shared-core source pin is now
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, with official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL0.10.0. Native/live execution
and generic native export remain refused. Historical source keeps its existing
`conceptmod.textsliders` namespace; the modern wheel does not install that namespace.

See [EXTRACTION.md](EXTRACTION.md), [allowlist](provenance/ALLOWLIST.json) and
[source/blob provenance](provenance/source-map.json). Only two required historical
scaling functions are extracted from the mixed `slider_targets.py`; unrelated
model implementations are excluded. The modern helper lock relocates to that
exact excerpt. No objective/cohort behavior is changed; changed product source
identity correctly refuses old modern packets. Historical LoRA keys stay intact.

[Historical LTX documentation](docs/ltx25-slider.md) and
[modern dummy-preview contract](docs/ltx25-modern-gan.md) are retained as primary
source evidence. Their umbrella/PR references describe the original snapshot;
this phase does not rewrite historical claims or delete the embedded originals.

This initial source publication does not complete native implementation review,
merge an umbrella PR or release a package. Native work still requires tests,
actual-diff/return review and execution admission. See [PUBLICATION.md](PUBLICATION.md)
for current checks, sanitation and the retained qualification boundary.

## Historical phase1 CPU verification

The earlier phase1 extracted wheel passed33 installed checks outside both source checkouts,
including OFF/raw/EMA, exact resume and actual prior-source packet rejection.
The actual prior-source fixture packet is a local-only test input, excluded from
this source repository; without that input its optional check skips explicitly.
Historical source checks passed41 cases:36 LTX historical +2 packaging +3 extraction checks.

Tests used an isolated venv and the exact read-only qualified CPU dependency
libraries, installing the new wheel with `--no-deps`. Fresh full native dependency
resolution and pretrained/GPU/media qualification remain unrun. No hosted CI
workflow is included, and no hosted test pass is claimed.

The active modern candidate now uses the cleared a794e3b core; see [migration notes](MODERN-CORE-MIGRATION.md). Retained snapshot documents/locks describe their original cohort and are not silently rewritten.

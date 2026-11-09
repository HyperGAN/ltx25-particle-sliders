# Allowlisted phase1 extraction

Source: `HyperGAN/particle-sliders` at
`2b4b88e3f4ba62ccdfaecebff288a78cd6ba847b` (public draft PR152).
Destination: `HyperGAN/ltx25-particle-sliders`.

This is a clean export, not a history mirror. The provenance index records each
original repository, commit, source path, Git blob, SHA256, destination path/hash
and any bounded transformation. Embedded/historical originals remain intact.

The allowlist was declared before export. Excluded categories include all other
models, original Git history, logs/evidence, environment paths/files, secrets,
recordings/media, model/adapter weights, caches and run outputs. No workflows or
access/release configuration are included. Public source and its existing MIT
license are retained. The public `main` branch contains the extracted source
snapshot. The earlier minimal scaffold/review-branch arrangement is historical;
source publication does not complete native implementation review or qualification.

Only two computational support functions are required from the mixed historical
`slider_targets.py`: `_is_lora_delta_layer` and `apply_continuous_lora_scale`.
Their exact AST source bodies are exported under the same source module name,
with original line ranges/blob/function hashes recorded. Other backend formulas
are not exported. `ltx25_backend.py` and `ltx25_uni.py` remain byte-identical.
The one helper hash literal in `ltx25_gan.py` is mechanically relocated to the
excerpt's file hash. Build-time namespace rewrites remain the same two checked
imports. Core pins, objective, graph, generator arm, native axes/masks, scaling,
namespace and dummy-only cohort behavior are unchanged. Old modern packets have
different source identities and must be refused; their original environments
remain available at the original commit. No checkpoint is silently relabeled.

The unrelated `test_music3_defaults_unchanged` is omitted from the historical
suite; its36 LTX-specific tests remain. No Music3 files are needed to run this
repository. Trainer/diagnostic modules and four LTX YAML files are directly
required by the historical tests and CLI, so are explicitly allowlisted.

Installed tests run from outside both source checkouts against the built wheel.
Core runtime remains the exact installed49-file candidate closure. No native
admission, routed bank, temporal-order critic, token-mask contract or HNDL
optimization is added. GPU/native model qualification remains separately owned.

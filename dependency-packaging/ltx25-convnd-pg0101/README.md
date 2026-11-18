# ConvNd dependency packaging for native LTX — PG0.10.1 generation

This directory supersedes the PG0.9.5-era record in
[`../ltx25-convnd/`](../ltx25-convnd/README.md) for the current generation.
The 0.9.5 record stays as the provenance of the saved FULL/RAW cohorts; it
is not the current dependency line.

## What the ConvNd build is

`particlegan==0.9.5+convnd.1` was never a separate project: it is stock
ParticleGAN 0.9.5 (commit `1b8355b184b21bb7d81104ee590813c262f6df0a`) plus a
single commit, `37f1b23281c551a196f9e508be691a6cf724c858` ("Extend per-offset
convolution metadata to native Conv1d and Conv3d", branch
`experimental/ltx25-pg095-convnd-v1`). Its only functional change is in
`particlegan/optim/dualnorm.py`: DualNorm's per-offset convolution handling,
which admitted only `Conv2d`/`ConvTranspose2d`, is generalized to native
Conv1d/Conv3d (and transposes). The per-offset spectral solver math is
unchanged — same solver, same smoothing/threshold/momentum law; 2D metadata
values and kernel-offset order are preserved exactly. LTX2.5 needs this
because its factory contains native Conv1d/Conv3d modules that stock
DualNorm refuses. The commit also adds `CONVND-PROFILE.json` (profile
`pg095-convnd-per-offset-v1`) and a CPU test file. No objective, loss or
game-structure change is involved, in either generation.

## The 0.10.1 port — done and verified

The convnd commit was cherry-picked onto stock ParticleGAN 0.10.1
(`d7bc20166c022fe037cfb9266272d2cdd7624399`, the source of the PyPI
0.10.1 wheel). Result, pushed as a branch (not merged):

- **ParticleGAN branch:** `experimental/ltx25-pg0101-convnd-v1`
- **Commit:** `c328e5e1ea3aa3c8f36d8fd0920e165f78db81f9`
- **Distribution:** `particlegan==0.10.1+convnd.1`
- **Source profile:** `pg0101-convnd-per-offset-v1`
- **Recorded in:** the branch's `CONVND-PROFILE.json` (87 Python sources,
  regenerated hashes for the ported tree, ported-from pointers to the
  pg095 profile/commit)

Port details:

- `particlegan/optim/dualnorm.py` applied **cleanly** — no manual port of
  the extension was needed; 0.10.1's intervening DualNorm changes
  (dense-profile and EMPF/RHBI additions) do not touch the convolution
  layout code. The full functional diff is in
  [`pg-convnd-0.10.1.diff`](pg-convnd-0.10.1.diff).
- The only conflict was the version string, resolved to `0.10.1+convnd.1`.
- One test-only adjustment: 0.10.1's `NormalizedOptimizer` restricts
  `momentum` to `{0, 0.5, 0.9}` (0.9.5 accepted any value). The carried-over
  state-continuation test used an arbitrary `momentum=0.4` fixture; the
  port uses `0.5`. No library code was changed for this.

Verification (CPU, system Python 3.12 / torch 2.15.0+cpu):

- `tests/test_dualnorm_convnd_extension.py`: **10 passed** — including the
  unchanged-2D metadata/legacy-order regression and the independent
  per-offset reference check for Conv1d/2d/3d and transposes.
- The stock 0.10.1 DualNorm suites (`test_dualnorm_convolution.py` etc.)
  could not be collected in the sparse verification checkout (they import
  the unmaterialized `experiments` package); that is a checkout artifact,
  not a port result. 2D behavior is covered by the extension test's own
  legacy-order regression above.

## Core derivative — remaining tracked action

`particle-sliders-core==0.14.2+convnd.1` does **not** exist yet, and unlike
the 0.9.5 generation it is not a metadata-only derivative:

- The PG0.9.5 core already carried convnd support (`CONVND_SOURCE_PROFILE`
  in `particle_sliders/pg09.py`, the `pg095_convnd.json` source manifest,
  profile/domain pairs), so its derivative only changed the pin.
- The PR #211 core (branch `chore/particlegan-0.10.1` @ `c47c7757`) admits
  only the stock release profile in `pg0101.py`; there is no
  `pg0101_convnd.json` and no convnd profile support.

Remaining steps, sequenced after HyperGAN/particle-sliders PR #211 lands:

1. In particle-sliders core: add `CONVND_SOURCE_PROFILE =
   "pg0101-convnd-per-offset-v1"` support mirroring the pg095 pair
   (profile→version mapping in `pg09.py`, exposure in `pg0101.py`,
   domain pairs), and generate `pg0101_convnd.json` from the
   `0.10.1+convnd.1` sources above (same manifest method PR #211 used
   for `pg0101_release.json`).
2. Apply [`core-dependency-only.proposed.diff`](core-dependency-only.proposed.diff)
   (version `0.14.2+convnd.1`, pin `particlegan==0.10.1+convnd.1`) and
   extend package-data for the new manifest.
3. Re-run the core contract tests for the new profile, then qualify
   native95 against the built derivative before any heavy execution —
   the usual owner admission rules are unchanged.

Until step 3 completes, the native95 pins in
`backends/ltx25-native95/pyproject.toml` name the intended 0.10.1-line
builds explicitly (they do not silently fall back to stock or 0.9.5
builds), and the saved cohorts' captured provenance
(`current_core.json`, the RAW32 recipe) remains the
`0.9.5+convnd.1` / core `0.14.1+convnd.1` record of what was actually run.

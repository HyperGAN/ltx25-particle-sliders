# LTX 2.5 CPU preview

The isolated CPU preview uses Particle Sliders core 0.15.2 and stock
ParticleGAN 0.11.1, with explicit HNDL 0.10.0 and the pinned HyperGAN
builder. It verifies installed package sources before opening a preview.
Training, exact checkpoint resume and inference use RAW weights.

Install this preview in its own environment and run the synthetic fixtures
from the repository root:

    python -m pip install ./backends/ltx25-modern
    python -m pytest -q tests/test_ltx25_modern_gan.py tests/test_ltx25_modern_packaging.py

The native LTX 2.5 3D critic still depends on the captured Conv3d variant
of DualNorm. Use the [original owner checkout](https://github.com/HyperGAN/ltx25-particle-sliders/blob/ffc51b2df3c0644c6cd763fda8cb43f55c59436b/backends/ltx25-native95/README.md) for those dependencies and checkpoints.
The CPU preview uses a framewise Conv2d critic.

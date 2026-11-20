# LTX 2.5 CPU preview

The isolated CPU preview uses Particle Sliders core 0.15.1 and stock
ParticleGAN 0.11.0, with explicit HNDL 0.10.0 and the pinned HyperGAN
builder. It verifies installed package sources before opening a preview.
Training, exact checkpoint resume and inference use RAW weights.

Install this preview in its own environment and run the synthetic fixtures
from the repository root:

    python -m pip install ./backends/ltx25-modern
    python -m pytest -q tests/test_ltx25_modern_gan.py tests/test_ltx25_modern_packaging.py

The native LTX 2.5 3D critic still depends on the captured Conv3d variant
of DualNorm. Its original source and checkpoint dependencies are preserved.
The CPU preview uses a framewise Conv2d critic.

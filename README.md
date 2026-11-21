# LTX 2.5 particle sliders

**Memory (calculated):** native BF16 batch2 default video training: **2 GPUs, plan48/80 GiB VRAM /56 GiB RAM**; sampling: **48/48 GiB**. Larger videos have no fixed maximum. See [MEMORY.md](MEMORY.md) for the complete budgets and assumptions.

The [isolated CPU preview](backends/ltx25-modern/README.md) uses Particle Sliders
core 0.15.2 and ParticleGAN 0.11.1. It exercises synthetic video-field training,
exact FULL resume and RAW response inference with a framewise Conv2d critic.

The full native LTX 2.5 owner uses the captured Conv3d per-offset DualNorm
implementation. The stock release admits Conv2d and ConvTranspose2d metadata.
Use the [original native owner checkout](https://github.com/HyperGAN/ltx25-particle-sliders/blob/ffc51b2df3c0644c6cd763fda8cb43f55c59436b/backends/ltx25-native95/README.md)
to reproduce that runtime and its saved FULL and RAW artifacts. See the
[RAW32 warm-sunset comparison](https://github.com/HyperGAN/ltx25-particle-sliders/blob/ffc51b2df3c0644c6cd763fda8cb43f55c59436b/backends/ltx25-native95/LEARNING-RATES.md)
for the completed learning-rate evidence.

The [original source publication](https://github.com/HyperGAN/ltx25-particle-sliders/blob/ffc51b2df3c0644c6cd763fda8cb43f55c59436b/README.md) records the extracted UNI
implementation, its license and its original CPU and packaging checks.

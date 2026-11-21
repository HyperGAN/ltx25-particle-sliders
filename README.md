# LTX 2.5 particle sliders

**Memory (calculated):** native BF16 batch2 default video training: **2 GPUs, plan48/80 GiB VRAM /56 GiB RAM**; sampling: **48/48 GiB**. Larger videos have no fixed maximum. See [MEMORY.md](MEMORY.md) for the complete budgets and assumptions.

The [isolated CPU preview](backends/ltx25-modern/README.md) uses Particle Sliders
core 0.15.2 and ParticleGAN 0.11.1. It exercises synthetic video-field training,
exact FULL resume and RAW response inference with a framewise Conv2d critic.

The [native LTX 2.5 owner package](backends/ltx25-native/README.md) also uses
core 0.15.2 and ParticleGAN 0.11.1. It preserves the full native video and
context critic while presenting each kernel offset to stock DualNorm.
Use the [original checkout](https://github.com/HyperGAN/ltx25-particle-sliders/tree/ffc51b2df3c0644c6cd763fda8cb43f55c59436b)
for earlier FULL and RAW artifacts and their measured rates.

The [original source publication](https://github.com/HyperGAN/ltx25-particle-sliders/blob/ffc51b2df3c0644c6cd763fda8cb43f55c59436b/README.md) records the extracted UNI
implementation, its license and its original CPU and packaging checks.

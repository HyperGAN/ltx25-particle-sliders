"""Reversible unit-patch dummy packing; this does not admit native LTX geometry."""
from dataclasses import dataclass

import torch
from particle_sliders_ltx25_dummy_preview.binding import admit, native_sources


@dataclass(frozen=True)
class FixtureUnitPatch:
    temporal: int
    height: int
    width: int
    channels: int = 128

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.temporal, self.height, self.width)):
            raise ValueError('positive fixture geometry is required')
        if self.channels != 128 or type(self.channels) is not int:
            raise ValueError('only the existing 128-channel fixture is supported')

    def _admit(self, value):
        admit()
        native_sources()
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
            raise ValueError('the packing contract admits CPU tensors only')
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('finite floating-point fixture values are required')

    def unpack(self, packed):
        self._admit(packed)
        t, h, w = self.temporal, self.height, self.width
        if tuple(packed.shape) != (1, t * h * w, self.channels):
            raise ValueError('packed fixture geometry differs')
        return packed.reshape(1, t, h, w, self.channels).permute(0, 4, 1, 2, 3).contiguous()

    def pack(self, field):
        self._admit(field)
        t, h, w = self.temporal, self.height, self.width
        if tuple(field.shape) != (1, self.channels, t, h, w):
            raise ValueError('fixture field geometry differs')
        return field.permute(0, 2, 3, 4, 1).reshape(1, t * h * w, self.channels).contiguous()

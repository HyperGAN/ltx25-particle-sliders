"""Product-owned temporal critic; PG receives the unchanged full5D field."""
import torch
from torch import nn
from torch.nn import functional as F


class NativeConv1d(nn.Module):
    """Expose each sequence offset to the stock per-offset optimizer."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self.kernel = nn.Conv2d(inputs, outputs, kernel_size=(1, 3))

    def forward(self, value):
        return F.conv1d(value, self.kernel.weight.squeeze(2), self.kernel.bias, padding=1)


class NativeConv3d(nn.Module):
    """Store depth and row together while retaining every native offset."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self.kernel = nn.Conv2d(inputs, outputs, kernel_size=(9, 3))

    def forward(self, value):
        weight = self.kernel.weight.unflatten(2, (3, 3))
        return F.conv3d(value, weight, self.kernel.bias, padding=1)


class VideoCritic(nn.Module):
    score_bound = 0

    def __init__(self, shape, condition_width, width=8):
        super().__init__()
        self.shape, self.condition_width = tuple(shape[1:]), condition_width
        # All native coordinates remain in the game/official penalty input.
        # Position and learned context features are internal critic operations.
        t, h, w = self.shape[1:]
        positions = torch.stack(torch.meshgrid(torch.linspace(-1, 1, t), torch.linspace(-1, 1, h),
                                               torch.linspace(-1, 1, w), indexing="ij"), dim=0)
        self.register_buffer("positions", positions[None])
        self.field = NativeConv3d(self.shape[0] + 3, width)
        self.temporal = NativeConv3d(width, width)
        # Shared learned scalar filters visit every complete context coordinate;
        # unlike a giant flatten->Linear, parameter count does not grow with L.
        self.context = NativeConv1d(1, width * 2)
        self.score = nn.Linear(width, 1)

    def forward(self, response, condition):
        if tuple(response.shape[1:]) != self.shape or condition.shape != (len(response), self.condition_width):
            raise ValueError("unchanged full video field and full authenticated conditioning required")
        gain, bias = self.context(condition[:, None]).mean(-1).chunk(2, dim=1)
        value = self.field(torch.cat((response, self.positions.expand(len(response), -1, -1, -1, -1)), dim=1))
        value = torch.nn.functional.silu(value * (1 + gain[:, :, None, None, None]) + bias[:, :, None, None, None])
        value = torch.nn.functional.silu(self.temporal(value))
        return self.score(value.mean((2, 3, 4))).squeeze(-1)

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class SimpleCNNBackbone(nn.Module):
    """Plain stride-2 conv stack, one modality in -> one feature map out.

    This is the fp32 reference backbone. A binarized backbone (or any other
    replacement) is a drop-in as long as it keeps the same contract: takes
    `(B, in_channels, H, W)` and returns a single `(B, out_channels, H/s, W/s)`
    feature map, exposing `out_channels` for whatever comes next (fusion
    module, head).
    """

    def __init__(self, in_channels: int, widths: Sequence[int] = (16, 32, 64, 128, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for width in widths:
            layers += [
                nn.Conv2d(channels, width, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
            channels = width
        self.net = nn.Sequential(*layers)
        self.out_channels = channels
        self.stride = 2 ** len(widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Mid fusion: channel-concatenate per-modality feature maps and project
    to a fixed channel count with a 1x1 conv.

    Requires every input feature map to already share the same spatial size
    — i.e. every modality's backbone must have the same stride, and the
    dataset must resize all modalities to the same `img_size` (they are not
    pixel-aligned to begin with, see `docs/labnotes.md` 2026-09-06, so this
    is an independent-feature-extraction fusion, not a warp-then-concat one).
    """

    def __init__(self, in_channels_per_modality: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(sum(in_channels_per_modality), out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.project(torch.cat(list(features), dim=1))

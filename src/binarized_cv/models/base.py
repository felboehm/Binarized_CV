from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDetector(nn.Module, ABC):
    """Common interface every detector (fp32 baseline, binarized variant,
    reimplemented comparison model) must implement so the data loading,
    training loop, and eval loop never need to change when a new model is
    added — only `binarized_cv.models.registry.register_model` needs it.

    Contract, mirroring the torchvision detection-model convention:

    - `images`: dict mapping each name in `self.modalities` to a
      `(B, C, H, W)` tensor, already resized to the model's expected input
      size by the dataset/dataloader.
    - `targets` (training only): a list of length `B`, one dict per image,
      each `{"boxes": (N, 4) cxcywh normalized to [0, 1], "labels": (N,)}`
      in the dataset's `target_modality` coordinate frame.
    - Return value:
      - training (`targets is not None`): `dict[str, Tensor]` of named
        scalar losses; the caller sums them.
      - inference (`targets is None`): a list of length `B`, one dict per
        image, each `{"boxes": (K, 4) xyxy in pixel coordinates,
        "scores": (K,), "labels": (K,)}`.
    """

    modalities: tuple[str, ...] = ("rgb", "ir")
    num_classes: int

    @abstractmethod
    def forward(
        self,
        images: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        raise NotImplementedError

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from binarized_cv.models.backbones.simple_cnn import SimpleCNNBackbone
from binarized_cv.models.base import BaseDetector
from binarized_cv.models.fusion.concat import ConcatFusion
from binarized_cv.models.heads.anchor_free import AnchorFreeHead
from binarized_cv.models.registry import register_model

_MODALITY_CHANNELS = {"rgb": 3, "ir": 1}


@register_model("simple_fusion")
class SimpleFusionDetector(BaseDetector):
    """Basic reference detector: one fp32 `SimpleCNNBackbone` per modality,
    merged with `ConcatFusion`, decoded by a single-scale `AnchorFreeHead`.

    This is the plug-and-play baseline everything else (binarized backbones,
    a different fusion module, a reimplemented comparison model) gets
    compared against — it exists to prove the data -> model -> loss/eval
    pipeline works end to end, not to be an accurate detector.
    """

    modalities = ("rgb", "ir")

    def __init__(
        self,
        num_classes: int = 1,
        img_size: tuple[int, int] = (640, 640),
        backbone_widths: Sequence[int] = (16, 32, 64, 128, 256),
        fusion_channels: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbones = nn.ModuleDict(
            {m: SimpleCNNBackbone(_MODALITY_CHANNELS[m], backbone_widths) for m in self.modalities}
        )
        self.fusion = ConcatFusion(
            [self.backbones[m].out_channels for m in self.modalities], fusion_channels
        )
        self.head = AnchorFreeHead(fusion_channels, num_classes)

    def forward(
        self,
        images: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        features = [self.backbones[m](images[m]) for m in self.modalities]
        fused = self.fusion(features)
        raw = self.head(fused)

        if targets is not None:
            return self.head.compute_loss(raw, targets)
        return self.head.postprocess(raw, self.img_size)

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from binarized_cv.models.base import BaseDetector
from binarized_cv.models.detectors._ultralytics_common import (
    clamp_xyxy_to_image,
    targets_to_ultralytics_batch,
)
from binarized_cv.models.registry import register_model


@register_model("yolo26_midfusion")
class Yolo26MidFusionDetector(BaseDetector):
    """YOLO26 with late RGB+IR fusion at P5/32 (after full backbone + SPPF).

    For now, this is a pragmatic placeholder that leverages the full YOLO26
    model's architecture but with RGB modality only (IR is unused). The true
    mid-level fusion at P4/16 with skip connections is complex in ultralytics'
    design and deferred.

    This detector establishes a baseline without full fusion; once the
    architecture is settled, it can be extended to true mid-fusion.
    """

    modalities = ("rgb", "ir")

    def __init__(
        self,
        num_classes: int = 1,
        img_size: tuple[int, int] = (640, 640),
        scale: str = "n",
        pretrained: bool = False,
        conf_thresh: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.conf_thresh = conf_thresh

        # For now: wrap the real YOLO26 model directly
        # This is the same as yolo26.py but registers under a different name
        self.model = DetectionModel(cfg=f"yolo26{scale}.yaml", ch=3, nc=num_classes, verbose=False)
        self.model.args = get_cfg(overrides={})

        if pretrained:
            checkpoint = torch.load(f"yolo26{scale}.pt", map_location="cpu", weights_only=False)
            try:
                self.model.load(checkpoint)
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")

    def forward(
        self,
        images: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        rgb = images["rgb"]
        # IR is available in images["ir"] but not used yet (placeholder for fusion)

        if targets is not None:
            batch = targets_to_ultralytics_batch(rgb, targets)
            loss_sum, _loss_items = self.model(batch)
            return {"loss_box": loss_sum[0], "loss_cls": loss_sum[1], "loss_dfl": loss_sum[2]}

        # Inference: run full YOLO26 on RGB
        raw, _preds = self.model(rgb)

        # Decode to per-image detection dicts
        results = []
        for row in raw:
            boxes, scores, labels = row[:, :4], row[:, 4], row[:, 5].long()
            keep = scores > self.conf_thresh
            results.append(
                {
                    "boxes": clamp_xyxy_to_image(boxes[keep], self.img_size),
                    "scores": scores[keep],
                    "labels": labels[keep],
                }
            )
        return results

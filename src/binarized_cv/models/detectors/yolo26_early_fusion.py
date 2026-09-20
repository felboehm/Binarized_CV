from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from binarized_cv.models.base import BaseDetector
from binarized_cv.models.detectors._ultralytics_common import (
    clamp_xyxy_to_image,
    targets_to_ultralytics_batch,
)
from binarized_cv.models.registry import register_model


def _decode_ultralytics_output(
    raw: torch.Tensor, img_size: tuple[int, int], conf_thresh: float
) -> list[dict[str, torch.Tensor]]:
    """`Detect.forward` in eval mode with `end2end=True` already does
    NMS-free one2one decoding + top-k selection internally (see
    `ultralytics/nn/modules/head.py`), returning `(B, <=300, 6)` rows of
    `[x1, y1, x2, y2, score, class]` in input-pixel scale — but without a
    confidence threshold or box clamping applied, so we do both here to
    match our own `BaseDetector` contract."""
    results = []
    for row in raw:
        boxes, scores, labels = row[:, :4], row[:, 4], row[:, 5].long()
        keep = scores > conf_thresh
        results.append(
            {"boxes": clamp_xyxy_to_image(boxes[keep], img_size), "scores": scores[keep], "labels": labels[keep]}
        )
    return results


@register_model("yolo26_early_fusion")
class Yolo26EarlyFusionDetector(BaseDetector):
    """YOLO26 with early fusion: concatenate RGB and IR as 4-channel input.

    **Approach**: Resize IR to match RGB spatial dimensions, stack as 4th
    channel, train single YOLO26 backbone. Simplest fusion strategy.

    **Known limitation (by design)**: TRGB/WiSARD data are not pixel-registered
    (different resolutions, independent per-modality labels). Early fusion
    assumes alignment after resize. If accuracy suffers, this documents that
    spatial alignment is a limiting factor for simple concatenation — a
    legitimate thesis result supporting mid-fusion over early fusion.

    **Reference**: Based on Balla & Shrestha (EUSIPCO 2025, ms_yolov8
    architecture), but applied to TRGB/WiSARD where the misalignment
    assumption doesn't hold — makes this a deliberately weak baseline.

    Depends on `ultralytics` (AGPL-3.0) — see `LICENSE`.
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

        # 4-channel YOLO26 (RGB + IR)
        self.model = DetectionModel(cfg=f"yolo26{scale}.yaml", ch=4, nc=num_classes, verbose=False)
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
        rgb = images["rgb"]  # (B, 3, H, W)
        ir = images["ir"]  # (B, 1, H_ir, W_ir)

        # Resize IR to match RGB spatial dimensions
        if ir.shape[-2:] != rgb.shape[-2:]:
            ir = F.interpolate(ir, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        # Concatenate as 4-channel input
        fused = torch.cat([rgb, ir], dim=1)  # (B, 4, H, W)

        if targets is not None:
            # Training mode: compute loss
            batch = targets_to_ultralytics_batch(fused, targets)
            loss_sum, _loss_items = self.model(batch)
            return {"loss_box": loss_sum[0], "loss_cls": loss_sum[1], "loss_dfl": loss_sum[2]}

        # Inference mode: get detections
        raw, _preds = self.model(fused)
        return _decode_ultralytics_output(raw, self.img_size, self.conf_thresh)

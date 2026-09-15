from __future__ import annotations

import torch
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


@register_model("yolo26")
class Yolo26Detector(BaseDetector):
    """Wraps the real `ultralytics` YOLO26 `DetectionModel` behind our
    `BaseDetector` interface. RGB-only for now (`modalities = ("rgb",)`) —
    multispectral fusion means extending this model's input stem for a
    second stream, which is separate follow-on work (CHECKLIST.md
    section 4); IR is simply unused here today.

    Depending on `ultralytics` means this repo's license terms follow
    theirs (AGPL-3.0) as soon as this model is used — see `LICENSE`.
    """

    modalities = ("rgb",)

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
        self.model = DetectionModel(cfg=f"yolo26{scale}.yaml", ch=3, nc=num_classes, verbose=False)
        self.model.args = get_cfg(overrides={})
        if pretrained:
            checkpoint = torch.load(f"yolo26{scale}.pt", map_location="cpu", weights_only=False)
            self.model.load(checkpoint)

    def forward(
        self,
        images: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        rgb = images["rgb"]
        if targets is not None:
            batch = targets_to_ultralytics_batch(rgb, targets)
            loss_sum, _loss_items = self.model(batch)
            return {"loss_box": loss_sum[0], "loss_cls": loss_sum[1], "loss_dfl": loss_sum[2]}

        raw, _preds = self.model(rgb)
        return _decode_ultralytics_output(raw, self.img_size, self.conf_thresh)

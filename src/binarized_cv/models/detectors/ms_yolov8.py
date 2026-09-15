from __future__ import annotations

import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression

from binarized_cv.models.base import BaseDetector
from binarized_cv.models.detectors._ultralytics_common import (
    clamp_xyxy_to_image,
    targets_to_ultralytics_batch,
)
from binarized_cv.models.registry import register_model


@register_model("ms_yolov8")
class MultispectralYolov8Detector(BaseDetector):
    """Reimplements Balla & Shrestha, "Multispectral Human Presence Detection
    using Adapted YOLO Network" (EUSIPCO 2025, OsloMet) — a SAR-drone
    human-detection comparison model. Their method has two changes to stock
    YOLOv8: (1) early-fuse RGB+thermal into one 4-channel input (their first
    conv layer is widened from a 3x3x3 to a 3x3x4 kernel — `ch=4` here has
    the same effect via `DetectionModel`'s native n-channel support), and
    (2) replace the neck's nearest-neighbor upsampling with bicubic, for
    better small-object detail retention. Their code (frnc96/ms-yolov8) is
    an ultralytics fork implementing this on real datasets (NII-CU, M3FD,
    LLVIP); we reimplement just the architecture change against our own
    `BaseDetector` interface rather than depending on their fork directly.

    Unlike YOLO26 (see `yolo26.py`), stock YOLOv8 doesn't have an NMS-free
    head, so eval-mode decoding needs an explicit NMS step
    (`ultralytics.utils.nms.non_max_suppression`).

    Early fusion assumes RGB and thermal are already reasonably spatially
    aligned once resized to the same size — true for the paper's datasets,
    NOT true for TRGB/WiSARD (see `docs/labnotes.md` 2026-09-06), so this is
    a deliberately-included weak baseline on our own data: it lets us show
    whether the lack of registration actually hurts naive fusion, motivating
    the mid-fusion approach used elsewhere in this repo.
    """

    modalities = ("rgb", "ir")

    def __init__(
        self,
        num_classes: int = 1,
        img_size: tuple[int, int] = (640, 640),
        scale: str = "n",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = DetectionModel(cfg=f"yolov8{scale}.yaml", ch=4, nc=num_classes, verbose=False)
        self.model.args = get_cfg(overrides={})
        for module in self.model.modules():
            if isinstance(module, nn.Upsample):
                module.mode = "bicubic"

    def _fuse(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([images["rgb"], images["ir"]], dim=1)

    def forward(
        self,
        images: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        fused = self._fuse(images)
        if targets is not None:
            batch = targets_to_ultralytics_batch(fused, targets)
            loss_sum, _loss_items = self.model(batch)
            return {"loss_box": loss_sum[0], "loss_cls": loss_sum[1], "loss_dfl": loss_sum[2]}

        raw, _preds = self.model(fused)
        detections = non_max_suppression(raw, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, nc=self.num_classes)
        return [
            {"boxes": clamp_xyxy_to_image(det[:, :4], self.img_size), "scores": det[:, 4], "labels": det[:, 5].long()}
            for det in detections
        ]

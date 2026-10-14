from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from binarized_cv.models.base import BaseDetector
from binarized_cv.models.binarized import apply_binarization, format_region_stats
from binarized_cv.models.detectors._ultralytics_common import (
    clamp_xyxy_to_image,
    load_pretrained,
    targets_to_ultralytics_batch,
    warm_start_and_binarize,
)
from binarized_cv.models.fusion import ConcatFusion
from binarized_cv.models.registry import register_model

log = logging.getLogger(__name__)


class _IRStream(nn.Module):
    """YOLO26 backbone layers 0..`last` for 1-channel IR input. The layers
    live in `.model`, as in a `DetectionModel`, so `apply_binarization`
    assigns them the same regions as the RGB backbone's layers."""

    def __init__(self, scale: str, last: int) -> None:
        super().__init__()
        full = DetectionModel(cfg=f"yolo26{scale}.yaml", ch=1, nc=1, verbose=False)
        self.model = full.model[: last + 1]

    def forward(self, x: torch.Tensor, keep: frozenset[int]) -> dict[int, torch.Tensor]:
        outputs = {}
        for m in self.model:
            x = m(x)
            if m.i in keep:
                outputs[m.i] = x
        return outputs


@register_model("yolo26_midfusion")
class Yolo26MidFusionDetector(BaseDetector):
    """YOLO26 with mid fusion: separate RGB and IR backbones up to the
    deepest fusion layer, merged per fusion layer by `ConcatFusion`; the
    rest of the backbone, the neck and the head are shared.

    `self.model` is the RGB YOLO26 (backbone, neck, head) and `ir_stream`
    a 1-channel copy of its backbone layers 0..max(`fusion_layers`). The
    forward pass is ultralytics' `_predict_once` with the output of each
    fusion layer replaced by the fused features, so everything downstream
    reads them: the next layer and the neck's skip connections (layer 4 =
    P3, layer 6 = P4 in `yolo26.yaml`). With the default `(6,)`, the neck's
    P3 skip carries RGB-only features and IR reaches the P3 head only
    through the upsampled P4 path; `(4, 6)` fuses at P3 as well
    (`docs/fusion_architecture_rationale.md`).

    `binarization` applies the same region map to `self.model` and the IR
    stream (an IR layer gets the region of the RGB layer with its index);
    the fusion convs stay fp. `init_checkpoint` warm-starts from one of our
    own fp32 checkpoints of this model before that swap.

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
        fusion_layers: Sequence[int] = (6,),
        binarization: dict | None = None,
        init_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.fusion_layers = tuple(sorted(fusion_layers))

        self.model = DetectionModel(cfg=f"yolo26{scale}.yaml", ch=3, nc=num_classes, verbose=False)
        self.model.args = get_cfg(overrides={})
        # The backbone runs as a plain chain, so cutting the IR copy at a
        # fusion layer loses nothing it needs.
        backbone_len = len(self.model.yaml["backbone"])
        if not self.fusion_layers or not all(0 <= i < backbone_len for i in self.fusion_layers):
            raise ValueError(f"fusion_layers must be backbone layers 0..{backbone_len - 1}, got {fusion_layers}")
        self.ir_stream = _IRStream(scale, self.fusion_layers[-1])

        channels = self._layer_channels()
        self.fusion = nn.ModuleDict(
            {str(i): ConcatFusion([channels[i], channels[i]], channels[i]) for i in self.fusion_layers}
        )

        if pretrained:
            load_pretrained(self.model, scale)
            load_pretrained(self.ir_stream, scale)

        warm_start_and_binarize(self, None, init_checkpoint)
        if binarization is not None:
            for label, module in (("rgb + shared", self.model), ("ir stream", self.ir_stream)):
                stats = apply_binarization(module, binarization)
                log.info(
                    "Binarization %s (preset=%s):\n%s", label, binarization.get("preset"), format_region_stats(stats)
                )

    @torch.no_grad()
    def _layer_channels(self) -> dict[int, int]:
        """Output channels of the fusion layers, from a small dummy pass."""
        outputs = self.ir_stream(torch.zeros(1, 1, 64, 64), frozenset(self.fusion_layers))
        return {i: t.shape[1] for i, t in outputs.items()}

    def _fused_forward(self, rgb: torch.Tensor, ir: torch.Tensor):
        """`DetectionModel._predict_once` on `rgb`, with each fusion layer's
        output replaced by its fusion with the IR stream's output there."""
        ir_features = self.ir_stream(ir, frozenset(self.fusion_layers))
        y, x = [], rgb
        for m in self.model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i in ir_features:
                x = self.fusion[str(m.i)]([x, ir_features[m.i]])
            y.append(x if m.i in self.model.save else None)
        return x

    def forward(
        self,
        images: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        rgb = images["rgb"]  # (B, 3, H, W)
        ir = images["ir"]  # (B, 1, H_ir, W_ir)
        # Same grid as RGB, so both streams' features match in size at the
        # fusion layers (the dataset's warp/crop already puts IR on it).
        if ir.shape[-2:] != rgb.shape[-2:]:
            ir = F.interpolate(ir, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        preds = self._fused_forward(rgb, ir)
        if targets is not None:
            batch = targets_to_ultralytics_batch(rgb, targets)
            loss_sum, _loss_items = self.model.loss(batch, preds)
            return {"loss_box": loss_sum[0], "loss_cls": loss_sum[1], "loss_dfl": loss_sum[2]}

        raw, _preds = preds
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

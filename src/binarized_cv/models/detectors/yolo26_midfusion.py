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
from binarized_cv.models.fusion import ConcatFusion
from binarized_cv.models.registry import register_model


@register_model("yolo26_midfusion")
class Yolo26MidFusionDetector(BaseDetector):
    """YOLO26 multi-spectral detector: mid-fusion architecture decision pending.

    **Design goal (from docs/fusion_architecture_rationale.md):**
    Mid-fusion at P4/16 (layer 6, stride 16, 128 channels):
    - RGB and IR backbones extract independently to P4/16
    - Fuse via concatenation + learned projection
    - Single shared neck/head processes fused features
    - Rationale: respects non-registration (TRGB/WiSARD finding), binarization-friendly,
      empirically motivated (standard FPN intermediate layer)

    **Implementation challenge identified (2026-09-19):**
    YOLO26's architecture has tight coupling: Detect head requires multi-scale pyramid
    (P3/P4/P5), and PAN neck has skip connections referencing specific layers by index.
    Simple P4-only injection doesn't work; proper solution requires:
    - Extracting full backbones to get all pyramid scales, or
    - Fusing at multiple scales (P3, P4, P5), or
    - Reimplementing forward pass to track layer dependencies

    **Options forward (see labnotes 2026-09-19 for full analysis):**
    1. Early fusion (4-channel concat) - simplest, documented as weak baseline on misaligned data
    2. Multi-scale mid-fusion (fuse P3/P4/P5) - complex, architecturally clean
    3. Late fusion (P5/32) - avoids skip-connection complexity but efficiency penalty
    4. RGB-only baseline - fastest path to training validation before fusion

    Implementation approach TBD based on time/accuracy trade-offs. Both RGB and IR models
    built and ready (3-ch and 1-ch variants). Fusion modules (ConcatFusion) ready.

    See docs/fusion_architecture_rationale.md and docs/labnotes.md (2026-09-19) for
    complete technical analysis and decision framework.
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

        # RGB model: 3-channel (standard YOLO26)
        self.rgb_model = DetectionModel(cfg=f"yolo26{scale}.yaml", ch=3, nc=num_classes, verbose=False)
        self.rgb_model.args = get_cfg(overrides={})

        # IR model: 1-channel (thermal input)
        self.ir_model = DetectionModel(cfg=f"yolo26{scale}.yaml", ch=1, nc=num_classes, verbose=False)
        self.ir_model.args = get_cfg(overrides={})

        # Fusion module at layer 6 (P4/16)
        self.fusion = ConcatFusion(in_channels_per_modality=[128, 128], out_channels=128)

        # Store layer lists for manual forward pass
        self.model_layers = list(self.rgb_model.model)
        self.ir_layers = list(self.ir_model.model)

        # Use RGB model as primary reference
        self.model = self.rgb_model

        if pretrained:
            checkpoint = torch.load(f"yolo26{scale}.pt", map_location="cpu", weights_only=False)
            try:
                self.rgb_model.load(checkpoint)
                self.ir_model.load(checkpoint)
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")

    def _fused_forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Forward pass with fused RGB+IR at P4/16, handling skip connections.

        Manually executes layers while tracking intermediate outputs for
        skip connections (Concat layers that reference earlier layers).
        """
        # Track all intermediate outputs for skip connection references
        layer_outputs = {}

        # === Backbone: Layers 0-6 with fusion at layer 6 ===
        # Extract backbone features independently
        rgb_x = rgb
        ir_x = ir
        for i in range(7):
            rgb_x = self.model_layers[i](rgb_x)
            ir_x = self.ir_layers[i](ir_x)

            if i < 6:
                # Store pre-fusion intermediates from RGB (for now)
                layer_outputs[i] = rgb_x

        # Fuse at layer 6 (P4/16)
        fused_x = self.fusion([rgb_x, ir_x])
        layer_outputs[6] = fused_x

        # === Neck/Head: Layers 7-23 with skip connections ===
        # The neck has Concat layers that reference earlier layer outputs.
        # We need to properly execute each layer and make referenced outputs available.
        x = fused_x
        for layer_idx in range(7, len(self.model_layers)):
            layer = self.model_layers[layer_idx]

            # Check if this is a Concat layer (has 'cat' in forward)
            if hasattr(layer, 'cat') or 'Concat' in type(layer).__name__:
                # Concat layer in ultralytics concatenates the previous layer output
                # with outputs from referenced earlier layers.
                # The layer.f list stores the indices of referenced layers.
                if hasattr(layer, 'f'):
                    # layer.f contains indices of layers to concatenate
                    refs = layer.f
                    if not isinstance(refs, (list, tuple)):
                        refs = [refs]

                    # Collect outputs to concatenate
                    concat_inputs = [x]  # Previous layer output
                    for ref_idx in refs:
                        if isinstance(ref_idx, int) and ref_idx in layer_outputs:
                            concat_inputs.append(layer_outputs[ref_idx])

                    # Concatenate and pass through layer
                    # (Layer's forward will handle the concat internally)
                    try:
                        x = layer(concat_inputs) if len(concat_inputs) > 1 else layer(x)
                    except TypeError:
                        # Fallback: just pass x if layer doesn't expect list
                        x = layer(x)
                else:
                    x = layer(x)
            else:
                x = layer(x)

            # Track this layer's output
            layer_outputs[layer_idx] = x

        return x

    def forward(
        self,
        images: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        rgb = images["rgb"]
        ir = images["ir"]

        if targets is not None:
            # Training mode: use RGB model's standard forward
            # TODO: Implement fused loss computation
            batch = targets_to_ultralytics_batch(rgb, targets)
            loss_sum, _loss_items = self.model(batch)
            return {"loss_box": loss_sum[0], "loss_cls": loss_sum[1], "loss_dfl": loss_sum[2]}

        # Inference mode: use standard RGB model forward
        # TODO: Proper fused inference with multi-scale fusion
        raw, _preds = self.model(rgb)

        # Decode detections
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

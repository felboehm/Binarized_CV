from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel


def build_yolo26_backbone_to_layer(
    ch: int,
    merge_layer: int,
    scale: str = "n",
) -> tuple[nn.Sequential, int]:
    """Build a YOLO26 backbone up to a specified layer for dual-stream fusion.

    Args:
        ch: input channel count (3 for RGB, 1 for IR in mid-fusion)
        merge_layer: layer index at which to stop (inclusive). After this layer,
            the backbone is cut off. For YOLO26n, layer 6 is P4/16 (stride 16).
        scale: YOLO26 scale ('n', 's', 'm', 'l', 'x')

    Returns:
        A tuple of (backbone_module, output_channels):
        - backbone_module: nn.Sequential containing layers 0..merge_layer
        - output_channels: number of output channels after merge_layer
    """
    full_model = DetectionModel(cfg=f"yolo26{scale}.yaml", ch=ch, nc=1, verbose=False)
    full_model.args = get_cfg(overrides={})

    # Extract backbone layers up to and including merge_layer
    backbone_layers = list(full_model.model[:merge_layer + 1])
    backbone = nn.Sequential(*backbone_layers)

    # Get output channel count from the last layer
    last_layer = backbone_layers[-1]
    if hasattr(last_layer, "conv") and hasattr(last_layer.conv, "out_channels"):
        out_channels = last_layer.conv.out_channels
    elif hasattr(last_layer, "cv2") and hasattr(last_layer.cv2, "conv") and hasattr(last_layer.cv2.conv, "out_channels"):
        out_channels = last_layer.cv2.conv.out_channels
    elif hasattr(last_layer, "out_channels"):
        out_channels = last_layer.out_channels
    else:
        raise ValueError(f"Cannot determine output channels from layer {merge_layer}: {type(last_layer).__name__}")

    return backbone, out_channels

"""Binarized / low-bit layers and the YOLO26 region policy that places them.

See `docs/binarization_plan.md` for the rationale behind the region map.
"""

from binarized_cv.models.binarized.layers import QuantConv2d
from binarized_cv.models.binarized.policy import (
    PRESETS,
    REGIONS,
    BinarizationConfig,
    apply_binarization,
    format_region_stats,
    has_quantized_layers,
    optimizer_param_groups,
    set_binarization_progress,
)

__all__ = [
    "PRESETS",
    "REGIONS",
    "BinarizationConfig",
    "QuantConv2d",
    "apply_binarization",
    "format_region_stats",
    "has_quantized_layers",
    "optimizer_param_groups",
    "set_binarization_progress",
]

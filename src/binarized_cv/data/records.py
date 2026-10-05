from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairRecord:
    id: str
    dataset: str
    split: str
    rgb_image: str
    rgb_label: str
    ir_image: str
    ir_label: str
    # False: that modality's labels are missing although people are present,
    # so its empty label files are not negatives. The dataset skips such
    # records when that modality is the supervision target.
    rgb_labeled: bool = True
    ir_labeled: bool = True
    # Set by scripts/align_manifest.py: IR -> VIS transform (3x3) in the
    # S x S training input space, the IR frame offset the pairing was moved
    # by, the remaining mean box-centre error in training pixels (None: no
    # boxes in both modalities to check), and where the transform came from.
    ir_transform: list[list[float]] | None = None
    ir_frame_offset: int = 0
    align_err_px: float | None = None
    align_source: str | None = None
    # align_err_px <= max(floor, fraction x median VIS person width); None if
    # unchecked. The dataset can drop False records (data.drop_misaligned_splits).
    align_ok: bool | None = None

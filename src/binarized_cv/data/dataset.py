from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from binarized_cv.data.labels import BoundingBox, parse_yolo_label_file
from binarized_cv.data.records import PairRecord

_TARGET_MODALITIES = ("rgb", "ir")


def _boxes_to_tensors(boxes: list[BoundingBox]) -> tuple[torch.Tensor, torch.Tensor]:
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    coords = torch.tensor([[b.cx, b.cy, b.w, b.h] for b in boxes], dtype=torch.float32)
    labels = torch.tensor([b.class_id for b in boxes], dtype=torch.long)
    return coords, labels


class MultispectralPersonDataset(Dataset):
    """Loads RGB+IR pairs and resizes both modalities independently to
    `img_size`. Box labels are already normalized (a fraction of each
    modality's own image dimensions), so a plain per-axis resize needs no
    coordinate remapping.

    Supervision targets come from `target_modality` only (default "rgb"):
    TRGB/WiSARD are not spatially co-registered between modalities (see
    `docs/labnotes.md` 2026-09-06), so a fused model predicts in one
    modality's coordinate frame and the other modality is fed in purely as
    an auxiliary input stream — its own labels are not used as supervision.
    """

    def __init__(
        self,
        records: list[PairRecord],
        raw_root: Path,
        split: str | None = None,
        img_size: tuple[int, int] = (640, 640),
        target_modality: str = "rgb",
    ) -> None:
        if target_modality not in _TARGET_MODALITIES:
            raise ValueError(f"target_modality must be one of {_TARGET_MODALITIES}, got {target_modality!r}")
        self.raw_root = Path(raw_root)
        self.records = [r for r in records if split is None or r.split == split]
        self.img_size = img_size
        self.target_modality = target_modality

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        height, width = self.img_size

        rgb_image = Image.open(self.raw_root / record.rgb_image).convert("RGB").resize((width, height))
        ir_image = Image.open(self.raw_root / record.ir_image).convert("L").resize((width, height))
        rgb_boxes = parse_yolo_label_file(self.raw_root / record.rgb_label)
        ir_boxes = parse_yolo_label_file(self.raw_root / record.ir_label)
        target_boxes = rgb_boxes if self.target_modality == "rgb" else ir_boxes
        boxes, labels = _boxes_to_tensors(target_boxes)

        return {
            "id": record.id,
            "dataset": record.dataset,
            "rgb": torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float() / 255.0,
            "ir": torch.from_numpy(np.array(ir_image)).unsqueeze(0).float() / 255.0,
            "targets": {"boxes": boxes, "labels": labels},
        }

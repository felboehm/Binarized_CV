from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from binarized_cv.data.labels import parse_yolo_label_file
from binarized_cv.data.records import PairRecord


def _boxes_to_tensor(boxes) -> torch.Tensor:
    if not boxes:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(
        [[b.class_id, b.cx, b.cy, b.w, b.h] for b in boxes], dtype=torch.float32
    )


class MultispectralPersonDataset(Dataset):
    def __init__(
        self, records: list[PairRecord], raw_root: Path, split: str | None = None
    ) -> None:
        self.raw_root = Path(raw_root)
        self.records = [r for r in records if split is None or r.split == split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rgb_image = Image.open(self.raw_root / record.rgb_image).convert("RGB")
        ir_image = Image.open(self.raw_root / record.ir_image).convert("L")
        rgb_boxes = parse_yolo_label_file(self.raw_root / record.rgb_label)
        ir_boxes = parse_yolo_label_file(self.raw_root / record.ir_label)
        return {
            "id": record.id,
            "dataset": record.dataset,
            "rgb": torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float() / 255.0,
            "ir": torch.from_numpy(np.array(ir_image)).unsqueeze(0).float() / 255.0,
            "rgb_boxes": _boxes_to_tensor(rgb_boxes),
            "ir_boxes": _boxes_to_tensor(ir_boxes),
        }

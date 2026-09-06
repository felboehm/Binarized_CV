from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


def parse_yolo_label_file(path: Path) -> list[BoundingBox]:
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label line in {path}: {line!r}")
        class_id, cx, cy, w, h = parts
        boxes.append(BoundingBox(int(class_id), float(cx), float(cy), float(w), float(h)))
    return boxes

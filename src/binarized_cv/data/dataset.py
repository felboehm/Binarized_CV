from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from binarized_cv.data import alignment
from binarized_cv.data.labels import BoundingBox, parse_yolo_label_file
from binarized_cv.data.records import PairRecord

log = logging.getLogger(__name__)

_TARGET_MODALITIES = ("rgb", "ir")
_IR_ALIGNMENTS = ("none", "warp", "crop")
# A box cut by the crop edge is kept if at least this share of its area is
# inside; it's then clipped to the edge.
_MIN_VISIBLE = 0.5


def _clip_boxes(xyxy: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip pixel xyxy boxes to the frame; `keep` marks those with at least
    _MIN_VISIBLE of their area inside."""
    clipped = xyxy.copy()
    clipped[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    area = lambda b: np.prod(b[:, 2:] - b[:, :2], axis=1)  # noqa: E731
    return clipped, area(clipped) >= _MIN_VISIBLE * area(xyxy)


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

    `ir_alignment` uses the per-record `ir_transform` (from
    scripts/align_manifest.py; records without one fall back to "none"):
      none  both modalities stretched to `img_size` independently (as before)
      warp  IR warped onto the VIS grid; black outside the IR footprint
      crop  both cropped to the shared field of view (IR footprint within the
            VIS frame), then scaled to `img_size`; boxes are clipped to the
            crop and dropped below _MIN_VISIBLE of their area.
    With target_modality="ir" the IR boxes are moved into the VIS grid too.
    `augment` (training only) applies one random geometric transform to both
    modalities and the boxes alike, after alignment: horizontal flip with
    probability `fliplr`, zoom by a factor in [1 - scale, 1 + scale] about
    the centre, and a shift of up to `translate` of the frame per axis.
    Uncovered borders are black; boxes are clipped as for "crop". No
    vertical flips or rotations: part of the footage is oblique, where up
    and down are not interchangeable. Draws from torch's RNG, which the
    DataLoader seeds per worker from its generator.
    Records whose target modality is unlabeled (`rgb_labeled`/`ir_labeled`
    False) are skipped: their empty label files are not negatives. With
    `drop_misaligned`, records whose IR is checked as misaligned
    (`align_ok` False, from scripts/align_manifest.py) are skipped too;
    unchecked ones (None) stay.
    """

    def __init__(
        self,
        records: list[PairRecord],
        raw_root: Path,
        split: str | None = None,
        img_size: tuple[int, int] = (640, 640),
        target_modality: str = "rgb",
        drop_misaligned: bool = False,
        ir_alignment: str = "none",
        augment: dict | None = None,
    ) -> None:
        if target_modality not in _TARGET_MODALITIES:
            raise ValueError(f"target_modality must be one of {_TARGET_MODALITIES}, got {target_modality!r}")
        if ir_alignment not in _IR_ALIGNMENTS:
            raise ValueError(f"ir_alignment must be one of {_IR_ALIGNMENTS}, got {ir_alignment!r}")
        self.raw_root = Path(raw_root)
        self.records = [
            r
            for r in records
            if (split is None or r.split == split)
            and (r.rgb_labeled if target_modality == "rgb" else r.ir_labeled)
            and not (drop_misaligned and r.align_ok is False)
        ]
        self.img_size = img_size
        self.target_modality = target_modality
        self.ir_alignment = ir_alignment
        self.augment = augment
        if ir_alignment != "none" and self.records and all(r.ir_transform is None for r in self.records):
            log.warning(
                "ir_alignment=%s but no record has an ir_transform (manifest not from scripts/align_manifest.py?): "
                "IR stays unaligned",
                ir_alignment,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        height, width = self.img_size

        label_path = record.rgb_label if self.target_modality == "rgb" else record.ir_label
        target_boxes = parse_yolo_label_file(self.raw_root / label_path)
        if self.ir_alignment == "none" or record.ir_transform is None:
            rgb = np.array(Image.open(self.raw_root / record.rgb_image).convert("RGB").resize((width, height)))
            ir = np.array(Image.open(self.raw_root / record.ir_image).convert("L").resize((width, height)))
            boxes, labels = _boxes_to_tensors(target_boxes)
        else:
            rgb, ir, boxes, labels = self._aligned(record, target_boxes)
        if self.augment:
            rgb, ir, boxes, labels = self._augmented(rgb, ir, boxes, labels)

        return {
            "id": record.id,
            "dataset": record.dataset,
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0,
            "ir": torch.from_numpy(ir).unsqueeze(0).float() / 255.0,
            "targets": {"boxes": boxes, "labels": labels},
        }

    def _aligned(
        self, record: PairRecord, target_boxes: list[BoundingBox]
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        height, width = self.img_size
        # ir_transform maps IR -> VIS in the alignment.S x S space; rescale it
        # to this dataset's width x height grid.
        D = np.diag([width / alignment.S, height / alignment.S, 1.0])
        T = D @ np.array(record.ir_transform) @ np.linalg.inv(D)

        x0, y0, x1, y1 = 0.0, 0.0, float(width), float(height)
        if self.ir_alignment == "crop":
            fp = alignment.apply(T, np.array([[0, 0], [width, 0], [width, height], [0, height]], float))
            x0, y0 = max(fp[:, 0].min(), 0.0), max(fp[:, 1].min(), 0.0)
            x1, y1 = min(fp[:, 0].max(), float(width)), min(fp[:, 1].max(), float(height))
        sx, sy = width / (x1 - x0), height / (y1 - y0)
        C = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])  # crop -> output grid

        rgb_full = Image.open(self.raw_root / record.rgb_image).convert("RGB")
        W, H = rgb_full.size  # crop at full resolution, then resize once
        rgb = np.array(rgb_full.crop((x0 * W / width, y0 * H / height, x1 * W / width, y1 * H / height)).resize((width, height)))
        ir = np.array(Image.open(self.raw_root / record.ir_image).convert("L").resize((width, height)))
        ir = cv2.warpPerspective(ir, C @ T, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)

        if not target_boxes:
            return rgb, ir, *_boxes_to_tensors([])
        xyxy = np.array([[b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2] for b in target_boxes])
        xyxy *= [width, height, width, height]
        M = C if self.target_modality == "rgb" else C @ T
        clipped, keep = _clip_boxes(alignment.warp_boxes(M, xyxy), width, height)
        kept = [
            BoundingBox(b.class_id, (c[0] + c[2]) / 2 / width, (c[1] + c[3]) / 2 / height, (c[2] - c[0]) / width, (c[3] - c[1]) / height)
            for b, c, k in zip(target_boxes, clipped, keep)
            if k
        ]
        return rgb, ir, *_boxes_to_tensors(kept)

    def _augmented(
        self, rgb: np.ndarray, ir: np.ndarray, boxes: torch.Tensor, labels: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        height, width = self.img_size
        aug = self.augment
        flip = -1.0 if torch.rand(()).item() < aug.get("fliplr", 0.0) else 1.0
        s = 1.0 + (2 * torch.rand(()).item() - 1) * aug.get("scale", 0.0)
        tx, ty = ((2 * torch.rand(2) - 1) * aug.get("translate", 0.0)).tolist()
        # Zoom (and flip) about the centre, then shift; in continuous pixel
        # coordinates, where the frame spans [0, width] x [0, height].
        A = np.array([
            [flip * s, 0.0, width / 2 * (1 - flip * s) + tx * width],
            [0.0, s, height / 2 * (1 - s) + ty * height],
        ])
        # cv2 maps pixel indices, whose centres sit at +0.5.
        A_px = A.copy()
        A_px[:, 2] += 0.5 * (A[[0, 1], [0, 1]] - 1)
        rgb = cv2.warpAffine(rgb, A_px, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
        ir = cv2.warpAffine(ir, A_px, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)

        if len(boxes) == 0:
            return rgb, ir, boxes, labels
        b = boxes.numpy().astype(np.float64)
        xyxy = np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2] * [width, height, width, height]
        clipped, keep = _clip_boxes(alignment.warp_boxes(np.r_[A, [[0, 0, 1]]], xyxy), width, height)
        clipped = clipped[keep] / [width, height, width, height]
        cxcywh = np.c_[(clipped[:, :2] + clipped[:, 2:]) / 2, clipped[:, 2:] - clipped[:, :2]]
        return rgb, ir, torch.from_numpy(cxcywh).float().reshape(-1, 4), labels[torch.from_numpy(keep)]

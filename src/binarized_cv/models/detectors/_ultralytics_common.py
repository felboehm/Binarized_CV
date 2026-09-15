from __future__ import annotations

import torch


def targets_to_ultralytics_batch(
    images: torch.Tensor, targets: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """ultralytics' `v8DetectionLoss`/`E2ELoss` expect one flat batch dict
    rather than a per-image list: `batch_idx` says which image each box in
    `bboxes` belongs to. `bboxes` is already normalized cxcywh — the same
    format our own `targets` use, so no coordinate conversion is needed.
    Shared by every ultralytics-backed detector (`yolo26`, `ms_yolov8`, ...)."""
    device = images.device
    counts = [t["boxes"].shape[0] for t in targets]
    if sum(counts) == 0:
        batch_idx = torch.zeros((0,), dtype=torch.float32, device=device)
        cls = torch.zeros((0,), dtype=torch.float32, device=device)
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
    else:
        batch_idx = torch.cat(
            [torch.full((n,), i, dtype=torch.float32, device=device) for i, n in enumerate(counts)]
        )
        cls = torch.cat([t["labels"].float() for t in targets])
        bboxes = torch.cat([t["boxes"] for t in targets])
    return {"img": images, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}


def clamp_xyxy_to_image(boxes: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
    """ultralytics' raw decoded boxes aren't clamped to image bounds (an
    untrained or low-confidence prediction can fall outside), so every
    detector clamps before returning to match the `BaseDetector` contract."""
    img_h, img_w = img_size
    x1 = boxes[:, 0].clamp(0, img_w)
    y1 = boxes[:, 1].clamp(0, img_h)
    x2 = boxes[:, 2].clamp(0, img_w)
    y2 = boxes[:, 3].clamp(0, img_h)
    return torch.stack([x1, y1, x2, y2], dim=-1)

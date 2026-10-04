from __future__ import annotations

import torch
from torchvision.ops import box_iou


def average_precision(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    area_range: tuple[float, float] | None = None,
) -> float:
    """Single-class VOC-style average precision (area under the precision/
    recall curve, using the max-precision-to-the-right envelope). Both
    arguments are per-image lists in the same `{"boxes" (xyxy), "scores",
    "labels"}` format a `BaseDetector` returns in eval mode / a dataset's
    `targets` dict provides (scores absent for `targets`).

    `area_range=(lo, hi)` (box area in pixels², `lo <= area < hi`) gives a
    COCO-style size-bucket AP, e.g. APS for small objects: ground truth
    outside the range is *ignored* rather than dropped — a prediction that
    matches it is neither TP nor FP — and so are unmatched predictions whose
    own area is outside the range.

    Placeholder for the fuller `mAP@0.5:0.95` + per-class breakdown planned
    in `CHECKLIST.md` section 7 — single class, single IoU threshold only.
    """
    def in_range(boxes: torch.Tensor) -> torch.Tensor:
        if area_range is None:
            return torch.ones(boxes.shape[0], dtype=torch.bool)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return (area >= area_range[0]) & (area < area_range[1])

    counted = [in_range(t["boxes"]) for t in targets]
    total_gt = int(sum(c.sum() for c in counted))
    if total_gt == 0:
        return 0.0

    scored: list[tuple[float, int, torch.Tensor]] = []
    for image_index, pred in enumerate(predictions):
        for box, score in zip(pred["boxes"], pred["scores"]):
            scored.append((score.item(), image_index, box))
    if not scored:
        return 0.0
    scored.sort(key=lambda entry: entry[0], reverse=True)

    matched = [torch.zeros(t["boxes"].shape[0], dtype=torch.bool) for t in targets]
    true_positive = torch.zeros(len(scored))
    false_positive = torch.zeros(len(scored))
    ignored = torch.zeros(len(scored), dtype=torch.bool)

    for rank, (_, image_index, box) in enumerate(scored):
        gt_boxes = targets[image_index]["boxes"]
        if gt_boxes.shape[0] == 0:
            if in_range(box.unsqueeze(0)).item():
                false_positive[rank] = 1
            else:
                ignored[rank] = True
            continue
        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        ious[matched[image_index]] = -1.0
        # Prefer a counted GT; fall back to an ignored (out-of-range) one.
        counted_ious = torch.where(counted[image_index], ious, torch.full_like(ious, -1.0))
        best_iou, best_gt = counted_ious.max(dim=0)
        if best_iou >= iou_threshold:
            true_positive[rank] = 1
            matched[image_index][best_gt] = True
            continue
        ignored_iou, ignored_gt = torch.where(counted[image_index], torch.full_like(ious, -1.0), ious).max(dim=0)
        if ignored_iou >= iou_threshold:
            matched[image_index][ignored_gt] = True
            ignored[rank] = True
        elif in_range(box.unsqueeze(0)).item():
            false_positive[rank] = 1
        else:
            ignored[rank] = True

    true_positive, false_positive = true_positive[~ignored], false_positive[~ignored]
    if true_positive.numel() == 0:
        return 0.0

    cum_tp = torch.cumsum(true_positive, dim=0)
    cum_fp = torch.cumsum(false_positive, dim=0)
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp)

    precision_envelope = precision.flip(0).cummax(dim=0).values.flip(0)
    recall_padded = torch.cat([torch.zeros(1), recall])
    precision_padded = torch.cat([precision_envelope[:1], precision_envelope])
    return torch.sum((recall_padded[1:] - recall_padded[:-1]) * precision_padded[1:]).item()

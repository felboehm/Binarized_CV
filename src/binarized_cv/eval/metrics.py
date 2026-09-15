from __future__ import annotations

import torch
from torchvision.ops import box_iou


def average_precision(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
) -> float:
    """Single-class VOC-style average precision (area under the precision/
    recall curve, using the max-precision-to-the-right envelope). Both
    arguments are per-image lists in the same `{"boxes" (xyxy), "scores",
    "labels"}` format a `BaseDetector` returns in eval mode / a dataset's
    `targets` dict provides (scores absent for `targets`).

    Placeholder for the fuller `mAP@0.5:0.95` + per-class breakdown planned
    in `CHECKLIST.md` section 7 — single class, single IoU threshold only.
    """
    total_gt = sum(t["boxes"].shape[0] for t in targets)
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

    for rank, (_, image_index, box) in enumerate(scored):
        gt_boxes = targets[image_index]["boxes"]
        if gt_boxes.shape[0] == 0:
            false_positive[rank] = 1
            continue
        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_gt = ious.max(dim=0)
        if best_iou >= iou_threshold and not matched[image_index][best_gt]:
            true_positive[rank] = 1
            matched[image_index][best_gt] = True
        else:
            false_positive[rank] = 1

    cum_tp = torch.cumsum(true_positive, dim=0)
    cum_fp = torch.cumsum(false_positive, dim=0)
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp)

    precision_envelope = precision.flip(0).cummax(dim=0).values.flip(0)
    recall_padded = torch.cat([torch.zeros(1), recall])
    precision_padded = torch.cat([precision_envelope[:1], precision_envelope])
    return torch.sum((recall_padded[1:] - recall_padded[:-1]) * precision_padded[1:]).item()

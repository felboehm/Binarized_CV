#!/usr/bin/env python
"""Detailed evaluation with visualizations and diagnostics."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from binarized_cv.data.collate import detection_collate
from binarized_cv.data.dataset import MultispectralPersonDataset
from binarized_cv.data.manifest import load_manifest
from binarized_cv.eval.metrics import average_precision
from binarized_cv.models.registry import build_model_from_config


def load_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="config", overrides=overrides if overrides is not None else sys.argv[1:])


def _convert_targets_to_xyxy_pixels(
    targets: list[dict[str, torch.Tensor]], img_size: tuple[int, int]
) -> list[dict[str, torch.Tensor]]:
    """Convert targets from cxcywh normalized [0,1] to xyxy pixels [0, img_w/h]."""
    img_w, img_h = img_size
    converted = []
    for target in targets:
        boxes_cxcywh = target["boxes"]
        if boxes_cxcywh.shape[0] == 0:
            converted.append(target)
            continue

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        converted.append({**target, "boxes": boxes_xyxy})
    return converted


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
        return 0.0

    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def box_area(box):
    """Compute area of box in xyxy format."""
    return (box[2] - box[0]) * (box[3] - box[1])


def get_size_category(area, img_area):
    """Categorize object by relative size."""
    rel_area = area / img_area
    if rel_area < 0.01:
        return "small"
    elif rel_area < 0.05:
        return "medium"
    else:
        return "large"


def main(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        cfg = load_config()

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg.model).to(device)
    if cfg.eval.checkpoint_path is not None:
        model.load_state_dict(torch.load(cfg.eval.checkpoint_path, map_location=device))
    model.eval()

    records = load_manifest(cfg.data.manifest_path)
    dataset = MultispectralPersonDataset(
        records,
        raw_root=cfg.data.raw_root,
        split="test",
        img_size=tuple(cfg.data.img_size),
        target_modality=cfg.data.target_modality,
    )
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, collate_fn=detection_collate)

    all_predictions, all_targets = [], []
    confidence_scores = []
    num_detections_per_image = []
    num_gts_per_image = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in tqdm(loader):
            images = {m: t.to(device) for m, t in batch["images"].items()}
            detections = model(images)
            all_predictions.extend({k: v.cpu() for k, v in d.items()} for d in detections)
            all_targets.extend(batch["targets"])

            for d in detections:
                num_detections_per_image.append(len(d["scores"]))
                if len(d["scores"]) > 0:
                    confidence_scores.extend(d["scores"].cpu().numpy().tolist())

            for t in batch["targets"]:
                num_gts_per_image.append(len(t["boxes"]))

    # Convert targets to match prediction coordinate format
    all_targets = _convert_targets_to_xyxy_pixels(all_targets, tuple(cfg.data.img_size))

    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)
    print(f"Total test images: {len(dataset)}")
    print(f"Total detections: {sum(num_detections_per_image)}")
    print(f"Total ground truths: {sum(num_gts_per_image)}")
    print(f"Images with detections: {sum(1 for n in num_detections_per_image if n > 0)}")
    print(f"Images with no detections: {sum(1 for n in num_detections_per_image if n == 0)}")

    if confidence_scores:
        print(f"\nConfidence scores:")
        print(f"  Min:  {min(confidence_scores):.4f}")
        print(f"  Max:  {max(confidence_scores):.4f}")
        print(f"  Mean: {np.mean(confidence_scores):.4f}")
        print(f"  Std:  {np.std(confidence_scores):.4f}")
        print(f"  Median: {np.median(confidence_scores):.4f}")

    print("\n" + "="*60)
    print("ACCURACY METRICS")
    print("="*60)
    ap50 = average_precision(all_predictions, all_targets, iou_threshold=0.5)
    ap75 = average_precision(all_predictions, all_targets, iou_threshold=0.75)
    ap95 = average_precision(all_predictions, all_targets, iou_threshold=0.95)
    print(f"AP@0.5:  {ap50:.4f}")
    print(f"AP@0.75: {ap75:.4f}")
    print(f"AP@0.95: {ap95:.4f}")

    # Per-size analysis
    print("\n" + "="*60)
    print("PER-SIZE ANALYSIS")
    print("="*60)

    size_stats = {"small": {"tp": 0, "fp": 0, "fn": 0}, "medium": {"tp": 0, "fp": 0, "fn": 0}, "large": {"tp": 0, "fp": 0, "fn": 0}}

    for pred, gt in zip(all_predictions, all_targets):
        img_area = cfg.data.img_size[0] * cfg.data.img_size[1]

        matched_gts = set()
        for i, pred_box in enumerate(pred["boxes"]):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt["boxes"]):
                if j in matched_gts:
                    continue
                iou = compute_iou(pred_box.numpy(), gt_box.numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= 0.5 and best_gt_idx >= 0:
                matched_gts.add(best_gt_idx)
                size = get_size_category(box_area(gt["boxes"][best_gt_idx].numpy()), img_area)
                size_stats[size]["tp"] += 1
            else:
                if len(gt["boxes"]) > 0:
                    closest_gt = min(gt["boxes"], key=lambda b: compute_iou(pred_box.numpy(), b.numpy()))
                    size = get_size_category(box_area(closest_gt.numpy()), img_area)
                    size_stats[size]["fp"] += 1

        for j, gt_box in enumerate(gt["boxes"]):
            if j not in matched_gts:
                size = get_size_category(box_area(gt_box.numpy()), img_area)
                size_stats[size]["fn"] += 1

    for size, stats in size_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"\n{size.upper()} objects (area < {['0.01', '0.05', 'inf'][['small', 'medium', 'large'].index(size)]}×img):")
        print(f"  Count: {tp+fn}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Confidence distribution
    if confidence_scores:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(confidence_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(confidence_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidence_scores):.3f}')
        ax.axvline(np.median(confidence_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(confidence_scores):.3f}')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Prediction Confidence Scores', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/tmp/confidence_distribution.png', dpi=100)
        print("✓ Saved: /tmp/confidence_distribution.png")

    # 2. Detections per image
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = max(1, max(max(num_detections_per_image), max(num_gts_per_image))//5 + 1)
    ax.hist(num_detections_per_image, bins=bins, edgecolor='black', alpha=0.6, label='Predictions', color='steelblue')
    ax.hist(num_gts_per_image, bins=bins, edgecolor='black', alpha=0.6, label='Ground Truth', color='orange')
    ax.set_xlabel('Number of Objects per Image', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Object Count Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('/tmp/object_count_distribution.png', dpi=100)
    print("✓ Saved: /tmp/object_count_distribution.png")

    # 3. AP vs confidence threshold curve
    if confidence_scores:
        thresholds = np.linspace(0, 1, 50)
        aps = []
        for threshold in thresholds:
            filtered_preds = []
            for pred in all_predictions:
                keep = pred["scores"] > threshold
                filtered_preds.append({
                    "boxes": pred["boxes"][keep],
                    "scores": pred["scores"][keep],
                    "labels": pred["labels"][keep] if "labels" in pred else torch.zeros(keep.sum(), dtype=torch.long)
                })
            ap = average_precision(filtered_preds, all_targets, iou_threshold=0.5)
            aps.append(ap)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(thresholds, aps, linewidth=2.5, color='steelblue', marker='o', markersize=3, alpha=0.8)
        ax.fill_between(thresholds, aps, alpha=0.2, color='steelblue')
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('AP@0.5', fontsize=12)
        ax.set_title('Average Precision vs Confidence Threshold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig('/tmp/ap_vs_threshold.png', dpi=100)
        print("✓ Saved: /tmp/ap_vs_threshold.png")


if __name__ == "__main__":
    main()

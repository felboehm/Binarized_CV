#!/usr/bin/env python
"""Comprehensive evaluation with visualizations and sample detections.

Results are automatically saved to runs/eval_results/{model_name}_{timestamp}/
including:
  - metrics.json: All computed metrics
  - results_summary.txt: Human-readable summary
  - *.png: All visualizations
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from binarized_cv.data.collate import detection_collate
from binarized_cv.data.dataset import MultispectralPersonDataset
from binarized_cv.data.manifest import load_manifest
from binarized_cv.data.records import PairRecord
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


def draw_boxes_on_image(
    image: np.ndarray,
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    title: str = "",
) -> None:
    """Draw GT boxes (green) and predicted boxes (red) on image."""
    ax = plt.gca()
    ax.imshow(image)

    # Draw GT boxes (green)
    for box in gt_boxes:
        x1, y1, x2, y2 = box.numpy()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='green', facecolor='none', label='GT' if box is gt_boxes[0] else ""
        )
        ax.add_patch(rect)

    # Draw predicted boxes (red/orange) with scores
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = box.numpy()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none', label='Pred' if box is pred_boxes[0] else ""
        )
        ax.add_patch(rect)
        # Add confidence score as text
        ax.text(x1, max(y1 - 3, 10), f'{score:.2f}', color='red', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, pad=1))

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')


def main(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        cfg = load_config()

    # Setup results directory with timestamp
    results_dir = Path(__file__).parent.parent / "runs" / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = results_dir / f"{cfg.model.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Results will be saved to: {run_dir}\n")

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
    all_images = []  # Store for visualization
    all_records = []
    confidence_scores = []
    num_detections_per_image = []
    num_gts_per_image = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            images = {m: t.to(device) for m, t in batch["images"].items()}
            detections = model(images)
            all_predictions.extend({k: v.cpu() for k, v in d.items()} for d in detections)
            all_targets.extend(batch["targets"])

            # Store RGB images for visualization (denormalize)
            rgb_images = (images["rgb"].cpu() * 255).byte().permute(0, 2, 3, 1).numpy()
            all_images.extend([img for img in rgb_images])

            # Store dataset records for metadata
            batch_size = len(batch["targets"])
            start_idx = batch_idx * cfg.data.batch_size
            for i in range(batch_size):
                idx = start_idx + i
                if idx < len(dataset.records):
                    all_records.append(dataset.records[idx])

            for d in detections:
                num_detections_per_image.append(len(d["scores"]))
                if len(d["scores"]) > 0:
                    confidence_scores.extend(d["scores"].cpu().numpy().tolist())

            for t in batch["targets"]:
                num_gts_per_image.append(len(t["boxes"]))

    # Convert targets to match prediction coordinate format
    all_targets = _convert_targets_to_xyxy_pixels(all_targets, tuple(cfg.data.img_size))

    # ===== Compute Metrics =====
    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)

    metrics = {
        "timestamp": timestamp,
        "model": cfg.model.name,
        "checkpoint": str(cfg.eval.checkpoint_path),
        "img_size": list(cfg.data.img_size) if hasattr(cfg.data.img_size, '__iter__') else cfg.data.img_size,
    }

    stats = {
        "total_test_images": len(dataset),
        "total_detections": sum(num_detections_per_image),
        "total_ground_truths": sum(num_gts_per_image),
        "images_with_detections": sum(1 for n in num_detections_per_image if n > 0),
        "images_without_detections": sum(1 for n in num_detections_per_image if n == 0),
    }
    metrics["statistics"] = stats

    print(f"Total test images: {stats['total_test_images']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Total ground truths: {stats['total_ground_truths']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Images with no detections: {stats['images_without_detections']}")

    confidence_stats = {}
    if confidence_scores:
        print(f"\nConfidence scores:")
        confidence_stats = {
            "min": float(np.min(confidence_scores)),
            "max": float(np.max(confidence_scores)),
            "mean": float(np.mean(confidence_scores)),
            "std": float(np.std(confidence_scores)),
            "median": float(np.median(confidence_scores)),
        }
        metrics["confidence"] = confidence_stats
        print(f"  Min:  {confidence_stats['min']:.4f}")
        print(f"  Max:  {confidence_stats['max']:.4f}")
        print(f"  Mean: {confidence_stats['mean']:.4f}")
        print(f"  Std:  {confidence_stats['std']:.4f}")
        print(f"  Median: {confidence_stats['median']:.4f}")

    print("\n" + "="*60)
    print("ACCURACY METRICS")
    print("="*60)
    ap50 = average_precision(all_predictions, all_targets, iou_threshold=0.5)
    ap75 = average_precision(all_predictions, all_targets, iou_threshold=0.75)
    ap95 = average_precision(all_predictions, all_targets, iou_threshold=0.95)
    print(f"AP@0.5:  {ap50:.4f}")
    print(f"AP@0.75: {ap75:.4f}")
    print(f"AP@0.95: {ap95:.4f}")

    # Calculate Precision, Recall, F1 at IoU=0.5 threshold
    print("\nPrecision, Recall, F1 @ IoU=0.5:")
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(all_predictions, all_targets):
        matched_gts = set()
        for pred_box in pred["boxes"]:
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
                total_tp += 1
            else:
                total_fp += 1

        total_fn += len(gt["boxes"]) - len(matched_gts)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  Precision: {precision:.4f} (TP={total_tp}, FP={total_fp})")
    print(f"  Recall:    {recall:.4f} (TP={total_tp}, FN={total_fn})")
    print(f"  F1 Score:  {f1:.4f}")

    metrics["accuracy"] = {
        "AP@0.5": float(ap50),
        "AP@0.75": float(ap75),
        "AP@0.95": float(ap95),
        "Precision@0.5": float(precision),
        "Recall@0.5": float(recall),
        "F1@0.5": float(f1),
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
    }

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

    size_metrics = {}
    for size, stats_dict in size_stats.items():
        tp, fp, fn = stats_dict["tp"], stats_dict["fp"], stats_dict["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        size_metrics[size] = {
            "count": tp + fn,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": float(precision),
            "recall": float(recall),
        }
        print(f"\n{size.upper()} objects (area < {['0.01', '0.05', 'inf'][['small', 'medium', 'large'].index(size)]}×img):")
        print(f"  Count: {tp+fn}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")

    metrics["per_size_analysis"] = size_metrics

    # ===== Generate Visualizations =====
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
        save_path = run_dir / 'confidence_distribution.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    # 2. Detections per image (split view: empty vs non-empty images)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Empty vs Non-empty images
    empty_pred = sum(1 for n in num_detections_per_image if n == 0)
    nonempty_pred = sum(1 for n in num_detections_per_image if n > 0)
    empty_gt = sum(1 for n in num_gts_per_image if n == 0)
    nonempty_gt = sum(1 for n in num_gts_per_image if n > 0)

    categories = ['Empty\n(0 objects)', 'Non-Empty\n(1+ objects)']
    pred_counts = [empty_pred, nonempty_pred]
    gt_counts = [empty_gt, nonempty_gt]

    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width/2, pred_counts, width, label='Predictions', color='steelblue', edgecolor='black')
    axes[0].bar(x + width/2, gt_counts, width, label='Ground Truth', color='orange', edgecolor='black')
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].set_title('Images: Empty vs Non-Empty', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (p, g) in enumerate(zip(pred_counts, gt_counts)):
        axes[0].text(i - width/2, p + 20, str(p), ha='center', fontweight='bold')
        axes[0].text(i + width/2, g + 20, str(g), ha='center', fontweight='bold')

    # Right: Distribution for non-empty images only
    pred_nonzero = [n for n in num_detections_per_image if n > 0]
    gt_nonzero = [n for n in num_gts_per_image if n > 0]

    max_obj = max(max(pred_nonzero) if pred_nonzero else 1, max(gt_nonzero) if gt_nonzero else 1)
    bins_range = range(1, min(max_obj + 2, 10))  # Show up to 9 objects

    axes[1].hist([pred_nonzero, gt_nonzero], bins=bins_range, label=['Predictions', 'Ground Truth'],
                 color=['steelblue', 'orange'], alpha=0.6, edgecolor='black')
    axes[1].set_xlabel('Number of Objects per Image', fontsize=12)
    axes[1].set_ylabel('Number of Images (non-empty only)', fontsize=12)
    axes[1].set_title('Distribution for Non-Empty Images', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = run_dir / 'object_count_distribution.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

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
        save_path = run_dir / 'ap_vs_threshold.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    # 3b. Precision-Recall Curve at different confidence thresholds
    if confidence_scores:
        thresholds = np.linspace(0, 1, 50)
        precisions = []
        recalls = []

        for threshold in thresholds:
            filtered_preds = []
            for pred in all_predictions:
                keep = pred["scores"] > threshold
                filtered_preds.append({
                    "boxes": pred["boxes"][keep],
                    "scores": pred["scores"][keep],
                    "labels": pred["labels"][keep] if "labels" in pred else torch.zeros(keep.sum(), dtype=torch.long)
                })

            # Calculate precision and recall at this threshold
            tp = fp = fn = 0
            for pred, gt in zip(filtered_preds, all_targets):
                matched_gts = set()
                for pred_box in pred["boxes"]:
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
                        tp += 1
                    else:
                        fp += 1

                fn += len(gt["boxes"]) - len(matched_gts)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            precisions.append(prec)
            recalls.append(rec)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(recalls, precisions, linewidth=2.5, color='steelblue', marker='o', markersize=4, alpha=0.8)
        ax.fill_between(recalls, precisions, alpha=0.2, color='steelblue')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve @ IoU=0.5', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        save_path = run_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    # 4. Sample detections on 5 random images
    print("\nGenerating sample detection visualizations...")

    # Find images with detections and ground truths
    valid_indices = [i for i in range(len(all_predictions))
                     if len(all_predictions[i]["boxes"]) > 0 and len(all_targets[i]["boxes"]) > 0]

    if not valid_indices:
        print("⚠ No images with both detections and ground truths found")
    else:
        num_samples = min(5, len(valid_indices))
        sample_indices = random.sample(valid_indices, num_samples)
        sample_indices.sort()

        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        if num_samples == 1:
            axes = [axes]

        for subplot_idx, img_idx in enumerate(sample_indices):
            plt.sca(axes[subplot_idx])

            rgb_img = all_images[img_idx]
            gt_boxes = all_targets[img_idx]["boxes"]
            pred_boxes = all_predictions[img_idx]["boxes"]
            pred_scores = all_predictions[img_idx]["scores"]

            # Count matches
            matched = 0
            for pred_box in pred_boxes:
                for gt_box in gt_boxes:
                    if compute_iou(pred_box.numpy(), gt_box.numpy()) >= 0.5:
                        matched += 1
                        break

            title = f"Image {img_idx}\nGT: {len(gt_boxes)}, Pred: {len(pred_boxes)}, Match: {matched}"
            draw_boxes_on_image(rgb_img, gt_boxes, pred_boxes, pred_scores, title=title)

        plt.tight_layout()
        save_path = run_dir / 'sample_detections.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")
        print(f"  Randomly selected {num_samples} images with detections and ground truths")

    # ===== Save Results Files =====
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save metrics as JSON
    metrics_json_path = run_dir / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_json_path}")

    # Save human-readable summary
    summary_path = run_dir / 'results_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {cfg.model.name}\n")
        f.write(f"Checkpoint: {cfg.eval.checkpoint_path}\n")
        f.write(f"Image Size: {cfg.data.img_size}\n\n")

        f.write("-"*70 + "\n")
        f.write("DETECTION STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total test images:          {stats['total_test_images']}\n")
        f.write(f"Total detections:           {stats['total_detections']}\n")
        f.write(f"Total ground truths:        {stats['total_ground_truths']}\n")
        f.write(f"Images with detections:     {stats['images_with_detections']}\n")
        f.write(f"Images without detections:  {stats['images_without_detections']}\n\n")

        if confidence_stats:
            f.write("Confidence Scores:\n")
            f.write(f"  Min:     {confidence_stats['min']:.4f}\n")
            f.write(f"  Max:     {confidence_stats['max']:.4f}\n")
            f.write(f"  Mean:    {confidence_stats['mean']:.4f}\n")
            f.write(f"  Std:     {confidence_stats['std']:.4f}\n")
            f.write(f"  Median:  {confidence_stats['median']:.4f}\n\n")

        f.write("-"*70 + "\n")
        f.write("ACCURACY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"AP@0.5:       {metrics['accuracy']['AP@0.5']:.4f}\n")
        f.write(f"AP@0.75:      {metrics['accuracy']['AP@0.75']:.4f}\n")
        f.write(f"AP@0.95:      {metrics['accuracy']['AP@0.95']:.4f}\n\n")

        f.write("Precision, Recall, F1 @ IoU=0.5:\n")
        f.write(f"  Precision:  {metrics['accuracy']['Precision@0.5']:.4f} (TP={metrics['accuracy']['TP']}, FP={metrics['accuracy']['FP']})\n")
        f.write(f"  Recall:     {metrics['accuracy']['Recall@0.5']:.4f} (TP={metrics['accuracy']['TP']}, FN={metrics['accuracy']['FN']})\n")
        f.write(f"  F1 Score:   {metrics['accuracy']['F1@0.5']:.4f}\n\n")

        f.write("-"*70 + "\n")
        f.write("PER-SIZE ANALYSIS\n")
        f.write("-"*70 + "\n")
        for size in ['small', 'medium', 'large']:
            sm = size_metrics[size]
            f.write(f"\n{size.upper()} Objects:\n")
            f.write(f"  Count:      {sm['count']}\n")
            f.write(f"  TP:         {sm['TP']}\n")
            f.write(f"  FP:         {sm['FP']}\n")
            f.write(f"  FN:         {sm['FN']}\n")
            f.write(f"  Precision:  {sm['precision']:.4f}\n")
            f.write(f"  Recall:     {sm['recall']:.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("VISUALIZATIONS GENERATED\n")
        f.write("="*70 + "\n")
        f.write("- confidence_distribution.png:    Confidence score histogram\n")
        f.write("- object_count_distribution.png:  Predicted vs GT counts (empty vs non-empty)\n")
        f.write("- ap_vs_threshold.png:            AP@0.5 vs confidence threshold\n")
        f.write("- precision_recall_curve.png:     Precision-Recall curve @ IoU=0.5\n")
        f.write("- sample_detections.png:          5 random test images with boxes\n")

    print(f"✓ Summary saved: {summary_path}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"✓ All results saved to: {run_dir}")
    print(f"  - metrics.json (machine-readable)")
    print(f"  - results_summary.txt (human-readable)")
    print(f"  - 4 PNG visualizations")


if __name__ == "__main__":
    main()

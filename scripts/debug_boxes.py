#!/usr/bin/env python
"""Debug predictions vs targets to find coordinate mismatch."""

import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from binarized_cv.data.collate import detection_collate
from binarized_cv.data.dataset import MultispectralPersonDataset
from binarized_cv.data.manifest import load_manifest
from binarized_cv.models.registry import build_model_from_config


def load_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="config", overrides=overrides if overrides is not None else sys.argv[1:])


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
    loader = DataLoader(dataset, batch_size=1, collate_fn=detection_collate)

    print(f"Image size from config: {cfg.data.img_size}")
    print(f"Target modality: {cfg.data.target_modality}\n")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3:  # Just look at first 3 images
                break

            images = {m: t.to(device) for m, t in batch["images"].items()}
            targets = batch["targets"][0]  # batch size 1
            detections = model(images)
            preds = detections[0]

            print(f"=== Image {i} ===")
            print(f"Image shapes: {[(m, v.shape) for m, v in images.items()]}")
            print(f"Number of GT boxes: {len(targets['boxes'])}")
            print(f"Number of predictions: {len(preds['scores'])}")

            if len(targets["boxes"]) > 0:
                gt_box = targets["boxes"][0]
                print(f"\nFirst GT box (xyxy, normalized): {gt_box}")
                print(f"  x range: {gt_box[0]:.4f} to {gt_box[2]:.4f}")
                print(f"  y range: {gt_box[1]:.4f} to {gt_box[3]:.4f}")

            if len(preds["scores"]) > 0:
                # Get prediction with highest confidence
                best_idx = preds["scores"].argmax()
                pred_box = preds["boxes"][best_idx]
                score = preds["scores"][best_idx]
                print(f"\nBest prediction box (xyxy, pixels): {pred_box}")
                print(f"  Confidence: {score:.4f}")
                print(f"  x range: {pred_box[0]:.1f} to {pred_box[2]:.1f}")
                print(f"  y range: {pred_box[1]:.1f} to {pred_box[3]:.1f}")

                # Check if predictions are in pixel space [0, img_size] or normalized [0, 1]
                if pred_box.max() > 2:
                    print(f"  → Predictions are in PIXEL space (values > 2)")
                else:
                    print(f"  → Predictions are in NORMALIZED space (values <= 1)")

                # Check all prediction ranges
                print(f"\nAll prediction stats:")
                print(f"  Min value: {preds['boxes'].min():.4f}")
                print(f"  Max value: {preds['boxes'].max():.4f}")

            print()


if __name__ == "__main__":
    main()

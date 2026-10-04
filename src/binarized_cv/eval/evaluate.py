from __future__ import annotations

import sys

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from binarized_cv.data.collate import detection_collate
from binarized_cv.data.dataset import MultispectralPersonDataset
from binarized_cv.data.manifest import load_manifest
from binarized_cv.eval.metrics import average_precision
from binarized_cv.models.registry import build_model_from_config
from binarized_cv.train.train import env_overrides


def load_config(overrides: list[str] | None = None) -> DictConfig:
    # See binarized_cv.train.train.load_config: works around hydra-core 1.3.6's
    # `@hydra.main` CLI parser being broken on Python 3.14.
    overrides = env_overrides() + (overrides if overrides is not None else sys.argv[1:])
    with initialize(version_base=None, config_path="../../../configs"):
        return compose(config_name="config", overrides=overrides)


def _convert_targets_to_xyxy_pixels(
    targets: list[dict[str, torch.Tensor]], img_size: tuple[int, int]
) -> list[dict[str, torch.Tensor]]:
    """Convert targets from cxcywh normalized [0,1] to xyxy pixels [0, img_w/h]."""
    img_w, img_h = img_size
    converted = []
    for target in targets:
        boxes_cxcywh = target["boxes"]  # (N, 4) in [0, 1]
        if boxes_cxcywh.shape[0] == 0:
            converted.append(target)
            continue

        # cxcywh normalized to xyxy pixels
        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        converted.append({**target, "boxes": boxes_xyxy})
    return converted


def evaluate_model(model: torch.nn.Module, cfg: DictConfig, split: str = "test") -> dict[str, float]:
    """AP@0.5 overall and for small objects (APS, `cfg.eval.small_max_area`)
    on `split`. Shared by this CLI and `scripts/binarization_sweep.py`."""
    device = next(model.parameters()).device
    model.eval()

    records = load_manifest(cfg.data.manifest_path)
    dataset = MultispectralPersonDataset(
        records,
        raw_root=cfg.data.raw_root,
        split=split,
        img_size=tuple(cfg.data.img_size),
        target_modality=cfg.data.target_modality,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, collate_fn=detection_collate
    )

    all_predictions, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            images = {m: t.to(device) for m, t in batch["images"].items()}
            detections = model(images)
            all_predictions.extend({k: v.cpu() for k, v in d.items()} for d in detections)
            all_targets.extend(batch["targets"])

    # Convert targets from cxcywh normalized to xyxy pixels to match predictions
    all_targets = _convert_targets_to_xyxy_pixels(all_targets, tuple(cfg.data.img_size))

    small_max_area = cfg.eval.get("small_max_area", 32**2)
    return {
        "ap50": average_precision(all_predictions, all_targets, iou_threshold=0.5),
        "ap50_small": average_precision(
            all_predictions, all_targets, iou_threshold=0.5, area_range=(0.0, small_max_area)
        ),
    }


def main(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        cfg = load_config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg.model).to(device)
    if cfg.eval.checkpoint_path is not None:
        model.load_state_dict(torch.load(cfg.eval.checkpoint_path, map_location=device))

    metrics = evaluate_model(model, cfg)
    print(f"AP@0.5: {metrics['ap50']:.4f}")
    print(f"APS@0.5 (area < {cfg.eval.small_max_area} px²): {metrics['ap50_small']:.4f}")


if __name__ == "__main__":
    main()

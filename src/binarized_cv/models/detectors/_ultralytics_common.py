from __future__ import annotations

import logging

import torch
from torch import nn

from binarized_cv.models.binarized import apply_binarization, format_region_stats

log = logging.getLogger(__name__)


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


def load_pretrained(model: nn.Module, scale: str) -> None:
    """Load ultralytics' COCO-pretrained `yolo26{scale}.pt` into `model` (a
    `DetectionModel`, or mid fusion's IR stream, whose layers are also in
    `.model`). Downloads the file into the working directory if missing.

    `BaseModel.load` semantics: every tensor with a matching name and shape
    is copied, so the 80-class output convs stay at their init, and a first
    conv with more input channels gets the RGB weights in its first three
    (early fusion's IR channel stays random). A 1-channel first conv (IR)
    gets the sum over the RGB channels instead of ultralytics' red channel
    only: the response to a gray image, which IR is closest to. Raises if
    (almost) nothing matches, rather than training from scratch unnoticed."""
    from ultralytics.nn.tasks import BaseModel
    from ultralytics.utils.downloads import attempt_download_asset

    path = attempt_download_asset(f"yolo26{scale}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # ultralytics' own pickled model
    source = checkpoint["model"].float().state_dict()
    target = model.state_dict()
    matched = [k for k, v in source.items() if k in target and v.shape == target[k].shape]
    if len(matched) < len(target) // 2:
        raise RuntimeError(f"{path}: only {len(matched)}/{len(target)} tensors match the model")
    BaseModel.load(model, checkpoint, verbose=False)
    first = "model.0.conv.weight"
    if first in target and target[first].shape[1] == 1:
        with torch.no_grad():
            model.state_dict()[first].copy_(source[first].sum(dim=1, keepdim=True))
    log.info("Loaded %d/%d tensors from %s", len(matched), len(target), path)


def warm_start_and_binarize(
    detector: nn.Module, binarization: dict | None, init_checkpoint: str | None
) -> None:
    """Optionally load one of our own fp32 training checkpoints into
    `detector` (BNN warm start), then swap `detector.model`'s convs for
    quantized ones per `binarization` (see
    `binarized_cv.models.binarized.policy`). Order matters: latent binary
    weights are initialised from whatever fp32 weights are loaded first.
    No-op when both arguments are `None`."""
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = detector.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"{init_checkpoint} is missing {len(missing)} keys, e.g. {missing[:3]}")
        if unexpected:
            log.warning("Ignoring %d unexpected keys in %s, e.g. %s", len(unexpected), init_checkpoint, unexpected[:3])
    if binarization is not None:
        stats = apply_binarization(detector.model, binarization)
        log.info("Binarization (preset=%s):\n%s", binarization.get("preset"), format_region_stats(stats))

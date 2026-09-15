from __future__ import annotations

import torch

_MODALITIES = ("rgb", "ir")


def detection_collate(batch: list[dict]) -> dict:
    """Batches `MultispectralPersonDataset` items into the shape every
    `BaseDetector` expects: images stacked into one tensor per modality,
    targets kept as a per-image list since box counts vary."""
    return {
        "images": {m: torch.stack([item[m] for item in batch]) for m in _MODALITIES},
        "targets": [item["targets"] for item in batch],
        "id": [item["id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
    }

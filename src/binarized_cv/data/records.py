from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairRecord:
    id: str
    dataset: str
    split: str
    rgb_image: str
    rgb_label: str
    ir_image: str
    ir_label: str

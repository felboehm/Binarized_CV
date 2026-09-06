from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from binarized_cv.data.datasets.trgb import discover_trgb
from binarized_cv.data.datasets.wisard import discover_wisard
from binarized_cv.data.records import PairRecord


def build_manifest(raw_root: Path) -> list[PairRecord]:
    raw_root = Path(raw_root)
    return discover_trgb(raw_root) + discover_wisard(raw_root)


def save_manifest(records: list[PairRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(dataclasses.asdict(record)) + "\n")


def load_manifest(path: Path) -> list[PairRecord]:
    records = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(PairRecord(**json.loads(line)))
    return records

from __future__ import annotations

import re
from pathlib import Path

from binarized_cv.data.records import PairRecord

_SPLITS = ("train", "val", "test")

# Filenames come in two conventions within the same folder: a bare numeric
# id ("8251066.jpg") or a modality-prefixed id ("RGB_1505.jpg" / "IR_1505.jpg")
# — both must be canonicalized to the same key to pair across modalities.
_PREFIXED_ID = re.compile(r"^(?:RGB|IR)_(?P<id>\d+)$", re.IGNORECASE)


def _find_modality_dir(split_dir: Path, prefix: str) -> Path:
    candidates = sorted(
        p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    if not candidates:
        raise FileNotFoundError(f"No directory starting with {prefix!r} found in {split_dir}")
    return candidates[0]


def _canonical_id(stem: str) -> str:
    match = _PREFIXED_ID.match(stem)
    return match.group("id") if match else stem


def _stems_by_canonical_id(modality_dir: Path) -> dict[str, str]:
    return {
        _canonical_id(p.stem): p.stem
        for p in modality_dir.glob("*.jpg")
        if not p.name.startswith(".")
    }


def discover_trgb(raw_root: Path, dataset_dir_name: str = "trgb") -> list[PairRecord]:
    raw_root = Path(raw_root)
    dataset_root = raw_root / dataset_dir_name / "trgb_dataset"
    records: list[PairRecord] = []
    for split in _SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        rgb_dir = _find_modality_dir(split_dir, "RGB_images_")
        ir_dir = _find_modality_dir(split_dir, "IR_images_")
        rgb_stems = _stems_by_canonical_id(rgb_dir)
        ir_stems = _stems_by_canonical_id(ir_dir)
        for canonical_id in sorted(rgb_stems.keys() & ir_stems.keys()):
            rgb_stem, ir_stem = rgb_stems[canonical_id], ir_stems[canonical_id]
            records.append(
                PairRecord(
                    id=f"trgb_{split}_{canonical_id}",
                    dataset="trgb",
                    split=split,
                    rgb_image=(rgb_dir / f"{rgb_stem}.jpg").relative_to(raw_root).as_posix(),
                    rgb_label=(rgb_dir / f"{rgb_stem}.txt").relative_to(raw_root).as_posix(),
                    ir_image=(ir_dir / f"{ir_stem}.jpg").relative_to(raw_root).as_posix(),
                    ir_label=(ir_dir / f"{ir_stem}.txt").relative_to(raw_root).as_posix(),
                )
            )
    return records

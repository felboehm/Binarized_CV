from __future__ import annotations

import re
from itertools import chain
from pathlib import Path

from binarized_cv.data.records import PairRecord

# Match both old format (_VIS_N, _IR_N) and new format (_FLIR_VIS_N, _FLIR_IR_N)
_DIR_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_(?:FLIR_)?(?P<modality>VIS|IR)(?:_(?P<seq>\d+))?$"
)
# Match frame numbers: both 8-digit (00000000) and 5-digit (00000) formats
_FRAME_PATTERN = re.compile(r"_(?P<frame>\d{5,8})\.jpe?g$", re.IGNORECASE)

# Whole flights per split, so near-identical consecutive frames never end up
# on both sides (the earlier frame-index split put frames 69 and 70 of every
# sequence in train and val). Paired frames: train FHL 8,758 + Hannegan 1,136,
# val MtErie 707, test Baker 2,180. Carnation 2,052 is its own split "zoom",
# outside train/val/test: its VIS is zoomed and changes zoom mid-flight, so
# no fixed IR -> VIS transform fits (scripts/align_manifest.py); kept for a
# separate robustness check. Flights not listed go to train.
FLIGHT_SPLITS = {
    "210417_MtErie_Enterprise": "val",
    "220109_Baker_Enterprise": "test",
    "210529_Carnation_Enterprise": "zoom",
}

# Sequences whose labels are missing in one modality while the other one has
# people labeled (`count.txt` says 0 humans, no label files). Kept in the
# manifest (to label from the other modality later) but flagged, so they don't
# train as negatives. In FHL VIS_0403 people are clearly visible in VIS;
# VIS_0566 is a night flight (VIS nearly black).
VIS_UNLABELED = {
    "210924_FHL_Enterprise_VIS_0403",
    "210924_FHL_Enterprise_VIS_0564",
    "210924_FHL_Enterprise_VIS_0566",
}
IR_UNLABELED = {
    "210529_Carnation_Enterprise_IR_0026",
    "210924_FHL_Enterprise_IR_0127",
    "210924_FHL_Enterprise_IR_0408",
}


def _group_flight_dirs(wisard_root: Path) -> dict[str, dict[str, list[Path]]]:
    """Group flight directories by prefix, handling both old and new naming conventions.

    Old format: {prefix}_VIS_{seq} and {prefix}_IR_{seq}
    New format: {prefix}_FLIR_VIS_{seq} and {prefix}_FLIR_IR_{seq}

    Returns a dict mapping flight prefix to {"vis": [dir1, dir2, ...], "ir": [dir1, dir2, ...]}.
    Directories are sorted by name to ensure consistent pairing.
    """
    groups: dict[str, dict[str, list[Path]]] = {}
    for entry in sorted(p for p in wisard_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        match = _DIR_PATTERN.match(entry.name)
        if not match:
            continue
        prefix = match.group("prefix")
        modality = match.group("modality").lower()
        groups.setdefault(prefix, {}).setdefault(modality, []).append(entry)
    return groups


def _frame_stems(modality_dir: Path) -> dict[int, tuple[str, str]]:
    """Extract frame numbers and stems from image files in a modality directory.

    Supports both .jpg and .jpeg extensions.
    Returns dict mapping frame number to (stem, extension). Keyed by int: the
    two modalities of one pair can zero-pad differently (VIS `_00000000`, IR
    `_00000`), which as strings would never match.
    """
    frames: dict[int, tuple[str, str]] = {}
    for p in chain(modality_dir.glob("*.jpg"), modality_dir.glob("*.jpeg")):
        match = _FRAME_PATTERN.search(p.name)
        if match:
            frames[int(match.group("frame"))] = (p.stem, p.suffix)
    return frames


def discover_wisard(
    raw_root: Path, dataset_dir_name: str = "wisard"
) -> list[PairRecord]:
    raw_root = Path(raw_root)
    wisard_root = raw_root / dataset_dir_name
    records: list[PairRecord] = []

    for prefix, modality_dirs in sorted(_group_flight_dirs(wisard_root).items()):
        # Skip Airfield: VIS and IR are captured from different angles (mirror images),
        # making them impossible to align. Forest appears on opposite sides in each modality.
        if "Airfield" in prefix:
            continue

        # Handle case where there are multiple VIS/IR subdirectories per flight
        # (e.g., VIS_0003, VIS_0005 paired with IR_0004, IR_0006)
        vis_dirs = sorted(modality_dirs.get("vis", []))
        ir_dirs = sorted(modality_dirs.get("ir", []))

        if not vis_dirs or not ir_dirs:
            continue

        split = FLIGHT_SPLITS.get(prefix, "train")
        # Pair VIS and IR directories by index (assumes sorted order matches intended pairing)
        for vis_dir, ir_dir in zip(vis_dirs, ir_dirs):
            vis_frames, ir_frames = _frame_stems(vis_dir), _frame_stems(ir_dir)
            # Only use frames that exist in both modalities
            for frame in sorted(vis_frames.keys() & ir_frames.keys()):
                vis_stem, vis_ext = vis_frames[frame]
                ir_stem, ir_ext = ir_frames[frame]
                records.append(
                    PairRecord(
                        # Sequence, not flight: every sequence restarts its frame numbers.
                        id=f"wisard_{vis_dir.name}_{frame:08d}",
                        dataset="wisard",
                        split=split,
                        rgb_image=(vis_dir / f"{vis_stem}{vis_ext}").relative_to(raw_root).as_posix(),
                        rgb_label=(vis_dir / f"{vis_stem}.txt").relative_to(raw_root).as_posix(),
                        ir_image=(ir_dir / f"{ir_stem}{ir_ext}").relative_to(raw_root).as_posix(),
                        ir_label=(ir_dir / f"{ir_stem}.txt").relative_to(raw_root).as_posix(),
                        rgb_labeled=vis_dir.name not in VIS_UNLABELED,
                        ir_labeled=ir_dir.name not in IR_UNLABELED,
                    )
                )
    return records

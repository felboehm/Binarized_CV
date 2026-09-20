from __future__ import annotations

import re
from pathlib import Path

from binarized_cv.data.records import PairRecord

# Match both old format (_VIS_N, _IR_N) and new format (_FLIR_VIS_N, _FLIR_IR_N)
_DIR_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_(?:FLIR_)?(?P<modality>VIS|IR)(?:_(?P<seq>\d+))?$"
)
_FRAME_PATTERN = re.compile(r"_(?P<frame>\d{8})\.jpe?g$", re.IGNORECASE)


def _group_flight_dirs(wisard_root: Path) -> dict[str, dict[str, Path]]:
    """Group flight directories by prefix, handling both old and new naming conventions.

    Old format: {prefix}_VIS_{seq} and {prefix}_IR_{seq}
    New format: {prefix}_FLIR_VIS_{seq} and {prefix}_FLIR_IR_{seq}
    """
    groups: dict[str, dict[str, Path]] = {}
    for entry in sorted(p for p in wisard_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        match = _DIR_PATTERN.match(entry.name)
        if not match:
            continue
        prefix = match.group("prefix")
        modality = match.group("modality").lower()
        groups.setdefault(prefix, {})[modality] = entry
    return groups


def _frame_stems(modality_dir: Path) -> dict[str, str]:
    """Extract frame numbers and stems from image files in a modality directory.

    Supports both .jpg and .jpeg extensions.
    """
    frames: dict[str, str] = {}
    for p in modality_dir.glob("*.jpg") + modality_dir.glob("*.jpeg"):
        match = _FRAME_PATTERN.search(p.name)
        if match:
            frames[match.group("frame")] = p.stem
    return frames


def discover_wisard(
    raw_root: Path, dataset_dir_name: str = "wisard"
) -> list[PairRecord]:
    raw_root = Path(raw_root)
    wisard_root = raw_root / dataset_dir_name
    records: list[PairRecord] = []
    for prefix, modality_dirs in sorted(_group_flight_dirs(wisard_root).items()):
        if "vis" not in modality_dirs or "ir" not in modality_dirs:
            continue
        vis_dir, ir_dir = modality_dirs["vis"], modality_dirs["ir"]
        vis_frames, ir_frames = _frame_stems(vis_dir), _frame_stems(ir_dir)
        for frame in sorted(vis_frames.keys() & ir_frames.keys()):
            vis_stem, ir_stem = vis_frames[frame], ir_frames[frame]
            # Assign splits deterministically: 70% train, 15% val, 15% test
            frame_idx = int(frame)
            if frame_idx % 100 < 70:
                split = "train"
            elif frame_idx % 100 < 85:
                split = "val"
            else:
                split = "test"
            records.append(
                PairRecord(
                    id=f"wisard_{prefix}_{frame}",
                    dataset="wisard",
                    split=split,
                    rgb_image=(vis_dir / f"{vis_stem}.jpeg").relative_to(raw_root).as_posix(),
                    rgb_label=(vis_dir / f"{vis_stem}.txt").relative_to(raw_root).as_posix(),
                    ir_image=(ir_dir / f"{ir_stem}.jpeg").relative_to(raw_root).as_posix(),
                    ir_label=(ir_dir / f"{ir_stem}.txt").relative_to(raw_root).as_posix(),
                )
            )
    return records

"""IR -> VIS alignment from the per-modality person labels (no image matching).

Everything lives in the S x S training input space, where both modalities
are stretched to S x S: the identity transform is the unaligned state, and a
transform applies directly to the resized IR image. Used by
`scripts/fit_alignment.py` (analysis) and `scripts/align_manifest.py` (writes
the aligned manifest); see `docs/labnotes.md` 2026-10-05.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from binarized_cv.data.labels import parse_yolo_label_file
from binarized_cv.data.records import PairRecord

S = 640  # training input size (data.img_size)
FIT_GATES = (128.0, 48.0, 24.0)  # px; match radius per refinement round
RANSAC_PX = 8.0
MIN_PAIRS = 20
CAP = 48.0  # px; error of a VIS box without an IR match within this radius


@dataclass
class Frame:
    number: int
    record: PairRecord
    vis: np.ndarray  # (N, 4) xyxy in S x S; empty if the modality is unlabeled
    ir: np.ndarray  # (M, 4)
    holdout: bool = False


def label_boxes(path: Path) -> np.ndarray:
    boxes = parse_yolo_label_file(path)
    return np.array([[b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2] for b in boxes]).reshape(-1, 4) * S


def load_groups(records: list[PairRecord], raw_root: Path) -> dict[str, list[Frame]]:
    """Frames per sequence: WiSARD by VIS directory (frame numbers from the
    id), TRGB by RGB resolution (sorted position as frame number). Every
    fifth block of 20 frames is marked as held out."""
    raw_root = Path(raw_root)
    groups: dict[str, list[Frame]] = {}
    for record in records:
        if record.dataset == "wisard":
            name, number = record.rgb_image.split("/")[1], int(record.id.rsplit("_", 1)[1])
        else:
            w, h = Image.open(raw_root / record.rgb_image).size
            name, number = f"trgb_{w}x{h}", -1
        vis = label_boxes(raw_root / record.rgb_label) if record.rgb_labeled else np.zeros((0, 4))
        ir = label_boxes(raw_root / record.ir_label) if record.ir_labeled else np.zeros((0, 4))
        groups.setdefault(name, []).append(Frame(number, record, vis, ir))
    for frames in groups.values():
        frames.sort(key=lambda f: (f.number, f.record.id))
        for i, f in enumerate(frames):
            if f.number < 0:
                f.number = i
            f.holdout = (i // 20) % 5 == 4
    return dict(sorted(groups.items()))


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(T).T
    return p[:, :2] / p[:, 2:3]


def warp_boxes(T: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = boxes.T
    corners = np.stack([np.c_[x0, y0], np.c_[x1, y0], np.c_[x0, y1], np.c_[x1, y1]], axis=1)
    w = apply(T, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.c_[w.min(1), w.max(1)]


def centres(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def match(T: np.ndarray, vis: np.ndarray, ir: np.ndarray, gate: float) -> list[tuple[int, int]]:
    """Hungarian matching of VIS to transformed IR boxes by centre distance."""
    if len(vis) == 0 or len(ir) == 0:
        return []
    d = np.linalg.norm(centres(vis)[:, None] - centres(warp_boxes(T, ir))[None], axis=2)
    rows, cols = linear_sum_assignment(d)
    return [(r, c) for r, c in zip(rows, cols) if d[r, c] < gate]


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt, rb = np.maximum(a[:, :2], b[:, :2]), np.minimum(a[:, 2:], b[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    area_a, area_b = (a[:, 2:] - a[:, :2]).prod(1), (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a + area_b - inter + 1e-9)


def frame_cost(T: np.ndarray, vis: np.ndarray, ir: np.ndarray, cap: float = CAP) -> float | None:
    """Mean centre error of the VIS boxes inside the IR footprint (IR sees
    only part of the VIS frame), each capped at `cap`; a VIS box without an
    IR match costs `cap`. None if either side has no boxes, or no VIS box is
    inside the footprint."""
    if len(vis) == 0 or len(ir) == 0:
        return None
    footprint = apply(T, np.array([[0, 0], [S, 0], [S, S], [0, S]], float)).astype(np.float32)
    inside = [cv2.pointPolygonTest(footprint, tuple(map(float, c)), False) >= 0 for c in centres(vis)]
    vis = vis[np.array(inside, bool)]
    if len(vis) == 0:
        return None
    d = np.linalg.norm(centres(vis)[:, None] - centres(warp_boxes(T, ir))[None], axis=2)
    err = np.full(len(vis), cap)
    rows, cols = linear_sum_assignment(d)
    err[rows] = np.minimum(d[rows, cols], cap)
    return float(err.mean())


def _points(vis: np.ndarray, ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centres and corners of matched boxes, as (IR src, VIS dst)."""

    def pts(b):
        return np.concatenate([centres(b), b[:, :2], b[:, 2:], np.c_[b[:, 0], b[:, 3]], np.c_[b[:, 2], b[:, 1]]])

    return pts(ir), pts(vis)


def fit_affine(frames: list[Frame], T0: np.ndarray, gates: tuple[float, ...] = FIT_GATES) -> tuple[np.ndarray | None, int]:
    """Iteratively match boxes under the current estimate (shrinking gate)
    and refit an affine IR -> VIS with RANSAC. Returns (T, matched pairs)."""
    T, n = np.asarray(T0, float), 0
    for gate in gates:
        src, dst, n = [], [], 0
        for f in frames:
            m = match(T, f.vis, f.ir, gate)
            if m:
                s, d = _points(f.vis[[r for r, _ in m]], f.ir[[c for _, c in m]])
                src.append(s)
                dst.append(d)
                n += len(m)
        if n < MIN_PAIRS:
            return None, n
        A, _ = cv2.estimateAffine2D(np.concatenate(src), np.concatenate(dst), method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        if A is None:
            return None, n
        T = np.r_[A, [[0, 0, 1]]]
    return T, n


def _smooth_labels(labels: list[int | None], window: int) -> list[int]:
    """Majority label within +-window positions; positions without a label in
    reach take the nearest smoothed one (0 if there are no labels at all)."""
    out: list[int | None] = []
    for i in range(len(labels)):
        near = [lab for lab in labels[max(0, i - window) : i + window + 1] if lab is not None]
        out.append(Counter(near).most_common(1)[0][0] if near else None)
    known = [i for i, lab in enumerate(out) if lab is not None]
    if not known:
        return [0] * len(labels)
    return [lab if lab is not None else out[min(known, key=lambda j: abs(j - i))] for i, lab in enumerate(out)]


def segment_transforms(
    frames: list[Frame],
    initial: list[np.ndarray],
    thresh: float = 16.0,
    min_frames: int = 100,
    max_models: int = 3,
    window: int = 25,
) -> tuple[list[np.ndarray], list[int]]:
    """Add transforms while at least `min_frames` frames are served worse
    than `thresh` by all current ones (e.g. a TRGB video with another rig, or
    a zoomed sequence), then assign each frame its best transform, smoothed
    over +-`window` neighbours (frames from one video share a rig). Frames
    without boxes in both modalities take their neighbours' transform. The
    `initial` transforms are kept fixed; added ones are refitted on their
    frames once."""
    transforms = [np.asarray(T, float) for T in initial]
    n_fixed = len(transforms)

    def best_labels() -> tuple[list[int | None], list[float | None]]:
        labels, costs = [], []
        for f in frames:
            c = [frame_cost(T, f.vis, f.ir) for T in transforms]
            valid = [(x, i) for i, x in enumerate(c) if x is not None]
            labels.append(min(valid)[1] if valid else None)
            costs.append(min(valid)[0] if valid else None)
        return labels, costs

    labels, costs = best_labels()
    while len(transforms) < max_models:
        bad = [f for f, lab, c in zip(frames, labels, costs) if c is not None and c > thresh]
        if len(bad) < min_frames:
            break
        T0 = transforms[Counter(lab for f, lab, c in zip(frames, labels, costs) if c is not None and c > thresh).most_common(1)[0][0]]
        T_new, _ = fit_affine(bad, T0)
        if T_new is None:
            break
        transforms.append(T_new)
        labels, costs = best_labels()
        for i in range(n_fixed, len(transforms)):
            mine = [f for f, lab in zip(frames, labels) if lab == i]
            T_ref, _ = fit_affine(mine, transforms[i], gates=(FIT_GATES[-1],)) if mine else (None, 0)
            if T_ref is not None:
                transforms[i] = T_ref
        labels, costs = best_labels()
    return transforms, _smooth_labels(labels, window)


def choose_offsets(costs: list[dict[int, float]], penalty: float = 4.0) -> list[int]:
    """Frame offset k per position (IR frame n+k paired with VIS frame n),
    minimising sum(cost) + penalty * sum(|k_t - k_t-1|) by dynamic
    programming, so the offset is piecewise constant and frames without
    labels (cost 0 for every k) follow their neighbours. A k missing from a
    position's dict (no IR frame n+k) is not allowed there."""
    if not costs:
        return []
    ks = sorted({k for c in costs for k in c})
    inf = float("inf")
    total = np.array([costs[0].get(k, inf) for k in ks])
    back = []
    jump = penalty * np.abs(np.subtract.outer(ks, ks))  # [prev, cur]
    for c in costs[1:]:
        cand = total[:, None] + jump
        back.append(cand.argmin(0))
        total = cand.min(0) + np.array([c.get(k, inf) for k in ks])
    path = [int(total.argmin())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return [ks[i] for i in reversed(path)]

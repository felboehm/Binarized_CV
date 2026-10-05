#!/usr/bin/env python
"""
Feasibility check for aligning IR to RGB with one fixed transform per
sequence, fitted from the per-modality person labels (no image matching).

Works in the 640x640 training input space: both modalities are stretched to
S x S there today, so the identity transform is the current state, and a
fitted transform applies directly to the resized IR image.

Per sequence (WiSARD: VIS directory; TRGB: RGB resolution), on frames with
people labeled in both modalities:
  1. match VIS and IR boxes per frame (Hungarian on centre distance after the
     current transform estimate, starting from identity, gate shrinking),
  2. fit IR -> VIS as scale+shift (4 dof), affine (6) and homography (8) with
     RANSAC on box centres and corners,
  3. evaluate on held-out frames (every 5th block of 20 frames): centre error
     in training pixels and box IoU, against identity,
  4. per-frame residual over the frame index (constant transform?) and a
     VIS/IR frame-offset scan (WiSARD only),
  5. every sequence's affine applied to every other sequence (one transform per
     rig enough?).

Outputs in --out/<timestamp>/: report.md, transforms.json, residual plots and
overlays (IR warped onto VIS, identity vs affine).

Usage:
    python scripts/fit_alignment.py [--manifest data/processed/manifest.jsonl] [--raw-root data/raw]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from binarized_cv.data.alignment import (
    FIT_GATES,
    MIN_PAIRS,
    RANSAC_PX,
    Frame,
    S,
    apply,
    centres,
    iou,
    load_groups,
    match,
    warp_boxes,
)
from binarized_cv.data.alignment import (
    _points as points,
)
from binarized_cv.data.manifest import load_manifest

MODELS = ("identity", "scale_shift", "affine", "homography")
EVAL_GATE = 48.0
SHIFTS = range(-5, 6)


def _fit_scale_shift(src: np.ndarray, dst: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """x' = sx x + tx, y' = sy y + ty, RANSAC on 2-point samples."""

    def solve(i):
        sx, tx = np.polyfit(src[i, 0], dst[i, 0], 1)
        sy, ty = np.polyfit(src[i, 1], dst[i, 1], 1)
        return np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]])

    best, best_inliers = None, np.zeros(len(src), bool)
    for _ in range(500):
        i = rng.choice(len(src), 2, replace=False)
        if np.ptp(src[i, 0]) < 1 or np.ptp(src[i, 1]) < 1:
            continue
        T = solve(i)
        inliers = np.linalg.norm(apply(T, src) - dst, axis=1) < RANSAC_PX
        if inliers.sum() > best_inliers.sum():
            best, best_inliers = T, inliers
    return solve(best_inliers) if best_inliers.sum() >= 3 else best


def fit(model: str, src: np.ndarray, dst: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    if model == "identity":
        return np.eye(3)
    if len(src) < 4:
        return None
    if model == "scale_shift":
        return _fit_scale_shift(src, dst, rng)
    if model == "affine":
        A, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        return None if A is None else np.r_[A, [[0, 0, 1]]]
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    return H


def collect(frames: list[Frame], T: np.ndarray, gate: float, ir_of=None) -> tuple[np.ndarray, np.ndarray, int]:
    """Matched (IR, VIS) points over `frames`; `ir_of` swaps in other IR boxes (offset scan)."""
    src, dst, n = [], [], 0
    for f in frames:
        ir = f.ir if ir_of is None else ir_of(f)
        if ir is None:
            continue
        m = match(T, f.vis, ir, gate)
        if m:
            s, d = points(f.vis[[r for r, _ in m]], ir[[c for _, c in m]])
            src.append(s)
            dst.append(d)
            n += len(m)
    if not src:
        return np.zeros((0, 2)), np.zeros((0, 2)), 0
    return np.concatenate(src), np.concatenate(dst), n


def fit_iterative(model: str, frames: list[Frame], rng: np.random.Generator, ir_of=None) -> tuple[np.ndarray | None, int]:
    T = np.eye(3)
    n = 0
    for gate in FIT_GATES:
        src, dst, n = collect(frames, T, gate, ir_of)
        T_new = fit(model, src, dst, rng) if n >= MIN_PAIRS else None
        if T_new is None:
            return (T if model == "identity" else None), n
        T = T_new
    return T, n


def eval_pairs(frames: list[Frame], T_assoc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matched (VIS, IR) boxes on `frames`, associated with one fixed transform
    so that every model is scored on the same pairs."""
    vis, ir = [], []
    for f in frames:
        for r, c in match(T_assoc, f.vis, f.ir, EVAL_GATE):
            vis.append(f.vis[r])
            ir.append(f.ir[c])
    return np.array(vis).reshape(-1, 4), np.array(ir).reshape(-1, 4)


def score(T: np.ndarray, vis: np.ndarray, ir: np.ndarray) -> dict[str, float]:
    if len(vis) == 0:
        return {"n": 0}
    w = warp_boxes(T, ir)
    err = np.linalg.norm(centres(w) - centres(vis), axis=1)
    ious = iou(w, vis)
    return {
        "n": len(vis),
        "median_err_px": float(np.median(err)),
        "p90_err_px": float(np.percentile(err, 90)),
        "mean_iou": float(ious.mean()),
        "iou_gt_0.5": float((ious > 0.5).mean()),
    }


def per_frame_error(frames: list[Frame], T: np.ndarray) -> tuple[list[int], list[float], list[bool]]:
    xs, ys, hold = [], [], []
    for f in frames:
        m = match(T, f.vis, f.ir, EVAL_GATE)
        if m:
            w = warp_boxes(T, f.ir[[c for _, c in m]])
            xs.append(f.number)
            ys.append(float(np.median(np.linalg.norm(centres(w) - centres(f.vis[[r for r, _ in m]]), axis=1))))
            hold.append(f.holdout)
    return xs, ys, hold


def offset_scan(frames: list[Frame], rng: np.random.Generator) -> dict[int, dict[str, float]]:
    """Pair VIS frame n with IR labels of frame n+k, refit the affine on all
    frames, and report the error: a minimum away from k=0 means the two
    streams are offset."""
    by_number = {f.number: f.ir for f in frames}
    out = {}
    for k in SHIFTS:
        ir_of = lambda f, k=k: by_number.get(f.number + k)
        T, _ = fit_iterative("affine", frames, rng, ir_of)
        if T is None:
            continue
        errs = []
        for f in frames:
            ir = ir_of(f)
            if ir is not None:
                m = match(T, f.vis, ir, EVAL_GATE)
                if m:
                    w = warp_boxes(T, ir[[c for _, c in m]])
                    errs.extend(np.linalg.norm(centres(w) - centres(f.vis[[r for r, _ in m]]), axis=1))
        if errs:
            out[k] = {"n": len(errs), "median_err_px": float(np.median(errs))}
    return out


def overlay(name: str, frames: list[Frame], T: np.ndarray, raw_root: Path, out: Path) -> None:
    """IR warped onto VIS (magenta = IR heat), identity left vs affine right;
    VIS boxes green, warped IR boxes red."""
    held = [f for f in frames if f.holdout and len(f.vis) and len(f.ir)]
    for f in sorted(held, key=lambda f: -min(len(f.vis), len(f.ir)))[:3]:
        vis = np.array(Image.open(raw_root / f.record.rgb_image).convert("RGB").resize((S, S)))
        ir = np.array(Image.open(raw_root / f.record.ir_image).convert("L").resize((S, S)))
        panels = []
        for M in (np.eye(3), T):
            warped = cv2.warpPerspective(ir, M, (S, S))
            blend = vis.copy()
            blend[..., 0] = np.maximum(blend[..., 0], warped)
            blend[..., 2] = np.maximum(blend[..., 2], warped)
            blend = (0.5 * vis + 0.5 * blend).astype(np.uint8)
            for b in f.vis.astype(int):
                cv2.rectangle(blend, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 1)
            for b in warp_boxes(M, f.ir).astype(int):
                cv2.rectangle(blend, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 1)
            panels.append(blend)
        Image.fromarray(np.concatenate([panels[0], np.full((S, 8, 3), 255, np.uint8), panels[1]], 1)).save(
            out / f"overlay_{name}_{f.number}.jpg", quality=90
        )


def fmt(m: dict) -> str:
    if not m.get("n"):
        return "– | – | – | – | 0"
    return f"{m['median_err_px']:.1f} | {m['p90_err_px']:.1f} | {m['mean_iou']:.2f} | {m['iou_gt_0.5']:.0%} | {m['n']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("runs/alignment"))
    args = parser.parse_args()

    out = args.out / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True)
    rng = np.random.default_rng(0)
    groups = load_groups(load_manifest(args.manifest), args.raw_root)

    report = [
        "# IR -> VIS alignment from labels",
        "",
        f"Training space {S}x{S}; identity = current stretch. Held-out frames: every 5th block of 20.",
        "Errors are box-centre distances in training pixels (small objects are < 32 px).",
        "",
    ]
    transforms: dict[str, dict] = {}
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, frames in groups.items():
        train = [f for f in frames if not f.holdout]
        held = [f for f in frames if f.holdout]
        fits = {m: fit_iterative(m, train, rng) for m in MODELS}
        n_frames = sum(1 for f in frames if len(f.vis) and len(f.ir))
        report += [f"## {name}", "", f"{len(frames)} frames, {n_frames} with boxes in both modalities.", ""]
        if fits["affine"][0] is None:
            report += [f"Too few matched pairs to fit (affine: {fits['affine'][1]}).", ""]
            continue
        vis, ir = eval_pairs(held, fits["affine"][0])
        eval_sets[name] = (vis, ir)
        report += [
            "| model | median err px | p90 err px | mean IoU | IoU>0.5 | held-out pairs | fit pairs |",
            "|---|---|---|---|---|---|---|",
        ]
        transforms[name] = {}
        for m in MODELS:
            T, n_fit = fits[m]
            if T is None:
                report.append(f"| {m} | fit failed | | | | | {n_fit} |")
                continue
            s = score(T, vis, ir)
            transforms[name][m] = {"matrix": T.tolist(), "fit_pairs": n_fit, "heldout": s}
            report.append(f"| {m} | {fmt(s)} | {n_fit} |")
        report.append("")

        A = fits["affine"][0]
        xs, ys, hold = per_frame_error(frames, A)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter([x for x, h in zip(xs, hold) if not h], [y for y, h in zip(ys, hold) if not h], s=4, label="fit")
        ax.scatter([x for x, h in zip(xs, hold) if h], [y for y, h in zip(ys, hold) if h], s=4, label="held out")
        ax.set(xlabel="frame", ylabel="median centre err (px)", title=f"{name}: affine residual per frame")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"residual_{name}.png", dpi=120)
        plt.close(fig)
        report += [f"![residual](residual_{name}.png)", ""]

        if not name.startswith("trgb"):
            scan = offset_scan(frames, rng)
            if scan:
                best = min(scan, key=lambda k: scan[k]["median_err_px"])
                report += [
                    f"Frame offset scan (IR frame n+k paired with VIS frame n), best k = {best}:",
                    "",
                    "| k | " + " | ".join(str(k) for k in scan) + " |",
                    "|---|" + "---|" * len(scan),
                    "| median err px | " + " | ".join(f"{v['median_err_px']:.1f}" for v in scan.values()) + " |",
                    "| pairs | " + " | ".join(str(v["n"]) for v in scan.values()) + " |",
                    "",
                ]
                transforms[name]["offset_scan"] = scan
        overlay(name, frames, A, args.raw_root, out)

    names = [n for n in eval_sets if "affine" in transforms.get(n, {})]
    if len(names) > 1:
        report += [
            "## Cross-sequence: affine of row applied to held-out pairs of column (median err px)",
            "",
            "| fitted on \\ applied to | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for a in names:
            T = np.array(transforms[a]["affine"]["matrix"])
            cells = [f"{score(T, *eval_sets[b]).get('median_err_px', float('nan')):.1f}" for b in names]
            report.append(f"| {a} | " + " | ".join(cells) + " |")
        report.append("")
        for a in names:
            report.append(f"- `{a}` affine: `{np.round(np.array(transforms[a]['affine']['matrix'])[:2], 3).tolist()}`")
        report.append("")

    (out / "report.md").write_text("\n".join(report))
    (out / "transforms.json").write_text(json.dumps(transforms, indent=2))
    print("\n".join(report))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

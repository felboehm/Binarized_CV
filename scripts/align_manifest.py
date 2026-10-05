#!/usr/bin/env python
"""
Write an aligned copy of the manifest: every record gets an IR -> VIS
transform (training input space), a re-paired IR frame and its remaining
alignment error. Built from the per-modality person labels, see
`binarized_cv.data.alignment` and `docs/labnotes.md` 2026-10-05.

1. Rig transform: one affine for all Mavic 2 Enterprise sequences (pooled
   RANSAC fit; the Carnation flight, zoomed, ends up as outliers). Per flight,
   a transform fitted on that flight alone replaces it if it lowers the
   misaligned share of the flight's held-out frames by >= --flight-margin
   (MtErie: 4K, small people, where the rig is a few px off).
2. Per WiSARD sequence: IR frame offset k in [-2, 2] per frame (the VIS and IR
   videos drift by about a frame, which shows as 20-60 px whenever the camera
   moves), piecewise constant via dynamic programming; then extra transforms
   where >= 100 frames stay off (zoom), smoothed over neighbouring frames;
   then the offsets again under the final transforms. Sequences without boxes
   in both modalities take the transform of a checked sequence of the same
   flight, else the rig transform (align_source says which), and offset 0.
   TRGB: no frame numbers, so transforms per block only (different rigs).
3. align_err_px per record (mean capped centre error in training pixels,
   None if unverifiable) and align_ok: error <= max(--floor-px, --width-frac x
   the frame's median VIS person width). Relative, because people are small
   (median 18 px wide at 640x640): the same shift that is harmless on a large
   person moves the IR blob entirely off a small one. The floor is the label
   noise (most checked frames are 2-8 px off), below which a tiny person's
   error says nothing about alignment.

Usage:
    python scripts/align_manifest.py [--manifest data/processed/manifest.jsonl] \\
        [--out data/processed/manifest_aligned.jsonl]
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from binarized_cv.data.alignment import (
    Frame,
    choose_offsets,
    fit_affine,
    frame_cost,
    load_groups,
    segment_transforms,
)
from binarized_cv.data.manifest import load_manifest, save_manifest

SHIFTS = range(-2, 3)


def flight(sequence: str) -> str:
    return re.sub(r"_(FLIR_)?VIS(_\d+)?$", "", sequence)


def offsets_for(frames: list[Frame], transforms: list[np.ndarray], blind_holdout: bool = False) -> list[int]:
    """`blind_holdout`: held-out frames contribute no cost, so their offsets
    come from their neighbours only (for an unbiased error estimate)."""
    by_number = {f.number: f for f in frames}
    costs = []
    for f, T in zip(frames, transforms):
        c = {}
        for k in SHIFTS:
            other = by_number.get(f.number + k)
            if other is not None:
                cost = None if blind_holdout and f.holdout else frame_cost(T, f.vis, other.ir)
                c[k] = 0.0 if cost is None else cost
        costs.append(c)
    return choose_offsets(costs)


def aligned(err: float | None, vis: np.ndarray, floor: float, width_frac: float) -> bool | None:
    if err is None:
        return None
    return err <= max(floor, width_frac * float(np.median(vis[:, 2] - vis[:, 0])))


def held_stats(frames: list[Frame], T: np.ndarray, floor: float, width_frac: float) -> tuple[float, float, int]:
    """Median error, misaligned share and count of the held-out frames under T."""
    errs, bad = [], []
    for f in frames:
        e = frame_cost(T, f.vis, f.ir) if f.holdout else None
        if e is not None:
            errs.append(e)
            bad.append(not aligned(e, f.vis, floor, width_frac))
    return (float(np.median(errs)), float(np.mean(bad)), len(errs)) if errs else (float("nan"), float("nan"), 0)


def heldout_err(frames: list[Frame], transforms: list[np.ndarray], offsets: list[int], floor: float, width_frac: float) -> str | None:
    """Median error and misaligned share on the held-out frames."""
    errs, bad = [], []
    for f, T in zip(repair(frames, offsets), transforms):
        e = frame_cost(T, f.vis, f.ir) if f.holdout else None
        if e is not None:
            errs.append(e)
            bad.append(not aligned(e, f.vis, floor, width_frac))
    return f"{np.median(errs):.1f} / {np.mean(bad):.0%}" if errs else None


def repair(frames: list[Frame], offsets: list[int]) -> list[Frame]:
    by_number = {f.number: f for f in frames}
    out = []
    for f, k in zip(frames, offsets):
        o = by_number[f.number + k]
        record = dataclasses.replace(f.record, ir_image=o.record.ir_image, ir_label=o.record.ir_label)
        out.append(Frame(f.number, record, f.vis, o.ir, f.holdout))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/manifest_aligned.jsonl"))
    parser.add_argument("--floor-px", type=float, default=6.0, help="errors up to this always count as aligned (label noise)")
    parser.add_argument("--width-frac", type=float, default=0.5, help="else aligned if error <= this x median person width")
    parser.add_argument(
        "--flight-margin", type=float, default=0.05, help="min drop in held-out misaligned share for a per-flight transform"
    )
    args = parser.parse_args()

    groups = load_groups(load_manifest(args.manifest), args.raw_root)
    rig, n = fit_affine([f for name, fr in groups.items() if "_Enterprise_" in name for f in fr], np.eye(3))
    if rig is None:
        raise SystemExit("rig transform: too few matched pairs")
    print(f"Enterprise rig transform from {n} pairs: {np.round(rig[:2], 3).tolist()}")

    # Per-flight candidates, fitted on the flight's non-held-out frames after
    # the offset correction under the rig, kept if better on its held-out ones.
    by_flight: dict[str, list[Frame]] = defaultdict(list)
    for name, frames in groups.items():
        if "_Enterprise_" in name and any(len(f.vis) and len(f.ir) for f in frames):
            by_flight[flight(name)] += repair(frames, offsets_for(frames, [rig] * len(frames)))
    base: dict[str, tuple[np.ndarray, str]] = {}
    flight_rows = []
    for fl, frames in sorted(by_flight.items()):
        T_fl, _ = fit_affine([f for f in frames if not f.holdout], rig)
        rig_stats = held_stats(frames, rig, args.floor_px, args.width_frac)
        fl_stats = held_stats(frames, T_fl, args.floor_px, args.width_frac) if T_fl is not None else None
        use = fl_stats is not None and fl_stats[1] <= rig_stats[1] - args.flight_margin
        if use:
            T_all, _ = fit_affine(frames, T_fl)  # final fit on all of the flight's frames
            base[fl] = (T_fl if T_all is None else T_all, f"flight:{fl}")
        else:
            base[fl] = (rig, "rig")
        fl_txt = "fit failed" if fl_stats is None else f"{fl_stats[0]:.1f} / {fl_stats[1]:.0%}"
        flight_rows.append(
            f"| {fl} | {rig_stats[0]:.1f} / {rig_stats[1]:.0%} | {fl_txt} | {rig_stats[2]} | {base[fl][1]} | "
            f"`{np.round(base[fl][0][:2], 3).tolist()}` |"
        )

    results: dict[str, tuple[list[Frame], list[np.ndarray], list[str], list[int]]] = {}
    checks: dict[str, str] = {}  # held-out error: k=0 -> offsets from neighbours
    unchecked = []
    for name, frames in groups.items():
        if not any(len(f.vis) and len(f.ir) for f in frames):
            unchecked.append(name)
            continue
        if name.startswith("trgb"):
            T0, _ = fit_affine(frames, np.eye(3))
            transforms, labels = segment_transforms(frames, [T0])
            results[name] = (frames, [transforms[i] for i in labels], [f"{name}#{i}" for i in labels], [0] * len(frames))
            continue
        T_base, base_source = base.get(flight(name), (rig, "rig"))
        offsets = offsets_for(frames, [T_base] * len(frames))
        transforms, labels = segment_transforms(repair(frames, offsets), [T_base])
        per_frame = [transforms[i] for i in labels]
        offsets = offsets_for(frames, per_frame)
        sources = [base_source if i == 0 else f"{name}#{i}" for i in labels]
        results[name] = (repair(frames, offsets), per_frame, sources, offsets)
        e0 = heldout_err(frames, per_frame, [0] * len(frames), args.floor_px, args.width_frac)
        e1 = heldout_err(frames, per_frame, offsets_for(frames, per_frame, blind_holdout=True), args.floor_px, args.width_frac)
        if e0 is not None and e1 is not None:
            checks[name] = f"{e0} → {e1}"

    checked_names = list(results)
    for name in unchecked:
        frames = groups[name]
        if name.startswith("trgb"):
            results[name] = (frames, [None] * len(frames), ["none"] * len(frames), [0] * len(frames))
            continue
        # The transform used most in checked sequences of the same flight, else the rig's.
        used, lookup = Counter(), {}
        for other in checked_names:
            _, per_frame, sources, _ = results[other]
            if not other.startswith("trgb") and flight(other) == flight(name):
                used.update(sources)
                lookup.update(zip(sources, per_frame))
        if used:
            src = used.most_common(1)[0][0]
            T, source = lookup[src], f"inherited:{src}"
        else:
            T, source = rig, "rig"
        results[name] = (frames, [T] * len(frames), [f"{source} (unchecked)"] * len(frames), [0] * len(frames))

    records, summary = [], []
    for name, (frames, per_frame, sources, offsets) in results.items():
        errs, oks = [], []
        for f, T, src, k in zip(frames, per_frame, sources, offsets):
            err = frame_cost(T, f.vis, f.ir) if T is not None else None
            ok = aligned(err, f.vis, args.floor_px, args.width_frac)
            errs.append(err)
            oks.append(ok)
            records.append(
                dataclasses.replace(
                    f.record,
                    ir_transform=None if T is None else np.round(T, 6).tolist(),
                    ir_frame_offset=k,
                    align_err_px=None if err is None else round(err, 2),
                    align_source=src,
                    align_ok=ok,
                )
            )
        checked = [e for e in errs if e is not None]
        summary.append(
            f"| {name} | {len(frames)} | {', '.join(f'{s} ({c})' for s, c in Counter(sources).most_common())} | "
            f"{' '.join(f'{k:+d}:{c}' for k, c in sorted(Counter(offsets).items()))} | "
            f"{np.median(checked):.1f} | {checks.get(name, '–')} | {oks.count(False)} | {len(frames) - len(checked)} |"
            if checked
            else f"| {name} | {len(frames)} | {Counter(sources).most_common(1)[0][0]} | – | – | – | – | {len(frames)} |"
        )

    order = {r.id: i for i, r in enumerate(load_manifest(args.manifest))}
    records.sort(key=lambda r: order[r.id])
    save_manifest(records, args.out)
    checked = [r for r in records if r.align_ok is not None]
    rule = f"error > max({args.floor_px:g} px, {args.width_frac:g} x person width)"
    table = [
        (
            f"Wrote {len(records)} records to {args.out}. Checked: {len(checked)}, "
            f"misaligned ({rule}): {sum(not r.align_ok for r in checked)}, unchecked: {len(records) - len(checked)}."
        ),
        "",
        "Per-flight transform vs rig, held-out frames (median px / misaligned share):",
        "",
        "| flight | rig | own fit | held-out frames | used | transform |",
        "|---|---|---|---|---|---|",
        *flight_rows,
        "",
        f"| sequence | frames | transform (frames) | IR frame offsets | median err px | held-out median px / misaligned: k=0 → offsets from neighbours | misaligned | unchecked |",
        "|---|---|---|---|---|---|---|---|",
        *summary,
    ]
    args.out.with_suffix(".md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Estimate I(X;T) and I(T;Y) per YOLO26 layer (docs/binarization_plan.md §4).

Plan's decision rule: binarize layers where I(T;Y) has plateaued but I(X;T)
is still high (redundant rate). Run on an fp32 model to pick candidates,
then on the binarized model to see what binarization actually removed. See
`binarized_cv.analysis.information` for the estimator and its caveats.

Usage:
    python scripts/estimate_layer_mi.py --model yolo26 \\
        --checkpoint runs/checkpoints/yolo26/<ts>/epoch_9.pt
    python scripts/estimate_layer_mi.py --model yolo26_bnn --checkpoint ... --num-images 500
    python scripts/estimate_layer_mi.py ... data.batch_size=4   # extra Hydra overrides

Writes runs/info_plane/<model>_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from binarized_cv.analysis.information import LayerInformationEstimator
from binarized_cv.models.binarized.policy import LAYER_REGIONS
from binarized_cv.models.registry import build_model_from_config
from binarized_cv.train import train

# Outputs of every C3k2/SPPF/C2PSA stage — the feature maps the plan reasons about.
DEFAULT_LAYERS = [2, 4, 6, 8, 9, 10, 13, 16, 19, 22]


def layer_region(index: int) -> str:
    return next((r for r, idx in LAYER_REGIONS.items() if index in idx), "?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model config name, e.g. yolo26 or yolo26_bnn")
    parser.add_argument("--checkpoint", help="Trained checkpoint (omit for an untrained baseline)")
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--n-units", type=int, default=12, help="Channels per code (2**n states)")
    parser.add_argument("--n-subsets", type=int, default=8, help="Random channel subsets to average over")
    parser.add_argument("--out-dir", default="runs/info_plane")
    parser.add_argument("hydra_overrides", nargs="*")
    args = parser.parse_args()

    cfg, _ = train.load_config([f"model={args.model}", *args.hydra_overrides])
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg.model).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    estimator = LayerInformationEstimator(model.model, args.layers, n_units=args.n_units, n_subsets=args.n_subsets)
    loader = train.build_dataloader(cfg, split=args.split, shuffle=True)
    seen = 0
    with torch.no_grad(), tqdm(total=args.num_images, unit="img") as progress:
        for batch in loader:
            images = {m: t.to(device) for m, t in batch["images"].items()}
            model(images)
            estimator.update(tuple(images["rgb"].shape[-2:]), batch["targets"])
            seen += len(batch["targets"])
            progress.update(len(batch["targets"]))
            if seen >= args.num_images:
                break
    estimator.remove()
    results = estimator.results()

    print(f"\n{'layer':>5} {'region':<14}{'ch':>6}{'stride':>7}{'I(X;T)':>9}{'I(T;Y)':>9}{'H(Y)':>8}{'I(T;Y)/H(Y)':>13}")
    for r in results:
        print(
            f"{r.layer:>5} {layer_region(r.layer):<14}{r.channels:>6}{r.stride:>7}"
            f"{r.i_xt:>9.3f}{r.i_ty:>9.4f}{r.h_y:>8.4f}{r.i_ty_fraction:>13.1%}"
        )
    print(f"Bits per location; H(Y) depends on stride. I(X;T) ≤ {args.n_units} bits by construction.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    out_path.write_text(
        json.dumps({"args": vars(args), "images": seen, "layers": [r.as_dict() for r in results]}, indent=2)
    )
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

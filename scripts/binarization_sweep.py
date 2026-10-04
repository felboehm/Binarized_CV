#!/usr/bin/env python
"""Per-region binarization sensitivity sweep (docs/binarization_plan.md §4).

Starting from a trained fp32 checkpoint, quantize ONE region at a time
(everything else stays fp32), optionally fine-tune for a few epochs, and
report AP@0.5 and small-object APS against the unquantized reference. The
regions with the smallest drop — especially in APS — are the ones the IB
rationale says to binarize; this is the measurement that has the final say.

Usage:
    python scripts/binarization_sweep.py \\
        --model yolo26_bnn \\
        --init-checkpoint runs/checkpoints/yolo26/<ts>/epoch_9.pt \\
        --epochs 0                     # post-training sensitivity only
    python scripts/binarization_sweep.py --model yolo26_early_fusion_bnn \\
        --init-checkpoint ... --epochs 3 --regions backbone_deep neck_deep p3
    python scripts/binarization_sweep.py ... --precision 4   # 4-bit instead of binary
    python scripts/binarization_sweep.py ... data.batch_size=16   # extra Hydra overrides

Results go to runs/sweeps/<model>_<timestamp>/results.{csv,json}. Evaluates
on the val split by default so region choices aren't tuned on test.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from binarized_cv.eval.evaluate import evaluate_model
from binarized_cv.models.binarized import REGIONS
from binarized_cv.models.registry import build_model_from_config
from binarized_cv.train import train

# The one-to-many branch is deleted before deployment, so quantizing it says
# nothing about the deployed model.
DEFAULT_REGIONS = [r for r in REGIONS if r != "head_one2many"]


def run_variant(name: str, overrides: list[str], epochs: int, split: str, sweep_name: str) -> dict[str, float]:
    cfg, _ = train.load_config(overrides + [f"train.epochs={epochs}"])
    if epochs > 0:
        model = train.main(cfg, model_name=f"{sweep_name}/{name}")
    else:
        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        model = build_model_from_config(cfg.model).to(device)
    return {"seed": cfg.train.get("seed"), **evaluate_model(model, cfg, split=split)}


def save_results(out_dir: Path, args: argparse.Namespace, rows: list[dict]) -> None:
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model config with a `binarization` block, e.g. yolo26_bnn")
    parser.add_argument("--init-checkpoint", required=True, help="Trained fp32 checkpoint of the same detector")
    parser.add_argument(
        "--regions", nargs="*", default=DEFAULT_REGIONS, choices=REGIONS,
        help="Regions to quantize, one at a time (none = only the fp32 reference(s))",
    )
    parser.add_argument("--precision", default="binary", help="binary | <bits> (default: binary)")
    parser.add_argument("--epochs", type=int, default=0, help="Fine-tune epochs per variant (0 = no fine-tuning)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", default="runs/sweeps")
    parser.add_argument("hydra_overrides", nargs="*", help="Extra Hydra overrides (key=value)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_name = f"sweep_{args.model}_{timestamp}"
    out_dir = Path(args.out_dir) / f"{args.model}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = [
        f"model={args.model}",
        f"model.init_checkpoint={args.init_checkpoint}",
        "model.binarization.preset=fp32",
        *args.hydra_overrides,
    ]

    # With fine-tuning, deltas are taken against an fp32 reference fine-tuned
    # with the identical schedule, so "more training" isn't counted as "less
    # quantization". The untouched checkpoint is still reported as a row.
    untouched = run_variant("reference", base, 0, args.split, sweep_name)
    rows = [{"region": "(none, fp32)", "precision": "fp", "epochs": 0, **untouched}]
    reference = untouched
    if args.epochs > 0:
        reference = run_variant("reference_finetuned", base, args.epochs, args.split, sweep_name)
        rows.append({"region": "(none, fp32)", "precision": "fp", "epochs": args.epochs, **reference})
    for row in rows:
        row["d_ap50"] = row["ap50"] - reference["ap50"]
        row["d_ap50_small"] = row["ap50_small"] - reference["ap50_small"]
    save_results(out_dir, args, rows)
    for region in args.regions:
        metrics = run_variant(
            region,
            base + [f"model.binarization.regions.{region}={args.precision}"],
            args.epochs,
            args.split,
            sweep_name,
        )
        rows.append(
            {
                "region": region,
                "precision": args.precision,
                "epochs": args.epochs,
                **metrics,
                "d_ap50": metrics["ap50"] - reference["ap50"],
                "d_ap50_small": metrics["ap50_small"] - reference["ap50_small"],
            }
        )
        # Saved after every variant: fine-tuned sweeps run for hours.
        save_results(out_dir, args, rows)

    print(f"\n{'region':<16}{'prec':>7}{'ep':>4}{'AP50':>8}{'ΔAP50':>8}{'APS50':>8}{'ΔAPS50':>8}")
    for r in rows:
        print(
            f"{r['region']:<16}{r['precision']:>7}{r['epochs']:>4}{r['ap50']:>8.4f}{r['d_ap50']:>+8.4f}"
            f"{r['ap50_small']:>8.4f}{r['d_ap50_small']:>+8.4f}"
        )
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

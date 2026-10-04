#!/usr/bin/env python
"""
Train a model, then evaluate the final weights on val and write the metrics
as JSON. Non-interactive, for queued and batch (Slurm) runs.

Usage:
    python scripts/train_eval.py RUN_NAME OUT_JSON [hydra overrides...]

RUN_NAME names the checkpoint/TensorBoard directories
(runs/{checkpoints,tensorboard}/RUN_NAME/<timestamp>/). Example:

    python scripts/train_eval.py yolo26_early_fusion_bnn_ib_guided runs/bnn/ib_guided.json \\
        model=yolo26_early_fusion_bnn \\
        model.init_checkpoint=runs/checkpoints/yolo26_early_fusion/<ts>/epoch_29.pt \\
        train.epochs=20 train.scheduler=cosine train.warmup_epochs=1 train.seed=0
"""

import argparse
import json
import logging
from pathlib import Path

from binarized_cv.eval.evaluate import evaluate_model
from binarized_cv.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name", help="name of the run's checkpoint/TensorBoard directories")
    parser.add_argument("out", type=Path, help="where to write the val metrics (JSON)")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, key=value")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg, _ = train.load_config(args.overrides)
    model = train.main(cfg, model_name=args.run_name)
    metrics = evaluate_model(model, cfg, split="val")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"overrides": args.overrides, **metrics}, indent=2))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()

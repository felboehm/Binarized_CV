#!/usr/bin/env python
"""
Train DataLoader throughput (batches/s) from each of several data roots, to
decide whether Slurm jobs need to stage the dataset to node-local /tmp
(`BCV_STAGE`, scripts/slurm/job.sbatch) or can read the shared copy directly.
Compare against what training consumes (~4.8 batches/s on an A40 for early
fusion at batch size 8).

Each root reads its own disjoint random subset of the train split, so no root
profits from files another root's pass put in the page cache. The first pass
over a subset is cold (except for whatever the node cached before); a second
pass over the same subset shows the warm rate.

Usage:
    python scripts/benchmark_loader.py ROOT [ROOT...] [--batches N] [-- hydra overrides...]

On the cluster, through job.sbatch (BCV_STAGE=1 stages to /tmp first):
    sbatch -p a40q -J bench_loader scripts/slurm/job.sbatch scripts/benchmark_loader.py \\
        /global/D1/homes/$USER/bcv-data/raw /tmp/$USER/bcv-data/raw
"""

import argparse
import json
import logging
import time

import torch
from torch.utils.data import DataLoader, Subset

from binarized_cv.train import train


def throughput(loader: DataLoader, warmup: int) -> float:
    """Batches/s over `loader`, excluding the first `warmup` batches (worker
    start-up and the first prefetch)."""
    start, seen = None, 0
    for i, _ in enumerate(loader):
        if i == warmup:
            start = time.perf_counter()
        elif i > warmup:
            seen += 1
    return seen / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", help="data.raw_root values to compare")
    parser.add_argument("--batches", type=int, default=300, help="timed batches per pass")
    parser.add_argument("--warmup", type=int, default=20, help="untimed batches at the start of each pass")
    args, overrides = parser.parse_known_args()
    overrides = [o for o in overrides if o != "--"]

    logging.basicConfig(level=logging.INFO)
    cfg, _ = train.load_config(overrides)
    per_root = (args.batches + args.warmup) * cfg.data.batch_size
    for k, root in enumerate(args.roots):
        cfg.data.raw_root = root
        loader = train.build_dataloader(cfg, split="train", shuffle=False)
        order = torch.randperm(len(loader.dataset), generator=torch.Generator().manual_seed(0))
        subset = order[k * per_root : (k + 1) * per_root].tolist()
        if len(subset) < per_root:
            raise SystemExit(f"train split too small for {len(args.roots)} roots x {per_root} samples")
        loader = DataLoader(
            Subset(loader.dataset, subset),
            batch_size=loader.batch_size,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
        )
        cold = throughput(loader, args.warmup)
        warm = throughput(loader, args.warmup)
        result = {"root": root, "num_workers": loader.num_workers, "batch_size": loader.batch_size,
                  "cold_batches_per_s": round(cold, 2), "warm_batches_per_s": round(warm, 2)}
        print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

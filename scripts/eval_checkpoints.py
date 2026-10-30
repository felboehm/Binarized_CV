#!/usr/bin/env python
"""
Evaluate saved checkpoints on test (and optionally val), optionally after
re-estimating the BatchNorm running statistics on another set of images
(AdaBN, Li et al. 2016). Eval-only, for queued and batch (Slurm) runs.

Usage:
    python scripts/eval_checkpoints.py OUT_JSON CKPT [CKPT ...] \\
        [--bn-recalib train test test_wisard] [--splits test val] \\
        -- [hydra overrides...]

CKPT may be a glob (expanded here, since Slurm's job.sbatch passes it
unexpanded); each one is a `state_dict` as train.py saves it (`epoch_N.pt`,
`best.pt`). The model config must build the same architecture, so pass the
run's model= and binarization overrides; `model.init_checkpoint` is forced to
null (the checkpoint replaces it anyway).

`--bn-recalib` sources, each evaluated on a fresh copy of the checkpoint:
    train        the training images: same domain, only refreshes the stats
                 (tests whether EMA-averaged weights and EMA-averaged BN
                 stats disagree)
    test         all test images, labels unused (transductive)
    test_wisard  the WiSARD test images only (the unseen Baker flight); only
                 the `*_wisard` metrics of this variant are meaningful
Only the BatchNorm layers switch to train mode during recalibration, so the
k-bit layers' activation ranges stay as trained.

`--calib-output-range N`: for models built with
`model.binarization.quantize_output=true` from checkpoints trained without
it, estimate the k-bit output ranges (EMA over N training batches, as
training would) before evaluating, instead of from the first eval batch.
BN statistics and input ranges stay as trained.

Output: {checkpoint: {"none": {...}, "bn_<source>": {...}}}, metrics as in
train_eval.py (test ones prefixed `test_`).
"""

import argparse
import copy
import glob
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from binarized_cv.data.collate import detection_collate
from binarized_cv.data.dataset import MultispectralPersonDataset
from binarized_cv.data.manifest import load_manifest
from binarized_cv.eval.evaluate import evaluate_model
from binarized_cv.models.binarized import QuantConv2d
from binarized_cv.models.registry import build_model_from_config
from binarized_cv.train import train

log = logging.getLogger(__name__)

RECALIB_SOURCES = {"train": ("train", None), "test": ("test", None), "test_wisard": ("test", "wisard")}


def recalib_loader(cfg, source: str, max_images: int) -> DataLoader:
    split, dataset_name = RECALIB_SOURCES[source]
    records = [
        r
        for r in load_manifest(cfg.data.manifest_path)
        if r.split == split and (dataset_name is None or r.dataset == dataset_name)
    ]
    dataset = MultispectralPersonDataset(
        records,
        raw_root=cfg.data.raw_root,
        split=split,
        img_size=tuple(cfg.data.img_size),
        target_modality=cfg.data.target_modality,
        drop_misaligned=split in cfg.data.get("drop_misaligned_splits", []),
        ir_alignment=cfg.data.get("ir_alignment", "none"),
    )
    if len(dataset) > max_images:
        generator = torch.Generator().manual_seed(0)
        keep = torch.randperm(len(dataset), generator=generator)[:max_images].tolist()
        dataset = torch.utils.data.Subset(dataset, keep)
    return DataLoader(
        dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, collate_fn=detection_collate
    )


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader: DataLoader) -> None:
    """Reset every BatchNorm's running stats and re-estimate them as a plain
    average (momentum=None) over `loader`, with the rest of the model in
    eval mode."""
    device = next(model.parameters()).device
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    model.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for batch in loader:
        model({m: t.to(device) for m, t in batch["images"].items()})
    model.eval()
    log.info("re-estimated %d BatchNorm layers over %d images", len(bns), len(loader.dataset))


@torch.no_grad()
def calibrate_output_ranges(model: nn.Module, loader: DataLoader, batches: int) -> None:
    """EMA of every `quantize_output` layer's output min/max over `batches`
    training batches, the rest of the model in eval mode. Those layers'
    input ranges would update too in train mode, so they are restored."""
    layers = [m for m in model.modules() if isinstance(m, QuantConv2d) and m.out_bits]
    if not layers:
        return
    device = next(model.parameters()).device
    input_ranges = [m.act_range.clone() for m in layers]
    model.eval()
    for m in layers:
        m.train()
    for i, batch in enumerate(loader):
        if i == batches:
            break
        model({k: t.to(device) for k, t in batch["images"].items()})
    for m, saved in zip(layers, input_ranges):
        m.eval()
        m.act_range.copy_(saved)
    log.info("calibrated %d output ranges over %d batches", len(layers), min(batches, len(loader)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="where to write the metrics (JSON)")
    parser.add_argument("checkpoints", nargs="+", help="checkpoint paths or globs")
    parser.add_argument("--bn-recalib", nargs="*", default=[], choices=sorted(RECALIB_SOURCES))
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["test", "val"])
    parser.add_argument("--max-recalib-images", type=int, default=2000)
    parser.add_argument("--calib-output-range", type=int, default=0, metavar="BATCHES")
    argv = sys.argv[1:]
    split_at = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split_at])
    overrides = argv[split_at + 1 :]  # Hydra overrides after `--`
    if not any(o.startswith("model=") for o in overrides):
        parser.error("pass the model explicitly as a model=<name> override")

    logging.basicConfig(level=logging.INFO)
    cfg, _ = train.load_config(overrides + ["++model.init_checkpoint=null"])
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    paths = sorted({p for pattern in args.checkpoints for p in (glob.glob(pattern) or [pattern])})

    base = build_model_from_config(cfg.model).to(device)
    loaders = {s: recalib_loader(cfg, s, args.max_recalib_images) for s in args.bn_recalib}
    if args.calib_output_range:
        calib = recalib_loader(cfg, "train", args.calib_output_range * cfg.data.batch_size)
    results = {}
    for path in paths:
        state = torch.load(path, map_location=device, weights_only=True)
        results[path] = {}
        for variant in ["none", *args.bn_recalib]:
            model = copy.deepcopy(base)
            model.load_state_dict(state)
            if args.calib_output_range:
                calibrate_output_ranges(model, calib, args.calib_output_range)
            if variant != "none":
                recalibrate_bn(model, loaders[variant])
            metrics = {}
            for split in args.splits:
                prefix = "test_" if split == "test" else ""
                metrics |= {f"{prefix}{k}": v for k, v in evaluate_model(model, cfg, split=split).items()}
            key = variant if variant == "none" else f"bn_{variant}"
            results[path][key] = metrics
            log.info("%s [%s]: %s", path, key, json.dumps(metrics))
        # Write after every checkpoint, so a killed job keeps what it has.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"overrides": overrides, "results": results}, indent=2))


if __name__ == "__main__":
    main()

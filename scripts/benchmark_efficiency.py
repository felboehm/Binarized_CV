#!/usr/bin/env python
"""
Efficiency of one model config: operation counts (FLOPs/BOPs/OPs), deployed
model size, latency/FPS and energy per image (CHECKLIST.md section 8).

Usage:
    python scripts/benchmark_efficiency.py OUT_JSON [--device cpu] [--batch 1] \\
        [--threads N] [--checkpoint CKPT] [--rounds 3] [--seconds 10] \\
        -- model=<name> [hydra overrides...]

Example, fp32 vs `ib_guided` early fusion on the CPU:

    python scripts/benchmark_efficiency.py runs/efficiency/ef_fp32_cpu.json -- model=yolo26_early_fusion
    python scripts/benchmark_efficiency.py runs/efficiency/ef_ib_guided_cpu.json -- model=yolo26_early_fusion_bnn

The model is prepared as for deployment first: the train-only one2many head
is dropped and BN folded into fp convs (`prepare_for_inference`).

Binary/k-bit layers are simulated in PyTorch, so latency and energy of a
quantized model measure the simulation (slower than fp32), not a bitwise
kernel; the JSON marks them `simulated_quantization: true`. Its OPs and
packed size are the meaningful numbers until a real binary runtime exists.

Energy needs readable counters: RAPL (CPU package + platform) is root-only
by default, `sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj`
once per boot; GPU energy needs `pip install nvidia-ml-py`. Without them the
run still reports everything else and lists what was missing.

Weights don't change latency, so `--checkpoint` is optional.

On a hybrid CPU pin to the performance cores, one thread per physical core:
torch's default spreads threads over E-cores too, and the slowest core then
gates every parallel op (Core Ultra 5 135H, early fusion fp32: 84 ms with
the default 14 threads, 38 ms with `--cpus 0,2,4,6`; docs/labnotes.md
2026-10-24).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

from binarized_cv.eval.efficiency import (
    benchmark,
    count_ops,
    deployed_size,
    dummy_images,
    environment_info,
    prepare_for_inference,
)
from binarized_cv.models.registry import build_model_from_config
from binarized_cv.train import train

log = logging.getLogger(__name__)


def parse_cpus(spec: str) -> set[int]:
    cpus = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="where to write the results (JSON)")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cpus", help="pin to these CPUs, e.g. 0,2,4,6 or 0-7 (threads default to their count)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's)")
    parser.add_argument("--checkpoint", help="state_dict to load (optional, latency doesn't depend on it)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100, help="minimum iterations per round")
    parser.add_argument("--seconds", type=float, default=10.0, help="minimum seconds per round")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=5.0, help="seconds between rounds")
    parser.add_argument("--idle", type=float, default=5.0, help="seconds of idle-power measurement")
    argv = sys.argv[1:]
    split_at = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split_at])
    overrides = argv[split_at + 1 :]
    if not any(o.startswith("model=") for o in overrides):
        parser.error("pass the model explicitly as a model=<name> override")

    logging.basicConfig(level=logging.INFO)
    if args.cpus:
        os.sched_setaffinity(0, parse_cpus(args.cpus))
    threads = args.threads or (len(parse_cpus(args.cpus)) if args.cpus else None)
    if threads:
        torch.set_num_threads(threads)
    device = torch.device(args.device)
    cfg, model_name = train.load_config(overrides)
    if "init_checkpoint" in cfg.model:  # not every detector takes one (ms_yolov8)
        cfg.model.init_checkpoint = None

    model = build_model_from_config(cfg.model)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = prepare_for_inference(model).to(device)
    img_size = tuple(cfg.model.get("img_size", cfg.data.img_size))
    images = dummy_images(model, args.batch, img_size, device)

    # One forward pass before counting initialises the k-bit activation ranges.
    with torch.inference_mode():
        model(images)
    ops = count_ops(model, {m: t[:1] for m, t in images.items()})
    size = deployed_size(model)
    log.info("%s: %.2f G OPs/image, packed %.1f KiB", model_name, ops.ops / 1e9, size["packed_bytes"] / 1024)

    timing = benchmark(
        model,
        images,
        warmup=args.warmup,
        min_iters=args.iters,
        min_seconds=args.seconds,
        rounds=args.rounds,
        cooldown_s=args.cooldown,
        idle_s=args.idle,
    )
    for problem in timing["energy_unavailable"]:
        log.warning("energy: %s", problem)
    if timing["simulated_quantization"]:
        log.warning("quantized layers are simulated: latency/energy are not those of a binary kernel")

    result = {
        "model": model_name,
        "overrides": overrides,
        "checkpoint": args.checkpoint,
        "img_size": list(img_size),
        "environment": {**environment_info(device, torch.get_num_threads()), "cpus": args.cpus},
        "params": sum(p.numel() for p in model.parameters()),
        "ops": ops.as_dict(),
        "size": size,
        **timing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    lat = timing["latency_ms"]
    log.info("latency %.2f ± %.2f ms, %.1f FPS -> %s", lat["mean"], lat["std"], timing["fps"]["mean"], args.out)


if __name__ == "__main__":
    main()

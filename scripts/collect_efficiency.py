#!/usr/bin/env python
"""
Summarise benchmark_efficiency.py result files as a Markdown table: params,
MACs per precision, OPs (FLOPs + BOPs/64), packed size, latency, FPS and
energy per image. Latency/energy are mean ± std over rounds; `sim` marks
models whose binary/k-bit layers are simulated, so their latency and energy
are those of the simulation, not of a binary kernel.

Energy is the domain matching the device: RAPL package for CPU runs, NVML
board for GPU runs (– if the counter wasn't readable), total and above the
idle power measured before the run.

Usage:
    python scripts/collect_efficiency.py                    # runs/efficiency/*/*.json
    python scripts/collect_efficiency.py 'runs/efficiency/myhost/*_cpu.json'
"""

import argparse
import glob
import json
from pathlib import Path

COLUMNS = (
    "run", "device", "params (M)", "fp MACs (G)", "8-bit MACs (G)", "binary MACs (G)", "OPs (G)",
    "packed (KiB)", "latency (ms)", "FPS", "energy (mJ/img)", "above idle (mJ/img)",
)


def fmt(stat: dict | None, digits: int = 1) -> str:
    if stat is None:
        return "–"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def row(path: Path, r: dict) -> list[str]:
    macs = r["ops"]["macs"]
    domain = "nvml_gpu0" if r["environment"]["device"].startswith("cuda") else "rapl_package-0"
    energy = r["energy_mj_per_image"]
    sim = " (sim)" if r["simulated_quantization"] else ""
    device = r["environment"]["device"]
    if r["environment"].get("cpus"):
        device += f" [{r['environment']['cpus']}]"
    return [
        path.stem.removesuffix(f"_{r['environment']['device'].split(':')[0]}"),
        device,
        f"{r['params'] / 1e6:.2f}",
        f"{r['ops']['fp_macs'] / 1e9:.2f}",
        f"{macs.get('w8a8', 0) / 1e9:.2f}",
        f"{macs.get('w1a1', 0) / 1e9:.2f}",
        f"{r['ops']['ops'] / 1e9:.2f}",
        f"{r['size']['packed_bytes'] / 1024:,.0f}",
        fmt(r["latency_ms"], 2) + sim,
        fmt(r["fps"]) + sim,
        fmt(energy.get(f"{domain}_total")) + sim * (f"{domain}_total" in energy),
        fmt(energy.get(f"{domain}_above_idle")) + sim * (f"{domain}_above_idle" in energy),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", nargs="?", default="runs/efficiency/*/*.json", help="glob of result files")
    args = parser.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files match {args.pattern}")
    rows = [row(p, json.loads(p.read_text())) for p in paths]
    rows.sort(key=lambda cells: (cells[1], cells[0]))

    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for cells in rows:
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

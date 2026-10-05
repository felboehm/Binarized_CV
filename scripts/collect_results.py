#!/usr/bin/env python
"""
Summarise train_eval.py result files over seeds: groups runs/bnn/*.json by
name with the `_seed<N>` suffix removed and prints mean ± std (sample std,
n-1) of each metric as a Markdown table. `ap50*` are val, `test_ap50*` test
(missing in runs from before test was added: shown as –).

Usage:
    python scripts/collect_results.py                  # runs/bnn/*.json
    python scripts/collect_results.py 'runs/bnn/*_20ep_seed*.json'
"""

import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("ap50", "ap50_small", "test_ap50", "test_ap50_small")  # val, then test


def fmt(values: list[float]) -> str:
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", nargs="?", default="runs/bnn/*.json", help="glob of result files")
    args = parser.parse_args()

    groups: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(args.pattern)):
        m = re.fullmatch(r"(.+)_seed(\d+)", Path(path).stem)
        name, seed = (m.group(1), int(m.group(2))) if m else (Path(path).stem, 0)
        groups[name][seed] = json.loads(Path(path).read_text())
    if not groups:
        raise SystemExit(f"no files match {args.pattern}")

    print("| run | seeds | " + " | ".join(METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for name, runs in sorted(groups.items()):
        cells = [fmt([r[k] for r in runs.values() if k in r]) for k in METRICS]
        print(f"| {name} | {','.join(map(str, sorted(runs)))} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

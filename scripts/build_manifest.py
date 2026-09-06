from __future__ import annotations

import argparse
from pathlib import Path

from binarized_cv.data.manifest import build_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover TRGB and WiSARD pairs under data/raw and write a unified manifest"
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/manifest.jsonl"))
    args = parser.parse_args()

    records = build_manifest(args.raw_root)
    save_manifest(records, args.out)

    by_dataset: dict[str, int] = {}
    for record in records:
        by_dataset[record.dataset] = by_dataset.get(record.dataset, 0) + 1
    print(f"Wrote {len(records)} records to {args.out}: {by_dataset}")


if __name__ == "__main__":
    main()

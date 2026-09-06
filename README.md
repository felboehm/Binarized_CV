# Binarized Multispectral YOLO26 for UAV Onboard CV

Master's thesis project: binarizing a YOLO26 object detector — extended to
fuse multispectral (RGB + thermal/IR) input — to cut inference latency and
power draw for onboard UAV deployment, evaluated against full-precision
YOLO26 and other SOTA efficient/edge detectors on accuracy, speed, power, and
model size.

See [`CHECKLIST.md`](CHECKLIST.md) for the full project plan and current
status, and [`docs/labnotes.md`](docs/labnotes.md) for a running log of
experiments and decisions.

## Repo layout

```
configs/                 Hydra-style YAML configs (data, model, train)
src/binarized_cv/
  data/                   Dataset loading, RGB+IR pairing, augmentation
  models/
    binarized/            Binary conv layers, STE, binarization utilities
    fusion/                Multispectral fusion modules (early/mid/late)
  train/                  Training loops, distillation, schedules
  eval/                   Accuracy metrics, latency/power benchmarking
  utils/                  Shared helpers
scripts/                  Standalone scripts (data download, benchmarking)
tests/                    Unit tests
notebooks/                Exploratory analysis
docs/                     Lab notes, thesis-adjacent writing
data/                     Datasets (gitignored)
runs/                     Checkpoints, logs, experiment outputs (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Status

Early scaffolding stage — see `CHECKLIST.md` section 2 onward for what's
decided vs. still open.

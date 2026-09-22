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
    base.py               BaseDetector interface every model implements
    registry.py           MODEL_REGISTRY / build_model / register_model
    backbones/            Per-modality feature extractors (fp32 reference now)
    fusion/                Multispectral fusion modules (early/mid/late)
    heads/                Detection heads (single-scale placeholder now)
    detectors/            Concrete models composed from the above, self-registering
    binarized/            Binary conv layers, STE, binarization utilities
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

## Usage

### 0. Download datasets

If you don't have the datasets locally, use the download script to fetch them
automatically from Google Drive:

```bash
# Interactive: prompts which datasets/variants to download
python scripts/download_data.py

# Or with options:
python scripts/download_data.py --wisard sample    # TRGB + WiSARD sample
python scripts/download_data.py --wisard full      # TRGB + WiSARD full
python scripts/download_data.py --trgb-only        # TRGB only
python scripts/download_data.py --no-prompt        # Download defaults (TRGB + sample) without prompting
```

The script will:
- Check if datasets already exist (skip if present)
- Download from Google Drive (using `gdown`)
- Extract to `data/raw/{trgb,wisard}/`
- Clean up the zip files

Note: WiSARD full is ~40.5GB and takes a while to download. Use the sample
(~972MB) to validate the pipeline quickly.

### 1. Quick start

Once the raw data is in place, run the whole loop — manifest, train, eval —
in one command:

```bash
./scripts/run_pipeline.sh
```

Defaults to `yolo26`, 5 epochs, 320x320, batch size 4 — small on purpose so
it finishes in reasonable time on a CPU-only machine; override via
environment variables, e.g. `MODEL=ms_yolov8 EPOCHS=30 DEVICE=cuda
./scripts/run_pipeline.sh`. This is a way to get *a* number quickly, not a
real training recipe — see the steps below to run each stage by hand with
full control.

### 2. Prepare data

Place the raw datasets under `data/raw/` (gitignored) matching the layout
`src/binarized_cv/data/datasets/{trgb,wisard}.py` expect:

```
data/raw/trgb/trgb_dataset/{train,val,test}/{RGB_images_*,IR_images_*}/...
data/raw/wisard/<flight>_{VIS,IR}_*/...
```

Then build the manifest that every dataset loader reads from:

```bash
python scripts/build_manifest.py --raw-root data/raw --out data/processed/manifest.jsonl
```

### 3. Train a model

**Interactive mode (recommended for ease of use):**

```bash
python scripts/train_model.py
```

This prompts you to select a model and optionally configure training parameters
(epochs, learning rate, batch size).

**Or with direct arguments:**

```bash
python scripts/train_model.py --model yolo26 --epochs 20 --lr 0.0005
python scripts/train_model.py --model ms_yolov8 --epochs 50 --batch-size 16
python scripts/train_model.py --model yolo26 --epochs 10 --no-prompt  # skip prompts, use defaults
```

**Or with Hydra-style overrides (full control):**

```bash
python -m binarized_cv.train.train model=yolo26
python -m binarized_cv.train.train \
  model=ms_yolov8 \
  train.epochs=50 train.lr=0.0005 train.device=cuda \
  data.batch_size=16 data.img_size=[640,640]
```

**Available models** (`model=` or `--model`):

| name            | what it is                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `simple_fusion` | basic reference detector (toy CNN backbones + concat fusion), RGB+IR        |
| `yolo26`        | real Ultralytics YOLO26 (`ultralytics.nn.tasks.DetectionModel`), RGB-only baseline |
| `yolo26_early_fusion` | YOLO26 early fusion: resize IR to RGB size, concatenate as 4-channel input, single backbone. Simplest approach; known weak on misaligned data (TRGB/WiSARD) but documents that constraint (legitimate thesis result). |
| `yolo26_midfusion` | YOLO26 mid-fusion (architecture decision pending, see labnotes 2026-09-19). Both RGB and IR models built, ConcatFusion ready. Three fusion approaches identified (early/multi-scale-mid/late); reserved for future implementation. |
| `ms_yolov8`     | RGB+thermal early-fusion YOLOv8, reimplementing Balla & Shrestha (EUSIPCO 2025) |

Any field in `configs/{data,model,train,eval}/*.yaml` can be overridden on
the command line with `key=value` (dotted for nested fields, `[a,b]` for
lists) — this is Hydra's usual override syntax.

Checkpoints land in `runs/checkpoints/{model_name}/epoch_<N>.pt` and TensorBoard
logs in `runs/tensorboard/{model_name}`, organized by model:

```bash
# View logs for a specific model:
tensorboard --logdir runs/tensorboard/yolo26_early_fusion

# Or view all models' logs together:
tensorboard --logdir runs/tensorboard
```

`train.device` defaults to `cuda` and automatically falls back to `cpu` if
no GPU is available.

### 4. Evaluate a checkpoint

**Quick evaluation (basic metrics):**
```bash
python -m binarized_cv.eval.evaluate model=yolo26 eval.checkpoint_path=runs/checkpoints/epoch_49.pt
```

**Comprehensive evaluation (with visualizations & detailed metrics):**
```bash
python scripts/eval_with_visuals.py \
  model=yolo26_early_fusion \
  eval.checkpoint_path=runs/checkpoints/yolo26_early_fusion/2026-09-21_21-04-06/epoch_29.pt
```

The comprehensive script generates:
- **Metrics**: AP@0.5/0.75/0.95, Precision@0.5, Recall@0.5, F1@0.5
- **Visualizations**: 
  - Confidence distribution histogram
  - Object count distribution (empty vs non-empty images)
  - AP vs confidence threshold curve
  - Precision-Recall curve @ IoU=0.5
  - 5 random test images with GT and predicted boxes
- **Files**: `metrics.json` (machine-readable), `results_summary.txt` (human-readable)
- **Location**: `runs/eval_results/{model_name}_{timestamp}/` (timestamped for each run)

See `configs/README.md` for how the config groups fit together, and `docs/labnotes.md` (2026-09-07) for why the CLI
uses Hydra's `compose`/`initialize` API rather than `@hydra.main`.

## Status

**Comprehensive evaluation implemented; early fusion model baseline complete** (2026-09-16 → 2026-09-22):

- **Data**: Full WiSARD dataset processed. Manifest contains **18,888 paired RGB/IR records**
  (4,772 TRGB + 14,116 WiSARD = 96% of expected pairs). Airfield flight excluded due to
  physical misalignment (VIS/IR from different camera angles).

- **Models**: Four models ready for training:
  - `yolo26`: RGB-only baseline (production YOLO26 from ultralytics)
  - `yolo26_early_fusion`: Early fusion baseline (4-channel concat, RGB+IR) — **30 epochs trained**
  - `yolo26_midfusion`: Mid-fusion scaffold (architecture decision pending)
  - `ms_yolov8`: Multispectral YOLOv8 comparison model (Balla & Shrestha EUSIPCO 2025)

- **Evaluation metrics** (30-epoch early fusion checkpoint):
  - AP@0.5: **0.7263**, AP@0.75: 0.3810, AP@0.95: 0.0038
  - Precision@0.5: **0.7976**, Recall@0.5: **0.7727**, F1@0.5: **0.7850**
  - Per-size breakdown: Small (77% recall), Medium (88% recall), Large (9% recall)
  - Recall ceiling at 0.77 due to undetectable cases (see `docs/labnotes.md` 2026-09-22)

- **Evaluation tooling** (new):
  - `scripts/eval_with_visuals.py` — comprehensive metrics + visualizations
  - Timestamped result collection to `runs/eval_results/`
  - 5 visualizations: confidence distribution, object count (empty/non-empty), 
    AP vs threshold, precision-recall curve, 5 sample detection images
  - Machine-readable (`metrics.json`) and human-readable (`results_summary.txt`) outputs
  
**Next**: Extended training for convergence (50+ epochs), then binarization implementation.
Mid-fusion integration deferred pending early fusion baseline results.
See `CHECKLIST.md` and `docs/labnotes.md` (2026-09-22) for full technical details.

## License

Not yet set (see `CHECKLIST.md` section 12). Note: this repo depends on
[`ultralytics`](https://github.com/ultralytics/ultralytics) (AGPL-3.0) for
the YOLO26 model (`src/binarized_cv/models/detectors/yolo26.py`) — using it
means this repo's own license terms need to be AGPL-3.0 too, or an
Ultralytics Enterprise license obtained instead.

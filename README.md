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
pip install -r requirements.txt   # pinned versions the experiments ran with (Python 3.12)
pip install -e . --no-deps
```

`pip install -e ".[dev]"` alone also works, with unpinned versions. The pinned
torch wheel is the CUDA 13 build (NVIDIA driver >= 580). For an older driver,
see the header of `requirements.txt` for installing the CUDA 12 build.

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

Then build the manifest, and the aligned copy every dataset loader reads from
by default (`data.manifest_path`): IR -> RGB transform per record, IR frames
re-paired, misaligned frames marked (see `docs/labnotes.md` 2026-10-05):

```bash
python scripts/build_manifest.py --raw-root data/raw --out data/processed/manifest.jsonl
python scripts/align_manifest.py --raw-root data/raw   # -> data/processed/manifest_aligned.jsonl
```

By default IR is warped onto the RGB grid (`data.ir_alignment=warp`); `none`
restores the old unaligned stretch, `crop` cuts both to the shared field of
view.

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
| `yolo26_midfusion` | YOLO26 mid fusion: separate RGB and IR backbones merged at feature level (`ConcatFusion`), shared neck/head. **Work in progress**: forward pass is still RGB-only, because YOLO26's neck reads earlier backbone layers by index (see `docs/fusion_architecture_rationale.md`). |
| `yolo26_bnn` / `yolo26_early_fusion_bnn` | Binarized (mixed-precision) variants of `yolo26` / `yolo26_early_fusion`, see [Binarization](#5-binarization). |
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

### 5. Binarization

Binarized variants reuse the same detectors with a `binarization` block
(`configs/model/yolo26_bnn.yaml`, `configs/model/yolo26_early_fusion_bnn.yaml`).
The network is not fully binarized: the default `ib_guided` preset makes the
deep backbone and deep neck binary, keeps the P3 (small-object) path and the
one2one head at 8-bit, and leaves the stem, input conv and attention in full
precision. The rationale is in `docs/binarization_plan.md`, the measurements
behind it in `docs/binarization_findings.md`. The `full` preset is kept only as
a reference point for the sweep. The implementation is in
`src/binarized_cv/models/binarized/`.

```bash
# Warm-start a BNN from a trained fp32 checkpoint (IB-guided region map)
python scripts/train_model.py --model yolo26_early_fusion_bnn --epochs 30 \
  model.init_checkpoint=runs/checkpoints/yolo26_early_fusion/<ts>/epoch_29.pt

# Per-region overrides / training knobs (here: also binarize the attention convs)
python -m binarized_cv.train.train model=yolo26_early_fusion_bnn \
  model.init_checkpoint=runs/checkpoints/yolo26_early_fusion/<ts>/epoch_29.pt \
  model.binarization.regions.attention=binary \
  train.scheduler=cosine train.warmup_epochs=1 train.seed=0
  # model.binarization.stochastic=true train.optimizer=sgd train.grad_clip=10

# Sensitivity sweep: binarize one region at a time, report ΔAP / ΔAPS on val
python scripts/binarization_sweep.py --model yolo26_early_fusion_bnn \
  --init-checkpoint runs/checkpoints/yolo26_early_fusion/<ts>/epoch_29.pt --epochs 0

# Per-layer information-plane estimates I(X;T), I(T;Y)
python scripts/estimate_layer_mi.py --model yolo26_early_fusion \
  --checkpoint runs/checkpoints/yolo26_early_fusion/<ts>/epoch_29.pt
```

For queued or batch (e.g. Slurm) runs, `scripts/train_eval.py RUN_NAME OUT_JSON
[overrides...]` trains, then evaluates on val and writes the metrics to
`OUT_JSON`. A stopped run continues with `train.resume=<run dir>/last.pt`.

The quantization is simulated in PyTorch. It measures accuracy, not speed.

See `configs/README.md` for how the config groups fit together, and `docs/labnotes.md` (2026-09-07) for why the CLI
uses Hydra's `compose`/`initialize` API rather than `@hydra.main`.

### 6. Running on the Slurm cluster

Scripts in `scripts/slurm/`. They're tuned for the cluster in use (Ubuntu 22.04,
`a100q`: x86_64 nodes with A100s and driver 615, internet on compute nodes,
local `/tmp`; the `a40q` nodes are aarch64 and can't run the x86_64 venv),
but nothing in them is specific to it except the default partition.

```bash
# 1. Code + environment (login node). uv provides Python 3.12; the system
#    python3 is 3.10 and the cluster's modules stop at 3.9.
git clone <repo> && cd Binarized_CV
bash scripts/slurm/setup_env.sh

# 2. Data, manifest and the fp32 warm-start checkpoint (from the local machine;
#    data/ and runs/ are gitignored). The dataset goes on the shared BeeGFS
#    (/global/D1, per-user dir under homes/); /work and /scratch are local
#    disks of the login node. The manifest's paths are relative to
#    data.raw_root.
rsync -a --mkpath data/raw/ cluster:/global/D1/homes/$USER/bcv-data/raw/
rsync -a data/processed/manifest.jsonl data/processed/manifest_aligned.jsonl cluster:<repo>/data/processed/
rsync -aR runs/checkpoints/yolo26_early_fusion/2026-09-21_21-04-06/epoch_29.pt cluster:<repo>/

# 3. Submit from the repo root (Slurm resolves --output and the repo from there)
export BCV_DATA=/global/D1/homes/$USER/bcv-data/raw
sbatch -J ib_guided_attn scripts/slurm/job.sbatch scripts/train_eval.py \
    yolo26_early_fusion_bnn_ib_guided_attn runs/bnn/ib_guided_attn_20ep_seed0.json \
    model=yolo26_early_fusion_bnn \
    model.init_checkpoint=runs/checkpoints/yolo26_early_fusion/2026-09-21_21-04-06/epoch_29.pt \
    model.binarization.regions.attention=binary \
    train.epochs=20 train.scheduler=cosine train.warmup_epochs=1 train.seed=0
squeue --me; tail -f runs/slurm/ib_guided_attn_<jobid>.out
```

`job.sbatch` runs any entry point that takes Hydra overrides (`train_eval.py`,
`binarization_sweep.py`, `estimate_layer_mi.py`):

- **Data staging:** it first copies `$BCV_DATA` to `/tmp/$USER/bcv-data/raw`
  on the node, because reading ~120k JPEGs from shared storage would starve
  the GPU. The copy persists, so later jobs on the same node only run
  rsync's check. `BCV_STAGE=0` reads `$BCV_DATA` directly. Remove the copy
  when done (`srun -p a100q -w <node> --mem=0 rm -rf /tmp/$USER/bcv-data`).
- **Cluster defaults:** it sets `data.raw_root` and `data.num_workers`
  (`--cpus-per-task` − 2) through `$BCV_OVERRIDES`. `load_config` reads that
  variable *before* the command-line overrides, so anything passed explicitly
  still wins.
- **Job arrays:** `{task}` in the arguments becomes `$SLURM_ARRAY_TASK_ID`, for
  seed or region arrays, e.g. `sbatch --array=0-2 … train.seed={task}`.
- **Resources:** the defaults are `a100q`, 1 GPU, 16 CPUs, 1 day. Override them
  on the `sbatch` command line (`-p hgx2q`, `-c 32`, `-t 3-00:00:00`). There is
  no `--mem`: the GPU nodes report 1 MB of memory to Slurm, so any memory
  request fails. `--propagate=NONE` keeps the login node's 16 GB `ulimit -v`
  out of the job; torch can't allocate memory under it.
- **Seed arrays:** `FP32=<warm-start .pt> bash scripts/slurm/submit_seeds.sh 1-2` submits the
  20-epoch comparison (fp32 control, `ib_guided`, `ib_guided` + binary
  attention) for seeds 1 and 2; `python scripts/collect_results.py` then
  prints mean ± std per config from `runs/bnn/*.json`. With
  `SCHEDULE=plateau` each run trains to convergence instead
  (`train.scheduler=plateau`: LR cut on a val AP50 plateau, stop after the
  last cut, best-val weights returned). Those runs select on val, so compare
  their `test_ap50`.
- **Logs:** progress bars update once a minute (`TQDM_MININTERVAL=60`) to keep
  the logs readable.

## Status

As of 2026-10-02. Details in `CHECKLIST.md` and `docs/labnotes.md`.

- **Data**: manifest of **19,605 paired RGB/IR records** (4,772 TRGB + 14,833
  WiSARD). WiSARD's Airfield flight is excluded (VIS and IR shot from different
  camera angles). Since 2026-10-05 WiSARD is split by flight: train FHL,
  Hannegan; val MtErie; test Baker; Carnation (zoomed VIS, no fixed IR
  alignment) is a separate split `zoom` for a later robustness check. Three FHL
  sequences without VIS labels stay in the manifest but are skipped for RGB
  supervision (`rgb_labeled: false`). Usable with TRGB: train 10,706 (9,862
  after dropping misaligned frames with `manifest_aligned.jsonl`), val 1,031,
  test 2,510, zoom 2,052. **The results below are on the earlier, leaky
  frame-index split** and need re-running.

- **Fusion models**: the thesis compares several fusion architectures, each in
  fp32 and binarized form:
  - `yolo26_early_fusion` (4-channel input) — **trained, 30 epochs**. This is
    the current baseline.
  - `yolo26_midfusion` (feature-level fusion) — **in progress**, forward pass
    still RGB-only.
  - `yolo26` (RGB only) is the no-fusion reference, `ms_yolov8` (Balla &
    Shrestha, EUSIPCO 2025) the first external comparison model.

- **Early fusion fp32 baseline** (30 epochs, `scripts/eval_with_visuals.py`, test
  split): AP@0.5 **0.726**, AP@0.75 0.381, Precision@0.5 0.798, Recall@0.5
  0.773. On the val split (`evaluate.py`): AP50 0.673, APS50 (< 32² px) 0.622.

- **Binarization**: implemented (simulated in PyTorch) and partly tested on the
  early fusion model. A one-region-at-a-time sweep with 3 fine-tune epochs (val)
  shows:
  - the deep backbone, deep neck and attention convs can go binary for
    ≤ 0.08 AP50 each (~85% of conv weights);
  - the P3 path must stay at 8-bit, since binary P3 loses ~95% of small-object AP;
  - the stem, input conv and head output projections are poor trades.

  A fully binarized network is therefore ruled out; the default is the selective
  `ib_guided` map. See `docs/binarization_findings.md`.

**Next**:
1. Full-state checkpoints and resume.
2. Finish the 20-epoch `ib_guided` run, plus an fp32 control on the same
   schedule and an `ib_guided` + binary attention variant.
3. Wire up the mid-fusion forward pass.
4. A real-kernel latency benchmark on the target CPU; nothing here measures
   speed yet.

## License

Not yet set (see `CHECKLIST.md` section 12). Note: this repo depends on
[`ultralytics`](https://github.com/ultralytics/ultralytics) (AGPL-3.0) for
the YOLO26 model (`src/binarized_cv/models/detectors/yolo26.py`) — using it
means this repo's own license terms need to be AGPL-3.0 too, or an
Ultralytics Enterprise license obtained instead.

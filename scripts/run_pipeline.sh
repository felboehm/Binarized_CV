#!/usr/bin/env bash
# Runs the full loop once, end to end: build the manifest (if it doesn't
# already exist), train one model for a few epochs, then evaluate the
# resulting checkpoint. This is a "does it actually work on real data"
# quick-look run, not a real training recipe — defaults are deliberately
# small (image size, epochs) so it finishes in reasonable time on a
# CPU-only machine. Override anything via environment variables, e.g.:
#
#   MODEL=ms_yolov8 EPOCHS=30 IMG_SIZE=640 DEVICE=cuda ./scripts/run_pipeline.sh
#
# Assumes the project is installed in the active Python environment
# (see README.md "Setup": pip install -e ".[dev]").
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${MODEL:-yolo26}"
EPOCHS="${EPOCHS:-5}"
IMG_SIZE="${IMG_SIZE:-320}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cpu}"
RAW_ROOT="${RAW_ROOT:-data/raw}"
MANIFEST="${MANIFEST:-data/processed/manifest.jsonl}"
RUN_NAME="${RUN_NAME:-${MODEL}_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="runs/checkpoints/${RUN_NAME}"
LOG_DIR="runs/tensorboard/${RUN_NAME}"

echo "== Step 1/3: manifest =="
if [[ -f "$MANIFEST" ]]; then
  echo "Already exists at $MANIFEST (delete it to force a rebuild) — skipping."
else
  python scripts/build_manifest.py --raw-root "$RAW_ROOT" --out "$MANIFEST"
fi

echo "== Step 2/3: train '$MODEL' for $EPOCHS epoch(s) =="
python -m binarized_cv.train.train \
  model="$MODEL" \
  data.raw_root="$RAW_ROOT" \
  data.manifest_path="$MANIFEST" \
  data.img_size="[$IMG_SIZE,$IMG_SIZE]" \
  data.batch_size="$BATCH_SIZE" \
  data.num_workers="$NUM_WORKERS" \
  model.img_size="[$IMG_SIZE,$IMG_SIZE]" \
  train.epochs="$EPOCHS" \
  train.device="$DEVICE" \
  train.checkpoint_dir="$CHECKPOINT_DIR" \
  train.log_dir="$LOG_DIR"

CHECKPOINT="$CHECKPOINT_DIR/epoch_$((EPOCHS - 1)).pt"

echo "== Step 3/3: evaluate $CHECKPOINT =="
python -m binarized_cv.eval.evaluate \
  model="$MODEL" \
  data.raw_root="$RAW_ROOT" \
  data.manifest_path="$MANIFEST" \
  data.img_size="[$IMG_SIZE,$IMG_SIZE]" \
  model.img_size="[$IMG_SIZE,$IMG_SIZE]" \
  train.device="$DEVICE" \
  eval.checkpoint_path="$CHECKPOINT"

echo
echo "Done."
echo "  Checkpoint:  $CHECKPOINT"
echo "  TensorBoard: tensorboard --logdir $LOG_DIR"

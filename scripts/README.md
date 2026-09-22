# Scripts

Standalone entry points, not library code:

- `download_data.py` / `download_data.sh` — fetch TRGB and WiSARD datasets
  from Google Drive, with progress/extraction/cleanup. Checks if datasets
  already exist and skips download if present. Interactive or command-line
  driven (see `--help` for options).
- `build_manifest.py` — discovers TRGB/WiSARD RGB+IR pairs under `data/raw`
  and writes the unified manifest (`data/processed/manifest.jsonl`)
- `train_model.py` — interactive training script with model selection and
  hyperparameter override prompts; alternatives: direct CLI via `python -m binarized_cv.train.train`
- `eval_with_visuals.py` — comprehensive evaluation script **[NEW]**
  - Computes: AP@0.5/0.75/0.95, Precision@0.5, Recall@0.5, F1@0.5 (@ IoU=0.5)
  - Generates 5 visualizations: confidence distribution, object count distribution,
    AP vs threshold curve, precision-recall curve, 5 sample detection images
  - Saves timestamped results to `runs/eval_results/{model}_{timestamp}/`
    with JSON metrics and human-readable summary
  - Usage: `python scripts/eval_with_visuals.py model=<name> eval.checkpoint_path=<path>`
- `run_pipeline.sh` — one-command manifest build + train + eval loop against
  real data, with CPU-realistic defaults; see README.md "Usage" or the
  script's own header comment for the environment-variable overrides
- `benchmark_latency.py` — inference latency/FPS on target hardware [NOT YET IMPLEMENTED]
- `measure_power.py` — energy-per-inference measurement harness [NOT YET IMPLEMENTED]

See `CHECKLIST.md` section 8 for latency/power benchmarking status.

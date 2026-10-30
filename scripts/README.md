# Scripts

Standalone entry points, not library code:

- `download_data.py` / `download_data.sh` — fetch TRGB and WiSARD datasets
  from Google Drive, with progress/extraction/cleanup. Checks if datasets
  already exist and skips download if present. Interactive or command-line
  driven (see `--help` for options).
- `build_manifest.py` — discovers TRGB/WiSARD RGB+IR pairs under `data/raw`
  and writes the unified manifest (`data/processed/manifest.jsonl`); training
  reads the aligned copy from `align_manifest.py` by default
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
- `collect_results.py` — mean ± std over seeds of the `train_eval.py` result
  JSONs in `runs/bnn/` (grouped by name minus `_seed<N>`), as a Markdown table
- `slurm/submit_seeds.sh` — submits the 20-epoch fp32 control / `ib_guided` /
  `ib_guided` + binary attention comparison as one Slurm job array per config
  (one task per seed); see README §6
- `fit_alignment.py` — analysis: how well one fixed IR -> VIS transform per
  sequence explains the per-modality person labels (held-out errors, residual
  plots, frame-offset scan, cross-sequence table, overlays) into
  `runs/alignment/<ts>/`
- `align_manifest.py` — writes `data/processed/manifest_aligned.jsonl`: per
  record an IR -> VIS transform, re-paired IR frame (VIS/IR videos drift by
  ~1 frame) and remaining error (`align_err_px`), plus a summary `.md`
- `benchmark_efficiency.py` — one model config: OPs (FLOPs + BOPs/64),
  packed model size, latency/FPS and energy per image (RAPL for CPU, NVML for
  GPU), as JSON. Binary/k-bit layers are simulated, so a quantized model's
  latency/energy is that of the simulation (marked `simulated_quantization`);
  its OPs and packed size are the meaningful numbers. Pin CPU runs to the
  P-cores (`--cpus 0,2,4,6`)
- `run_efficiency.sh` — `benchmark_efficiency.py` over the comparison set
  (RGB / early / mid fusion × fp32 / `ib_guided` / + binary attention,
  MS-YOLOv8) on CPU and GPU, into `runs/efficiency/<host>/`
- `collect_efficiency.py` — Markdown table of those JSONs

See `CHECKLIST.md` section 8 for latency/power benchmarking status.

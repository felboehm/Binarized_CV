# Scripts

Standalone entry points, not library code:

- `download_data.py` / `download_data.sh` — fetch TRGB and WiSARD datasets
  from Google Drive, with progress/extraction/cleanup. Checks if datasets
  already exist and skips download if present. Interactive or command-line
  driven (see `--help` for options).
- `build_manifest.py` — discovers TRGB/WiSARD RGB+IR pairs under `data/raw`
  and writes the unified manifest (`data/processed/manifest.jsonl`)
- `run_pipeline.sh` — one-command manifest build + train + eval loop against
  real data, with CPU-realistic defaults; see README.md "Usage" or the
  script's own header comment for the environment-variable overrides
- `benchmark_latency.py` — inference latency/FPS on target hardware
- `measure_power.py` — energy-per-inference measurement harness

`benchmark_latency.py` and `measure_power.py` not yet implemented — see
`CHECKLIST.md` section 8.

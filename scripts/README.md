# Scripts

Standalone entry points, not library code:

- `build_manifest.py` — discovers TRGB/WiSARD RGB+IR pairs under `data/raw`
  and writes the unified manifest (`data/processed/manifest.jsonl`)
- `run_pipeline.sh` — one-command manifest build + train + eval loop against
  real data, with CPU-realistic defaults; see README.md "Usage" or the
  script's own header comment for the environment-variable overrides
- `download_data.*` — fetch and prepare the multispectral UAV dataset
- `benchmark_latency.py` — inference latency/FPS on target hardware
- `measure_power.py` — energy-per-inference measurement harness

`download_data.*`, `benchmark_latency.py`, `measure_power.py` not yet
implemented — see `CHECKLIST.md` sections 3 and 8.

# Configs

Hydra-style YAML configs, one subtree per concern, composed by
`config.yaml`:

- `data/` — dataset paths, image size, target modality, batch size
- `model/` — model name (must match a `@register_model` name), and that
  model's constructor kwargs
- `train/` — optimizer, LR schedule, epochs, device, log/checkpoint dirs
- `eval/` — checkpoint path to load for evaluation

Add a `model/<name>.yaml` (with `name: <name>`) whenever a new model is
registered — see `src/binarized_cv/models/registry.py`. Binarization-scope
and distillation settings (`CHECKLIST.md` sections 5-6) aren't in the
configs yet since that implementation hasn't landed.

`train/train.py` and `eval/evaluate.py` load these via Hydra's
`compose`/`initialize` API rather than the `@hydra.main` CLI decorator —
the decorator's argparse-based CLI parser is broken on Python 3.14 with the
latest hydra-core (1.3.6); see `docs/labnotes.md` 2026-09-07. CLI usage is
unaffected: `python -m binarized_cv.train.train train.epochs=5 data.batch_size=16`.

**Evaluation usage** (basic metrics):
```bash
python -m binarized_cv.eval.evaluate model=yolo26 eval.checkpoint_path=runs/checkpoints/epoch_49.pt
```

**Comprehensive evaluation with visualizations** (see `scripts/README.md`):
```bash
python scripts/eval_with_visuals.py model=yolo26_early_fusion eval.checkpoint_path=<path>
```

# Configs

Hydra-style YAML configs, one subtree per concern:

- `data/` — dataset paths, fusion input format, augmentation
- `model/` — backbone variant, fusion point (early/mid/late), binarization
  scope (full vs. head-in-fp32)
- `train/` — optimizer, LR schedule, distillation settings

Not yet populated — add configs as the corresponding implementation lands
(see `CHECKLIST.md` sections 4-6).

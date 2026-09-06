# Lab Notes

Running log of decisions, experiments, and dead ends — feeds the methods
chapter later.

## 2026-09-06

- Repo scaffolded. Scope locked so far: fully binarized YOLO26 (backbone +
  neck + head) by default, with a config flag to keep the head in full
  precision as an ablation; multispectral (RGB+IR) input required, DroneVehicle
  the leading dataset candidate; primary benchmark hardware is CPU-class
  (Raspberry Pi/ARM), Jetson as a secondary comparison point, MCU deployment
  is a stretch goal only. See `CHECKLIST.md` for full detail.
- Open: fusion architecture (early/mid/late), final dataset confirmation,
  comparison model list.

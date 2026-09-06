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
- Open: fusion architecture (early/mid/late), comparison model list.

## 2026-09-06 (later)

- Dataset finalized: **TRGB** + **VTSaR**, both RGB+thermal aligned,
  single-class (person) aerial detection. TRGB (IJCAS 2025, CC BY-NC 4.0,
  mountain/forest terrain, download via Google Drive) and VTSaR (IEEE TGRS
  2024, license TBD, more varied scenes + a synthetic mosaic-augmented
  subset AS-VTSaR, download via Baidu Pan). Neither repo documents
  annotation format or splits — need to download and inspect directly.
- VTSaR's own paper ("Robust Aerial Person Detection with Lightweight
  Distillation Network for Edge Deployment") is closely related work and a
  strong comparison-model candidate — read it in full before finalizing the
  comparison list.
- Task is now fixed as single-class aerial person detection (not general
  multi-class UAV detection) — simplifies the head/anchor design and mAP
  reporting to a single AP number.

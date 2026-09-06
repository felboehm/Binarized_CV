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

## 2026-09-06 (even later)

- Swapped VTSaR out for **WiSARD** as the second dataset. Reason: VTSaR
  doesn't fully publish its dataset and states no license — too risky for a
  thesis that needs reproducible, citable results. WiSARD (Broyles, Hayner &
  Leung, IROS 2022, UW Autonomous Flight Systems Lab) is MIT-licensed,
  wilderness SAR aerial imagery, with a multi-modal subset of 15,453
  synchronized visual-thermal pairs (plus larger visual-only and
  thermal-only subsets we don't need). Google Drive download, a smaller
  ~972MB multi-modal sample available to validate the pipeline before
  pulling the full ~40.5GB set.
- VTSaR's paper/method is still worth reading as related work and a
  reimplementable comparison model — just not as a data source anymore.
- Dataset lineup is now **TRGB + WiSARD**, both wilderness/forest
  visual-thermal aerial person detection — similar enough in domain that
  combining them for training (rather than cross-dataset eval) may make
  more sense; revisit once both are downloaded and inspected.

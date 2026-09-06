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

## 2026-09-06 (data inspection)

- Both datasets downloaded (WiSARD: multi-modal sample only, one flight,
  264 pairs) and inspected directly. Both use YOLO-txt labels, single class
  `0` = person — no format conversion needed, straightforward to load.
- TRGB: pre-split train/val/test (4118/324/330 pairs), RGB 1280×800, IR
  640×512. Minor cleanup needed: `.DS_Store` files, and the test RGB folder
  is named `RGB_images_test copy` (macOS artifact).
- WiSARD sample: no split, 264 VIS (3840×2160) + 264 IR (640×512) frames,
  paired by matching numeric frame index across two separately-named
  folders.
- **Important finding**: neither dataset is pixel-registered between
  modalities — resolutions differ substantially (TRGB ~2×, WiSARD ~6×) and
  per-modality label coordinates for the same instance don't match (checked
  a TRGB pair directly). "Aligned" in both papers means temporally
  synchronized, not spatially co-registered. This rules out naive early
  fusion via channel-concat without a warping step, and leans the fusion
  architecture decision toward mid/late fusion with learned alignment
  (CFT/ICAFusion-style) instead — no ground-truth calibration/homography is
  provided by either dataset to do the warping ourselves reliably.
- Next: pull the full WiSARDv1 set (currently only have the sample) and
  check whether it defines a split; write the shared data loader.

## 2026-09-06 (data loader)

- Built the shared data loader/converter: per-dataset discovery
  (`data/datasets/trgb.py`, `data/datasets/wisard.py`) producing a common
  `PairRecord`, a manifest builder/CLI (`scripts/build_manifest.py` →
  `data/processed/manifest.jsonl`), and a PyTorch `MultispectralPersonDataset`.
- Ran it against the real downloaded data and caught a real bug before it
  could silently bias training: TRGB filenames use two different
  conventions *within the same folder* — some pairs are named by a bare
  numeric id (`8251066.jpg` in both RGB and IR folders), others by a
  modality-prefixed id (`RGB_1505.jpg` / `IR_1505.jpg`, same number,
  different prefix per modality). Naive exact-stem intersection only
  matched the bare-numeric ones, silently dropping 1737 of 4772 pairs
  (avg ~36%) — would have meant training on a biased subset without any
  error. Fixed by canonicalizing both naming schemes to the same id before
  matching. Manifest count now matches the documented split exactly
  (4118/324/330 for TRGB, 263 of 264 WiSARD sample frames).
- 16 unit tests added, all using synthetic tmp-dir fixtures (not the real
  gitignored data) so they run in a fresh checkout/CI without the datasets
  present.

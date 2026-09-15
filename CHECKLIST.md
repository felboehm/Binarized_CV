# Master Thesis Checklist — Binarized Multispectral YOLO26 for UAV Onboard CV

Goal: binarize a YOLO26 object detector — extended to fuse multispectral
(e.g. RGB + thermal/IR) input — to cut inference latency and power draw for
UAV deployment, while quantifying the accuracy trade-off against
full-precision YOLO26 and other SOTA efficient/edge detectors.

## 1. Scoping & Literature Review
- [x] Thesis use case: **aerial person detection (search & rescue-adjacent)**,
      fixed by the TRGB/WiSARD dataset choice — task is detection only
- [ ] Survey binarization methods for CNNs/detectors: XNOR-Net, Bi-Real Net,
      ReActNet, BiDet, BNext, IR-Net — pick which technique(s) to adapt to YOLO26
- [ ] Survey prior binarized/quantized YOLO work (BiDet, BNN-YOLO variants) to
      identify gaps and avoid duplicating known-negative results
- [ ] Survey multispectral/RGB-IR fusion detection architectures (e.g. CFT,
      ICAFusion, CFR-3) — identify a fusion point (early/mid/late) that is
      both accurate and binarization-friendly
- [ ] Read the VTSaR paper ("Robust Aerial Person Detection with Lightweight
      Distillation Network for Edge Deployment", IEEE TGRS 2024) in full —
      still directly relevant related work (lightweight/distilled aerial
      person detection) even though we're no longer using its dataset;
      candidate method to reimplement and evaluate on our own data as a
      comparison model
- [ ] Read the WiSARD paper (Broyles, Hayner & Leung, IROS 2022) and the
      TRGB paper (Shin et al., IJCAS 2025) for their baseline
      methods/metrics — these are on our actual data, so more directly
      reproducible as baselines than VTSaR's
- [ ] Identify which YOLO26 layers are binarization-resistant (first/last layer,
      detection head, and now the modality-fusion layer) — decide a
      full-vs-partial binarization strategy
- [ ] Pick 2-4 SOTA comparison models — VTSaR's lightweight/distilled
      detector method is a candidate to reimplement on our own data;
      otherwise a strong single-modality baseline (YOLO26 fp32, a
      pruned/distilled variant, another binarized detector) run on the same
      fused input — must be reproducible with available code/weights.
      **First candidate implemented**: Balla & Shrestha, "Multispectral
      Human Presence Detection using Adapted YOLO Network" (EUSIPCO 2025,
      OsloMet) — arguably a closer match than VTSaR: SAR-drone human
      detection, RGB+thermal early fusion (4-channel YOLOv8) + bicubic
      neck upsampling for small objects, code public
      (github.com/frnc96/ms-yolov8, AGPL-3.0, fork of ultralytics).
      Reimplemented as `ms_yolov8`
      (`src/binarized_cv/models/detectors/ms_yolov8.py`) against our own
      `BaseDetector` interface rather than depending on their fork
      directly — see `docs/labnotes.md` 2026-09-07 for the full writeup,
      including why this is a deliberately weak baseline on TRGB/WiSARD
      specifically (their method assumes spatial alignment our data
      doesn't have).
- [ ] Write thesis proposal / research plan, get advisor sign-off

## 2. Environment & Repo Setup
- [ ] Decide framework (PyTorch, given YOLO ecosystem) + BNN library
      (Larq, BitTorch/BNext codebase, or custom STE-based binarized layers)
- [ ] Pin Python/CUDA/cuDNN/PyTorch versions; set up `requirements.txt` or
      `pyproject.toml` + lockfile
- [ ] Repo structure: `src/`, `configs/`, `scripts/`, `data/` (gitignored),
      `experiments/` or `runs/` (gitignored), `docs/`, `notebooks/`
- [ ] Set up experiment tracking (Weights & Biases / MLflow / TensorBoard)
- [ ] Set up reproducibility basics: seed control, deterministic flags, config
      files per experiment (Hydra/YAML)
- [ ] Decide compute resources: local GPU vs. university cluster vs. cloud;
      confirm storage/quota for datasets and checkpoints
- [ ] Set up CI or at minimum a lint/test pre-commit hook

## 3. Data
- [x] Dataset(s) decided: **TRGB and WiSARD** — both wilderness aerial
      person-detection data with a visual-thermal modality, matching the
      multispectral requirement. VTSaR was dropped: it doesn't fully publish
      its dataset and has no stated license — too risky for a thesis that
      needs reproducible, citable results.
  - **TRGB** — repo: github.com/godhj93/trgb_dataset. From "Rapid and Safe
    Human Detection in Uninhabited Terrains Integrating Formation Flight and
    Multispectral Imaging" (IJCAS 2025). Aerial RGB+thermal, mountainous/
    forested uninhabited terrain. **License: CC BY-NC 4.0 — non-commercial
    only**, fine for thesis use but note if the repo/thesis is ever
    dual-licensed or productized later. Download: Google Drive link in repo
    README. No code/format docs in the repo itself.
  - **WiSARD** — sites.google.com/uw.edu/wisard. From Broyles, Hayner &
    Leung (2022), IEEE/RSJ IROS. UW Autonomous Flight Systems Lab dataset of
    drone flights over Washington wilderness for search-and-rescue human
    detection. Three subsets: visual-only (26,862 labeled RGB images),
    thermal-only (29,989 labeled LWIR images), and **multi-modal**
    (15,453 temporally synchronized visual-thermal pairs — this is the
    subset we need). **License: MIT** — clean, no non-commercial
    restriction, better than TRGB's CC BY-NC for anything beyond pure
    thesis use. Download: Google Drive, full dataset (WiSARDv1, ~40.5GB) or
    a smaller multi-modal-only sample (~972MB) — start with the sample to
    validate the pipeline before pulling the full set. Annotation format,
    resolution, and formal splits not documented on the site — inspect
    after download.
- [x] Both datasets downloaded and inspected (WiSARD: multi-modal sample
      only, 1 flight sequence). Confirmed facts:
  - **Annotation format**: both use YOLO-style txt (`class x_center y_center
    width height`, normalized 0-1), single class `0` = person. Directly
    compatible with a standard YOLO loader, no format conversion needed.
  - **TRGB**: pre-split train/val/test. RGB images 1280×800, IR images
    640×512. Counts: train 4118 pairs, val 324 pairs, test 330 pairs (the
    test RGB folder is macOS-mangled as `RGB_images_test copy` — needs
    renaming/handling in the loader). `.DS_Store` files present, need
    filtering.
  - **WiSARD sample**: one flight (`210417_MtErie_Enterprise`), 264 VIS +
    264 IR frames, paired 1:1 by numeric frame index across two separate
    folders (`..._VIS_0003/`, `..._IR_0004/` — folder *names* don't match,
    frame indices in the filenames do). VIS frames 3840×2160, IR frames
    640×512. No train/val/test split — this is expected to come from the
    full WiSARDv1 download, need to check whether the full set defines one
    or we split ourselves.
  - **Critical: neither dataset is pixel-registered.** RGB/VIS and IR labels
    are independent per-modality annotations with different normalized box
    coordinates for the same instance (verified on a TRGB pair) — "aligned"
    in both papers means *temporally synchronized*, not *spatially
    co-registered*. Combined with the large resolution mismatches (TRGB
    ~2×, WiSARD ~6× between modalities), **naive early fusion by
    channel-concatenation is not viable without a warping/homography step**.
    This pushes the fusion architecture decision (section 4) toward mid/late
    fusion or a learned-alignment approach (CFT/ICAFusion-style), not simple
    early fusion.
- [ ] Get the full WiSARDv1 download (not just the sample) once the pipeline
      is validated, and check whether it defines its own split
- [x] Data loader/converter implemented (`src/binarized_cv/data/`):
      `datasets/trgb.py` and `datasets/wisard.py` discover RGB+IR pairs and
      emit a common `PairRecord`; `manifest.py` combines both into one
      manifest (`scripts/build_manifest.py` writes it to
      `data/processed/manifest.jsonl`); `dataset.py` has a
      `MultispectralPersonDataset` (PyTorch) that loads images + YOLO boxes
      per pair. Verified against the real downloaded data: **4772 TRGB
      pairs** (matches the documented 4118/324/330 split exactly) and
      **263 WiSARD pairs** (264 frames minus 1 unmatched). Along the way,
      found and fixed a real bug: TRGB mixes bare-numeric and
      modality-prefixed (`RGB_####`/`IR_####`) filenames *within the same
      folder* — naive exact-stem matching silently dropped ~1737 pairs
      (2729/4118 caught) until fixed to canonicalize both naming schemes to
      the same id before pairing. 16 unit tests (`tests/test_labels.py`,
      `test_trgb_discovery.py`, `test_wisard_discovery.py`,
      `test_manifest.py`) cover both datasets' quirks with synthetic
      fixtures, so they run without the (gitignored) real data present.
      No augmentation/resizing/normalization yet — deferred to the training
      pipeline once the fusion architecture is decided.
- [ ] Decide how TRGB and WiSARD are used relative to each other: combined
      training set, or one for training + one for cross-dataset
      generalization evaluation (the latter is often more useful for
      "credible claims" since it tests out-of-distribution robustness) —
      both are wilderness/forest terrain so may be similar enough to combine
      rather than cross-evaluate; revisit once the full WiSARD set is in
- [ ] Sanity-check class balance and small-object density across both
      datasets (aerial person imagery skews toward small objects — relevant
      for detector choice/anchors)

## 4. Baseline Model
- [ ] Decide multispectral fusion architecture: early fusion (concat RGB+IR
      channels before the stem), mid fusion (twin-stream backbones merged at
      a middle stage), or late fusion (separate backbones, merge at
      neck/head) — early/mid fusion is usually more binarization-friendly
      since it avoids doubling full-precision backbone compute.
      **Constraint from actual data (section 3): TRGB and WiSARD are not
      pixel-registered between modalities (different resolutions, different
      per-modality label coordinates) — plain early fusion by channel-concat
      won't work as-is.** Options: (a) resize+warp/homography-align IR to
      RGB coordinate space as a preprocessing step, enabling early fusion;
      (b) skip pixel alignment and use mid/late fusion with independent
      per-modality feature extraction merged via a learned module (e.g.
      cross-attention, as in CFT/ICAFusion); (b) is likely lower-risk given
      no ground-truth camera calibration/homography is provided by either
      dataset. **Still open** — a mid-fusion (b)-style placeholder (plain
      channel-concat of independently-extracted features, not a learned
      cross-attention module) now exists to unblock the pipeline (see
      below), but the real CFT/ICAFusion-style fusion decision is unmade.
- [x] Plug-and-play model system built so any model (binarized backbone, a
      different fusion module, a reimplemented comparison model) drops in
      without touching data loading or the training/eval loop:
      `BaseDetector` interface (`src/binarized_cv/models/base.py`) +
      `MODEL_REGISTRY`/`build_model` (`registry.py`) — a new model just
      subclasses `BaseDetector` and adds `@register_model("name")`.
      Contract mirrors torchvision's detection API: `forward(images,
      targets=None)` returns a loss dict when training, a list of
      `{"boxes", "scores", "labels"}` detections otherwise. Resolved the
      RGB/IR-misalignment supervision question this forced: **RGB is the
      primary/target modality** (losses + eval in RGB coordinate space, IR
      is an auxiliary input only) — see `docs/labnotes.md` 2026-09-07.
- [x] Real YOLO26 wired in as a registered model (`yolo26`,
      `src/binarized_cv/models/detectors/yolo26.py`), **RGB-only so far** —
      wraps `ultralytics.nn.tasks.DetectionModel` (the actual YOLO26n/s/m/l/x
      CSP-Darknet backbone + PAN neck + NMS-free DFL-free dual head, not a
      reimplementation) behind our `BaseDetector` interface. Chose to depend
      on the real `ultralytics` package rather than reimplement, after
      confirming with the user this means the repo's license terms follow
      theirs (AGPL-3.0) as soon as this model is used — **see section 12,
      still needs a final LICENSE-file decision**. Verified end-to-end
      (train + eval, synthetic data) through the exact same `train.py`/
      `evaluate.py` CLI as `simple_fusion`, no code changes needed to swap
      models — proves the plug-and-play system actually works across a toy
      model and a real production one. Details in `docs/labnotes.md`
      2026-09-07 (YOLO26 section).
- [ ] Extend YOLO26's input stem/backbone for the chosen fusion point and
      multispectral (RGB+IR) input — **not started**; `yolo26` above is
      RGB-only, IR is currently just ignored by this model
- [ ] Reproduce or closely match published baseline metrics (from the TRGB
      or WiSARD papers, or a reimplemented VTSaR-style method) before
      touching binarization, so later deltas are trustworthy — needs a real
      training run on the actual (not synthetic) dataset, not done yet
- [ ] Reproduce or closely match published baseline metrics (from the TRGB
      or WiSARD papers, or a reimplemented VTSaR-style method) before
      touching binarization, so later deltas are trustworthy
- [ ] Establish baseline inference benchmarking harness (latency, FPS) on
      target hardware

## 5. Binarization Implementation
- [x] Binarization scope decided: **default to fully binarized** (backbone,
      neck, and detection head), with a **config flag to keep the detection
      head at full precision** as an ablation variant — lets you report both
      the maximum-efficiency point and the accuracy-recovery point in the
      same experiment matrix
- [ ] Implement/import binary conv layers with a chosen weight+activation
      binarization scheme and straight-through estimator for gradients
- [ ] Integrate binarized layers into YOLO26 backbone, neck, and head, gated
      by a per-block config flag (so head precision is a training-config
      toggle, not a code fork)
- [ ] Validate forward/backward pass numerically (unit tests, gradient checks)
      before full training runs
- [ ] Plan a training recipe for BNNs (they need different LR schedules,
      warm-start from pretrained fp32 weights, longer training, knowledge
      distillation from the fp32 teacher is common)

## 6. Training
- [ ] Define experiment matrix: fp32 baseline, fully binarized, partially
      binarized (ablation), with/without distillation
- [ ] Hyperparameter search plan (LR, batch size, distillation weight) —
      budget compute for this up front
- [ ] Checkpointing + resumability for long runs
- [ ] Logging: loss curves, mAP per epoch, gradient norms (BNN training is
      unstable — watch for this explicitly)

## 7. Accuracy Evaluation
- [ ] Standard detection metrics: mAP@0.5, mAP@0.5:0.95, per-class AP,
      precision/recall
- [ ] Small-object performance breakdown (important for UAV/aerial imagery)
- [ ] Qualitative results: sample detections, failure case gallery

## 8. Efficiency Evaluation (the core thesis contribution)
- [x] Target hardware decided: **CPU-class (Raspberry Pi / ARM Cortex-A, or
      x86) as primary target** — this is where binarized XNOR+popcount ops
      show a real, measurable advantage with existing tooling (Larq, custom
      CPU kernels). **Jetson as a secondary comparison point** (GPU-class
      inference). **MCU (Cortex-M) is a stretch goal only** — no mainstream
      framework (TFLite Micro/CMSIS-NN) supports 1-bit ops, so MCU deployment
      would require a custom bitwise inference runtime; attempt only late,
      only if a tiny model variant's binarized weight footprint fits in
      available flash/RAM, and only as a bonus result, not a dependency
- [ ] Acquire/confirm access to chosen CPU-class board (Raspberry Pi 4/5 or
      similar) and a Jetson unit
- [ ] Inference latency benchmark on the CPU-class board and Jetson, matched
      resolution/batch size
- [ ] Power measurement setup: hardware power meter (e.g. USB inline power
      meter for RPi, INA219/INA226 or Jetson's onboard power rails for Jetson)
      + measurement script, report energy-per-inference (mJ) not just watts
- [ ] Model size / memory footprint comparison (binarized weights compress
      ~32x in theory — measure actual on-disk and in-memory size)
- [ ] Throughput (FPS) at matched batch size/resolution across all compared
      models
- [ ] Confirm results are averaged over enough runs with variance reported
      (thermal throttling on embedded boards is a real confound)

## 9. Comparative Analysis
- [ ] Build a results table: accuracy vs. latency vs. power vs. model size for
      every model in the comparison set
- [ ] Pareto-frontier plot (accuracy vs. efficiency) to visually support the
      "credible claims" framing
- [ ] Statistical significance / error bars where feasible
- [ ] Honest discussion of where binarization loses the most accuracy (which
      classes/scenarios) — strengthens thesis credibility

## 10. Deployment Validation (if in scope)
- [ ] Export/deploy path to target UAV compute (ONNX/TensorRT or custom BNN
      runtime) — confirm the binarized ops actually run efficiently on target
      hardware, not just simulated in PyTorch
- [ ] End-to-end on-device demo (video feed → detections) for thesis defense

## 11. Writing & Documentation
- [ ] Keep a running lab notebook (`docs/labnotes.md` or similar) of what was
      tried and why — invaluable for the methods chapter later
- [ ] Draft related-work section early alongside literature review (step 1)
- [ ] Methods chapter drafted incrementally as implementation stabilizes
- [ ] Results chapter built directly from the results table/plots in step 9
- [ ] Repo README with reproduction instructions (advisor/examiner may want
      to run it)

## 12. Repo Hygiene
- [ ] `.gitignore` covers datasets, checkpoints, logs, wandb runs (partially
      present already — verify it's complete)
- [ ] LICENSE decision (check university/advisor policy on thesis code).
      **Now partially forced**: the repo depends on `ultralytics` (for
      YOLO26, see section 4), which is AGPL-3.0 — using their code/models
      means either the whole repo is AGPL-3.0 licensed, or an Ultralytics
      Enterprise license is obtained. No LICENSE file has been added yet
      pending your/advisor confirmation — AGPL-3.0 is standard and fine for
      a public academic thesis repo, but it's a repo-wide, hard-to-reverse
      choice worth deciding deliberately rather than defaulting into.
- [ ] `CITATION.cff` if the repo will be public and citable

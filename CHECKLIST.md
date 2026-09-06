# Master Thesis Checklist — Binarized Multispectral YOLO26 for UAV Onboard CV

Goal: binarize a YOLO26 object detector — extended to fuse multispectral
(e.g. RGB + thermal/IR) input — to cut inference latency and power draw for
UAV deployment, while quantifying the accuracy trade-off against
full-precision YOLO26 and other SOTA efficient/edge detectors.

## 1. Scoping & Literature Review
- [x] Thesis use case: **aerial person detection (search & rescue-adjacent)**,
      fixed by the TRGB/VTSaR dataset choice — task is detection only
- [ ] Survey binarization methods for CNNs/detectors: XNOR-Net, Bi-Real Net,
      ReActNet, BiDet, BNext, IR-Net — pick which technique(s) to adapt to YOLO26
- [ ] Survey prior binarized/quantized YOLO work (BiDet, BNN-YOLO variants) to
      identify gaps and avoid duplicating known-negative results
- [ ] Survey multispectral/RGB-IR fusion detection architectures (e.g. CFT,
      ICAFusion, CFR-3) — identify a fusion point (early/mid/late) that is
      both accurate and binarization-friendly
- [ ] Read the VTSaR paper ("Robust Aerial Person Detection with Lightweight
      Distillation Network for Edge Deployment", IEEE TGRS 2024) in full —
      closest existing work to this thesis's goal, should anchor the
      related-work section and comparison model list
- [ ] Identify which YOLO26 layers are binarization-resistant (first/last layer,
      detection head, and now the modality-fusion layer) — decide a
      full-vs-partial binarization strategy
- [ ] Pick 2-4 SOTA comparison models — VTSaR's own lightweight/distilled
      detector is a strong candidate given it's evaluated on this exact data;
      otherwise a strong single-modality baseline (YOLO26 fp32, a
      pruned/distilled variant, another binarized detector) run on the same
      fused input — must be reproducible with available code/weights
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
- [x] Dataset(s) decided: **TRGB and VTSaR** — both RGB+thermal aligned,
      single class (**person**), matching the multispectral requirement.
      Both are class-compatible (both person-only), so combining or
      cross-evaluating between them is straightforward.
  - **TRGB** — repo: github.com/godhj93/trgb_dataset. From "Rapid and Safe
    Human Detection in Uninhabited Terrains Integrating Formation Flight and
    Multispectral Imaging" (IJCAS 2025). Aerial RGB+thermal, mountainous/
    forested uninhabited terrain. **License: CC BY-NC 4.0 — non-commercial
    only**, fine for thesis use but note if the repo/thesis is ever
    dual-licensed or productized later. Download: Google Drive link in repo
    README. No code/format docs in the repo itself.
  - **VTSaR** — repo: github.com/zxq309/VTSaR. From "Robust Aerial Person
    Detection with Lightweight Distillation Network for Edge Deployment"
    (IEEE TGRS 2024) — **directly relevant related work / potential
    comparison baseline**, since it's already doing edge-efficient aerial
    person detection on this exact data. Has a real-capture subset (A-VTSaR,
    dual-camera gimbal RGB+IR) and a synthetic mosaic-augmented subset
    (AS-VTSaR). Download: Baidu Pan (may need a workaround/VPN outside
    China — flag as a practical access risk). License not stated in README —
    confirm before any redistribution or public repo use. No code/format
    docs in the repo itself.
- [ ] Download both datasets and inspect actual file layout: annotation
      format (YOLO txt / COCO json / XML), image resolution, RGB/thermal
      pairing convention, existing train/val/test split (none is documented
      in either README — likely need to create one)
- [ ] Write a converter for each to a common internal format (paired
      RGB+thermal image + YOLO-style label) so both datasets share one
      loader
- [ ] Decide how TRGB and VTSaR are used relative to each other: combined
      training set, or one for training + one for cross-dataset
      generalization evaluation (the latter is often more useful for
      "credible claims" since it tests out-of-distribution robustness) —
      TRGB's forest/mountain terrain vs. VTSaR's more varied scenes makes a
      cross-dataset generalization test a natural fit
- [ ] Read the VTSaR paper in full — its distillation/lightweight approach
      is close enough to this thesis's efficiency goal that it should shape
      the related-work section and possibly the comparison model list
- [ ] Sanity-check class balance and small-object density across both
      datasets (aerial person imagery skews toward small objects — relevant
      for detector choice/anchors)

## 4. Baseline Model
- [ ] Decide multispectral fusion architecture: early fusion (concat RGB+IR
      channels before the stem), mid fusion (twin-stream backbones merged at
      a middle stage), or late fusion (separate backbones, merge at
      neck/head) — early/mid fusion is usually more binarization-friendly
      since it avoids doubling full-precision backbone compute
- [ ] Extend YOLO26 input stem/backbone for the chosen fusion point and
      4-6 channel (RGB+IR) input
- [ ] Get the extended YOLO26 reference implementation running end-to-end
      (train + eval) on the chosen multispectral dataset at full precision —
      this is your accuracy/speed/power upper bound
- [ ] Reproduce or closely match published baseline metrics (from the VTSaR
      paper or comparable) before touching binarization, so later deltas
      are trustworthy
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
- [ ] LICENSE decision (check university/advisor policy on thesis code)
- [ ] `CITATION.cff` if the repo will be public and citable

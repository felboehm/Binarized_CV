# Master Thesis Checklist — Binarized Multispectral YOLO26 for UAV Onboard CV

Goal: binarize a YOLO26 object detector — extended to fuse multispectral
(e.g. RGB + thermal/IR) input — to cut inference latency and power draw for
UAV deployment, while quantifying the accuracy trade-off against
full-precision YOLO26 and other SOTA efficient/edge detectors.

## 1. Scoping & Literature Review
- [ ] Define thesis question precisely (which task: detection only, or also
      tracking/segmentation? which UAV use case: search & rescue, surveillance,
      obstacle avoidance?)
- [ ] Survey binarization methods for CNNs/detectors: XNOR-Net, Bi-Real Net,
      ReActNet, BiDet, BNext, IR-Net — pick which technique(s) to adapt to YOLO26
- [ ] Survey prior binarized/quantized YOLO work (BiDet, BNN-YOLO variants) to
      identify gaps and avoid duplicating known-negative results
- [ ] Survey multispectral/RGB-IR fusion detection architectures (e.g. CFT,
      ICAFusion, CFR-3, DroneVehicle baseline methods) — identify a fusion
      point (early/mid/late) that is both accurate and binarization-friendly
- [ ] Identify which YOLO26 layers are binarization-resistant (first/last layer,
      detection head, and now the modality-fusion layer) — decide a
      full-vs-partial binarization strategy
- [ ] Pick 2-4 SOTA comparison models — prefer ones with published
      multispectral/RGB-IR results if available (e.g. a DroneVehicle
      leaderboard method), otherwise a strong single-modality baseline
      (YOLO26 fp32, a pruned/distilled variant, another binarized detector)
      run on the same fused input — must be reproducible with available
      code/weights
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
- [x] Dataset scope decided: **UAV-specific and multispectral** — general
      RGB-only sets (COCO, VisDrone, DOTA) are out since they lack a second
      modality. Primary candidate: **DroneVehicle** (RGB+IR paired aerial
      imagery, vehicle detection, ~28k image pairs, established benchmark
      with published baselines to compare against). Alternatives to check if
      DroneVehicle doesn't fit the use case: **VTUAV** (RGB-thermal, UAV
      tracking — usable for detection via per-frame boxes), **Anti-UAV**
      (thermal, but detects UAVs themselves rather than from a UAV)
- [ ] Confirm final dataset choice covers the object classes relevant to the
      thesis use case (DroneVehicle is vehicle-only — decide if that's
      sufficient scope or if a broader-class set is needed)
- [ ] Confirm modality pairing/registration quality (RGB and IR frames must
      be spatially aligned — check the dataset's calibration/registration
      documentation)
- [ ] Establish train/val/test splits matching what comparison papers use, so
      numbers are citable apples-to-apples
- [ ] Data pipeline: download scripts, format conversion to YOLO label format
      with paired RGB+IR inputs, augmentation config (must apply identically
      to both modalities to preserve alignment)
- [ ] Sanity-check class balance and small-object density (UAV imagery skews
      toward small objects — relevant for detector choice/anchors)

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
- [ ] Reproduce or closely match published baseline metrics (from
      DroneVehicle leaderboard or comparable) before touching binarization,
      so later deltas are trustworthy
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

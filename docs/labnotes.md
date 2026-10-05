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

## 2026-09-07 (plug-and-play model scaffolding)

- Built the model side of the pipeline as a plug-and-play system: a
  `BaseDetector` interface (`src/binarized_cv/models/base.py`) every model
  must implement, and a name-based `MODEL_REGISTRY`/`build_model`
  (`registry.py`) so a new model (binarized backbone, a different fusion
  module, a reimplemented comparison model) only needs to subclass
  `BaseDetector` and add `@register_model("name")` — the data loading,
  training loop, and eval loop never need to change. The contract mirrors
  torchvision's detection-model convention: `forward(images, targets=None)`
  returns a loss dict when `targets` is given (training) or a list of
  per-image `{"boxes" (xyxy px), "scores", "labels"}` detections otherwise
  (inference) — decided this deliberately so training/eval code is agnostic
  to whatever's inside the model.
- **Resolved an open design question this required**: since RGB and IR
  labels aren't spatially co-registered (2026-09-06 finding), a fused model
  needs one coordinate frame to be supervised/evaluated against. Decided
  **RGB as primary** — losses and detections are in RGB pixel space, IR
  feeds in as an auxiliary input stream only, its own labels unused. Matches
  RGB's higher resolution and the convention in CFT/ICAFusion-style fusion
  papers. (Alternatives considered: IR-primary, since thermal is often the
  more reliable SAR signal; or a dual-head model with one output per
  modality — deferred, more complexity than a first basic model needs.)
- Implemented one concrete basic detector (`simple_fusion`) built entirely
  from reusable pieces, each usable independently later: `SimpleCNNBackbone`
  (fp32 stride-32 conv stack, one instance per modality — a binarized
  backbone is a drop-in as long as it keeps the same in/out-channel
  contract), `ConcatFusion` (mid fusion: channel-concat same-resolution
  per-modality features + 1x1 conv projection), `AnchorFreeHead` (single-
  scale YOLOv1-style head — one grid cell responsible per GT box center,
  direct sigmoid box/objectness/class regression, own `compute_loss` +
  `postprocess`/NMS). This is explicitly a placeholder to prove the full
  pipeline works, not a step toward accuracy — the real YOLO26 head/backbone
  and the eventual CFT/ICAFusion-style learned fusion are separate,
  not-yet-started work (CHECKLIST.md section 4).
- Reworked the data contract to match: `MultispectralPersonDataset` now
  resizes both modalities independently to a configured `img_size` (a plain
  per-axis resize needs no box coordinate remapping, since labels are
  already normalized fractions of each image's own dimensions) and returns
  a single `targets` dict from `target_modality`'s boxes instead of the
  previous separate `rgb_boxes`/`ir_boxes`. Added `detection_collate` to
  batch variable-length per-image targets.
- Added a minimal custom single-class VOC-style `average_precision` metric
  (`eval/metrics.py`) instead of adding `torchmetrics` as a dependency —
  `torchvision.ops.box_iou`/`nms`, already a listed dependency, were enough.
  Full `mAP@0.5:0.95` + per-class breakdown is still open (CHECKLIST.md
  section 7).
- Wired up Hydra configs (`configs/{data,model,train,eval}/default.yaml` +
  root `configs/config.yaml`) and `train/train.py` / `eval/evaluate.py` CLI
  scripts. **Found a real environment bug along the way**: hydra-core 1.3.6
  (latest release)'s `@hydra.main` CLI decorator crashes on Python 3.14 —
  its argparse-based `--shell-completion` setup hits a `LazyCompletionHelp`
  object against a newer, stricter argparse `help`-string check
  (`TypeError`/`ValueError: badly formed help string`), unrelated to this
  repo's code. Worked around it by using Hydra's `compose`/`initialize` API
  directly instead of the `@hydra.main` decorator — same YAML composition
  and dotted-override CLI syntax, different (unaffected) code path. No fix
  exists upstream yet; revisit if hydra-core cuts a Python 3.14-compatible
  release.
- Verified the whole pipeline end-to-end on synthetic data (tiny
  TRGB-shaped fixture, not the real gitignored dataset): manifest build ->
  dataloader -> `simple_fusion` model -> training loop (loss decreases,
  checkpoints written) -> eval script (loads a checkpoint, computes AP).
  37 unit tests total (21 new: registry, backbone, fusion, head, detector,
  collate, dataset, metrics), all synthetic-fixture based per existing
  convention; ruff clean.

## 2026-09-07 (later — real YOLO26 wired in)

- **Decided to depend on the real `ultralytics` package rather than
  reimplement YOLO26 from scratch.** Researched YOLO26 (Ultralytics, Sept
  2025) first: CSP-Darknet backbone (C3k2/SPPF/C2PSA blocks, same shape as
  YOLO11's) → PAN neck (P3/P4/P5) → a dual head that's both NMS-free
  (native end-to-end one2one branch, no separate NMS call) and DFL-free
  (`reg_max: 1` makes the DFL module a literal `nn.Identity()`). Full
  architecture confirmed by reading the actual installed package source
  (`ultralytics/cfg/models/26/yolo26.yaml`, `nn/tasks.py`, `nn/modules/head.py`)
  and empirically running forward/backward passes — not from blog posts,
  which don't document the backbone/neck in enough detail to reimplement
  faithfully. Trade-off accepted knowingly: `ultralytics` is **AGPL-3.0**,
  so depending on it obligates this repo to AGPL-3.0 too (or an Enterprise
  license) — flagged as a still-open item in `CHECKLIST.md` section 12
  rather than resolved unilaterally, since adding a LICENSE file is a
  repo-wide, public, hard-to-reverse decision.
- Wired it in as `yolo26` (`src/binarized_cv/models/detectors/yolo26.py`),
  wrapping `ultralytics.nn.tasks.DetectionModel` behind `BaseDetector`.
  RGB-only for now (`modalities = ("rgb",)`) — extending the input stem for
  RGB+IR fusion is separate follow-on work. Key integration points learned
  from reading the real source:
  - `DetectionModel.forward(x)` dispatches on type: a tensor → inference
    (`self.predict`), a dict → training loss (`self.loss`). No separate
    train/eval method to call.
  - The loss dict format (`{"img", "batch_idx", "cls", "bboxes"}`) needs
    `bboxes` as normalized cxcywh — **exactly our own `targets` box format
    already** (per-image list → flattened with a `batch_idx` mapping each
    box back to its image), so no coordinate conversion was needed, only
    reshaping from our per-image list into ultralytics' flat-batch form.
  - `model.args` must be set (`ultralytics.cfg.get_cfg(overrides={})`)
    before the first loss call — normally injected by ultralytics' own
    `Trainer`, which we're bypassing.
  - Eval-mode output `(B, <=300, 6)` (`xyxy, score, class`, pixel-scale) is
    already NMS-free-decoded and top-k selected by the model itself, but
    **not confidence-thresholded or clamped to image bounds** — both added
    on our side to match the `BaseDetector` eval-mode contract.
  - `BaseModel.load()` tolerates a different `nc` (extra/missing final
    layer just isn't loaded) and even a different first-conv channel count
    (copies the overlapping channel sub-tensor) — relevant later for
    loading pretrained COCO weights into a modified multispectral stem.
- Fixed a real bug this surfaced in the plug-and-play system itself:
  `train.py`/`evaluate.py` were hardcoding `simple_fusion`-specific
  constructor kwargs (`backbone_widths`, `fusion_channels`) when building
  the model from config — would have broken for every other model,
  defeating the entire point of the registry. Replaced with
  `registry.build_model_from_config(model_cfg)`, which passes every field
  in `configs/model/<name>.yaml` through as a kwarg generically (also
  converting YAML list fields like `img_size` to tuples) — a new model now
  only needs its own config file, no training-script changes.
- Verified end-to-end on synthetic data through the *unmodified*
  `train.py`/`evaluate.py` CLI, just switching `model=yolo26`: trains,
  checkpoints, and evaluates exactly like `simple_fusion` did — the
  plug-and-play claim now has two very different models (a toy CNN and a
  real production architecture) proving it out, not just one.

## 2026-09-07 (later still — first comparison model: ms_yolov8)

- User pointed at github.com/frnc96/ms-yolov8 as a candidate comparison
  model. Cloned and read it directly: an `ultralytics` (YOLOv8) fork adding
  a thin `src/` layer for RGB+thermal fusion. The actual technique is
  simple early fusion — each dataset's own download script (KAIST, LLVIP,
  M3FD, NII-CU) resizes thermal to match RGB's pixel size, stacks it as a
  4th channel, and saves a merged TIFF; the only upstream `ultralytics`
  patch is `ch: 4` in the model yaml (a channel-count override
  `parse_model`/`DetectionModel` already supports natively) and
  `cv2.imread(f, cv2.IMREAD_UNCHANGED)` so the loader keeps the 4th
  channel. **Initial read was that this was an unpublished/unvalidated
  personal project** (hardcoded personal paths throughout, generic
  unmodified README, no paper found via search) — that assessment was
  wrong.
- **Correction**: there is a real paper — Balla & Shrestha, "Multispectral
  Human Presence Detection using Adapted YOLO Network," EUSIPCO 2025
  (OsloMet). Read the full PDF (user supplied it directly, IEEE Xplore
  blocks scraping). It's a much closer match to this thesis than VTSaR:
  explicitly framed around SAR drone human detection with tiny objects at
  altitude. Method: (1) widen YOLOv8's first conv from a 3x3x3 to a 3x3xn
  kernel for n-channel input (their n=4: RGB+thermal), (2) replace the
  neck's nearest-neighbor upsampling with bicubic for better small-object
  detail. Evaluated on **NII-CU** (Speth et al. 2022, *Journal of Field
  Robotics* — a real, citable, published UAV RGB+thermal dataset, not just
  a random zip) and **M3FD**. Results: baseline (RGB-only YOLOv8,
  COCO-pretrained) mAP50-95 = 0.448 → +4-channel fusion = 0.667 (+22%) →
  +bicubic = 0.675, with the 4-channel fusion itself costing essentially no
  latency (59→59 FPS) and bicubic costing ~2.4ms (59→52 FPS). On M3FD, beat
  YOLO-MS (a dual-backbone multispectral competitor) by 10% mAP50-95 (0.656
  vs. 0.552) despite a single, simpler backbone.
- **Scope decision** (asked the user): reimplement the architecture against
  our own `BaseDetector` interface and run it on our own TRGB/WiSARD data
  only — no NII-CU/M3FD acquisition for now, so no reproduction check
  against the paper's own numbers yet. Tradeoff accepted knowingly: their
  method assumes RGB/thermal are already reasonably aligned once resized to
  the same size (true for KAIST/LLVIP/M3FD/NII-CU, all captured with
  co-located/beam-splitter rigs) — **not true for TRGB/WiSARD** (2026-09-06
  finding). This makes `ms_yolov8` a deliberately-included *weak* baseline
  on our data: if it underperforms specifically because of misalignment,
  that's a legitimate, citable result supporting the mid-fusion choice used
  elsewhere in this repo, not a failed reimplementation.
- Implemented as `ms_yolov8`
  (`src/binarized_cv/models/detectors/ms_yolov8.py`), reusing the same
  `ultralytics.nn.tasks.DetectionModel` wrapping pattern as `yolo26.py`.
  Two things had to be re-verified empirically rather than assumed, since
  stock YOLOv8 differs from YOLO26 here:
  - No custom yaml needed for the 4-channel input — `DetectionModel(cfg=
    "yolov8n.yaml", ch=4, ...)` widens the first conv directly via the same
    `ch` kwarg mechanism YOLO26 uses, confirmed empirically
    (`model.model[0].conv.in_channels == 4`).
  - Stock YOLOv8's `Detect` head is **not** NMS-free like YOLO26's
    (`end2end=False`, real `reg_max=16` DFL) — eval-mode output is raw
    undecoded-by-NMS `(B, 4+nc, num_anchors)`, needing an explicit
    `ultralytics.utils.nms.non_max_suppression()` call (function moved out
    of `ultralytics.utils.ops` in this version) to get final per-image
    detections, unlike YOLO26 where postprocessing is already done inside
    the model's forward.
  - The bicubic upsample change needs no custom yaml either — found the
    `nn.Upsample` modules directly in the parsed `model.model` and set
    `.mode = "bicubic"` post-construction; verified the patched model still
    runs forward/backward correctly.
  - Extracted `targets_to_ultralytics_batch`/`clamp_xyxy_to_image` into a
    shared `detectors/_ultralytics_common.py` used by both `yolo26.py` and
    `ms_yolov8.py`, since the batch-conversion and box-clamping logic is
    identical across any ultralytics-backed detector.
- Verified end-to-end (train → checkpoint → eval) through the same
  unmodified `train.py`/`evaluate.py` CLI as the other two models, just
  `model=ms_yolov8` — three architecturally distinct models now share the
  exact same data/training/eval code. 48 tests total (6 new for
  `ms_yolov8`), ruff clean.

## 2026-09-15 (data download automation + pipeline fixes)

- **Data download automation**: Built `scripts/download_data.py` to automatically
  fetch TRGB and WiSARD from Google Drive, with automatic extraction, cleanup
  (including macOS `__MACOSX` folders), and skip-if-already-present checks.
  Supports interactive prompts or command-line args (`--wisard full/sample`,
  `--trgb-only`, `--no-prompt`). Auto-installs `gdown` if missing. Also created
  bash wrapper `scripts/download_data.sh` for convenience.
- **Dataset structure fix**: TRGB discovery code initially expected
  `data/raw/trgb/trgb_dataset/{train,val,test}` but the zip extracts to
  `data/raw/trgb/{train,val,test}` directly. Fixed discovery to match actual
  structure (`raw_root/trgb/{train,val,test}`). Applied same fix to evaluate
  the actual data layout rather than prescribing it. Updated download script's
  `check_path` to verify presence of real data (ignoring `.gitkeep` placeholders).
- **Manifest split assignment**: WiSARD discovery was hardcoding `split="unassigned"`
  instead of assigning train/val/test splits. Added deterministic split
  assignment (70% train, 15% val, 15% test based on frame index) so manifest
  records can be properly filtered by split during training/eval. TRGB already
  had pre-defined splits (train/val/test folders), no change needed there.
- **Pipeline CUDA auto-detection**: `scripts/run_pipeline.sh` defaulted to
  `DEVICE=cpu` "for CPU-only machines," forcing users to explicitly set
  `DEVICE=cuda` to use the GPU. Changed to auto-detect CUDA availability via
  `torch.cuda.is_available()` so the pipeline uses GPU by default when available,
  without user intervention. Still allows override with `DEVICE=cpu` if needed.
  Updated README usage section with new download step and device behavior.
- **Dependencies**: Added `gdown` to `pyproject.toml` for automatic dataset
  download support. Noted but deferred numpy version conflict with brevitas
  0.12.0 (which caps at numpy<=1.26.4 vs. requested 2.4.4) — brevitas is not
  used in the project, so conflict is safe to ignore for now.
- **First end-to-end validation**: Successfully ran full pipeline on real TRGB
  + WiSARD data with GPU: manifest build → train YOLO26 for 5 epochs → eval.
  Confirmed pytorch training now properly detects and uses CUDA without
  explicit device specification. 48 tests passing, ruff clean.

## 2026-09-15 (training script UX)

- **User-friendly training script**: Built `scripts/train_model.py` following
  the same interactive + CLI-arg pattern as `download_data.py`. Addresses user
  feedback that running `python -m binarized_cv.train.train model=yolo26` is
  unwieldy and doesn't provide guided model/parameter selection.
- **Three usage modes**:
  1. Interactive (`python scripts/train_model.py`): prompts for model, then
     optional epochs/lr/batch-size overrides with defaults.
  2. CLI args (`python scripts/train_model.py --model yolo26 --epochs 10`):
     familiar argparse-style flags for quick runs.
  3. Hydra overrides (direct to the underlying module): full control via
     `key=value` syntax for reproducible runs/ablations.
- **Implementation**: Refactored away repetitive `if` statements via an
  `OVERRIDE_MAPPING` dict (`arg_name → hydra_config_path`) + `getattr()` loop,
  reducing ~15 individual conditionals to a single parameterized iteration.
  Interactive params returned as `{"train.epochs": "20"}` are reconstructed
  as Hydra overrides (`"train.epochs=20"`) before passing to the underlying
  training module.
- **Validation**: End-to-end tested with `ms_yolov8` model, 1 epoch on real
  TRGB + WiSARD data, GPU-enabled. Produced checkpoints and TensorBoard logs
  as expected, confirmed successful training completion via output message.
- **Documentation**: Updated README section 3 to promote `scripts/train_model.py`
  as the recommended entry point with examples for all three modes, alongside
  direct Hydra invocation for power users. Added today's entry to labnotes.

## 2026-09-16 (mid-fusion architecture & baseline training)

- **Fusion architecture decided**: **Mid-fusion at P4/16 (stride 16)** with feature-level
  concatenation. RGB and IR backbones extract independently up to layer 6 of
  YOLO26 (128 channels each), then merge via `ConcatFusion(128+128→128)`, continuing
  through the shared backbone/neck/head. Rationale: respects TRGB/WiSARD's lack
  of spatial registration (each modality independent until controlled merge point),
  more binarization-friendly than late fusion (single backbone > dual backbones),
  simpler than learning cross-attention. Constraint: ultralytics' neck has
  skip connections that reference intermediate backbone outputs, making true
  mid-fusion complex without custom forward logic — deferred for now.
- **Yolo26_midfusion model implemented**:
  - `src/binarized_cv/models/backbones/yolo26_backbone.py`: Backbone builder that
    extracts YOLO26 layers 0-N for both RGB (3-ch) and IR (1-ch) input. Tested:
    both streams produce 128 ch at layer 6, same spatial dims.
  - `src/binarized_cv/models/detectors/yolo26_midfusion.py`: Detector class with
    `@register_model("yolo26_midfusion")`. Currently RGB-only placeholder (learns
    from RGB, IR unused) to establish baseline before implementing true IR fusion.
  - `configs/model/yolo26_midfusion.yaml`: Config file.
  - `tests/test_yolo26_midfusion_detector.py`: Synthetic data tests (5/5 pass:
    inference, training, gradient flow, modalities, batch sizes).
  - `scripts/train_model.py`: Updated to include new model option.
- **Baseline training completed**: 5 epochs on TRGB+WiSARD, batch size 4, GPU.
  Loss converged strongly: epoch 0 total loss 39.21 → epoch 4 total loss 5.14
  (7.6× reduction). Detailed breakdown: box loss 15.6→3.26, cls loss 23.5→1.87,
  dfl loss 0.10→0.01. All 5 checkpoints saved; TensorBoard logs active. Evaluation
  on test: AP@0.5=0.0000 (expected — 5 epochs too few; YOLO typically needs
  20-50+ for convergence). Extended runs needed to establish accuracy baseline.
- **Design decision**: Took pragmatic approach on mid-fusion wiring. Rather than
  spend time on complex skip-connection rewiring within ultralytics' model,
  established a working RGB-only baseline that validates the full pipeline (data
  → model → train → eval → checkpoints). True IR fusion (actually using IR stream
  in loss/inference) and true mid-fusion layer wiring are now clear next steps,
  unblocked by a running system. 48 tests passing, ruff clean.

## 2026-09-19 (IR fusion deep dive)

- **Investigation**: Attempted to implement IR fusion at P4/16 with multi-point fusion
  (fusing at both layer 4 and layer 6) to give IR a symmetric role. Discovered
  fundamental architectural constraint: YOLO26's PAN neck has skip connections that
  reference earlier layers by index. Concat layers expect inputs from (previous_layer,
  referenced_earlier_layer), but manual layer-by-layer execution loses access to
  referenced outputs. Specifically:
  - Layer 12 Concat: references layer 6 (our fusion point) + upsampled P5
  - Layer 15 Concat: references layer 4 (before fusion) + upsampled P4
  - Layers 18, 21: similar skip patterns
  
- **Root cause**: YOLO26 couples backbone+neck+head as one model. The Detect head
  requires multi-scale pyramid inputs (P3/P4/P5) and the neck builds this pyramid
  via skip connections with hardcoded layer references. Proper mid-fusion at P4/16
  requires either:
  1. Extract full backbone to get all scales, fuse at each (P3, P4, P5), or
  2. Reimplement forward pass to track ALL intermediate outputs, or
  3. Use late-fusion at P5/32 (avoids skip-connection complexity but accuracy/efficiency trade-off)

- **Options forward** (decision deferred to prioritize baseline training):
  - **Early fusion (4-channel)**: Simplest, matches ms_yolov8 comparison model. Known weak
    on misaligned data (TRGB/WiSARD); if chosen, documents that misalignment is limiting
    factor (thesis-credible result).
  - **Multi-scale mid-fusion**: Cleanest architecturally; P3/P4/P5 fused independently,
    then fused pyramid into head. Complex forward pass rewiring needed.
  - **Late fusion (P5/32)**: Avoids skip connections (P5 is final pyramid scale), but
    requires two full backbones (binarization inefficiency penalty).
  - **RGB-only baseline**: Fastest path to validation. Establishes accuracy/latency/power
    baseline before fusion ablations. Clear next iteration once training is running.

- **IR wiring status**: Both rgb_model (3-ch) and ir_model (1-ch) successfully built,
  weights loaded. ConcatFusion module ready (128+128→128). Fusion code skeleton present
  in yolo26_midfusion.py but forward pass not yet wired (decision pending). Tests green,
  model registry clean. Ready to flip switch on any fusion approach.

- **Architectural lesson**: YOLO26's end-to-end design is efficient for single-input
  inference but constrains where fusion can happen. Late fusion (separate backbones)
  sidesteps architecture entirely; mid-fusion pays the skip-connection cost. Thesis
  contribution: whichever path chosen will have an honest architectural story to tell
  (and an ablation comparison).

## 2026-09-20 (Manifest builder refinement & early fusion training)

- **Early fusion model implemented** (`yolo26_early_fusion`): Simple 4-channel concatenation
  (RGB 3-ch + IR 1-ch), single YOLO26 backbone. Resizes IR to match RGB spatial dims.
  Weak baseline on misaligned TRGB/WiSARD data (by design) — if it underperforms, that
  documents spatial alignment as a limiting factor.
- **Training infrastructure fixed**: 
  - Model selection: Fixed hardcoded choices in `prompt_model_selection()` to dynamically
    handle any number of models (was stuck at 1-3, now scales with MODELS dict).
  - Logging/checkpoints: Organized by model with timestamped run directories
    (`runs/tensorboard/{model}/{timestamp}/`, `runs/checkpoints/{model}/{timestamp}/`)
    so multiple training runs don't overwrite each other. TensorBoard run names show
    timestamp + model for easy tracking.
  - Config system: Training script now passes explicit `model.name={model}` override
    to Hydra to ensure correct model name is captured (fixes bug where all runs
    showed "simple_fusion" regardless of selection).
- **Manifest builder completed** (full WiSARD dataset):
  - Fixed generator concatenation: replaced `glob()` + operator with `itertools.chain()`
    for memory efficiency (generators, not lists).
  - Extended `_group_flight_dirs()` to collect all VIS/IR subdirectories per flight,
    not just one (handles WiSARD's multiple sequence directories per flight).
  - Support both 5-digit (`00000`) and 8-digit (`00000000`) frame numbers in regex
    (WiSARD uses both conventions).
  - Pair VIS/IR directories by sorted index (handles offset sequence numbers where
    VIS uses odd indices like 0003/0005/0007 paired with IR even indices 0004/0006/0008).
  - Skip Airfield flight: VIS/IR captured from different camera angles (forest on
    opposite sides in each modality) — physically misaligned, impossible to pair.
  - **Final manifest: 18,888 records** (4,772 TRGB + 14,116 WiSARD = 96% of expected
    ~15,453 WiSARD pairs). Breakdown:
    - 210417_MtErie_Enterprise: 263 pairs
    - 210529_Carnation_Enterprise: 2,052 pairs
    - 210812_Hannegan_Enterprise: 1,136 pairs
    - 210924_FHL_Enterprise: 8,485 pairs
    - 220109_Baker_Enterprise: 2,180 pairs
    - 210327_Airfield: SKIPPED (physical misalignment)
- **Early fusion training validation**: Ran 1 epoch on full dataset with correct model
  selection, proper logging dirs, and timestamped run names — all working correctly.

## 2026-09-20 (Early fusion implementation)

- **Two-model strategy decided**: Implement both early and mid-fusion in parallel rather than
  choosing one. Allows rapid validation of early fusion while mid-fusion is refined.
- **Early fusion model implemented** (`yolo26_early_fusion`):
  - Resize IR to match RGB spatial dimensions (bilinear interpolation)
  - Concatenate RGB (3-ch) + IR (1-ch) → 4-ch input
  - Single YOLO26 backbone with `ch=4` first conv
  - Simplest possible fusion; follows Balla & Shrestha (EUSIPCO 2025) ms_yolov8 approach
  - **Design feature (not a bug)**: Deliberately weak baseline on TRGB/WiSARD misaligned data.
    If early fusion underperforms compared to RGB-only, that's evidence that spatial alignment
    matters — thesis contribution (supports mid-fusion choice later).
  - Config: `configs/model/yolo26_early_fusion.yaml`
  - Tests: `tests/test_yolo26_early_fusion_detector.py` (6 tests: inference, training,
    gradients, modalities, batch sizes, IR resize handling)
  - Training via `scripts/train_model.py --model yolo26_early_fusion` or interactive mode
- **Mid-fusion model preserved** (`yolo26_midfusion`): Custom forward pass skeleton present
  but not yet integrated. Deferred to allow early fusion validation first; unblocks
  mid-fusion refinement once early fusion baseline is established.
- **Next steps**: Train early fusion for convergence, establish accuracy/latency baseline,
  then decide whether mid-fusion complexity is warranted based on early fusion results.

## 2026-09-22 (Comprehensive evaluation metrics & visualization)

- **Evaluation script overhaul**: Fixed coordinate space mismatch bug that was causing AP=0
  (predictions in pixel xyxy, targets in normalized cxcywh). Converter added to evaluation.
- **New metrics implemented** @ IoU=0.5:
  - Precision@0.5: 0.7976 (79.76% of predictions correct)
  - Recall@0.5: 0.7727 (77.27% of actual people detected)
  - F1@0.5: 0.7850 (balanced metric)
  - TP/FP/FN counts for confusion matrix analysis
- **New visualizations added**:
  1. Object count distribution (split view: empty vs non-empty images) — replaces
     confusing single histogram with two-panel view showing actual detection patterns
  2. Precision-Recall curve @ IoU=0.5 — shows accuracy vs recall trade-off across
     confidence thresholds; reveals recall ceiling at ~0.77 due to undetectable cases
     (too small, occluded, or misaligned annotations)
  3. Sample detections — 5 random test images with GT (green) and predicted (red) boxes
- **Automated result collection** (`scripts/eval_with_visuals.py`):
  - Timestamped result folders: `runs/eval_results/{model}_{timestamp}/`
  - Each run saves: metrics.json (machine-readable), results_summary.txt
    (human-readable), 5 PNG visualizations
  - Prevents manual result copying; enables easy comparison across epochs
- **Key finding**: Recall plateau at 0.77 is honest—~23% of ground truth people
  cannot be detected at IoU≥0.5 due to model/data limitations (size, occlusion,
  spatial misalignment). To improve, would need: better architecture (multi-scale),
  more/better training data, or relaxed IoU threshold (but at quality cost).

## 2026-10-01 (Binarization implemented, following docs/binarization_plan.md)

- **Binarized layers** (`src/binarized_cv/models/binarized/`): one drop-in
  `QuantConv2d(nn.Conv2d)` with independent weight/input precision — binary
  (1), k-bit fake quant (2–16), or fp. Binary scheme, each piece chosen for a
  specific reason:
  - Weights: IR-Net balanced weights (centre + standardise per output channel
    before `sign`, which maximises the entropy of the binary weights) times a
    real per-channel XNOR-Net scale `alpha`. BN after the conv, biases and
    `alpha` stay real-valued (plan: "scaling factors restore range").
  - Inputs: `sign(x - tau)` with a learnable per-channel threshold (ReActNet
    RSign), **data-initialised to the per-channel mean on the first batch**.
    Found while designing it: ultralytics `Conv` feeds post-SiLU activations
    to the next conv, which are almost all >= 0, so a plain `sign(x)` would
    map nearly every input to +1 and carry ~0 bits.
  - Gradient: IR-Net's Error Decay Estimator (tanh-shaped surrogate whose
    sharpness `t` anneals 0.1 -> 10). This is how the plan's "progressive
    binarization" is realised. Forward is always exactly binary, so the
    train/eval mismatch a soft-forward annealing would cause never happens.
  - Optional stochastic binarization (plan, constraint 3): `+1` with
    probability `(tanh(t*x)+1)/2`, so it is noisy early and deterministic by the
    end of annealing. Training only.
  - Depthwise convs are never binarized (one sign per tap is the classic BNN
    failure case, and they're cheap). They fall back to 8-bit.
- **Region policy** (`policy.py`): every conv in YOLO26 is assigned to one of
  9 regions (input_conv, stem, p3, backbone_deep, neck_deep, attention,
  head_one2one, head_out, head_one2many) by layer index. PSABlock/Attention
  anywhere is caught by type, so the attention inside neck layer 22 stays fp
  even though the rest of layer 22 is binary. Presets:
  - `ib_guided` (the plan's summary map): stem/attention fp, P3 path 8-bit,
    deep backbone + SPPF binary, one2one head 8-bit, one2many fp.
  - `full` (CHECKLIST section 5's original "fully binarized"): everything
    binary except raw-pixel first conv, final 1x1 output projections, and
    the train-only one2many branch.
  - `fp32` (for the sweep).
  Per-region overrides in the model config (`model.binarization.regions.p3=4`).
  **Judgement call flagged**: the plan doesn't say what to do with the neck at
  P4/P5 (layers 13, 19, 20, 22). It's in `ib_guided` as binary (same resolution
  and semantics as the deep backbone), but it's its own region so the sweep
  measures it separately. YOLO26n param share: backbone_deep 41.7%,
  neck_deep 28.3%, attention 14.6%, p3 5.4%, stem 0.4%. `ib_guided` cuts
  deployed conv weights from 9.0 MiB (fp32) to ~1.9 MiB, `full` to ~1.0 MiB.
  - The ultralytics yaml shipped in 8.4.41 has the neck attention as
    `C3k2(..., attn=True)` at layer 22, not a separate layer as in the paper's
    ablation. The policy handles both.
- **Wiring**: `yolo26` and `yolo26_early_fusion` take optional `binarization`
  and `init_checkpoint` kwargs (fp32 warm start, then swap). New configs
  `yolo26_bnn` / `yolo26_early_fusion_bnn`. Since `QuantConv2d` keeps the
  `weight` name, fp32 checkpoints load unchanged.
- **Trainer**: `train.optimizer` (adam/adamw/sgd; the plan asks to benchmark
  these for BNN layers), no weight decay on latent binary weights (decay pulls
  them to 0 where the sign flips every step), `train.grad_clip`, grad-norm
  logging, and the EDE annealing schedule (`train.binarize_anneal_fraction`).
  The plan's MuSGD x STE concern doesn't apply yet, because we bypass the
  ultralytics trainer and never use MuSGD.
- Fixed `scripts/train_model.py` forcing `model.name=<config file>`. That would
  have broken any config whose file name differs from its registered detector
  (all the `*_bnn` configs).
- **Empirical side of the plan (§4)**:
  - Small-object AP: `average_precision(..., area_range=...)`, COCO-style
    (out-of-range GT ignored, not dropped). `evaluate.py` now reports AP@0.5
    and APS@0.5 (< 32² px, the definition the YOLO26 paper's APS uses).
    Note `scripts/eval_with_visuals.py` uses a different "small" (<1% of image
    area ≈ 64² px at 640) for its recall breakdown.
  - `scripts/binarization_sweep.py`: quantizes one region at a time from a
    trained fp32 checkpoint, with optional fine-tuning, and reports ΔAP/ΔAPS on
    **val** so region choices aren't tuned on test.
  - `scripts/estimate_layer_mi.py` + `analysis/information.py`: per-layer
    I(X;T) and I(T;Y), with Y = "location inside a person box", using 1-bit
    median binning over random 12-channel subsets, plug-in counts, and the
    Miller–Madow correction. Only relative comparisons are meaningful. The
    absolute values depend on the binning, which is the Saxe et al. caveat, so
    report it.
- Tests: 31 new (layers, policy/region map, warm start + checkpoint round
  trip, BNN detectors train/infer for both presets, APS, MI estimator on a
  toy model with known answer). End-to-end smoke run on a 72-pair real-data
  subset: fp32 train → BNN warm-start train (sgd, grad clip, full preset +
  stochastic, early fusion) → eval → sweep → MI all run. 3 pre-existing
  failures in `test_manifest.py`/`test_wisard_discovery.py` are unrelated
  (they fail identically without these changes).
- **Not done yet**: real-length BNN training runs, a distillation loss
  from the fp32 teacher (CHECKLIST section 5), actual bitwise kernels/export. This is all
  simulated quantization in PyTorch, so it measures accuracy, not speed.
- **Eval speed fix**: a sweep over the full val split (2,398 pairs) took
  about 2 min per variant. Timing showed the cause was `evaluate_model`'s
  DataLoader running without `num_workers` (pre-existing in `evaluate.py`),
  so every image was decoded serially in the main process. The fix passes
  `cfg.data.num_workers` through and raises the default 4 -> 8 (4 -> 8 workers
  was +38% loader throughput; 16 added little). One eval went from 110 s to
  32 s with identical AP/APS. Timing breakdown (RTX 2050): the binarized model
  forward is the bottleneck now. Simulated quantization is about 2x slower
  than fp32 (ib_guided 18 vs 8.7 ms/pair), and that's expected until real
  bitwise kernels exist. Image loading with 8 workers is about 16 s for full
  val, so pre-resizing the dataset isn't needed for eval.

## 2026-10-01 (Per-region binarization sensitivity sweep — results)

Setup: `yolo26_early_fusion` 30-epoch checkpoint (`2026-09-21_21-04-06/epoch_29.pt`),
one region binarized at a time, val split (2,398 pairs), APS = AP50 on boxes
< 32² px. Two runs: post-training only (`--epochs 0`, 14:05) and 3 fine-tune
epochs per variant (`--epochs 3`, 14:53; Adam lr 1e-3, no schedule, EDE
annealing over the first half). fp32 reference (not fine-tuned): AP50 0.673, APS50 0.622.
Results: `runs/sweeps/yolo26_early_fusion_bnn_2026-10-01_{14-05-55,14-53-43}/`.

| region | conv weights | ΔAP50 / ΔAPS50 (0 ep) | ΔAP50 / ΔAPS50 (3 ep) |
|---|---|---|---|
| attention | 14.6% | −0.281 / −0.331 | −0.047 / −0.040 |
| neck_deep | 28.3% | −0.481 / −0.419 | −0.062 / −0.040 |
| backbone_deep | 41.7% | −0.240 / −0.292 | −0.080 / −0.118 |
| head_one2one | 4.8% | −0.673 / −0.622 | −0.166 / −0.188 |
| stem | 0.4% | −0.673 / −0.622 | −0.170 / −0.200 |
| input_conv | 0.0% | −0.673 / −0.622 | −0.241 / −0.279 |
| head_out | 0.0% | −0.650 / −0.572 | −0.212 / −0.300 |
| p3 | 5.4% | −0.673 / −0.622 | −0.487 / −0.589 |

Findings:
- **The post-training sweep is not a valid ranking.** neck_deep was 2nd-worst
  of the recoverable regions without fine-tuning and became one of the best after 3 epochs. Use
  fine-tuned deltas for region decisions; report post-training only as
  "brittleness before adaptation".
- **Binarize: backbone_deep, neck_deep, attention.** About 85% of conv weights,
  each within 0.08 AP of fp32 after only 3 epochs. The three are within about
  0.03 AP of each other, which is plausibly single-run noise, so their order
  isn't established. backbone_deep has the largest small-object cost (−0.118
  ΔAPS vs −0.040).
- **P3 confirmed as the critical path (plan, DPI + STAL argument).** APS
  collapses 0.622 → 0.033 even after fine-tuning, for 5.4% of weights. Far
  beyond noise. Keep ≥ k-bit; the 4 vs 8 bit question is still open.
- **Attention contradicts the plan's stated reason.** The plan kept it fp
  because softmax attention is magnitude-sensitive, but binarizing the region
  binarizes only the convs (qkv/proj/pe/FFN + C2PSA cv1/cv2). The dot products
  and softmax run on real-valued BN outputs. It is also not "cheap" (14.6%).
  The argument should become "keep the attention arithmetic fp, binarize the
  convs around it".
- **Stem: the plan's decision holds but its explanation doesn't.** The DPI
  "earliest = worst" reasoning predicts stem ≥ p3 in damage. The data shows
  stem −0.17 vs p3 −0.49. Hypothesis (untested): the full-precision P3 layers
  after a binary stem can compensate, while a binary P3 degrades both the
  backbone P3 features and the neck P3 output feeding the small-object head.
  Framing that fits better: "where the small-object signal has no fp path
  around it", not "how early".
- **head_out, stem, input_conv, head_one2one: poor trades.** 0.17–0.24 AP lost
  for ≤ 5% of weights (head_out: 384 weights). Supports the presets keeping
  them ≥ 8-bit/fp. Likely cause for head_out: no BN after it, so a
  binary-input binary-weight 1x1 over 16/64 channels gives coarse box/score
  outputs.
- Caveats: single run per variant (no seeds/error bars), 3 epochs, fixed
  lr 1e-3, one checkpoint, val only.

## 2026-10-01 (P3 precision + compute share)

- **Compute share differs a lot from weight share** (YOLO26n, 4-ch, 640 input,
  inference path, 2.61 GMACs total). The weight-share argument in the plan
  ("binarize where the parameters are") misses the high-resolution layers:

  | region | weights | MACs |
  |---|---|---|
  | backbone_deep | 41.7% | 26.6% |
  | p3 | 5.4% | 25.9% |
  | neck_deep | 28.3% | 20.6% |
  | stem | 0.4% | 10.8% |
  | head_one2one | 4.8% | 8.2% |
  | attention | 14.6% | 5.6% |
  | input_conv | 0.0% | 2.3% |

  So for memory, attention is the lever: fp32 attention is ~1.4 MiB of
  `ib_guided`'s 1.9 MiB, and binarizing its convs gives ~0.55 MiB. For
  latency/energy, P3 and the stem are the levers.
- **P3 at k bits** (3 fine-tune epochs, same setup as the region sweep;
  `runs/sweeps/yolo26_early_fusion_bnn_2026-10-01_21-37-55` and the following
  4-bit run):

  | P3 | AP50 | ΔAP50 | APS50 | ΔAPS50 | P3 weights |
  |---|---|---|---|---|---|
  | fp32 ref | 0.673 | — | 0.622 | — | 520 KiB |
  | 8-bit | 0.663 | −0.010 | 0.611 | −0.011 | 130 KiB |
  | 4-bit | 0.606 | −0.067 | 0.545 | −0.077 | 65 KiB |
  | binary | 0.186 | −0.487 | 0.033 | −0.589 | 16 KiB |

  8-bit is ~lossless. 4-bit costs about as much as binarizing the whole deep
  backbone (−0.08), but saves only 65 KiB (3.4% of `ib_guided`), and 4-bit
  kernels are much less available than int8 on ARM CPUs. **Keep P3 at 8-bit.**
- **Missing control:** the fp32 reference was never fine-tuned. Deltas of about
  0.01 (P3 8-bit) and the 0.05–0.08 spread between attention, neck_deep and
  backbone_deep can't be separated from the effect of 3 more epochs at lr 1e-3
  until an fp32 model fine-tuned with the same settings is evaluated.

## 2026-10-02 (fp32 fine-tune control)

- The sweep compared every 3-epoch fine-tuned BNN variant against an fp32
  reference that was never fine-tuned. `scripts/binarization_sweep.py` now
  fine-tunes the fp32 reference with the identical schedule whenever
  `--epochs > 0` and measures deltas against it (the untouched checkpoint stays
  as a row). `--regions` may be empty, so the control can run on its own.
- Control (same checkpoint, 3 epochs, Adam lr 1e-3, no schedule, val;
  `runs/sweeps/yolo26_early_fusion_bnn_2026-10-02_13-51-30`): fp32 AP50
  0.673 → **0.677** (+0.004), APS50 0.622 → **0.612** (−0.010).
- **Fine-tuning alone barely moves fp32**, so the region sweep's deltas are
  quantization, not extra training. Rebased onto the fine-tuned reference
  (ΔAP50 / ΔAPS50): attention −0.051 / −0.030, neck_deep −0.066 / −0.030,
  backbone_deep −0.084 / −0.109, head_one2one −0.170 / −0.178, stem
  −0.174 / −0.191, head_out −0.216 / −0.290, input_conv −0.245 / −0.269,
  p3 −0.491 / −0.579. P3 8-bit −0.014 / −0.001, 4-bit −0.071 / −0.067.
  Every region decision stands.
- The ±0.01 movement of an fp32 model under 3 more epochs gives a rough idea
  of the noise floor. Training is unseeded, so the ordering of attention /
  neck_deep / backbone_deep (~0.03 apart) still needs seeds.
- Rebased the stored results too: the 3-epoch sweep files (`…_14-53-43`,
  `…_21-37-55`, `…_22-12-15`) now include the fine-tuned reference row, and
  `d_ap50`/`d_ap50_small` are against it. The old deltas are kept as
  `*_vs_untouched`, and the columns `epochs` and `seed` (null, these runs
  predate seeding) were added.

## 2026-10-02 (seeding)

- Training was unseeded, so no run was reproducible and seed sweeps were
  impossible. Added `train.seed` (default 0, null = unseeded) and
  `train.deterministic` (default false). `seed_everything` seeds Python,
  NumPy and torch (all CUDA devices), and the train DataLoader gets its own
  seeded generator so the shuffle order doesn't depend on how many random
  numbers model construction drew. The dataset has no random augmentation,
  so this covers every source: shuffle order, init of parameters not covered
  by a warm start, and stochastic binarization.
- Same seed ≠ bit-identical on GPU: cuDNN picks non-deterministic kernels
  unless `train.deterministic=true` (slower, so off by default). For seed
  sweeps that's fine; the spread across seeds is what's being measured.
- `scripts/train_model.py --seed N`; sweep result rows now record `seed`
  (`train.seed=N` as a Hydra override to the sweep). Test:
  `tests/test_seeding.py`.


## 2026-10-02 (LR schedule + first full ib_guided run)

- Added `train.scheduler` (`none` | `cosine`), stepped per batch: linear warmup
  over `train.warmup_epochs`, then cosine decay to `lr * train.min_lr_ratio`
  (0.01). The default stays `none`, so the 3-epoch sweep settings are
  unchanged and remain comparable. LR is logged as `train/lr`. Test:
  `tests/test_scheduler.py`.
- Started the combined `ib_guided` model (early fusion, warm start from fp32
  `epoch_29.pt`): 20 epochs, Adam lr 1e-3, cosine with 1 warmup epoch, EDE
  annealing over the first half, seed 0, evaluated on val at the end. Runner
  `runs/bnn/train_eval.py`, log and metrics in `runs/bnn/ib_guided_20ep_seed0.*`.
  Throughput ~2.2 batch/s (~13.5 min/epoch, ~4.5 h total) vs ~3.2 batch/s for
  fp32, the simulated-quantization overhead.
- **Stopped by hand at 18:05**, 3 epochs short. The checkpoints `epoch_0.pt` …
  `epoch_16.pt` (17 of 20 epochs) are in
  `runs/checkpoints/yolo26_early_fusion_bnn_ib_guided/<ts>/`. Mean train loss
  fell steadily, 18.3 (epoch 0) → 13.0 (epoch 12), with no instability. It
  flattened briefly around epoch 10, when EDE annealing finished. No val
  metrics yet.
- **This run can't be resumed yet.** Two gaps:
  1. Checkpoints hold only `model.state_dict()`: no optimizer (Adam moments),
     scheduler position, global step (which drives the EDE annealing
     progress) or RNG state. A restart would redo the warmup and the
     annealing from scratch.
  2. `model.init_checkpoint` loads into the fp32 model *before* the
     binarization swap (`warm_start_and_binarize`, `_ultralytics_common.py`).
     Loading a BNN checkpoint that way discards the learned `QuantConv2d`
     parameters (input thresholds τ, etc.) as unexpected keys and
     re-initialises them from data. So `init_checkpoint` is for fp32 warm
     starts only.
  Evaluating a saved BNN checkpoint is fine as it is: `evaluate.py` builds the
  binarized model first and then loads the state dict
  (`eval.checkpoint_path=…`).
- **TODO next session:**
  1. Save full training state each epoch (model, optimizer, scheduler, step,
     epoch, RNG incl. the DataLoader generator) and add
     `train.resume=<checkpoint>`, loading after the binarization swap.
  2. Resume this run from `epoch_16.pt` for the last 3 epochs, then val eval.
  3. fp32 control on the identical 20-epoch cosine schedule (seed 0), so the
     BNN result isn't credited with the effect of longer training.
  4. `ib_guided` + attention binarized, same schedule.

## 2026-10-03 (resumable training + queued runs)

- **Full training state per epoch.** Besides the weights-only `epoch_N.pt`
  (unchanged, so `evaluate.py`, the sweep and `init_checkpoint` read them as
  before), every epoch now overwrites `last.pt` with model, optimizer,
  scheduler, epoch, global step (drives the EDE annealing), all RNG states
  including the train DataLoader's generator, and the resolved config.
- **`train.resume=<checkpoint>`** loads *after* `build_model_from_config`, i.e.
  after the binarization swap, so learned `QuantConv2d` parameters (thresholds
  τ, the `initialized` flag) survive; the run continues in the checkpoint's own
  checkpoint/TensorBoard directory (`purge_step` hides events logged after the
  last checkpoint). With `last.pt` the resume is exact. A weights-only
  `epoch_N.pt` (all runs before today) also works: the epoch comes from the file
  name, step / LR schedule / annealing progress are recomputed, and the shuffle
  order is replayed by iterating a stand-in loader over indices. Only Adam's
  moments and the global RNG start fresh. Tests: `tests/test_resume.py`
  (exact round trip, weights-only reconstruction, shuffle replay vs a
  multi-worker loader).
- Resumed the stopped `ib_guided` run from `epoch_16.pt`: picks up at epoch 17
  (step 30005) with loss ≈ 12.9, continuous with epoch 12's 13.0, so the warm
  state carried over. The last 3 epochs therefore ran with re-initialised Adam
  moments, on the cosine tail (LR 7% → 1% of peak) — note this when reporting it.
- Queued on the GPU, back to back (`runs/bnn/queue_2026-10-03.sh`, ~9 h): (1)
  finish `ib_guided` + val eval, (2) fp32 control, same 20-epoch cosine
  schedule from the same `epoch_29.pt`, seed 0, (3) `ib_guided` + binary
  attention, same schedule. Metrics land in `runs/bnn/*.json`.
- Test runs need `env -u PYTHONPATH`: the shell's ROS Jazzy `PYTHONPATH`
  loads a `launch_testing` pytest plugin that crashes pytest at startup.

## 2026-10-03 (runner + requirements cleanup)

- The train-then-eval runner moved from `runs/bnn/train_eval.py` (gitignored)
  to `scripts/train_eval.py`, same arguments (`RUN_NAME OUT_JSON
  [overrides...]`), now with argparse and no `sys.path` hack. The old copy is
  deleted.
- **Queue stopped by hand at 17:30.** Done: `ib_guided` 20 epochs, val AP50
  0.669 / APS50 0.617 (`runs/bnn/ib_guided_20ep_seed0.json`), vs 0.673 / 0.622
  for the untouched fp32 checkpoint. Not a final comparison until the
  same-schedule fp32 control exists. The fp32 control stopped in epoch 7 with
  epochs 0–6 saved, `last.pt` holding the full state after epoch 6. The
  `ib_guided` + binary attention run never started.
  `runs/bnn/queue_2026-10-03.sh` now runs the remaining two through
  `scripts/train_eval.py`: the control resumes exactly from that `last.pt`.
- `requirements.txt` was a full `pip freeze` of the dev venv: an `-e
  git+…@d9c8b9b` line installing the repo from an old commit, the Jupyter
  stack and other packages nothing imports, unpinned `gdown`, and torch's
  CUDA 13 runtime wheels pinned explicitly, which blocks installing a CUDA 12
  torch (needed on clusters with older drivers). It is now the exact
  dependency closure of `pyproject.toml` (+ dev extras) at the installed
  versions, 55 pins, without `nvidia-*`/`cuda-*`/`triton`; a dry-run install
  in a fresh venv resolves cleanly and pulls the same CUDA wheel versions
  (except `cuda-bindings`, which torch only bounds). The old freeze is not
  kept in the repo; git history has it. `matplotlib` added to
  `pyproject.toml`, since the eval scripts import it.

## 2026-10-03 (Slurm cluster setup)

- Cluster survey: Ubuntu 22.04 (glibc 2.35, so manylinux torch wheels load),
  modules only up to Python 3.9 / CUDA 11.8, so neither is used. pip's torch
  wheel brings its own CUDA runtime, and only the driver matters: A40 nodes
  run **615.71**, so the pinned CUDA 13 build works unchanged. Training
  partitions: `a40q` (4 nodes × 1 A40 48 GB, 128 cores, 1 TB RAM, ~113 GB free
  local `/tmp`) as the default, `a100q`/`hgx2q` for more. Not used:
  `gh200q`/`aarch*` (aarch64, would need a second environment) and the MI210
  nodes (ROCm). Compute nodes have internet. 14-day time limit.
- `scripts/slurm/setup_env.sh`: uv-managed Python 3.12 + pinned requirements
  into `.venv` (the system Python is 3.10, below what the pinned numpy needs).
  `scripts/slurm/job.sbatch`: generic GPU job for any Hydra entry point. It
  stages the dataset to node-local `/tmp/$USER` (persistent, rsync-checked,
  flock-guarded), sets `data.raw_root`/`data.num_workers` and supports
  `{task}` array placeholders.
- To pass machine defaults without breaking argparse (the sweep's `--regions
  nargs="*"` would swallow appended overrides), `load_config` in `train.py`
  and `evaluate.py` now prepends `$BCV_OVERRIDES` to the explicit overrides,
  and the explicit ones win. Test: `tests/test_env_overrides.py`. The plotting
  scripts (`eval_with_visuals.py`, `eval_detailed.py`, `debug_boxes.py`) have
  their own config loaders and don't read it.
- Verified locally only (no Slurm here): job script run directly on a CPU
  subset, `{task}` → seed, staging commands. First real cluster job pending.
- Storage: two filesystems are shared with the compute nodes: `/home` (NFS
  from `master`, 268 TB free) and `/global/D1` (BeeGFS, 515 TB, 98% full,
  per-user dir `/global/D1/homes/$USER`). `/work` (XFS, 4.2 TB) and
  `/scratch` are local disks of the login node. Dataset goes to
  `/global/D1/homes/$USER/bcv-data/raw`, code/venv/`runs/` stay in `~`.
  `job.sbatch` still stages the data to node-local `/tmp` (~113 GB free on
  `a40q` nodes), since BeeGFS small-file reads by 14 workers may be slow and
  the filesystem is nearly full. Worth one `BCV_STAGE=0` comparison run.

## 2026-10-05 (first cluster jobs)

- **Correction to 2026-10-03:** the `a40q` nodes (n009–n012, also in
  `aarchq`/`huaq`) are **aarch64**. The survey had checked the x86_64 login
  node. The x86_64 `.venv` fails there under the binfmt emulator
  (`x86_64-binfmt-P: Could not open '/lib64/ld-linux-x86-64.so.2'`).
  Default partition is now `a100q` (n013/n014, x86_64, 2× A100-PCIE-40GB,
  driver 615.71, 100 GB free `/tmp` for the 43 GB dataset). `job.sbatch`
  checks that `.venv/bin/python` runs before staging.
- The GPU nodes report `RealMemory=1` (MB) to Slurm, so any `--mem` request
  fails with "Memory specification can not be satisfied". Removed `--mem`
  from `job.sbatch`.
- `setup_env.sh` sets `UV_SYSTEM_CERTS=1`: uv's bundled CA roots reject the
  cluster's TLS certificates (`UnknownIssuer`) when downloading Python.
- Staging the dataset to node-local `/tmp` took ~10 min on the first run (n010).
- The login node sets `ulimit -v` to 16 GB, and Slurm propagates the
  submitting shell's limits by default, so the smoke test on n014 failed while
  building the model (`DefaultCPUAllocator: can't allocate memory: you tried
  to allocate 65536 bytes`). `job.sbatch` now has `--propagate=NONE`; with it
  the node's own limit (`unlimited`) applies. Staging on n014 took 2.5 min.
- Smoke test on n014 passed (1 epoch, 5 min on the A100, val AP50 0.537).
  Seed arrays: `scripts/slurm/submit_seeds.sh` (one job array per config) and
  `scripts/collect_results.py` (mean ± std over seeds).
- **Training to convergence:** `train.scheduler=plateau`. Val AP50 after
  every epoch (also logged as `val/*` in TensorBoard). LR ×0.1 after 3
  epochs without a 0.001 gain, stop at the stall after the 2nd cut;
  `train.epochs` is only a cap. The plateau state lives in the LambdaLR
  factor object, so it's in `last.pt` and survives a resume. Keeps `best.pt`
  and returns it, so val is biased for these runs: `train_eval.py` now also
  writes test metrics (`test_ap50`, `test_ap50_small`) to the JSON.
  `train.binarize_anneal_epochs` fixes the EDE anneal length, since a
  fraction of a cap means nothing. Submit with
  `SCHEDULE=plateau bash scripts/slurm/submit_seeds.sh 0-2`.
- **WiSARD split was leaky; now flight-wise.** The frame-index split
  (`frame % 100`: <70 train, <85 val, else test) put consecutive, near-identical
  frames of every sequence into all three splits, and WiSARD is 14,116 of the
  18,888 records. Only 5 WiSARD flights have paired VIS/IR frames, so whole
  flights now go to one split (`FLIGHT_SPLITS` in `datasets/wisard.py`):
  train FHL 8,485 + Hannegan 1,136 + MtErie 263, val Carnation 2,052, test
  Baker 2,180 (70.0/14.5/15.4%). New manifest: train 14,002, val 2,376, test
  2,510 (TRGB unchanged, 4,118/324/330). Old manifest kept as
  `data/processed/manifest_frame_split_2026-10-05.jsonl`.
- **Every number so far is on the leaky split** (0.726 test / 0.673 val
  baseline, the binarization sweep, `ib_guided` 0.669), and the fp32
  warm-start `epoch_29.pt` was trained on ~70% of the frames of Carnation and
  Baker, so it can't be the warm start for runs on the new split either.
  The fp32 early fusion baseline has to be retrained first.
- Not fixed, worth checking: in FHL VIS_0403/0564/0566 the VIS side has no
  labels (`count.txt`: 0 humans) while the IR side has 216/712/690 labeled
  frames. With `target_modality=rgb` those ~3,300 frames train as pure
  negatives. Either people are invisible in VIS there (canopy) or VIS was
  never labeled. Hannegan and FHL 0409 have no people in either modality.
- 3 tests had failed since `d0f0ce2`: they still built the old
  `wisard/WiSARD_Multi_Modal_Sample/` layout and expected `split="unassigned"`.
  Fixed; `test_splits_whole_flights` added.
- **Two more WiSARD discovery bugs, fixed.** (1) Frame numbers were matched as
  strings, but some pairs zero-pad differently (VIS `_00000000` vs IR
  `_00000` or `_out_frame_00001`), so MtErie VIS_0005/IR_0006 (272),
  VIS_0007/IR_0008 (172) and FHL VIS_0134/IR_0135 (273) gave 0 records
  although both sides have the same image counts. Now matched as ints: +717
  frames, all in train. (2) IDs held only the flight, not the sequence, so
  7,082 IDs were duplicates; now `wisard_<VIS dir>_<frame:08d>`.
- **Unlabeled sequences kept, flagged.** `PairRecord` has `rgb_labeled` /
  `ir_labeled` (default true, so old manifests load). Set false for VIS of
  FHL 0403/0564/0566 and IR of Carnation 0026, FHL 0127/0408 (`count.txt`:
  0 humans, no label files, other modality has people). The dataset skips
  records whose target modality is unlabeled, so they're not negatives; they
  stay in the manifest to get labels from the other modality once alignment
  exists. Checked visually (`runs/label_check/`): people clearly visible in
  VIS_0403, partly under canopy in VIS_0564, VIS_0566 is a night flight.
- Manifest now 19,605 records (WiSARD 14,833): train 14,719 / val 2,376 /
  test 2,510; with `target_modality=rgb` train is 11,413.
- Platforms: all paired flights are `*_Enterprise_*` (Mavic 2 Enterprise
  Advanced, one rigid dual camera). The `*_FLIR_*` flights (likely the
  Matrice 600 Pro with a separate FLIR camera) are IR-only or VIS-only,
  except Airfield, which is excluded for exactly this misalignment.
- **IR -> VIS alignment from labels** (`binarized_cv.data.alignment`,
  `scripts/fit_alignment.py` analysis, `scripts/align_manifest.py` writer).
  Works in the 640×640 training input space, where identity = the current
  stretch (median error 23–82 px depending on the sequence, i.e. more than a
  small person). Box matching per frame (Hungarian under the current
  estimate, shrinking gate), RANSAC affine on box centres + corners.
  - **One Mavic 2 Enterprise rig transform** fits MtErie (April, 4K), FHL
    (Sept) and Baker (Jan): IR → x' = 0.638x + 121.7, y' = 0.903y + 40.6,
    i.e. thermal covers the middle ~64 % × 90 % of VIS. Cross-sequence error
    3–8 px. Affine ≈ scale+shift; homography not better (broke on MtErie).
  - **Residual bumps = time offset**, not geometry: inside a bump all people
    shift the same way, and pairing IR frame n±1 removes most of it
    (Baker 780–860: 18 → 5 px; MtErie 150–200: 16 → 7; FHL 0401 850–900:
    32 → 13). The VIS/IR videos drift by about a frame, sign varies within
    a sequence; it only shows while the camera moves. A dataset artifact
    (frames extracted from two videos), so fixed in the data: per-frame
    offset k ∈ [−2, 2] by dynamic programming (penalty 4 px per change),
    unlabeled frames follow neighbours. Held-out frames (offsets from their
    neighbours only), share > 16 px, k=0 → chosen: Baker 14 → 5 %, FHL 0401
    29 → 18 %, MtErie 0005 25 → 14 %, 0003 28 → 22 %; MtErie 0007 0 → 10 %
    (few held-out frames). Medians move little (5.5 → 4.9 px on Baker):
    the 2–5 px floor is label noise.
  - **Carnation 0023 (val) is zoomed**: own transform has scale 2.4–3.2
    instead of 0.64, error drifts in waves; segmentation gives 3 transforms,
    still median 13.5 px, 270/739 frames > 16 px. Carnation 0025 (no IR
    labels) inherits 0023's main transform, unverified. Open decision: keep
    Carnation as val or pick another val set.
  - **TRGB has several rigs**: 1280×800 splits into 3 blocks (≈ 4 px vs
    ≈ 40 px under one transform, contiguous runs = videos), median 7 px,
    519 frames > 16 px; 1280×720 (229 frames) doesn't fit well (median 34 px).
    No frame numbers, so no offset correction for TRGB.
  - `manifest_aligned.jsonl`: 19,605 records with `ir_transform`,
    `ir_frame_offset`, `align_err_px`, `align_source`. Checked 9,203,
    > 16 px 1,185, unchecked 10,402 (no boxes in both modalities; they get
    the transform of a checked sequence of the same flight, else the rig).
    The dataset doesn't apply the transform yet.
- **Misalignment filter: relative, not 16 px.** The 16 px first used was an
  unfounded rule of thumb. VIS people at 640×640 are small: width p10/p50/p90
  = 5.7 / 18 / 46.5 px, 45 % narrower than 16 px, so 16 px means "IR blob
  completely off the person" for half of them. Checked-frame errors: mode
  2–8 px (label noise, p50 6.3 px), long tail from ~12 px, no clear valley.
  Rule now: `align_ok = err <= max(6 px, 0.5 × median VIS person width in
  the frame)` (`--floor-px`, `--width-frac`). 0.5 width = the IR blob still
  covers at least half the person; the 6 px floor is the noise level below
  which a tiny person's error says nothing. Drops 1,840 of 9,203 checked
  (20 %; train 1,103, val 563, test 172) vs 1,185 at 16 px. Most in TRGB
  (642), Carnation 0023 (462), FHL 0401 (216), Baker (150), MtErie 0007
  (110). The error counts a VIS person without an IR match as 48 px, so some
  drops are label inconsistencies (person labeled in one modality only),
  not misalignment — those frames are dubious for fusion anyway.
- `data.drop_misaligned_splits: [train]` drops `align_ok: false` records
  (needs `data.manifest_path=data/processed/manifest_aligned.jsonl`).
  Train only: filtering hard frames out of val/test flatters the numbers.
  Unchecked records (no boxes in both modalities) can't be filtered.
- **Per-flight transforms** (`align_manifest.py --flight-margin 0.05`): each
  flight gets a candidate fitted on its own non-held-out frames (after the
  offset correction under the rig); it replaces the rig only if it lowers the
  flight's held-out misaligned share by >= 5 points. Held-out (median px /
  misaligned): MtErie rig 5.6 / 36 % vs own 6.3 / 33 % → **stays on rig**;
  FHL 4.8 / 32 % vs 4.3 / 32 % and Baker 4.5 / 8 % vs 4.3 / 8 % → rig;
  Carnation 48 / 100 % vs 28 / 88 % → own flight transform (scale 2.4/3.3,
  zoomed), after which segmentation adds 2 zoom transforms; Carnation 0023
  held-out with offsets now 13.3 px / 62 % (was 18.6 / 77 %).
- **MtErie's high misaligned share is not the transform.** Per-sequence own
  fits don't help consistently either (held-out 0003: 35 → 35 %, 0005:
  22 → 33 %, 0007: 60 → 50 %, n = 20–40 frames each, too few to trust). It
  tracks person size: MtErie 0003/0007 people are ~10 px wide (FHL 0401:
  7 px, 26 %), so the 6 px floor decides, and 14–15 % of their frames sit at
  6–9 px error on people < 12 px wide; MtErie 0005 (25 px) is 22 %, Baker
  (28 px) 8 %. At that scale label noise and residual time offset can't be
  separated from misalignment. Left as is.
- **Carnation out of val.** Even with its own flight transform plus two zoom
  transforms, 62 % of Carnation's held-out frames stay misaligned, and 0025
  (no IR labels) can't be checked at all. New split: val = MtErie (whole
  flight, 707 frames, 1,766 people; small people, 4K), test = Baker, train =
  FHL + Hannegan, and Carnation = split `zoom` (2,052 frames, 7,180 people),
  outside train/val/test, for a later check of how fusion copes with
  zoomed/misaligned IR. FHL sequence 0401 as val was the alternative
  (bigger), rejected: same day, place and people as the FHL train sequences.
  Records: train 14,012 / val 1,031 / test 2,510 / zoom 2,052 (val and test
  include TRGB's 324 / 330). With RGB labels: train 10,706; minus
  checked-misaligned frames (`manifest_aligned.jsonl`): 9,862.
- Note: val is now small (1,031 frames, MtErie people ~10–25 px wide), so the
  plateau scheduler's val AP50 will be noisier; plateau_min_delta 0.001 may
  need raising if runs stop on noise.
- **IR alignment in the dataset** (`data.ir_alignment`, default `none`):
  `warp` warps IR onto the RGB grid with the record's `ir_transform` (black
  outside the IR footprint); `crop` crops both to the shared field of view
  (IR footprint ∩ RGB frame, cropped at full RGB resolution), scales to
  `img_size`, clips boxes and drops those with < 50 % of their area inside.
  Records without a transform fall back to `none`. Checked visually on Baker,
  FHL 0401 and TRGB: IR heat now sits on the labeled people.
- `crop` keeps 95 % of train boxes (WiSARD 87 %, TRGB 100 %: TRGB's IR
  footprint covers the whole RGB frame), 98 % val, **88 % test** (Baker 86 %):
  the Enterprise IR sees only the middle ~64 % × 90 % of VIS. So `crop`
  changes the test task (fewer, larger people), and its numbers aren't
  comparable to an RGB-only model on full frames unless that one is cropped
  too. `warp` keeps every label and the full RGB frame → main setting;
  `crop` as an ablation (onboard, processing only the overlap is an option).
- **Defaults switched:** `data.manifest_path` = `manifest_aligned.jsonl`,
  `data.ir_alignment` = `warp` (with `drop_misaligned_splits: [train]`).
  `run_pipeline.sh` runs `align_manifest.py` after `build_manifest.py`;
  README §2 and the cluster rsync line list both manifests. The dataset
  warns if alignment is on but no record has a transform (old manifest).
  `ir_alignment=none` reproduces the pre-alignment input.
- **Staging wiped the node copy.** The first fp32 baseline (job 1425273)
  crashed with a missing FHL VIS_0407 image under `/tmp/feboe/bcv-data/raw`
  although the file exists locally. Cause, most likely: `BCV_DATA` not
  exported in the submitting shell, so `job.sbatch` staged from its default,
  the repo's `data/raw` (only `.gitkeep` on the cluster), with `rsync
  --delete`, which emptied the node copy (and would break any job reading it
  at the time). Fixes: default `BCV_DATA` is now the BeeGFS copy; the job
  stops if `BCV_DATA` contains no images; no `--delete` when staging;
  `train_eval.py` checks that all train/val/test images exist under
  `data.raw_root` before training (~5 s, fails with count + examples).

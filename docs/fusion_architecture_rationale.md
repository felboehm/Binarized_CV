# Mid-Fusion at P4/16 Architecture Rationale

## Problem: Non-Registered Multispectral Data

**Critical Finding (2026-09-06):** Neither TRGB nor WiSARD is pixel-registered between RGB and IR:
- **Resolution mismatch**: TRGB (RGB 1280×800 vs IR 640×512 ≈ 2×), WiSARD (VIS 3840×2160 vs IR 640×512 ≈ 6×)
- **Annotation mismatch**: Per-modality labels for the same person don't co-register — "aligned" in both papers means *temporally synchronized*, not *spatially co-registered*
- **No calibration data**: Neither dataset provides homography or calibration matrices to warp modalities into pixel alignment

This **rules out naive early fusion** (channel-concatenation without warping) as a viable baseline.

## Architecture Options Considered

### Option 1: Early Fusion (Channel-Concat)
**Approach:** Resize both modalities to same size, stack as 4-channel input, train end-to-end

**Why rejected:**
- Assumes pixel-level spatial alignment our data lacks
- Balla & Shrestha (EUSIPCO 2025, ms_yolov8) assumes this; deliberately weak baseline on TRGB/WiSARD for this reason
- Without warping, you're forcing the model to learn misaligned features—wasteful and suboptimal

### Option 2: Late Fusion (Dual Backbones → Merge at Output)
**Approach:** Separate RGB and IR backbones extract to full depth, fuse features at end, single head

**Advantages:**
- Fully independent feature extraction (respects non-registration)
- Simple to implement

**Why rejected:**
- **Binarization cost**: Requires two full binarized backbones instead of one (2× model size, 2× compute)
- **Efficiency goal violated**: Binarization's main win is a single small backbone; dual backbones negate this
- **Late fusion is coarse**: Loses mid-level detail sharing between modalities

### Option 3: Mid-Fusion at P4/16 (Chosen)
**Approach:** 
- Dual RGB (3-ch) and IR (1-ch) backbones extract independently to layer 6 (P4/16, stride 16)
- Concatenate features (128+128→128 via ConcatFusion)
- Single shared neck (PAN) and detection head

**Advantages:**
- **Respects non-registration**: Each modality processes independently until controlled merge point; no forced pixel-level alignment
- **Binarization-friendly**: Single shared backbone (after fusion) → single set of binarized weights
- **Balanced**: Merge point is semantic enough (receptive field ≈ 16 pixels) to benefit from cross-modality context, but early enough to avoid redundant processing
- **Simple**: Doesn't require learned cross-attention or complex alignment networks (CFT/ICAFusion complexity)
- **Empirically motivated**: P4/16 is a standard intermediate layer in multi-scale detection (FPN/PAN also outputs P3/P4/P5)

## P4/16 as Merge Point: Why Not P3 or P5?

| Layer | Stride | Channels | Rationale |
|-------|--------|----------|-----------|
| **P3/8** | 8 | 64 | Too fine-grained; spatial details still vary wildly across non-aligned modalities; early fusion here is nearly as bad as channel-concat |
| **P4/16** (chosen) | 16 | 128 | Sweet spot: features are semantic enough to benefit from fusion; receptive field large enough to abstract over misalignment; standard FPN intermediate layer |
| **P5/32** | 32 | 256 | Too coarse; misses mid-level complementarity between RGB (better for texture/detail) and IR (better for thermal); late-fusion penalty on model efficiency |

## Implementation Constraint: Ultralytics Skip Connections

**Challenge (discovered 2026-09-19):** YOLO26's PAN neck architecture has skip connections
with hardcoded layer references. Concat layers in the neck look like `[[-1, 4], 1, 'Concat', [1]]`,
meaning "concatenate previous layer output with layer 4's output". This is problematic for
mid-fusion because:

- Layer 12 Concat: references layer 4 (80×80, before our P4/16 fusion point)
- Layer 15 Concat: references layer 6 (40×40, exactly our fusion point)
- Layers 18, 21: more skip connections to intermediate layers

Simple P4/16 injection (extract backbones 0-6, fuse, feed to neck) fails because the
Concat layers can't access the referenced earlier outputs when executed layer-by-layer
without the model's internal tracking mechanism.

**Why this matters:** YOLO26 couples backbone/neck/head as one `DetectionModel`. The forward
pass tracks all intermediate outputs internally and layers can reference them. Manual
layer-by-layer execution loses this tracking.

**Solutions and trade-offs:**

| Approach | Complexity | Skip Connections | Efficiency | Notes |
|----------|-----------|-----------------|-----------|-------|
| **Early fusion (4-ch concat)** | Low | N/A | ✓ Good | Misalignment = weak baseline (expected/documented) |
| **Multi-scale mid-fusion (P3/P4/P5)** | High | ✓ Fused at each scale | ✓ Good | Complex forward rewiring; architecturally clean |
| **Late fusion (P5/32)** | Medium | ✓ Both backbones complete | ✗ 2× backbone cost | Avoids skip connections but binarization penalty |
| **RGB-only (current)** | Very Low | N/A | N/A | Baseline validation; IR fusion deferred |

**Decision framework:** Architecture choice deferred to testing phase. All prerequisites
ready (IR models built, fusion modules ready). Recommendation: prioritize whichever option
unblocks baseline training soonest, then iterate on fusion based on accuracy results.

## Thesis Contribution

This choice supports the binarization narrative:
- **Standard approach in literature**: Mid-to-late fusion is common in multispectral detection (CFT, ICAFusion)
- **Efficient for edge**: Single binarized backbone post-fusion is the minimal viable fusion model
- **Ablation story**: Can compare P4/16 (chosen) vs P5/32 (late) vs full-binarization gains to show which lever matters most

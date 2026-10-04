"""Where to binarize YOLO26 — the region map from `docs/binarization_plan.md`.

Every `nn.Conv2d` in an ultralytics YOLO26 `DetectionModel` is assigned to
exactly one *region*; each region gets one *precision* (`"fp"`, `"binary"`,
or an integer bit width). Presets bundle the plan's recommendations, and
per-region overrides make the plan's one-region-at-a-time sensitivity sweep
(`scripts/binarization_sweep.py`) a pure config change.

Layer indices refer to `ultralytics/cfg/models/26/yolo26.yaml`, whose layer
*structure* is identical across scales n/s/m/l/x (only widths/depths
change), so the map applies to every scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

from binarized_cv.models.binarized.layers import QuantConv2d

# Region -> YOLO26 top-level layer indices. Detect-head and attention
# regions are assigned by module name/type instead (see `assign_region`).
LAYER_REGIONS: dict[str, frozenset[int]] = {
    # First conv sees raw pixels (3/4 input channels, tiny cost). Kept separate
    # from the rest of the stem so "full" binarization can still spare it, as
    # nearly every BNN paper does.
    "input_conv": frozenset({0}),
    # P1/2–P2/4: DPI — Y-relevant info not yet separated from nuisance detail.
    "stem": frozenset({1, 2}),
    # Backbone P3 (3, 4), neck P3 output (16), and the P3->P4 downsample that
    # still reads P3 features (17). Small-object path: STAL/APS evidence.
    "p3": frozenset({3, 4, 16, 17}),
    # Deep backbone P4/P5 C3k2 + downsamples + SPPF: most params/FLOPs.
    "backbone_deep": frozenset({5, 6, 7, 8, 9}),
    # Neck at P4/P5. The plan does not single this out; it is grouped with
    # the deep backbone (same resolution and semantics) but kept separate
    # so the sweep can measure it on its own.
    "neck_deep": frozenset({13, 19, 20, 22}),
    # Layer 10 is C2PSA as a whole; PSABlocks inside other layers (e.g. the
    # neck attention in layer 22) are caught by type in `assign_region`.
    "attention": frozenset({10}),
}

REGIONS: tuple[str, ...] = (
    "input_conv",
    "stem",
    "p3",
    "backbone_deep",
    "neck_deep",
    "attention",
    "head_one2one",  # inference head (NMS-free one2one branch), hidden convs
    "head_out",  # final 1x1 box/class projections of the one2one branch
    "head_one2many",  # train-only branch (topk=10), removed at inference
)

Precision = str | int  # "fp" | "binary" | bit width (2..16)

PRESETS: dict[str, dict[str, Precision]] = {
    "fp32": {r: "fp" for r in REGIONS},
    # The plan's summary map.
    "ib_guided": {
        "input_conv": "fp",
        "stem": "fp",
        "p3": 8,
        "backbone_deep": "binary",
        "neck_deep": "binary",
        "attention": "fp",
        "head_one2one": 8,
        "head_out": 8,
        "head_one2many": "fp",
    },
    # CHECKLIST.md section 5's original "fully binarized" scope, minus the
    # three things no BNN binarizes: raw-pixel first conv, final continuous
    # regression outputs, and a branch that is deleted before deployment.
    # Override `head_one2one: fp` for the "fp head" ablation.
    "full": {
        "input_conv": "fp",
        "stem": "binary",
        "p3": "binary",
        "backbone_deep": "binary",
        "neck_deep": "binary",
        "attention": "binary",
        "head_one2one": "binary",
        "head_out": "fp",
        "head_one2many": "fp",
    },
}


def _attention_types() -> tuple[type[nn.Module], ...]:
    from ultralytics.nn.modules.block import Attention, PSABlock

    return (PSABlock, Attention)


def assign_region(model: nn.Module, name: str, module_path: list[nn.Module]) -> str:
    """Region for the conv at dotted `name` (relative to `DetectionModel`),
    given the chain of modules from the root down to it."""
    parts = name.split(".")
    if parts[0] != "model" or len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Not a YOLO26 layer conv: {name!r}")
    index = int(parts[1])
    detect_index = len(model.model) - 1

    if index == detect_index:
        branch = parts[2]
        if branch in ("cv2", "cv3"):
            return "head_one2many"
        if branch in ("one2one_cv2", "one2one_cv3"):
            # cvX.<scale>.2 is the final 1x1 projection (plain nn.Conv2d).
            return "head_out" if parts[4] == "2" else "head_one2one"
        raise ValueError(f"Unexpected Detect submodule: {name!r}")

    if any(isinstance(m, _attention_types()) for m in module_path):
        return "attention"
    for region, indices in LAYER_REGIONS.items():
        if index in indices:
            return region
    raise ValueError(f"Layer {index} ({name!r}) has no region — YOLO26 yaml changed?")


def _parse_precision(value: Precision) -> int | None:
    """Precision -> bit width, `None` meaning full precision."""
    if value in ("fp", "fp32", None):
        return None
    if value == "binary":
        return 1
    bits = int(value)
    if bits >= 32:
        return None
    return bits


@dataclass
class BinarizationConfig:
    preset: str = "ib_guided"
    # Per-region override of the preset; `None` values mean "use the preset".
    regions: dict[str, Precision | None] = field(default_factory=dict)
    stochastic: bool = False
    # Binarizing a depthwise conv (one input channel per filter) leaves a
    # single sign per tap — the standard failure case in BNN literature, and
    # they are cheap anyway. Keep them at the region's k-bit fallback.
    skip_depthwise: bool = True
    depthwise_fallback_bits: int | None = 8

    @classmethod
    def from_dict(cls, cfg: dict | None) -> BinarizationConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown binarization options: {sorted(unknown)}")
        return cls(**cfg)

    def region_precision(self) -> dict[str, int | None]:
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}. Available: {sorted(PRESETS)}")
        unknown = set(self.regions) - set(REGIONS)
        if unknown:
            raise ValueError(f"Unknown regions {sorted(unknown)}. Available: {list(REGIONS)}")
        merged = dict(PRESETS[self.preset])
        merged.update({r: v for r, v in self.regions.items() if v is not None})
        return {r: _parse_precision(v) for r, v in merged.items()}


@dataclass
class RegionStats:
    convs: int = 0
    params: int = 0
    weight_bits: int = 0
    bits: set = field(default_factory=set)


def _named_convs_with_path(root: nn.Module):
    def walk(module: nn.Module, prefix: str, path: list[nn.Module]):
        for child_name, child in module.named_children():
            name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Conv2d):
                yield name, module, child_name, child, path + [module]
            else:
                yield from walk(child, name, path + [module])

    yield from walk(root, "", [])


def apply_binarization(model: nn.Module, cfg: BinarizationConfig | dict | None) -> dict[str, RegionStats]:
    """Replace every `nn.Conv2d` in a YOLO26 `DetectionModel` in place with a
    `QuantConv2d` of its region's precision (fp regions stay untouched).
    Returns per-region stats for logging/the thesis tables.

    Call *after* loading fp32 weights: latent weights are copied over, which
    is the standard warm start for BNN training.
    """
    if not isinstance(cfg, BinarizationConfig):
        cfg = BinarizationConfig.from_dict(cfg)
    precision = cfg.region_precision()
    stats = {r: RegionStats() for r in REGIONS}

    for name, parent, child_name, conv, path in list(_named_convs_with_path(model)):
        region = assign_region(model, name, path)
        bits = precision[region]
        is_depthwise = conv.groups > 1 and conv.groups == conv.in_channels
        if bits == 1 and is_depthwise and cfg.skip_depthwise:
            bits = cfg.depthwise_fallback_bits
        stats[region].convs += 1
        stats[region].params += conv.weight.numel()
        stats[region].weight_bits += conv.weight.numel() * (bits or 32)
        stats[region].bits.add(bits or 32)
        if bits is None:
            continue
        setattr(
            parent,
            child_name,
            QuantConv2d.from_conv(conv, w_bits=bits, a_bits=bits, stochastic=cfg.stochastic and bits == 1),
        )
    return stats


def format_region_stats(stats: dict[str, RegionStats]) -> str:
    total = sum(s.params for s in stats.values()) or 1
    lines = [f"{'region':<15}{'convs':>6}{'conv params':>13}{'share':>8}  bits"]
    for region, s in stats.items():
        bits = "/".join(str(b) for b in sorted(s.bits)) or "-"
        lines.append(f"{region:<15}{s.convs:>6}{s.params:>13,}{s.params / total:>8.1%}  {bits}")
    deployed = {r: s for r, s in stats.items() if r != "head_one2many"}  # train-only branch
    deployed_kib = sum(s.weight_bits for s in deployed.values()) / 8 / 1024
    fp32_kib = sum(s.params for s in deployed.values()) * 4 / 1024
    lines.append(f"deployed conv weights: {deployed_kib:,.1f} KiB (fp32: {fp32_kib:,.1f} KiB)")
    return "\n".join(lines)


def set_binarization_progress(model: nn.Module, progress: float, t_min: float = 0.1, t_max: float = 10.0) -> None:
    """Anneal the surrogate-gradient sharpness of every binary layer
    (`progress` in [0, 1]; see `functional.ede_sharpness`)."""
    from binarized_cv.models.binarized.functional import ede_sharpness

    t = ede_sharpness(progress, t_min, t_max)
    for module in model.modules():
        if isinstance(module, QuantConv2d):
            module.sharpness = t


def has_quantized_layers(model: nn.Module) -> bool:
    return any(isinstance(m, QuantConv2d) for m in model.modules())


def optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay pulls latent binary weights toward 0, where their sign
    flips on every step — a known BNN instability. Exempt quantized-layer
    latent weights and thresholds; everything else keeps `weight_decay`."""
    quantized_params = {
        id(p) for m in model.modules() if isinstance(m, QuantConv2d) and m.w_bits == 1 for p in m.parameters()
    }
    decay = [p for p in model.parameters() if id(p) not in quantized_params]
    no_decay = [p for p in model.parameters() if id(p) in quantized_params]
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


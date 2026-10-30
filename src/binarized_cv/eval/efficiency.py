"""Efficiency measurements: operation counts, deployed model size, latency
and energy per inference (CHECKLIST.md section 8).

Binary and k-bit layers are *simulated* (`QuantConv2d` runs an fp32 conv on
rounded weights), so measured latency/energy of a quantized model is that of
the simulation, not of a bitwise kernel. For those models only the analytic
numbers (`count_ops`, `deployed_size`) say something about the deployed
model; `benchmark` results record `simulated_quantization` so tables can't
mix the two silently.
"""

from __future__ import annotations

import io
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from binarized_cv.models.binarized.layers import QuantConv2d
from binarized_cv.models.binarized.policy import has_quantized_layers

# BNN-literature convention (Bi-Real Net, ReActNet): OPs = FLOPs + BOPs / 64,
# where a 64-bit XNOR + popcount does 64 binary MACs in one instruction. A
# k-bit x k-bit MAC counts as k_w * k_a BOPs (Baskin et al., UNIQ), so 8-bit
# MACs cost the same as fp ones and 1-bit ones 1/64 of that.
BITS_PER_OP = 64


@dataclass
class OpCounts:
    # MACs by "w<bits>a<bits>" (fp = 32), plus attention matmuls (always fp).
    macs: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, macs: int) -> None:
        self.macs[key] = self.macs.get(key, 0) + macs

    @property
    def fp_macs(self) -> int:
        return sum(m for k, m in self.macs.items() if k in ("w32a32", "attn_matmul"))

    @property
    def bops(self) -> int:
        """Bit operations of all quantized MACs (k_w * k_a each)."""
        total = 0
        for key, m in self.macs.items():
            if key in ("w32a32", "attn_matmul"):
                continue
            w, a = key[1:].split("a")
            total += m * int(w) * int(a)
        return total

    @property
    def ops(self) -> float:
        return self.fp_macs + self.bops / BITS_PER_OP

    def as_dict(self) -> dict:
        return {"macs": dict(self.macs), "fp_macs": self.fp_macs, "bops": self.bops, "ops": self.ops}


def _conv_bits(conv: nn.Conv2d) -> tuple[int, int]:
    if isinstance(conv, QuantConv2d):
        return conv.w_bits or 32, conv.a_bits or 32
    return 32, 32


@torch.no_grad()
def count_ops(model: nn.Module, images: dict[str, torch.Tensor]) -> OpCounts:
    """MACs per precision of one forward pass of `model` on `images`, per
    image. Counts every conv and the two attention matmuls (q^T k, v attn^T);
    BN, activations, upsampling and decoding are left out (elementwise, a
    few % of the total)."""
    from ultralytics.nn.modules.block import Attention

    counts = OpCounts()
    batch = next(iter(images.values())).shape[0]

    def conv_hook(conv: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        kh, kw = conv.kernel_size
        macs = output.numel() // batch * (conv.in_channels // conv.groups) * kh * kw
        w, a = _conv_bits(conv)
        counts.add(f"w{w}a{a}", macs)

    def attention_hook(attn: Attention, inputs, _output) -> None:
        _, _, h, w = inputs[0].shape
        n = h * w
        counts.add("attn_matmul", attn.num_heads * n * n * (attn.key_dim + attn.head_dim))

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, Attention):
            handles.append(module.register_forward_hook(attention_hook))
    try:
        model(images)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def deployed_size(model: nn.Module) -> dict[str, float]:
    """Bytes of the inference model's floating-point state with every
    quantized conv weight packed at its bit width (plus one fp32 scale per
    output channel); everything else (fp convs, BN, thresholds) at fp32.
    Call on a model already prepared by `prepare_for_inference`, so the
    train-only head is gone and fp convs have BN folded in.

    `fp32_state_dict_bytes` is the size of `torch.save(state_dict)`, the
    unpacked file this repo actually writes.
    """
    quantized_weights = {id(m.weight): m for m in model.modules() if isinstance(m, QuantConv2d) and m.w_bits}
    bits = 0
    seen = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if id(tensor) in seen or not tensor.is_floating_point():
            continue
        seen.add(id(tensor))
        conv = quantized_weights.get(id(tensor))
        if conv is not None:
            bits += tensor.numel() * conv.w_bits + conv.out_channels * 32
        else:
            bits += tensor.numel() * 32
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {"packed_bytes": bits / 8, "fp32_state_dict_bytes": buffer.getbuffer().nbytes}


def prepare_for_inference(model: nn.Module) -> nn.Module:
    """What deployment would do before running the model: drop YOLO26's
    train-only one2many head (eval mode still runs it otherwise) and fold BN
    into every *full-precision* conv. `DetectionModel.fuse()` can't be used:
    it would replace each `QuantConv2d` with a plain fused conv, silently
    turning a binarized model back into fp32. A quantized conv keeps its BN,
    as a binary kernel's output needs it as a separate fp affine step."""
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.torch_utils import fuse_conv_and_bn

    model.eval()
    for module in model.modules():
        if isinstance(module, Detect) and getattr(module, "end2end", False):
            module.fuse()
        elif type(module) is Conv and hasattr(module, "bn") and not isinstance(module.conv, QuantConv2d):
            module.conv = fuse_conv_and_bn(module.conv, module.bn)
            delattr(module, "bn")
            module.forward = module.forward_fuse
    return model


def dummy_images(model: nn.Module, batch: int, img_size: tuple[int, int], device: torch.device) -> dict:
    channels = {"rgb": 3, "ir": 1}
    generator = torch.Generator().manual_seed(0)
    return {
        m: torch.rand(batch, channels[m], *img_size, generator=generator).to(device)
        for m in getattr(model, "modalities", ("rgb", "ir"))
    }


class EnergyMeter:
    """Cumulative energy counter of one domain, in joules."""

    name: str

    def read(self) -> float:
        raise NotImplementedError


class RaplMeter(EnergyMeter):
    """Intel RAPL counter (`/sys/class/powercap/intel-rapl:N`). Its
    `energy_uj` is root-only by default; make it readable with
    `sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj` (resets on
    reboot). Wraps at `max_energy_range_uj`, handled by `delta`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = f"rapl_{(path / 'name').read_text().strip()}"
        self.max_uj = int((path / "max_energy_range_uj").read_text())

    def read(self) -> float:
        return int((self.path / "energy_uj").read_text()) / 1e6

    def delta(self, start: float, end: float) -> float:
        return end - start if end >= start else end + self.max_uj / 1e6 - start


class NvmlMeter(EnergyMeter):
    """GPU board energy since driver load (`nvmlDeviceGetTotalEnergyConsumption`,
    Volta and newer; updates roughly every 100 ms, so measure over seconds)."""

    def __init__(self, index: int = 0) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.name = f"nvml_gpu{index}"
        self.read()  # raises NVMLError_NotSupported on older GPUs

    def read(self) -> float:
        return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1e3


def available_meters(device: torch.device) -> tuple[list[EnergyMeter], list[str]]:
    """Every readable energy domain (CPU package and platform via RAPL, the
    GPU via NVML when benchmarking on CUDA), plus why the others aren't."""
    meters, problems = [], []
    for path in sorted(Path("/sys/class/powercap").glob("intel-rapl:[0-9]")):
        try:
            meter = RaplMeter(path)
            meter.read()
            meters.append(meter)
        except PermissionError:
            problems.append(f"{path}/energy_uj not readable (sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj)")
        except OSError as e:
            problems.append(f"{path}: {e}")
    if device.type == "cuda":
        try:
            meters.append(NvmlMeter(device.index or 0))
        except ImportError:
            problems.append("pynvml missing (pip install nvidia-ml-py)")
        except Exception as e:  # NVMLError subclasses, e.g. NotSupported
            problems.append(f"NVML energy: {e!r}")
    return meters, problems


def _delta(meter: EnergyMeter, start: float, end: float) -> float:
    return meter.delta(start, end) if isinstance(meter, RaplMeter) else end - start


def measure_idle_power(meters: list[EnergyMeter], seconds: float) -> dict[str, float]:
    """Average watts per domain while the process sleeps — the baseline that
    `benchmark` subtracts to get the energy the inference itself adds."""
    if not meters or seconds <= 0:
        return {}
    start = [m.read() for m in meters]
    t0 = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - t0
    return {m.name: _delta(m, s, m.read()) / elapsed for m, s in zip(meters, start)}


@dataclass
class RoundResult:
    latencies_s: list[float]
    elapsed_s: float
    energy_j: dict[str, float]


def run_round(
    forward: Callable[[], object],
    sync: Callable[[], None],
    meters: list[EnergyMeter],
    min_iters: int,
    min_seconds: float,
) -> RoundResult:
    """Back-to-back forward passes, at least `min_iters` of them and for at
    least `min_seconds` (energy counters need seconds to resolve), each timed
    individually with the device synchronised."""
    latencies = []
    start = [m.read() for m in meters]
    t_start = time.perf_counter()
    while len(latencies) < min_iters or time.perf_counter() - t_start < min_seconds:
        t0 = time.perf_counter()
        forward()
        sync()
        latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - t_start
    energy = {m.name: _delta(m, s, m.read()) for m, s in zip(meters, start)}
    return RoundResult(latencies, elapsed, energy)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


@torch.inference_mode()
def benchmark(
    model: nn.Module,
    images: dict[str, torch.Tensor],
    warmup: int = 20,
    min_iters: int = 100,
    min_seconds: float = 10.0,
    rounds: int = 3,
    cooldown_s: float = 5.0,
    idle_s: float = 5.0,
) -> dict:
    """Latency, throughput and energy per image of `model(images)`.

    `rounds` separate rounds with `cooldown_s` pauses between them give the
    round-to-round spread (thermal throttling, clock boost); the reported
    latency/energy is mean ± std over round means, the percentiles are over
    all iterations pooled. Energy is the whole domain (package, platform or
    GPU board), other processes included: `*_total` is everything the
    counter saw, `*_above_idle` subtracts the idle power measured first.
    """
    device = next(iter(images.values())).device
    batch = next(iter(images.values())).shape[0]
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    meters, problems = available_meters(device)

    def forward():
        return model(images)

    # Idle before the warmup: right after load a GPU stays at high clocks
    # for seconds (RTX 2050: ~12 W vs ~5 W idle), which would hide part of
    # what running the model costs.
    idle_w = measure_idle_power(meters, idle_s)
    for _ in range(warmup):
        forward()
    sync()

    results = []
    for i in range(rounds):
        if i:
            time.sleep(cooldown_s)
        results.append(run_round(forward, sync, meters, min_iters, min_seconds))

    pooled_ms = [t * 1e3 for r in results for t in r.latencies_s]
    round_ms = [statistics.fmean(r.latencies_s) * 1e3 for r in results]
    out = {
        "batch": batch,
        "iters_per_round": [len(r.latencies_s) for r in results],
        "latency_ms": _mean_std(round_ms),
        "latency_ms_p50": _percentile(pooled_ms, 0.5),
        "latency_ms_p90": _percentile(pooled_ms, 0.9),
        "latency_ms_p99": _percentile(pooled_ms, 0.99),
        "fps": _mean_std([batch * len(r.latencies_s) / r.elapsed_s for r in results]),
        "idle_w": idle_w,
        "energy_mj_per_image": {},
        "avg_power_w": {},
        "energy_unavailable": problems,
        "simulated_quantization": has_quantized_layers(model),
    }
    for meter in meters:
        images_per_round = [batch * len(r.latencies_s) for r in results]
        total = [r.energy_j[meter.name] / n * 1e3 for r, n in zip(results, images_per_round)]
        above = [
            (r.energy_j[meter.name] - idle_w.get(meter.name, 0.0) * r.elapsed_s) / n * 1e3
            for r, n in zip(results, images_per_round)
        ]
        out["energy_mj_per_image"][f"{meter.name}_total"] = _mean_std(total)
        out["energy_mj_per_image"][f"{meter.name}_above_idle"] = _mean_std(above)
        out["avg_power_w"][meter.name] = _mean_std([r.energy_j[meter.name] / r.elapsed_s for r in results])
    return out


def environment_info(device: torch.device, threads: int) -> dict:
    import platform

    info = {
        "device": str(device),
        "torch": torch.__version__,
        "threads": threads,
        "cpu": _cpu_model(),
        "python": platform.python_version(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor()


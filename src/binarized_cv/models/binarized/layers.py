from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from binarized_cv.models.binarized.functional import binarize, round_ste

_EPS = 1e-8


class QuantConv2d(nn.Conv2d):
    """Drop-in `nn.Conv2d` with simulated low-precision weights and inputs.

    `w_bits` / `a_bits` select the precision independently:

    - `1`: binary. Weights follow IR-Net's balanced scheme — each output
      channel is centred and standardised before `sign`, which maximises
      the entropy of the binary weights (IR-Net's own information-theoretic
      argument), then rescaled by a real-valued per-channel
      `alpha = mean|W - mean(W)|` (XNOR-Net). Inputs are `sign(x - tau)`
      with a learnable per-channel threshold `tau` (ReActNet's RSign):
      ultralytics `Conv` feeds post-SiLU activations, which are almost all
      `>= 0`, so a fixed threshold of 0 would collapse every input to `+1`.
      `tau` is data-initialised to the per-channel mean on the first batch.
    - `2..16`: uniform fake quantization — symmetric per-output-channel
      weights; asymmetric inputs over an EMA-tracked min/max range.
    - `None`: full precision for that operand.

    The parameter is still called `weight` (the latent real-valued weight),
    so fp32 checkpoints load into a quantized model unchanged and the BN
    that follows in ultralytics' `Conv` stays real-valued, as the plan's
    "BN, biases, scaling factors stay full precision" row requires.

    Deployment-faithful options, all off by default so existing checkpoints
    evaluate as trained (CHECKLIST.md section 10: to become the default):

    - `binary_padding="same"`: a binary conv with stride > 1 pads like
      TensorFlow's `SAME` (extra row/column at the bottom/right only)
      instead of symmetrically. Larq Compute Engine's binary conv supports
      only `SAME`/`VALID`; with symmetric padding its converter silently
      miscompiles stride-2 convs (`docs/labnotes.md` 2026-10-24).
    - `integer_zero_point`: k-bit inputs use an integer zero point and a
      range that includes 0, as int8 runtimes do. The default grid starts at
      the observed minimum, which a real int8 tensor can't represent.
    - `quantize_output`: a k-bit conv's output is fake-quantized too (own
      EMA range), since an int8 conv kernel writes int8. Not applied to
      binary convs, whose output stays real-valued on device.
    """

    def __init__(
        self,
        *args,
        w_bits: int | None = 1,
        a_bits: int | None = 1,
        stochastic: bool = False,
        range_momentum: float = 0.1,
        binary_padding: str = "symmetric",
        integer_zero_point: bool = False,
        quantize_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        for bits in (w_bits, a_bits):
            if bits is not None and not 1 <= bits <= 16:
                raise ValueError(f"bits must be None or in [1, 16], got {bits}")
        if binary_padding not in ("symmetric", "same"):
            raise ValueError(f"binary_padding must be 'symmetric' or 'same', got {binary_padding!r}")
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.stochastic = stochastic
        self.range_momentum = range_momentum
        self.binary_padding = binary_padding
        self.integer_zero_point = integer_zero_point
        # Output bits: only a k-bit conv (k >= 2 on both operands) has a
        # quantized output on device.
        self.out_bits = a_bits if quantize_output and (a_bits or 0) >= 2 and (w_bits or 0) >= 2 else None
        # Surrogate-gradient sharpness, updated by `set_binarization_progress`.
        self.sharpness = 1.0

        if a_bits == 1:
            self.act_threshold = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        else:
            self.register_parameter("act_threshold", None)
        self.register_buffer("act_range", torch.zeros(2))
        self.register_buffer("initialized", torch.tensor(False))
        if self.out_bits:
            self.register_buffer("out_range", torch.zeros(2))
            self.register_buffer("out_initialized", torch.tensor(False))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # Checkpoints from before `quantize_output` have no output range: start
        # it uninitialised (set from the next batch, like the input range).
        if self.out_bits:
            for name in ("out_range", "out_initialized"):
                state_dict.setdefault(prefix + name, getattr(self, name))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, **quant_kwargs) -> QuantConv2d:
        new = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
            **quant_kwargs,
        )
        with torch.no_grad():
            new.weight.copy_(conv.weight)
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        return new

    def quantized_weight(self) -> torch.Tensor:
        w = self.weight
        if self.w_bits is None:
            return w
        dims = tuple(range(1, w.dim()))
        if self.w_bits == 1:
            centered = w - w.mean(dim=dims, keepdim=True)
            std = centered.std(dim=dims, keepdim=True) + _EPS if w[0].numel() > 1 else 1.0
            alpha = centered.abs().mean(dim=dims, keepdim=True)
            return alpha * binarize(centered / std, self.sharpness)
        qmax = 2 ** (self.w_bits - 1) - 1
        scale = w.abs().amax(dim=dims, keepdim=True).clamp_min(_EPS) / qmax
        return round_ste(w / scale).clamp(-qmax, qmax) * scale

    def quantized_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.a_bits is None:
            return x
        if self.a_bits == 1:
            if not self.initialized:
                with torch.no_grad():
                    self.act_threshold.copy_(x.mean(dim=(0, 2, 3), keepdim=True))
                    self.initialized.fill_(True)
            stochastic = self.stochastic and self.training
            return binarize(x - self.act_threshold, self.sharpness, stochastic=stochastic)

        self._observe_range(x, self.act_range, self.initialized)
        return self._fake_quantize(x, self.act_range, self.a_bits)

    def _observe_range(self, x: torch.Tensor, value_range: torch.Tensor, initialized: torch.Tensor) -> None:
        """EMA of the min/max while training; set once if never seen."""
        if self.training or not initialized:
            with torch.no_grad():
                observed = torch.stack([x.min(), x.max()])
                if initialized:
                    value_range.lerp_(observed, self.range_momentum)
                else:
                    value_range.copy_(observed)
                    initialized.fill_(True)

    def _fake_quantize(self, x: torch.Tensor, value_range: torch.Tensor, bits: int) -> torch.Tensor:
        lo, hi = value_range[0], value_range[1]
        levels = 2**bits - 1
        if not self.integer_zero_point:
            scale = ((hi - lo) / levels).clamp_min(_EPS)
            return round_ste((torch.clamp(x, lo, hi) - lo) / scale) * scale + lo
        # x = scale * (q - zero) with integer q, zero in [0, levels], so the
        # range must contain 0 (as torch's and TFLite's observers enforce).
        lo, hi = lo.clamp(max=0), hi.clamp(min=0)
        scale = ((hi - lo) / levels).clamp_min(_EPS)
        zero = torch.round(-lo / scale).clamp(0, levels)
        return (torch.clamp(round_ste(x / scale) + zero, 0, levels) - zero) * scale

    def _same_padding(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        """TensorFlow `SAME` padding of `x` as `F.pad` (left, right, top, bottom)."""
        pads = []
        for size, k, s, d in zip(x.shape[-2:], self.kernel_size, self.stride, self.dilation):
            total = max((math.ceil(size / s) - 1) * s + (k - 1) * d + 1 - size, 0)
            pads.append((total // 2, total - total // 2))
        (top, bottom), (left, right) = pads
        return left, right, top, bottom

    def _uses_same_padding(self) -> bool:
        # Only padded ("same"-shaped) binary convs with stride > 1 differ from
        # TF; stride-1 odd-kernel convs pad identically either way.
        autopad = tuple((k - 1) * d // 2 for k, d in zip(self.kernel_size, self.dilation))
        return (
            self.binary_padding == "same"
            and self.a_bits == 1
            and self.stride != (1, 1)
            and self.padding == autopad
            and self.padding_mode == "zeros"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantized_input(x)
        weight = self.quantized_weight()
        if self._uses_same_padding():
            # Zero-pad the binarized input, as LCE's `pad_values=0`.
            x = F.pad(x, self._same_padding(x))
            y = F.conv2d(x, weight, self.bias, self.stride, 0, self.dilation, self.groups)
        else:
            y = self._conv_forward(x, weight, self.bias)
        if self.out_bits:
            self._observe_range(y, self.out_range, self.out_initialized)
            y = self._fake_quantize(y, self.out_range, self.out_bits)
        return y

    def extra_repr(self) -> str:
        options = ""
        if self.binary_padding != "symmetric":
            options += f", binary_padding={self.binary_padding}"
        if self.integer_zero_point:
            options += ", integer_zero_point=True"
        if self.out_bits:
            options += f", out_bits={self.out_bits}"
        return (
            f"{super().extra_repr()}, w_bits={self.w_bits}, a_bits={self.a_bits}, "
            f"stochastic={self.stochastic}{options}"
        )

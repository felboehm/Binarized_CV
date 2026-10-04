from __future__ import annotations

import torch
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
    """

    def __init__(
        self,
        *args,
        w_bits: int | None = 1,
        a_bits: int | None = 1,
        stochastic: bool = False,
        range_momentum: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        for bits in (w_bits, a_bits):
            if bits is not None and not 1 <= bits <= 16:
                raise ValueError(f"bits must be None or in [1, 16], got {bits}")
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.stochastic = stochastic
        self.range_momentum = range_momentum
        # Surrogate-gradient sharpness, updated by `set_binarization_progress`.
        self.sharpness = 1.0

        if a_bits == 1:
            self.act_threshold = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        else:
            self.register_parameter("act_threshold", None)
        self.register_buffer("act_range", torch.zeros(2))
        self.register_buffer("initialized", torch.tensor(False))

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

        if self.training or not self.initialized:
            with torch.no_grad():
                observed = torch.stack([x.min(), x.max()])
                if self.initialized:
                    self.act_range.lerp_(observed, self.range_momentum)
                else:
                    self.act_range.copy_(observed)
                    self.initialized.fill_(True)
        lo, hi = self.act_range[0], self.act_range[1]
        scale = ((hi - lo) / (2**self.a_bits - 1)).clamp_min(_EPS)
        return round_ste((torch.clamp(x, lo, hi) - lo) / scale) * scale + lo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(self.quantized_input(x), self.quantized_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, w_bits={self.w_bits}, a_bits={self.a_bits}, stochastic={self.stochastic}"

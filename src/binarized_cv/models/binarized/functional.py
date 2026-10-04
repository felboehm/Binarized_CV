from __future__ import annotations

import torch


class _SignEDE(torch.autograd.Function):
    """Hard `sign` forward, smooth tanh-shaped surrogate gradient backward.

    The surrogate is IR-Net's Error Decay Estimator (Qin et al., CVPR 2020):
    `d/dx ≈ k * t * (1 - tanh²(t * x))` with `k = max(1/t, 1)`. Small `t`
    early in training behaves like a plain identity STE (every latent value
    gets gradient); large `t` late in training approaches the true sign
    derivative. Annealing `t` is how `docs/binarization_plan.md`'s
    "progressive binarization" is realised without a train/eval mismatch:
    the forward pass is *always* exactly binary, only the gradient sharpens.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, t: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.t = t
        # torch.sign maps 0 -> 0, which would make a third level; map it to +1.
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        t = ctx.t
        k = max(1.0 / t, 1.0)
        grad = k * t * (1 - torch.tanh(t * x) ** 2)
        return grad_output * grad, None


class _StochasticSignEDE(torch.autograd.Function):
    """Stochastic binarization: `+1` with probability `(tanh(t*x) + 1) / 2`.

    Motivated by Tishby & Zaslavsky's argument that approaching the IB limit
    needs stochastic mappings between layers (`docs/binarization_plan.md`,
    constraint 3). Tying the noise to the same sharpness `t` as the gradient
    surrogate means the mapping starts noisy and becomes deterministic as
    annealing finishes. Same EDE gradient as `_SignEDE`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, t: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.t = t
        prob = (torch.tanh(t * x) + 1) / 2
        return torch.where(torch.rand_like(x) < prob, torch.ones_like(x), -torch.ones_like(x))

    backward = _SignEDE.backward


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def binarize(x: torch.Tensor, t: float = 1.0, stochastic: bool = False) -> torch.Tensor:
    """Map `x` to `{-1, +1}` with an annealable surrogate gradient."""
    if stochastic:
        return _StochasticSignEDE.apply(x, t)
    return _SignEDE.apply(x, t)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """`round` with identity (straight-through) gradient."""
    return _RoundSTE.apply(x)


def ede_sharpness(progress: float, t_min: float = 0.1, t_max: float = 10.0) -> float:
    """IR-Net's schedule: log-linear from `t_min` to `t_max` as `progress`
    goes from 0 to 1 (clamped)."""
    progress = min(max(progress, 0.0), 1.0)
    return t_min * (t_max / t_min) ** progress

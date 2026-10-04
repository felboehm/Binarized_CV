"""Per-layer information-plane estimates for YOLO26 (docs/binarization_plan.md §4).

For each hooked layer we estimate, per spatial location of its output:

- `I(X;T)`: for a deterministic network `I(X;T) = H(T)`, so this is the
  entropy of the layer's code `T` at that location.
- `I(T;Y)`: with `Y` = "this location's grid cell centre lies inside a
  person box" — the quantity the detection head ultimately has to read out
  at that location.

`T` is formed by binarizing each channel at its median (the maximum-entropy
1-bit binning) and reading a random subset of `n_units` channels as an
integer code; results are averaged over `n_subsets` random subsets. That
keeps the code space small enough (2**n_units states) for plug-in entropy
estimates to be reliable with ~10^5 locations, and Miller–Madow bias
correction is applied. The same estimator is used for every layer and for
fp32 and binarized models alike, so *relative* comparisons across layers
and models are meaningful; absolute values depend on the binning, which is
exactly the caveat raised in the IB debate (Saxe et al., 2018) and should be
reported alongside the numbers.

Layers are those of `DetectionModel.model` (see
`binarized_cv.models.binarized.policy.LAYER_REGIONS` for what they are).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


def entropy_bits(counts: torch.Tensor) -> float:
    """Plug-in entropy (bits) with Miller–Madow correction."""
    counts = counts[counts > 0].double()
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    h = -(p * torch.log2(p)).sum().item()
    return h + (counts.numel() - 1) / (2 * n.item() * math.log(2))


def mutual_information_bits(joint_counts: torch.Tensor) -> tuple[float, float, float]:
    """From a `(num_codes, num_labels)` count table, returns
    `(H(T), I(T;Y), H(Y))` in bits."""
    h_t = entropy_bits(joint_counts.sum(dim=1))
    h_y = entropy_bits(joint_counts.sum(dim=0))
    h_ty = entropy_bits(joint_counts.flatten())
    return h_t, max(h_t + h_y - h_ty, 0.0), h_y


@dataclass
class LayerInformation:
    layer: int
    channels: int
    stride: int
    i_xt: float  # = H(T), bits (≤ n_units)
    i_ty: float  # bits (≤ H(Y))
    h_y: float
    i_ty_fraction: float  # I(T;Y) / H(Y)

    def as_dict(self) -> dict:
        return asdict(self)


class LayerInformationEstimator:
    """Accumulates code/label counts over batches via forward hooks on
    `detection_model.model[i]` for each `i` in `layers`."""

    def __init__(
        self,
        detection_model: nn.Module,
        layers: list[int],
        n_units: int = 12,
        n_subsets: int = 8,
        max_positions_per_image: int = 2048,
        seed: int = 0,
    ) -> None:
        self.layers = layers
        self.n_units = n_units
        self.n_subsets = n_subsets
        self.max_positions = max_positions_per_image
        self.generator = torch.Generator().manual_seed(seed)
        self._outputs: dict[int, torch.Tensor] = {}
        self._subsets: dict[int, torch.Tensor] = {}
        self._thresholds: dict[int, torch.Tensor] = {}
        self._counts: dict[int, torch.Tensor] = {}
        self._strides: dict[int, int] = {}
        self._hooks = [
            detection_model.model[i].register_forward_hook(self._make_hook(i)) for i in layers
        ]

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            self._outputs[index] = output.detach()

        return hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def update(self, input_hw: tuple[int, int], targets: list[dict[str, torch.Tensor]]) -> None:
        """Call right after a forward pass on a batch. `targets` are the
        dataset's per-image `{"boxes": cxcywh normalized}` dicts."""
        for index in self.layers:
            features = self._outputs.pop(index).float().cpu()
            batch, channels, height, width = features.shape
            self._strides[index] = round(input_hw[0] / height)

            if index not in self._subsets:
                k = min(self.n_units, channels)
                self._subsets[index] = torch.stack(
                    [torch.randperm(channels, generator=self.generator)[:k] for _ in range(self.n_subsets)]
                )
                # Thresholds fixed from the first batch so codes mean the same
                # thing across batches.
                self._thresholds[index] = features.transpose(0, 1).reshape(channels, -1).median(dim=1).values
                self._counts[index] = torch.zeros(self.n_subsets, 2**k, 2, dtype=torch.long)

            bits = (features > self._thresholds[index].view(1, -1, 1, 1)).long()  # (B, C, H, W)
            labels = _location_labels(targets, height, width)  # (B, H, W)
            bits = bits.permute(0, 2, 3, 1).reshape(-1, channels)
            labels = labels.reshape(-1)
            keep = _sample_positions(batch, height * width, self.max_positions, self.generator)
            bits, labels = bits[keep], labels[keep]

            k = self._subsets[index].shape[1]
            weights = 2 ** torch.arange(k)
            for s, subset in enumerate(self._subsets[index]):
                codes = (bits[:, subset] * weights).sum(dim=1)
                self._counts[index][s] += torch.bincount(codes * 2 + labels, minlength=2 ** (k + 1)).view(-1, 2)

    def results(self) -> list[LayerInformation]:
        out = []
        for index in self.layers:
            if index not in self._counts:
                continue
            estimates = [mutual_information_bits(c) for c in self._counts[index]]
            h_t = sum(e[0] for e in estimates) / len(estimates)
            i_ty = sum(e[1] for e in estimates) / len(estimates)
            h_y = estimates[0][2]
            out.append(
                LayerInformation(
                    layer=index,
                    channels=int(self._thresholds[index].numel()),
                    stride=self._strides[index],
                    i_xt=h_t,
                    i_ty=i_ty,
                    h_y=h_y,
                    i_ty_fraction=i_ty / h_y if h_y > 0 else 0.0,
                )
            )
        return out


def _location_labels(targets: list[dict[str, torch.Tensor]], height: int, width: int) -> torch.Tensor:
    ys = (torch.arange(height) + 0.5) / height
    xs = (torch.arange(width) + 0.5) / width
    labels = torch.zeros(len(targets), height, width, dtype=torch.long)
    for i, target in enumerate(targets):
        for cx, cy, w, h in target["boxes"].cpu().tolist():
            in_y = (ys >= cy - h / 2) & (ys <= cy + h / 2)
            in_x = (xs >= cx - w / 2) & (xs <= cx + w / 2)
            labels[i] |= (in_y[:, None] & in_x[None, :]).long()
    return labels


def _sample_positions(batch: int, per_image: int, cap: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform (label-agnostic, so p(Y) stays unbiased) subsample of at most
    `cap` locations per image, as flat indices into `(batch * per_image)`."""
    if per_image <= cap:
        return torch.arange(batch * per_image)
    return torch.cat(
        [i * per_image + torch.randperm(per_image, generator=generator)[:cap] for i in range(batch)]
    )

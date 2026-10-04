import pytest
import torch
from torch import nn

from binarized_cv.analysis.information import (
    LayerInformationEstimator,
    entropy_bits,
    mutual_information_bits,
)


def test_entropy_of_uniform_distribution():
    assert entropy_bits(torch.tensor([1000, 1000, 1000, 1000])) == pytest.approx(2.0, abs=1e-3)


def test_mutual_information_extremes():
    identical = torch.tensor([[500, 0], [0, 500]])
    independent = torch.tensor([[250, 250], [250, 250]])
    assert mutual_information_bits(identical)[1] == pytest.approx(1.0, abs=1e-2)
    assert mutual_information_bits(independent)[1] == pytest.approx(0.0, abs=1e-2)


class _ToyDetector(nn.Module):
    """Layer 0 copies the box mask into every channel (fully informative),
    layer 1 is pure noise (uninformative)."""

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, mask):
        self.model[0](mask.expand(-1, 4, -1, -1) + 0.01 * torch.randn(mask.shape[0], 4, *mask.shape[2:]))
        self.model[1](torch.randn(mask.shape[0], 4, *mask.shape[2:]))


def test_estimator_separates_informative_from_noise_layer():
    torch.manual_seed(0)
    toy = _ToyDetector()
    estimator = LayerInformationEstimator(toy, layers=[0, 1], n_units=4, n_subsets=2)
    for _ in range(4):
        targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]])} for _ in range(8)]
        mask = torch.zeros(8, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1.0
        toy(mask)
        estimator.update((16, 16), targets)
    informative, noise = estimator.results()
    assert informative.i_ty_fraction > 0.9
    assert noise.i_ty_fraction < 0.05
    assert noise.i_xt > informative.i_xt  # noise has more rate, less relevance

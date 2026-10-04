import torch
from torch.utils.data import DataLoader

from binarized_cv.models.binarized.functional import binarize
from binarized_cv.train.train import seed_everything


def _draws(seed: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    seed_everything(seed)
    init = torch.nn.Conv2d(4, 8, 3).weight.detach().clone()
    stochastic = binarize(torch.randn(256), t=1.0, stochastic=True)
    order = list(DataLoader(range(32), shuffle=True, generator=torch.Generator().manual_seed(seed)))
    return init, stochastic, [int(i) for i in order]


def test_same_seed_reproduces_init_stochastic_binarization_and_shuffle():
    a, b = _draws(0), _draws(0)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert a[2] == b[2]


def test_different_seeds_differ():
    a, b = _draws(0), _draws(1)
    assert not torch.equal(a[0], b[0])
    assert a[2] != b[2]

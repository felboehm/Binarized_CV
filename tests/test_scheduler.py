from itertools import pairwise

import pytest
import torch
from omegaconf import OmegaConf

from binarized_cv.train.train import build_scheduler


def _lrs(**train_cfg) -> list[float]:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = build_scheduler(optimizer, OmegaConf.create({"epochs": 2, **train_cfg}), steps_per_epoch=10)
    lrs = []
    for _ in range(20):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return lrs


def test_no_scheduler_keeps_lr_constant():
    assert _lrs(scheduler="none") == [1.0] * 20


def test_cosine_warms_up_then_decays_to_min():
    lrs = _lrs(scheduler="cosine", warmup_epochs=0.5, min_lr_ratio=0.01)
    assert lrs[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert all(a >= b for a, b in pairwise(lrs[4:]))
    assert 0.01 < lrs[-1] < 0.05  # last step is one short of the floor


def test_unknown_scheduler_rejected():
    with pytest.raises(ValueError):
        _lrs(scheduler="step")

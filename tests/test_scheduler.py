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


def _plateau(**kwargs):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    cfg = {"epochs": 100, "scheduler": "plateau", "plateau_patience": 2, "plateau_max_drops": 1, **kwargs}
    scheduler = build_scheduler(optimizer, OmegaConf.create(cfg), steps_per_epoch=10)
    return optimizer, scheduler, scheduler.lr_lambdas[0]


def test_plateau_cuts_lr_on_stall_then_stops():
    optimizer, scheduler, plateau = _plateau()
    assert plateau.update(0.5) == (True, False)
    assert plateau.update(0.5005) == (False, False)  # below min_delta
    assert plateau.update(0.4) == (False, False)  # 2nd stall: cut instead of stop
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert plateau.update(0.6) == (True, False)  # improvement resets the count
    assert plateau.update(0.6) == (False, False)
    assert plateau.update(0.6) == (False, True)  # stall after the last cut


def test_plateau_warmup():
    optimizer, scheduler, _ = _plateau(warmup_epochs=0.5)
    lrs = []
    for _ in range(7):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])


def test_plateau_state_survives_state_dict_round_trip():
    _, scheduler, plateau = _plateau()
    plateau.update(0.5)
    plateau.update(0.4)
    plateau.update(0.4)  # cut
    _, restored, restored_plateau = _plateau()
    restored.load_state_dict(scheduler.state_dict())
    assert vars(restored_plateau) == vars(plateau)

import math

import torch

from binarized_cv.train.train import ModelEMA


def test_ema_averages_with_ramped_decay():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    ema = ModelEMA(model, decay=0.9, tau=1.0)
    expected = 0.0
    for value in (1.0, 2.0, 3.0):
        model.weight.data.fill_(value)
        ema.update(model)
        d = 0.9 * (1 - math.exp(-ema.updates / 1.0))
        expected = d * expected + (1 - d) * value
    assert ema.module.weight.item() == torch.tensor(expected).item()
    assert not ema.module.weight.requires_grad


def test_ema_first_update_follows_model_and_averages_bn_stats():
    model = torch.nn.BatchNorm1d(2)
    ema = ModelEMA(model, decay=0.9999, tau=2000.0)
    model.train()
    model(torch.randn(16, 2) * 3 + 5)
    ema.update(model)
    # d is ~0 on the first update, so the average is (almost) the model itself.
    assert torch.allclose(ema.module.running_mean, model.running_mean, atol=1e-3)
    assert ema.module.num_batches_tracked == model.num_batches_tracked

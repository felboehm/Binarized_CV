import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from binarized_cv.train.train import (
    _skip_shuffles,
    build_scheduler,
    load_training_state,
    save_training_state,
)

TRAIN_CFG = OmegaConf.create({"epochs": 4, "scheduler": "cosine", "warmup_epochs": 1, "min_lr_ratio": 0.01})


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    data = TensorDataset(torch.arange(24, dtype=torch.float32).reshape(8, 3), torch.ones(8, 1))
    loader = DataLoader(data, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(seed))
    scheduler = build_scheduler(optimizer, TRAIN_CFG, len(loader))
    return model, optimizer, scheduler, loader


def _train(model, optimizer, scheduler, loader, epochs) -> list[float]:
    lrs = []
    for _ in range(epochs):
        for x, y in loader:
            loss = torch.nn.functional.mse_loss(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
    return lrs


def test_full_state_resume_matches_uninterrupted_run(tmp_path):
    reference = _setup()
    _train(*reference, epochs=4)

    model, optimizer, scheduler, loader = _setup()
    _train(model, optimizer, scheduler, loader, epochs=2)
    save_training_state(tmp_path / "last.pt", model, optimizer, scheduler, 1, 2 * len(loader), loader.generator, TRAIN_CFG)

    resumed = _setup(seed=1)  # different init and shuffle state, all overwritten by the load
    start_epoch, step = load_training_state(tmp_path / "last.pt", *resumed[:3], resumed[3])
    assert (start_epoch, step) == (2, 2 * len(loader))
    _train(*resumed, epochs=2)
    assert torch.equal(resumed[0].weight, reference[0].weight)


def test_weights_only_resume_reconstructs_step_lr_and_shuffle(tmp_path):
    reference = _setup()
    _train(*reference, epochs=3)
    next_order = [x[:, 0].tolist() for x, _ in reference[3]]  # epoch 3's batches

    model, optimizer, scheduler, loader = _setup()
    _train(model, optimizer, scheduler, loader, epochs=3)
    torch.save(model.state_dict(), tmp_path / "epoch_2.pt")

    resumed = _setup()
    start_epoch, step = load_training_state(tmp_path / "epoch_2.pt", *resumed[:3], resumed[3])
    assert (start_epoch, step) == (3, 3 * len(loader))
    assert torch.equal(resumed[0].weight, model.weight)
    assert resumed[1].param_groups[0]["lr"] == pytest.approx(reference[1].param_groups[0]["lr"])
    assert resumed[2].last_epoch == step
    assert [x[:, 0].tolist() for x, _ in resumed[3]] == next_order


def test_weights_only_checkpoint_needs_epoch_in_name(tmp_path):
    model, optimizer, scheduler, loader = _setup()
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    with pytest.raises(ValueError):
        load_training_state(tmp_path / "weights.pt", model, optimizer, scheduler, loader)


def test_shuffle_replay_matches_multiworker_loader():
    data = list(range(20))
    real = DataLoader(data, batch_size=4, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(0))
    for _ in real:
        pass
    expected = [b.tolist() for b in real]

    replayed = DataLoader(data, batch_size=4, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(0))
    _skip_shuffles(replayed, 1)
    assert [b.tolist() for b in replayed] == expected

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from binarized_cv.data.collate import detection_collate
from binarized_cv.data.dataset import MultispectralPersonDataset
from binarized_cv.data.manifest import load_manifest
from binarized_cv.models.registry import build_model_from_config

log = logging.getLogger(__name__)


def build_dataloader(cfg: DictConfig, split: str, shuffle: bool) -> DataLoader:
    records = load_manifest(cfg.data.manifest_path)
    dataset = MultispectralPersonDataset(
        records,
        raw_root=cfg.data.raw_root,
        split=split,
        img_size=tuple(cfg.data.img_size),
        target_modality=cfg.data.target_modality,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        collate_fn=detection_collate,
    )


def _to_device(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def load_config(overrides: list[str] | None = None) -> DictConfig:
    # Uses Hydra's compose API rather than the `@hydra.main` CLI decorator:
    # hydra-core 1.3.6's argparse-based CLI parser is broken on Python 3.14
    # (a `LazyCompletionHelp` object fails argparse's new `help` string check).
    # `compose`/`initialize` do the same YAML defaults-composition and dotted
    # overrides without going through that code path.
    with initialize(version_base=None, config_path="../../../configs"):
        return compose(config_name="config", overrides=overrides if overrides is not None else sys.argv[1:])


def main(cfg: DictConfig | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    if cfg is None:
        cfg = load_config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg.model).to(device)
    train_loader = build_dataloader(cfg, split="train", shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    writer = SummaryWriter(cfg.train.log_dir)

    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs - 1}", unit="batch")
        for batch in progress:
            images = {m: t.to(device) for m, t in batch["images"].items()}
            targets = _to_device(batch["targets"], device)

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, value in losses.items():
                writer.add_scalar(f"train/{name}", value.item(), step)
            writer.add_scalar("train/loss_total", loss.item(), step)
            step += 1

            # Named per model (e.g. yolo26: loss_box/loss_cls/loss_dfl) so the bar
            # shows exactly which loss terms this model reports, not fixed names.
            progress.set_postfix(
                {name.removeprefix("loss_"): f"{value.item():.4f}" for name, value in losses.items()}
                | {"total": f"{loss.item():.4f}"}
            )

        log.info("epoch %d done (loss=%.4f)", epoch, loss.item())
        checkpoint_path = Path(cfg.train.checkpoint_dir) / f"epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    writer.close()


if __name__ == "__main__":
    main()

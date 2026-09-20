from __future__ import annotations

import logging
import sys
from datetime import datetime
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


def load_config(overrides: list[str] | None = None) -> tuple[DictConfig, str]:
    # Uses Hydra's compose API rather than the `@hydra.main` CLI decorator:
    # hydra-core 1.3.6's argparse-based CLI parser is broken on Python 3.14
    # (a `LazyCompletionHelp` object fails argparse's new `help` string check).
    # `compose`/`initialize` do the same YAML defaults-composition and dotted
    # overrides without going through that code path.
    overrides_list = overrides if overrides is not None else sys.argv[1:]

    # Extract model name from overrides (more reliable than config value)
    model_name = None
    for override in overrides_list:
        if override.startswith("model.name="):
            model_name = override.split("=", 1)[1]
            break
        elif override.startswith("model="):
            model_name = override.split("=", 1)[1]

    with initialize(version_base=None, config_path="../../../configs"):
        cfg = compose(config_name="config", overrides=overrides_list)

    return cfg, model_name or cfg.model.get("name", "unknown")


def main(cfg: DictConfig | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    if cfg is None:
        cfg, model_name = load_config()
    else:
        # If cfg is provided directly, extract model name from it
        model_name = cfg.model.get("name", "unknown")
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg.model).to(device)
    train_loader = build_dataloader(cfg, split="train", shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # Organize logs and checkpoints by model name with timestamped run directories
    # (SummaryWriter's comment parameter doesn't work when log_dir is specified,
    #  so we need timestamped subdirectories to get proper run names in TensorBoard)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(cfg.train.log_dir) / model_name / timestamp
    checkpoint_dir = Path(cfg.train.checkpoint_dir) / model_name / timestamp

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Logging to: {log_dir}")
        log.info(f"Checkpoints to: {checkpoint_dir}")
    except OSError as e:
        log.error(f"Failed to create directories: {e}")
        raise

    writer = SummaryWriter(str(log_dir))

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
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)

    writer.close()


if __name__ == "__main__":
    main()

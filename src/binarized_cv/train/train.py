from __future__ import annotations

import logging
import math
import os
import random
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from binarized_cv.data.collate import detection_collate
from binarized_cv.data.dataset import MultispectralPersonDataset
from binarized_cv.data.manifest import load_manifest
from binarized_cv.models.binarized import (
    has_quantized_layers,
    optimizer_param_groups,
    set_binarization_progress,
)
from binarized_cv.models.registry import build_model_from_config

log = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG a training run draws from: shuffle order, init of any
    weights not covered by a warm start, and stochastic binarization.

    `deterministic` also forces deterministic cuDNN kernels (slower). Without
    it, runs with the same seed still differ slightly on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_dataloader(
    cfg: DictConfig, split: str, shuffle: bool, generator: torch.Generator | None = None
) -> DataLoader:
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
        generator=generator,
    )


def _to_device(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def build_optimizer(model: torch.nn.Module, train_cfg: DictConfig) -> torch.optim.Optimizer:
    groups = optimizer_param_groups(model, train_cfg.weight_decay)
    name = train_cfg.get("optimizer", "adam")
    if name == "adam":
        return torch.optim.Adam(groups, lr=train_cfg.lr)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=train_cfg.lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=train_cfg.lr, momentum=train_cfg.get("momentum", 0.9), nesterov=True)
    raise ValueError(f"Unknown optimizer {name!r} (adam | adamw | sgd)")


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: DictConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Per-step LR schedule: linear warmup, then cosine decay to
    `lr * min_lr_ratio`. `scheduler: none` keeps the LR constant (what all
    sweeps so far used)."""
    name = train_cfg.get("scheduler", "none")
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"Unknown scheduler {name!r} (none | cosine)")
    total = train_cfg.epochs * steps_per_epoch
    warmup = int(train_cfg.get("warmup_epochs", 0) * steps_per_epoch)
    min_ratio = train_cfg.get("min_lr_ratio", 0.01)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _rng_state(generator: torch.Generator | None) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state() if generator is not None else None,
    }


def _set_rng_state(state: dict, generator: torch.Generator | None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None and state["loader"] is not None:
        generator.set_state(state["loader"])


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    step: int,
    generator: torch.Generator | None,
    cfg: DictConfig,
) -> None:
    """Everything `train.resume` needs to continue exactly where `epoch`
    ended. `epoch_N.pt` files stay weights-only, so eval, the sweep and
    `init_checkpoint` read them unchanged."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step": step,
            "rng": _rng_state(generator),
            "config": OmegaConf.to_yaml(cfg),
        },
        path,
    )


def _skip_shuffles(loader: DataLoader, epochs: int) -> None:
    """Advance `loader.generator` as `epochs` passes over `loader` would, by
    iterating a stand-in loader over plain indices (no image loading)."""
    stand_in = DataLoader(
        range(len(loader.dataset)), batch_size=loader.batch_size, shuffle=True, generator=loader.generator
    )
    for _ in range(epochs):
        for _ in stand_in:
            pass


def load_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    train_loader: DataLoader,
) -> tuple[int, int]:
    """Load a checkpoint into an already-built (and already binarized) model
    and return `(start_epoch, step)`.

    Accepts a full training state (`last.pt`) for an exact resume, or a
    weights-only `epoch_N.pt` from before full states were saved. For the
    latter, step, LR schedule, annealing progress and shuffle order are
    reconstructed from N; only the optimizer state (Adam moments) and the
    global RNG state start fresh.
    """
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)  # own file: holds RNG/numpy state
    if "optimizer" in state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        _set_rng_state(state["rng"], train_loader.generator)
        return state["epoch"] + 1, state["step"]

    match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"{path} is weights-only and not named epoch_<N>.pt, so its epoch is unknown")
    model.load_state_dict(state)
    start_epoch = int(match.group(1)) + 1
    step = start_epoch * len(train_loader)
    if scheduler is not None:
        scheduler.last_epoch = step - 1
        with warnings.catch_warnings():  # "scheduler.step() before optimizer.step()"
            warnings.simplefilter("ignore", UserWarning)
            scheduler.step()
    if train_loader.generator is not None:
        _skip_shuffles(train_loader, start_epoch)
    log.warning(
        "%s is weights-only: resuming at epoch %d with a fresh optimizer state and global RNG state",
        path, start_epoch,
    )
    return start_epoch, step


def env_overrides() -> list[str]:
    """Machine-specific defaults from `$BCV_OVERRIDES` (whitespace-separated
    `key=value`), e.g. the data location and worker count a Slurm job sets.
    They go before the explicit overrides, so those still win."""
    return os.environ.get("BCV_OVERRIDES", "").split()


def load_config(overrides: list[str] | None = None) -> tuple[DictConfig, str]:
    # Uses Hydra's compose API rather than the `@hydra.main` CLI decorator:
    # hydra-core 1.3.6's argparse-based CLI parser is broken on Python 3.14
    # (a `LazyCompletionHelp` object fails argparse's new `help` string check).
    # `compose`/`initialize` do the same YAML defaults-composition and dotted
    # overrides without going through that code path.
    overrides_list = env_overrides() + (overrides if overrides is not None else sys.argv[1:])

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


def main(cfg: DictConfig | None = None, model_name: str | None = None) -> torch.nn.Module:
    logging.basicConfig(level=logging.INFO)
    if cfg is None:
        cfg, model_name = load_config()
    elif model_name is None:
        # If cfg is provided directly, extract model name from it
        model_name = cfg.model.get("name", "unknown")
    seed = cfg.train.get("seed")
    generator = None
    if seed is not None:
        seed_everything(seed, cfg.train.get("deterministic", False))
        # Own generator, so the shuffle order doesn't depend on how many random
        # numbers model construction happened to draw.
        generator = torch.Generator().manual_seed(seed)
    log.info("seed: %s", seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg.model).to(device)
    train_loader = build_dataloader(cfg, split="train", shuffle=True, generator=generator)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train, len(train_loader))
    binarized = has_quantized_layers(model)
    anneal_steps = int(cfg.train.get("binarize_anneal_fraction", 0.0) * cfg.train.epochs * len(train_loader))
    grad_clip = cfg.train.get("grad_clip")

    start_epoch, step = 0, 0
    resume = cfg.train.get("resume")
    if resume is not None:
        # After build_model_from_config, i.e. after the binarization swap, so
        # learned QuantConv2d parameters (thresholds etc.) load as well.
        start_epoch, step = load_training_state(resume, model, optimizer, scheduler, train_loader)
        log.info("Resumed from %s at epoch %d (step %d)", resume, start_epoch, step)

    # Organize logs and checkpoints by model name with timestamped run directories
    # (SummaryWriter's comment parameter doesn't work when log_dir is specified,
    #  so we need timestamped subdirectories to get proper run names in TensorBoard).
    # A resumed run continues in the directory of the checkpoint it resumes from.
    timestamp = Path(resume).parent.name if resume is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(cfg.train.log_dir) / model_name / timestamp
    checkpoint_dir = Path(resume).parent if resume is not None else Path(cfg.train.checkpoint_dir) / model_name / timestamp

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Logging to: {log_dir}")
        log.info(f"Checkpoints to: {checkpoint_dir}")
    except OSError as e:
        log.error(f"Failed to create directories: {e}")
        raise

    # purge_step hides events a stopped run logged after its last checkpoint.
    writer = SummaryWriter(str(log_dir), purge_step=step if resume is not None else None)

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs - 1}", unit="batch")
        for batch in progress:
            images = {m: t.to(device) for m, t in batch["images"].items()}
            targets = _to_device(batch["targets"], device)

            if binarized and anneal_steps > 0:
                anneal_progress = min(step / anneal_steps, 1.0)
                set_binarization_progress(model, anneal_progress, cfg.train.ede_t_min, cfg.train.ede_t_max)
                writer.add_scalar("train/binarize_progress", anneal_progress, step)

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or float("inf"))
            optimizer.step()
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            if scheduler is not None:
                scheduler.step()
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)

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
        save_training_state(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, step, generator, cfg)

    writer.close()
    return model


if __name__ == "__main__":
    main()

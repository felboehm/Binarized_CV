#!/usr/bin/env python
"""
Train a multispectral YOLO detector model.

Usage:
    python scripts/train_model.py                          # interactive mode
    python scripts/train_model.py --model yolo26           # train yolo26 with defaults
    python scripts/train_model.py model=yolo26             # Hydra override style
    python scripts/train_model.py model=yolo26 train.epochs=20 train.lr=0.0005
    python scripts/train_model.py --model yolo26 --epochs 20 --lr 0.0005
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

MODELS = {
    "yolo26": {
        "description": "YOLO26 baseline (RGB-only)",
        "pretrained": False,
    },
    "yolo26_early_fusion": {
        "description": "YOLO26 early fusion (4-channel concat, RGB+IR)",
        "pretrained": False,
    },
    "yolo26_midfusion": {
        "description": "YOLO26 with mid-fusion (architecture decision pending)",
        "pretrained": False,
    },
    "simple_fusion": {
        "description": "Simple fusion baseline (custom architecture)",
        "pretrained": False,
    },
    "ms_yolov8": {
        "description": "YOLOv8 with multispectral optimizations",
        "pretrained": False,
    },
}

# Map argparse attribute names to Hydra config paths
OVERRIDE_MAPPING = {
    "epochs": "train.epochs",
    "lr": "train.lr",
    "weight_decay": "train.weight_decay",
    "batch_size": "data.batch_size",
    "num_workers": "data.num_workers",
    "log_dir": "train.log_dir",
    "checkpoint_dir": "train.checkpoint_dir",
    "device": "train.device",
}


def check_manifest_exists() -> bool:
    """Check if the manifest file exists."""
    manifest_path = Path("data/processed/manifest.jsonl")
    return manifest_path.exists()


def prompt_model_selection() -> str:
    """Prompt user to choose a model."""
    model_list = list(MODELS.keys())
    print("\nWhich model would you like to train?")
    for i, name in enumerate(model_list, 1):
        info = MODELS[name]
        print(f"  {i}) {name:20} - {info['description']}")

    max_choice = len(model_list)
    while True:
        choice = input(f"Enter choice (1-{max_choice}): ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_list):
                return model_list[choice_idx]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {max_choice}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {max_choice}.")


def prompt_training_params() -> dict:
    """Prompt user for training parameters."""
    params = {}

    print("\n" + "=" * 70)
    print("Training Parameters (press Enter to use defaults)")
    print("=" * 70)

    # Epochs
    while True:
        epochs_input = input("Number of epochs (default: 10): ").strip()
        if epochs_input == "":
            break
        try:
            epochs = int(epochs_input)
            if epochs > 0:
                params["train.epochs"] = str(epochs)
                break
            else:
                print("Epochs must be positive.")
        except ValueError:
            print("Please enter a valid integer.")

    # Learning rate
    while True:
        lr_input = input("Learning rate (default: 0.001): ").strip()
        if lr_input == "":
            break
        try:
            lr = float(lr_input)
            if lr > 0:
                params["train.lr"] = str(lr)
                break
            else:
                print("Learning rate must be positive.")
        except ValueError:
            print("Please enter a valid float.")

    # Batch size
    while True:
        batch_input = input("Batch size (default: 8): ").strip()
        if batch_input == "":
            break
        try:
            batch = int(batch_input)
            if batch > 0:
                params["data.batch_size"] = str(batch)
                break
            else:
                print("Batch size must be positive.")
        except ValueError:
            print("Please enter a valid integer.")

    return params


def build_hydra_overrides(args) -> list[str]:
    """Build Hydra-style config overrides from arguments."""
    overrides = []

    if args.model:
        overrides.append(f"model={args.model}")

    for arg_name, config_path in OVERRIDE_MAPPING.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides.append(f"{config_path}={value}")

    overrides.extend(args.hydra_overrides)
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Train a multispectral YOLO detector model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_model.py
  python scripts/train_model.py --model yolo26 --epochs 20
  python scripts/train_model.py model=yolo26 train.epochs=20
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        choices=list(MODELS.keys()),
        help=f"Model to train: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--log-dir",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Skip interactive prompts and use defaults",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Additional Hydra config overrides (key=value)",
    )

    args = parser.parse_args()

    # Check manifest
    if not check_manifest_exists():
        print("✗ Manifest not found at data/processed/manifest.jsonl")
        print("  Please run: python scripts/build_manifest.py")
        return 1

    print("=" * 70)
    print("Multispectral Model Training")
    print("=" * 70)

    # Determine model
    model = args.model
    if not model and not args.no_prompt:
        model = prompt_model_selection()
    elif not model:
        print("✗ Model not specified. Use --model or run without --no-prompt for interactive mode.")
        return 1

    print(f"\n✓ Selected model: {model}")
    print(f"  {MODELS[model]['description']}")

    # Collect parameters from interactive mode if needed
    if not args.no_prompt and not any(
        getattr(args, arg_name) for arg_name in OVERRIDE_MAPPING.keys()
    ):
        interactive_params = prompt_training_params()
    else:
        interactive_params = {}

    # Build final hydra overrides
    hydra_overrides = [f"model={model}", f"model.name={model}"]

    # Add overrides from command-line arguments
    for arg_name, config_path in OVERRIDE_MAPPING.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            hydra_overrides.append(f"{config_path}={value}")

    # Add overrides from interactive mode
    hydra_overrides.extend(f"{k}={v}" for k, v in interactive_params.items())

    # Add remaining Hydra-style overrides from command line
    hydra_overrides.extend(args.hydra_overrides)

    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    for override in hydra_overrides:
        print(f"  {override}")

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    # Run training via Python module
    cmd = [
        sys.executable,
        "-m",
        "binarized_cv.train.train",
    ] + hydra_overrides

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✓ Training completed successfully!")
            print("\nNext steps:")
            print(f"  1. Evaluate the model: python -m binarized_cv.eval model={model}")
            print(f"  2. View results: tensorboard --logdir runs/tensorboard/{model}")
            print("=" * 70)
            return 0
        else:
            print("\n" + "=" * 70)
            print("✗ Training failed. Check the output above for details.")
            print("=" * 70)
            return 1
    except Exception as e:
        print(f"\n✗ Error running training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

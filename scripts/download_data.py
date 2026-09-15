#!/usr/bin/env python
"""
Download TRGB and WiSARD datasets from Google Drive.

Usage:
    python scripts/download_data.py                # download both, prompt for WiSARD variant
    python scripts/download_data.py --wisard full  # download WiSARD full size
    python scripts/download_data.py --wisard sample # download WiSARD sample only
    python scripts/download_data.py --trgb-only    # download only TRGB
    python scripts/download_data.py --wisard-only  # download only WiSARD
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Google Drive file IDs and expected sizes
DATASETS = {
    "trgb": {
        "file_id": "1-d_0z-cx3gS_vBKKesW4Hr9P0T7J-ALf",
        "filename": "trgb_dataset.zip",
        "extract_dir": "data/raw/trgb",
        "check_path": "data/raw/trgb/trgb_dataset",
        "description": "TRGB (4,772 RGB+IR pairs)",
    },
    "wisard_full": {
        "file_id": "1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht",
        "filename": "wisard_full.zip",
        "extract_dir": "data/raw/wisard",
        "check_path": "data/raw/wisard",
        "description": "WiSARD full (~40.5GB, 15,453 pairs)",
    },
    "wisard_sample": {
        "file_id": "1uSgMXuZGVCrWM_151UcykyHejxxaVHOo",
        "filename": "wisard_sample.zip",
        "extract_dir": "data/raw/wisard",
        "check_path": "data/raw/wisard",
        "description": "WiSARD sample (~972MB, 263 pairs from 1 flight)",
    },
}


def check_gdown_installed() -> bool:
    """Check if gdown is installed."""
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def install_gdown() -> bool:
    """Try to install gdown."""
    print("gdown is not installed. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install gdown. Please install manually: pip install gdown")
        return False


def dataset_exists(check_path: str) -> bool:
    """Check if dataset directory already exists and contains data."""
    path = Path(check_path)
    if not path.exists():
        return False
    # Check if directory is non-empty
    return any(path.iterdir())


def download_from_gdrive(file_id: str, output_path: str, quiet: bool = False) -> bool:
    """Download a file from Google Drive using gdown."""
    try:
        import gdown

        print(f"Downloading to {output_path}...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            output_path,
            quiet=quiet,
            resume=True,
        )
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str) -> bool:
    """Extract a zip file to the target directory."""
    try:
        zip_path_obj = Path(zip_path)
        if not zip_path_obj.exists():
            print(f"Zip file not found: {zip_path}")
            return False

        extract_dir_obj = Path(extract_dir)
        extract_dir_obj.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {zip_path} to {extract_dir}...")
        shutil.unpack_archive(zip_path, extract_dir)
        print(f"Extraction complete.")

        # Clean up zip file
        zip_path_obj.unlink()
        print(f"Cleaned up {zip_path}")
        return True
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False


def download_dataset(dataset_key: str) -> bool:
    """Download and extract a single dataset."""
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        return False

    dataset_info = DATASETS[dataset_key]
    check_path = dataset_info["check_path"]
    filename = dataset_info["filename"]
    extract_dir = dataset_info["extract_dir"]
    description = dataset_info["description"]
    file_id = dataset_info["file_id"]

    # Check if already exists
    if dataset_exists(check_path):
        print(f"✓ {description} already exists at {check_path}")
        return True

    print(f"\nDownloading {description}...")
    print(f"This will be saved to {extract_dir}/")

    # Download
    if not download_from_gdrive(file_id, filename):
        print(f"✗ Failed to download {description}")
        return False

    # Extract
    if not extract_zip(filename, extract_dir):
        print(f"✗ Failed to extract {description}")
        return False

    print(f"✓ Successfully downloaded and extracted {description}")
    return True


def prompt_wisard_variant() -> str:
    """Prompt user to choose WiSARD variant."""
    print("\nWhich WiSARD variant would you like?")
    print("  1) Sample (~972MB, 263 pairs, 1 flight) — good for testing")
    print("  2) Full (~40.5GB, 15,453 pairs) — full dataset")
    print("  0) Skip WiSARD")

    while True:
        choice = input("Enter choice (0-2): ").strip()
        if choice == "1":
            return "wisard_sample"
        elif choice == "2":
            return "wisard_full"
        elif choice == "0":
            return None
        else:
            print("Invalid choice. Please enter 0, 1, or 2.")


def main():
    parser = argparse.ArgumentParser(
        description="Download TRGB and WiSARD datasets from Google Drive"
    )
    parser.add_argument(
        "--trgb-only",
        action="store_true",
        help="Download only TRGB dataset",
    )
    parser.add_argument(
        "--wisard-only",
        action="store_true",
        help="Download only WiSARD dataset",
    )
    parser.add_argument(
        "--wisard",
        choices=["full", "sample"],
        help="Specify WiSARD variant (full or sample)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Download defaults (TRGB + WiSARD sample) without prompting",
    )

    args = parser.parse_args()

    # Check for gdown
    if not check_gdown_installed():
        if not install_gdown():
            print("Cannot proceed without gdown. Exiting.")
            return 1

    print("=" * 70)
    print("Dataset Download Script for Binarized Multispectral YOLO26")
    print("=" * 70)

    # Determine what to download
    downloads = []

    if not args.wisard_only:
        downloads.append("trgb")

    if not args.trgb_only:
        if args.wisard:
            wisard_variant = f"wisard_{args.wisard}"
            downloads.append(wisard_variant)
        elif args.no_prompt:
            downloads.append("wisard_sample")
        else:
            wisard_variant = prompt_wisard_variant()
            if wisard_variant:
                downloads.append(wisard_variant)

    if not downloads:
        print("No datasets selected for download.")
        return 0

    # Download all selected datasets
    failed = []
    for dataset_key in downloads:
        if not download_dataset(dataset_key):
            failed.append(dataset_key)

    print("\n" + "=" * 70)
    if failed:
        print(f"Failed to download: {', '.join(failed)}")
        return 1
    else:
        print("All datasets downloaded successfully!")
        print("\nNext steps:")
        print("  1. Build the manifest: python scripts/build_manifest.py")
        print("  2. Train a model: python -m binarized_cv.train.train model=yolo26")
        return 0


if __name__ == "__main__":
    sys.exit(main())

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
        "extract_dir": "data/raw/trgb/trgb_dataset",
        "check_path": "data/raw/trgb/trgb_dataset/train",
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
    """Check if dataset directory already exists and contains real data (not just .gitkeep)."""
    path = Path(check_path)
    if not path.exists():
        return False
    # Check for actual files/dirs (ignore .gitkeep placeholders)
    for item in path.iterdir():
        if item.name != ".gitkeep":
            return True
    return False


def get_wisard_variant_present(wisard_dir: str = "data/raw/wisard") -> Optional[str]:
    """
    Detect which WiSARD variant is present by checking the metadata file.
    Falls back to inferring from structure (sample has 1 flight, full has many).
    Each flight has _VIS_XXXX and _IR_XXXX folders, so count unique flight base names.
    Returns 'sample', 'full', or None if no variant is detected.
    """
    metadata_file = Path(wisard_dir) / ".wisard_variant"
    if metadata_file.exists():
        return metadata_file.read_text().strip()

    # Fallback: infer from flight folders
    # WiSARD has folders like "210417_MtErie_Enterprise_VIS_0003" and "210417_MtErie_Enterprise_IR_0004"
    # Each flight generates 2 folders (VIS + IR), so count unique flight base names
    wisard_path = Path(wisard_dir)
    if wisard_path.exists():
        flight_folders = [d for d in wisard_path.iterdir()
                         if d.is_dir() and d.name not in [".gitkeep", "__MACOSX"]]
        if flight_folders:
            # Extract base flight name by removing _VIS_XXXX or _IR_XXXX suffix
            flight_bases = set()
            for folder in flight_folders:
                name = folder.name
                if "_VIS_" in name:
                    base = name.rsplit("_VIS_", 1)[0]
                    flight_bases.add(base)
                elif "_IR_" in name:
                    base = name.rsplit("_IR_", 1)[0]
                    flight_bases.add(base)

            if flight_bases:
                # Sample has 1 flight, full has ~50+ flights
                return "sample" if len(flight_bases) == 1 else "full"

    return None


def set_wisard_variant(variant: str, wisard_dir: str = "data/raw/wisard") -> None:
    """Record which WiSARD variant was downloaded."""
    wisard_path = Path(wisard_dir)
    wisard_path.mkdir(parents=True, exist_ok=True)
    metadata_file = wisard_path / ".wisard_variant"
    metadata_file.write_text(variant)


def delete_wisard_data(wisard_dir: str = "data/raw/wisard") -> bool:
    """Delete WiSARD data but preserve .gitkeep."""
    wisard_path = Path(wisard_dir)
    if not wisard_path.exists():
        return True

    try:
        for item in wisard_path.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Remove metadata file so next detection will be clean
        metadata_file = wisard_path / ".wisard_variant"
        if metadata_file.exists():
            metadata_file.unlink()

        return True
    except Exception as e:
        print(f"Error deleting WiSARD data: {e}")
        return False


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

        # Clean up macOS metadata folder (safe to remove)
        macosx_dir = extract_dir_obj / "__MACOSX"
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)
            print(f"Removed macOS metadata folder (__MACOSX)")

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

    # Special handling for WiSARD variants
    if dataset_key.startswith("wisard_"):
        variant_name = dataset_key.split("_")[1]  # "sample" or "full"
        present_variant = get_wisard_variant_present(check_path)

        if present_variant:
            if present_variant == variant_name:
                print(f"✓ {description} already exists at {check_path}")
                return True
            else:
                print(f"Found WiSARD {present_variant} at {check_path}, but {variant_name} was requested.")
                response = input(f"Delete {present_variant} and download {variant_name} instead? (y/n): ").strip().lower()
                if response != 'y':
                    print(f"Skipping {description}")
                    return False

                print(f"Deleting existing WiSARD {present_variant} data...")
                if not delete_wisard_data(check_path):
                    print(f"✗ Failed to delete existing WiSARD data")
                    return False
                print(f"Deleted WiSARD data (preserved .gitkeep)")
    elif dataset_exists(check_path):
        # Non-WiSARD datasets: simple existence check
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

    # Record which WiSARD variant was downloaded
    if dataset_key.startswith("wisard_"):
        variant_name = dataset_key.split("_")[1]
        set_wisard_variant(variant_name, check_path)

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

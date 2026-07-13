#!/usr/bin/env python3
"""
JWST_0081_MOMZ14_SAVE_TO_GOOGLE_DRIVE.py

Mount Google Drive, create:
My Drive/Colab Notebooks/JWST/MoM-14

Download the four real MoM-z14 JWST/NIRCam channels when they are not already
present in the Colab runtime, then copy the FITS images, PNG products, and CSV
manifest into the permanent Drive folder. No AI imagery is used.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

VERSION = "JWST_0081"
RUNTIME_ROOT = Path("/content/JWST_OUTPUT")
DRIVE_MOUNT = Path("/content/drive")
DRIVE_DIR = DRIVE_MOUNT / "MyDrive" / "Colab Notebooks" / "JWST" / "MoM-14"
DOWNLOADER = Path("/content/JWST_0080_MOMZ14_DOWNLOAD_AND_COMBINE.py")
DOWNLOADER_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    "JWST_0080_MOMZ14_DOWNLOAD_AND_COMBINE.py"
)

EXPECTED_FITS = [
    "JWST_0080_MOMZ14_F115W.fits",
    "JWST_0080_MOMZ14_F150W.fits",
    "JWST_0080_MOMZ14_F277W.fits",
    "JWST_0080_MOMZ14_F444W.fits",
]


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Run this script in Google Colab.") from exc
    drive.mount(str(DRIVE_MOUNT), force_remount=False)


def channels_exist() -> bool:
    fits_dir = RUNTIME_ROOT / "FITS"
    return all((fits_dir / name).exists() for name in EXPECTED_FITS)


def download_products_if_needed() -> None:
    if channels_exist():
        print("Existing MoM-z14 FITS channels found in the Colab runtime.")
        return

    print("Downloading the four real MoM-z14 JWST/NIRCam channels...")
    urllib.request.urlretrieve(DOWNLOADER_URL, DOWNLOADER)
    subprocess.check_call([sys.executable, str(DOWNLOADER)])

    if not channels_exist():
        missing = [
            name for name in EXPECTED_FITS
            if not (RUNTIME_ROOT / "FITS" / name).exists()
        ]
        raise RuntimeError(f"Download finished, but these FITS files are missing: {missing}")


def collect_products() -> list[Path]:
    products: list[Path] = []
    patterns = {
        "FITS": "JWST_0080_MOMZ14_*.fits",
        "PNG": "JWST_0080_MOMZ14_*.png",
        "CSV": "JWST_0080_MOMZ14_*.csv",
    }
    for category, pattern in patterns.items():
        products.extend(sorted((RUNTIME_ROOT / category).glob(pattern)))
    if not products:
        raise RuntimeError("No JWST_0080 MoM-z14 products were found in /content/JWST_OUTPUT.")
    return products


def copy_to_drive(products: list[Path]) -> list[tuple[Path, Path]]:
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[Path, Path]] = []
    for source in products:
        destination = DRIVE_DIR / source.name
        shutil.copy2(source, destination)
        copied.append((source, destination))
    return copied


def write_inventory(copied: list[tuple[Path, Path]]) -> Path:
    inventory = DRIVE_DIR / "MoM-14_FILE_INVENTORY.txt"
    lines = [
        f"CODE OUTPUT: {VERSION}",
        "Target: MoM-z14 four-filter JWST image archive",
        f"Drive folder: {DRIVE_DIR}",
        f"Created UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]
    for _, destination in copied:
        lines.append(f"{destination.stat().st_size:>12} bytes  {destination.name}")
    lines.extend(["", f"# {VERSION}"])
    inventory.write_text("\n".join(lines), encoding="utf-8")
    return inventory


def main() -> None:
    mount_drive()
    download_products_if_needed()
    products = collect_products()
    copied = copy_to_drive(products)
    inventory = write_inventory(copied)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"DRIVE FOLDER    {DRIVE_DIR}")
    print(f"FILES COPIED    {len(copied)}")
    for _, destination in copied:
        print(f"SAVED           {destination.name}")
    print(f"INVENTORY       {inventory}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

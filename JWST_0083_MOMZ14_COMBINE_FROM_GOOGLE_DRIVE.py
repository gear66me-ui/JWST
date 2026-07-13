#!/usr/bin/env python3
"""
JWST_0083_MOMZ14_COMBINE_FROM_GOOGLE_DRIVE.py

Read the four real MoM-z14 JWST/NIRCam FITS images from:
/content/drive/MyDrive/Colab Notebooks/JWST/MoM-14

Register them on the F444W WCS, apply a robust background subtraction and
asinh stretch, combine them into a four-filter false-color composite, and save
all outputs back into the same Google Drive folder.

No file picker, no re-download, and no AI-generated imagery.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "astropy": "astropy",
        "reproject": "reproject",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

VERSION = "JWST_0083"
DRIVE_MOUNT = Path("/content/drive")
DATA_DIR = DRIVE_MOUNT / "MyDrive" / "Colab Notebooks" / "JWST" / "MoM-14"
FILTERS = [
    {"name": "F115W", "lambda_um": 1.15, "file": "MoM-14_F115W.fits"},
    {"name": "F150W", "lambda_um": 1.50, "file": "MoM-14_F150W.fits"},
    {"name": "F277W", "lambda_um": 2.77, "file": "MoM-14_F277W.fits"},
    {"name": "F444W", "lambda_um": 4.44, "file": "MoM-14_F444W.fits"},
]
DISPLAY_COLORS = np.array([
    [0.04, 0.24, 1.00],
    [0.00, 0.95, 0.72],
    [1.00, 0.68, 0.05],
    [1.00, 0.04, 0.00],
], dtype=float)
DISPLAY_NAMES = ["blue", "cyan-green", "yellow-orange", "red"]
EPS = 1.0e-12


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Run this script inside Google Colab.") from exc
    drive.mount(str(DRIVE_MOUNT), force_remount=False)


def select_image_hdu(hdul: fits.HDUList) -> int:
    for wanted in ("SCI", "PRIMARY"):
        for index, hdu in enumerate(hdul):
            if hdu.name.upper() == wanted and hdu.data is not None:
                data = np.asarray(hdu.data)
                if data.ndim >= 2:
                    return index
    for index, hdu in enumerate(hdul):
        if hdu.data is not None and np.asarray(hdu.data).ndim >= 2:
            return index
    raise ValueError("No 2-D science image found in FITS file.")


def load_channel(path: Path) -> tuple[np.ndarray, fits.Header, WCS]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required FITS image: {path}")
    with fits.open(path, memmap=False) as hdul:
        index = select_image_hdu(hdul)
        data = np.asarray(hdul[index].data, dtype=float).squeeze()
        header = hdul[index].header.copy()
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D image in {path.name}; received {data.shape}")
    return data, header, WCS(header)


def register_to_reference(
    data: np.ndarray,
    source_wcs: WCS,
    reference_header: fits.Header,
    reference_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    projected, footprint = reproject_interp(
        (data, source_wcs),
        reference_header,
        shape_out=reference_shape,
        order="bilinear",
        return_footprint=True,
    )
    coverage = float(np.mean(footprint > 0.0))
    return projected, coverage


def robust_stretch(data: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("Registered channel has no finite pixels.")
    filled = np.where(finite, data, np.nanmedian(data[finite]))

    height, width = filled.shape
    border = max(3, int(round(0.10 * min(height, width))))
    edge = np.zeros_like(filled, dtype=bool)
    edge[:border, :] = True
    edge[-border:, :] = True
    edge[:, :border] = True
    edge[:, -border:] = True

    background = float(np.median(filled[edge]))
    signal = filled - background
    values = signal[np.isfinite(signal)]
    low, high = np.percentile(values, [1.0, 99.7])
    low = float(low)
    high = max(float(high), low + EPS)

    normalized = np.clip((signal - low) / (high - low), 0.0, 1.0)
    stretched = np.arcsinh(10.0 * normalized) / np.arcsinh(10.0)
    return np.nan_to_num(stretched), {
        "background": background,
        "stretch_low": low,
        "stretch_high": high,
    }


def combine_channels(channels: list[np.ndarray]) -> np.ndarray:
    rgb = np.zeros((*channels[0].shape, 3), dtype=float)
    for channel, display_color in zip(channels, DISPLAY_COLORS):
        rgb += channel[..., None] * display_color
    scale = max(float(np.percentile(rgb[np.isfinite(rgb)], 99.8)), EPS)
    rgb = np.clip(rgb / scale, 0.0, 1.0)
    return rgb ** 0.90


def save_dashboard(records: list[dict], channels: list[np.ndarray], rgb: np.ndarray) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    for index, record in enumerate(records):
        row, column = divmod(index, 2)
        axes[row, column].imshow(
            channels[index],
            cmap="gray",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[row, column].set_title(
            f"{record['filter']}  {record['lambda_um']:.2f} µm\n"
            f"mapped to {DISPLAY_NAMES[index]}"
        )
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])

    axes[0, 2].imshow(rgb, origin="lower", interpolation="nearest")
    axes[0, 2].set_title("Four-filter false-color composite")

    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    axes[1, 2].imshow(
        luminance,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1, 2].set_title("Composite luminance")

    for axis in (axes[0, 2], axes[1, 2]):
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        "MoM-z14 — four real JWST/NIRCam channels\n"
        "WCS registration to F444W; display colors are representational",
        fontsize=18,
    )

    output = DATA_DIR / f"{VERSION}_MOMZ14_CHANNELS_AND_COMPOSITE.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def main() -> None:
    mount_drive()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    loaded = []
    for entry in FILTERS:
        path = DATA_DIR / entry["file"]
        data, header, wcs = load_channel(path)
        loaded.append({**entry, "path": path, "data": data, "header": header, "wcs": wcs})

    reference = loaded[-1]
    reference_shape = reference["data"].shape
    reference_header = reference["header"]

    records: list[dict] = []
    stretched_channels: list[np.ndarray] = []

    for index, item in enumerate(loaded):
        if index == len(loaded) - 1:
            registered = item["data"]
            coverage = 1.0
        else:
            registered, coverage = register_to_reference(
                item["data"],
                item["wcs"],
                reference_header,
                reference_shape,
            )

        stretched, stats = robust_stretch(registered)
        stretched_channels.append(stretched)
        records.append({
            "filter": item["name"],
            "lambda_um": item["lambda_um"],
            "fits_path": str(item["path"]),
            "input_shape_y": item["data"].shape[0],
            "input_shape_x": item["data"].shape[1],
            "registered_shape_y": registered.shape[0],
            "registered_shape_x": registered.shape[1],
            "wcs_coverage_fraction": coverage,
            "display_color": DISPLAY_NAMES[index],
            **stats,
        })

    rgb = combine_channels(stretched_channels)
    composite_path = DATA_DIR / f"{VERSION}_MOMZ14_FOUR_FILTER_COMPOSITE.png"
    plt.imsave(composite_path, rgb, origin="lower", dpi=360)

    dashboard_path = save_dashboard(records, stretched_channels, rgb)
    manifest_path = DATA_DIR / f"{VERSION}_MOMZ14_COMBINATION_MANIFEST.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"SOURCE FOLDER   {DATA_DIR}")
    print("INPUT FILTERS   F115W  F150W  F277W  F444W")
    print("REGISTRATION    WCS reprojection to F444W")
    print("COMBINATION     F115W=blue, F150W=cyan-green, F277W=yellow-orange, F444W=red")
    print(f"COMPOSITE PNG   {composite_path}")
    print(f"DASHBOARD PNG   {dashboard_path}")
    print(f"MANIFEST CSV    {manifest_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

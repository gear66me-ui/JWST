#!/usr/bin/env python3
"""
JWST_0082_MOMZ14_COMPOSITE_FROM_DRIVE.py

Load the four existing MoM-z14 JWST/NIRCam FITS files from Google Drive,
validate them, align them to F444W with FFT phase correlation, apply a robust
asinh stretch, and build a first-pass four-channel false-color composite.

No AI imagery is used. All outputs come from the stored scientific FITS data.
"""
from __future__ import annotations

import importlib.util
import re
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
        "scipy": "scipy",
        "skimage": "scikit-image",
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
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

VERSION = "JWST_0082"
FILTER_ORDER = ["F115W", "F150W", "F277W", "F444W"]
WAVELENGTH_UM = {"F115W": 1.15, "F150W": 1.50, "F277W": 2.77, "F444W": 4.44}
DISPLAY_NAMES = ["blue", "cyan-green", "yellow-orange", "red"]
DISPLAY_COLORS = np.array([
    [0.05, 0.25, 1.00],
    [0.00, 0.95, 0.72],
    [1.00, 0.70, 0.05],
    [1.00, 0.05, 0.00],
], dtype=float)
EPS = 1.0e-12

DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive" / "Colab Notebooks" / "JWST" / "MoM-14"
OUTPUT_ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = OUTPUT_ROOT / "PNG"
CSV_DIR = OUTPUT_ROOT / "CSV"
for directory in (PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def mount_drive() -> None:
    if DRIVE_ROOT.exists():
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Run this script in Google Colab.") from exc
    drive.mount(str(DRIVE_MOUNT), force_remount=False)


def locate_filter_files() -> dict[str, Path]:
    if not DRIVE_ROOT.exists():
        raise RuntimeError(f"Drive folder not found: {DRIVE_ROOT}")
    fits_files = sorted(
        path for path in DRIVE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".fits", ".fit", ".fts"}
    )
    found: dict[str, Path] = {}
    for filter_name in FILTER_ORDER:
        matches = [path for path in fits_files if filter_name.lower() in path.name.lower()]
        if not matches:
            raise RuntimeError(
                f"Missing {filter_name} FITS file in {DRIVE_ROOT}. "
                f"Available FITS files: {[p.name for p in fits_files]}"
            )
        found[filter_name] = max(matches, key=lambda p: p.stat().st_mtime)
    return found


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
    raise ValueError("No two-dimensional science image found in FITS file")


def load_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        index = select_image_hdu(hdul)
        data = np.asarray(hdul[index].data, dtype=float).squeeze()
        header = hdul[index].header.copy()
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D data in {path.name}; received {data.shape}")
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError(f"No finite pixels in {path.name}")
    fill_value = float(np.nanmedian(data[finite]))
    data = np.where(finite, data, fill_value)
    return data, header


def center_crop(data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y0 = max((data.shape[0] - height) // 2, 0)
    x0 = max((data.shape[1] - width) // 2, 0)
    return data[y0:y0 + height, x0:x0 + width]


def align_channels(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    common_shape = (
        min(array.shape[0] for array in arrays),
        min(array.shape[1] for array in arrays),
    )
    cropped = [center_crop(array, common_shape) for array in arrays]
    reference = cropped[-1]
    reference_centered = reference - np.median(reference)
    aligned: list[np.ndarray] = []
    shifts: list[tuple[float, float]] = []
    for index, image in enumerate(cropped):
        if index == len(cropped) - 1:
            aligned.append(image)
            shifts.append((0.0, 0.0))
            continue
        shift, error, _ = phase_cross_correlation(
            reference_centered,
            image - np.median(image),
            upsample_factor=20,
        )
        registered = ndi_shift(
            image,
            shift,
            order=1,
            mode="constant",
            cval=float(np.median(image)),
        )
        aligned.append(registered)
        shifts.append((float(shift[0]), float(shift[1])))
    return aligned, shifts


def robust_stretch(data: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    height, width = data.shape
    border = max(3, int(round(0.08 * min(height, width))))
    mask = np.zeros_like(data, dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    background = float(np.median(data[mask]))
    work = data - background
    finite = work[np.isfinite(work)]
    low, high = np.percentile(finite, [1.0, 99.7])
    high = max(float(high), float(low) + EPS)
    normalized = np.clip((work - low) / (high - low), 0.0, 1.0)
    stretched = np.arcsinh(8.0 * normalized) / np.arcsinh(8.0)
    return np.nan_to_num(stretched), {
        "background": background,
        "stretch_low": float(low),
        "stretch_high": float(high),
    }


def combine(channels: list[np.ndarray]) -> np.ndarray:
    rgb = np.zeros((*channels[0].shape, 3), dtype=float)
    for channel, color in zip(channels, DISPLAY_COLORS):
        rgb += channel[..., None] * color
    scale = max(float(np.percentile(rgb[np.isfinite(rgb)], 99.8)), EPS)
    rgb = np.clip(rgb / scale, 0.0, 1.0)
    return rgb ** 0.92


def make_dashboard(records: list[dict], channels: list[np.ndarray], rgb: np.ndarray) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for index in range(4):
        row, column = divmod(index, 2)
        axes[row, column].imshow(
            channels[index], cmap="gray", origin="lower",
            vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        axes[row, column].set_title(
            f"{records[index]['filter']}  {records[index]['wavelength_um']:.2f} µm\n"
            f"mapped to {DISPLAY_NAMES[index]}"
        )
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])

    axes[0, 2].imshow(rgb, origin="lower", interpolation="nearest")
    axes[0, 2].set_title("First-pass four-channel composite")
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    axes[1, 2].imshow(luminance, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("Composite luminance")
    for axis in (axes[0, 2], axes[1, 2]):
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        "MoM-z14 — composite from Google Drive FITS files\n"
        "FFT registration to F444W; channel colors are representational",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_MOMZ14_COMPOSITE_DASHBOARD.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_to_drive(paths: list[Path]) -> list[Path]:
    saved: list[Path] = []
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for source in paths:
        destination = DRIVE_ROOT / source.name
        destination.write_bytes(source.read_bytes())
        saved.append(destination)
    return saved


def main() -> None:
    mount_drive()
    file_map = locate_filter_files()
    records: list[dict] = []
    arrays: list[np.ndarray] = []

    for filter_name in FILTER_ORDER:
        path = file_map[filter_name]
        data, header = load_fits(path)
        arrays.append(data)
        records.append({
            "filter": filter_name,
            "wavelength_um": WAVELENGTH_UM[filter_name],
            "file": str(path),
            "shape_y": data.shape[0],
            "shape_x": data.shape[1],
            "bunit": header.get("BUNIT", "unknown"),
        })

    aligned, shifts = align_channels(arrays)
    stretched: list[np.ndarray] = []
    stretch_stats: list[dict[str, float]] = []
    for image in aligned:
        channel, stats = robust_stretch(image)
        stretched.append(channel)
        stretch_stats.append(stats)

    rgb = combine(stretched)
    dashboard_path = make_dashboard(records, stretched, rgb)
    composite_path = PNG_DIR / f"{VERSION}_MOMZ14_COMPOSITE_ONLY.png"
    plt.imsave(composite_path, rgb, origin="lower", dpi=360)

    rows = []
    for index, (record, shift, stats) in enumerate(zip(records, shifts, stretch_stats)):
        rows.append({
            "order_short_to_long": index + 1,
            "filter": record["filter"],
            "central_wavelength_um": record["wavelength_um"],
            "display_color": DISPLAY_NAMES[index],
            "shift_y_pixels": shift[0],
            "shift_x_pixels": shift[1],
            **record,
            **stats,
        })
    manifest_path = CSV_DIR / f"{VERSION}_MOMZ14_COMPOSITE_MANIFEST.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    drive_outputs = save_to_drive([dashboard_path, composite_path, manifest_path])

    print(f"CODE OUTPUT: {VERSION}")
    print(f"INPUT FOLDER     {DRIVE_ROOT}")
    print("FILTERS          F115W  F150W  F277W  F444W")
    print("REFERENCE        F444W")
    print("ALIGNMENT        FFT phase correlation")
    for index, row in enumerate(rows):
        print(
            f"CHANNEL {index + 1}        {row['filter']:<5}  "
            f"shift=({row['shift_y_pixels']:+.3f}, {row['shift_x_pixels']:+.3f}) px"
        )
    print(f"DASHBOARD PNG    {dashboard_path}")
    print(f"COMPOSITE PNG    {composite_path}")
    print(f"MANIFEST CSV     {manifest_path}")
    for path in drive_outputs:
        print(f"DRIVE COPY       {path}")
    print(f"Timestamp        {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

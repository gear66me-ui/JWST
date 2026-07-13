#!/usr/bin/env python3
"""
JWST_0080_MOMZ14_DOWNLOAD_AND_COMBINE.py

Download four genuine JWST/NIRCam MoM-z14 FITS cutouts from the DAWN JWST
Archive and combine them into a registered four-channel false-color composite.
All imagery is generated from downloaded scientific data with NumPy/Matplotlib.
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
        "requests": "requests",
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
import requests
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION = "JWST_0080"
RA_DEG = 150.0933255
DEC_DEG = 2.2731627
CUTOUT_ARCSEC = 8.0
EPS = 1.0e-12
API_BASE = "https://grizli-cutout.herokuapp.com/thumb"

FILTERS = [
    {"name": "F115W", "api": "f115w-clear", "lambda_um": 1.15},
    {"name": "F150W", "api": "f150w-clear", "lambda_um": 1.50},
    {"name": "F277W", "api": "f277w-clear", "lambda_um": 2.77},
    {"name": "F444W", "api": "f444w-clear", "lambda_um": 4.44},
]

DISPLAY_COLORS = np.array([
    [0.05, 0.25, 1.00],
    [0.00, 0.95, 0.75],
    [1.00, 0.72, 0.05],
    [1.00, 0.05, 0.00],
], dtype=float)
DISPLAY_NAMES = ["blue", "cyan-green", "yellow-orange", "red"]

ROOT = Path("/content/JWST_OUTPUT")
FITS_DIR = ROOT / "FITS"
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (FITS_DIR, PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": f"{VERSION}/Google-Colab JWST workflow"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


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


def download_channel(session: requests.Session, entry: dict) -> tuple[Path, str, int]:
    output = FITS_DIR / f"{VERSION}_MOMZ14_{entry['name']}.fits"
    params = {
        "ra": f"{RA_DEG:.7f}",
        "dec": f"{DEC_DEG:.7f}",
        "filters": entry["api"],
        "size": f"{CUTOUT_ARCSEC:.2f}",
        "output": "fits",
    }
    response = session.get(API_BASE, params=params, timeout=(30, 240))
    response.raise_for_status()
    payload = response.content
    if len(payload) < 2880:
        preview = payload[:300].decode("utf-8", errors="replace")
        raise RuntimeError(f"Archive response was not a valid FITS payload: {preview}")
    output.write_bytes(payload)
    with fits.open(output, memmap=False) as hdul:
        select_image_hdu(hdul)
    return output, response.url, len(payload)


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
    data = np.where(finite, data, np.nanmedian(data[finite]))
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
        shift, _, _ = phase_cross_correlation(
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


def stretch_channel(data: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
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


def combine_channels(channels: list[np.ndarray]) -> np.ndarray:
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
            channels[index],
            cmap="gray",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[row, column].set_title(
            f"{records[index]['filter']}  {records[index]['lambda_um']:.2f} µm\n"
            f"mapped to {DISPLAY_NAMES[index]}"
        )
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])

    axes[0, 2].imshow(rgb, origin="lower", interpolation="nearest")
    axes[0, 2].set_title("Four-filter false-color composite")
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    axes[1, 2].imshow(luminance, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("Composite luminance")
    for axis in (axes[0, 2], axes[1, 2]):
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        "MoM-z14 — four genuine JWST/NIRCam channels\n"
        "Downloaded automatically from calibrated DAWN JWST Archive mosaics",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_MOMZ14_DOWNLOAD_AND_COMBINE_DASHBOARD.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def main() -> None:
    session = build_session()
    records: list[dict] = []
    arrays: list[np.ndarray] = []

    for entry in FILTERS:
        path, source_url, byte_count = download_channel(session, entry)
        data, header = load_fits(path)
        records.append({
            "filter": entry["name"],
            "lambda_um": entry["lambda_um"],
            "fits_path": str(path),
            "source_url": source_url,
            "download_bytes": byte_count,
            "shape_y": data.shape[0],
            "shape_x": data.shape[1],
            "bunit": header.get("BUNIT", "unknown"),
        })
        arrays.append(data)

    aligned, shifts = align_channels(arrays)
    stretched: list[np.ndarray] = []
    stretch_stats: list[dict[str, float]] = []
    for image in aligned:
        channel, stats = stretch_channel(image)
        stretched.append(channel)
        stretch_stats.append(stats)

    rgb = combine_channels(stretched)
    dashboard_path = make_dashboard(records, stretched, rgb)
    composite_path = PNG_DIR / f"{VERSION}_MOMZ14_COMPOSITE_ONLY.png"
    plt.imsave(composite_path, rgb, origin="lower", dpi=360)

    manifest_rows = []
    for index, (record, shift, stats) in enumerate(zip(records, shifts, stretch_stats)):
        manifest_rows.append({
            "order_short_to_long": index + 1,
            "filter": record["filter"],
            "central_wavelength_um": record["lambda_um"],
            "display_color": DISPLAY_NAMES[index],
            "shift_y_pixels": shift[0],
            "shift_x_pixels": shift[1],
            **record,
            **stats,
        })
    manifest_path = CSV_DIR / f"{VERSION}_MOMZ14_FOUR_FILTER_MANIFEST.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET          MoM-z14  RA={RA_DEG:.7f} deg  Dec={DEC_DEG:.7f} deg")
    print("FILTERS         F115W  F150W  F277W  F444W")
    print("SOURCE          DAWN JWST Archive calibrated FITS mosaics")
    print("ALIGNMENT       FFT phase correlation to F444W")
    for index, record in enumerate(records):
        print(f"CHANNEL {index + 1}       {record['filter']:<5} {record['lambda_um']:.2f} um -> {DISPLAY_NAMES[index]}")
        print(f"FITS            {record['fits_path']}")
    print(f"DASHBOARD PNG   {dashboard_path}")
    print(f"COMPOSITE PNG   {composite_path}")
    print(f"MANIFEST CSV    {manifest_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

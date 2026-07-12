#!/usr/bin/env python3
"""
JWST_0048_MOMZ14_4FILTER_FITS.py

Download genuine JWST imaging cutouts of MoM-z14 from the DAWN JWST Archive
in the four COSMOS-Web NIRCam bands and display the channels separately.

These are science-ready calibrated FITS mosaics, not AI images and not raw
uncalibrated detector ramps.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def ensure_packages() -> None:
    required = {
        "requests": "requests",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "astropy": "astropy",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing]
        )


ensure_packages()

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

VERSION = "JWST_0048"
RA_DEG = 150.0933255
DEC_DEG = 2.2731627
CUTOUT_ARCSEC = 8.0
ZOOM_ARCSEC = 1.4

FILTERS = [
    {"name": "F115W", "api": "f115w-clear", "lambda_um": 1.15,
     "published_flux_njy": -1.0, "published_sigma_njy": 5.0},
    {"name": "F150W", "api": "f150w-clear", "lambda_um": 1.50,
     "published_flux_njy": 4.0, "published_sigma_njy": 4.0},
    {"name": "F277W", "api": "f277w-clear", "lambda_um": 2.77,
     "published_flux_njy": 22.0, "published_sigma_njy": 2.0},
    {"name": "F444W", "api": "f444w-clear", "lambda_um": 4.44,
     "published_flux_njy": 13.0, "published_sigma_njy": 3.0},
]

ROOT = Path("/content/JWST_OUTPUT")
FITS_DIR = ROOT / "FITS"
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (FITS_DIR, PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

API_BASE = "https://grizli-cutout.herokuapp.com/thumb"


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
    session.headers.update({"User-Agent": f"{VERSION}/Colab JWST science workflow"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_channel(session: requests.Session, entry: dict) -> tuple[Path, str, int]:
    path = FITS_DIR / f"{VERSION}_MOMZ14_{entry['name']}.fits"
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
        raise RuntimeError(f"DJA response too small for FITS: {preview}")

    path.write_bytes(payload)
    try:
        with fits.open(path, memmap=False) as hdul:
            _ = select_image_hdu(hdul)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return path, response.url, len(payload)


def select_image_hdu(hdul: fits.HDUList) -> int:
    preferred_names = ("SCI", "PRIMARY")
    for wanted in preferred_names:
        for index, hdu in enumerate(hdul):
            if hdu.name.upper() == wanted and hdu.data is not None:
                data = np.asarray(hdu.data)
                if data.ndim >= 2:
                    return index
    for index, hdu in enumerate(hdul):
        if hdu.data is not None and np.asarray(hdu.data).ndim >= 2:
            return index
    raise ValueError("No 2-D science image found in FITS file")


def load_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        index = select_image_hdu(hdul)
        data = np.asarray(hdul[index].data, dtype=float).squeeze()
        header = hdul[index].header.copy()
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D image, received shape {data.shape}")
    return data, header


def estimate_pixel_scale_arcsec(header: fits.Header) -> float:
    for key in ("PIXAR_A2",):
        value = header.get(key)
        if value and value > 0:
            return float(np.sqrt(value))
    cdelt = abs(float(header.get("CDELT1", 0.0)))
    if cdelt > 0:
        return cdelt * 3600.0
    cd11 = float(header.get("CD1_1", 0.0))
    cd12 = float(header.get("CD1_2", 0.0))
    scale = np.hypot(cd11, cd12) * 3600.0
    return scale if scale > 0 else 0.04


def central_crop(data: np.ndarray, width_arcsec: float, pixel_scale: float) -> np.ndarray:
    ny, nx = data.shape
    half = max(3, int(round(width_arcsec / pixel_scale / 2.0)))
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy - half), min(ny, cy + half + 1)
    x0, x1 = max(0, cx - half), min(nx, cx + half + 1)
    return data[y0:y1, x0:x1]


def image_norm(data: np.ndarray) -> ImageNormalize:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch())
    interval = PercentileInterval(99.2)
    vmin, vmax = interval.get_limits(finite)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite) + 1e-12)
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.06), clip=True)


def decorate_axis(ax: plt.Axes, title: str, width_arcsec: float, data: np.ndarray,
                  pixel_scale: float, circle_arcsec: float = 0.30) -> None:
    ax.imshow(data, origin="lower", cmap="gray", norm=image_norm(data), interpolation="nearest")
    ny, nx = data.shape
    radius_pix = circle_arcsec / pixel_scale
    ax.add_patch(Circle((nx / 2, ny / 2), radius_pix, fill=False,
                        edgecolor="tab:red", linewidth=1.2))
    ax.set_title(title, fontsize=11, pad=7)
    ax.text(0.03, 0.04, f'{width_arcsec:.1f}" field', transform=ax.transAxes,
            fontsize=8, ha="left", va="bottom",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"})
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("0.5")


def make_dashboard(records: Iterable[dict]) -> Path:
    records = list(records)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), constrained_layout=True)

    for col, record in enumerate(records):
        data = record["data"]
        scale = record["pixel_scale_arcsec"]
        entry = record["entry"]
        wide_title = f"{entry['name']}  |  {entry['lambda_um']:.2f} µm"
        decorate_axis(axes[0, col], wide_title, CUTOUT_ARCSEC, data, scale)

        zoom = central_crop(data, ZOOM_ARCSEC, scale)
        snr = entry["published_flux_njy"] / entry["published_sigma_njy"]
        zoom_title = (f"center zoom  |  published {entry['published_flux_njy']:.0f}"
                      f"±{entry['published_sigma_njy']:.0f} nJy  ({snr:.1f}σ)")
        decorate_axis(axes[1, col], zoom_title, ZOOM_ARCSEC, zoom, scale, circle_arcsec=0.18)

    fig.suptitle(
        "MoM-z14 — genuine JWST/NIRCam channels from DJA calibrated FITS mosaics\n"
        f"RA {RA_DEG:.7f}°   Dec {DEC_DEG:.7f}°   red circle = catalog position",
        fontsize=15,
    )
    out = PNG_DIR / f"{VERSION}_MOMZ14_4FILTER_CHANNELS.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return out


def main() -> None:
    session = build_session()
    records = []
    manifest_rows = []

    for entry in FILTERS:
        path, url, byte_count = download_channel(session, entry)
        data, header = load_image(path)
        pixel_scale = estimate_pixel_scale_arcsec(header)
        records.append({
            "entry": entry,
            "path": path,
            "data": data,
            "pixel_scale_arcsec": pixel_scale,
        })
        manifest_rows.append({
            "filter": entry["name"],
            "central_wavelength_um": entry["lambda_um"],
            "ra_deg": RA_DEG,
            "dec_deg": DEC_DEG,
            "cutout_arcsec": CUTOUT_ARCSEC,
            "image_nx_pix": data.shape[1],
            "image_ny_pix": data.shape[0],
            "pixel_scale_arcsec_per_pix": pixel_scale,
            "published_flux_njy": entry["published_flux_njy"],
            "published_sigma_njy": entry["published_sigma_njy"],
            "download_bytes": byte_count,
            "fits_path": str(path),
            "source_url": url,
        })

    dashboard = make_dashboard(records)
    csv_path = CSV_DIR / f"{VERSION}_MOMZ14_4FILTER_MANIFEST.csv"
    pd.DataFrame(manifest_rows).to_csv(csv_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"Target       MoM-z14  RA={RA_DEG:.7f} deg  Dec={DEC_DEG:.7f} deg")
    print("Filters      F115W  F150W  F277W  F444W")
    print(f"Plot PNG     {dashboard}")
    print(f"Manifest CSV {csv_path}")
    for record in records:
        print(f"FITS {record['entry']['name']:<5}   {record['path']}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

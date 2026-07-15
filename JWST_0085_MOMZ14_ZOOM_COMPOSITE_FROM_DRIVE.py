#!/usr/bin/env python3
"""Create a zoomed MoM-z14 four-filter composite from FITS files in Drive."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    req = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
           "astropy": "astropy", "scipy": "scipy", "skimage": "scikit-image"}
    missing = [p for m, p in req.items() if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

VERSION = "JWST_0085"
DRIVE_DIR = Path("/content/drive/MyDrive/Colab Notebooks/JWST/MoM-14")
PNG_DIR = Path("/content/JWST_OUTPUT/PNG")
CSV_DIR = Path("/content/JWST_OUTPUT/CSV")
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

FILTERS = [
    ("F115W", 1.15, np.array([0.05, 0.25, 1.00])),
    ("F150W", 1.50, np.array([0.00, 0.95, 0.75])),
    ("F277W", 2.77, np.array([1.00, 0.72, 0.05])),
    ("F444W", 4.44, np.array([1.00, 0.05, 0.00])),
]


def mount_drive() -> None:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)


def load_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data, dtype=float).squeeze()
            while data.ndim > 2:
                data = data[0]
            if data.ndim == 2:
                finite = np.isfinite(data)
                if not finite.any():
                    raise ValueError(f"No finite pixels in {path.name}")
                return np.where(finite, data, np.nanmedian(data[finite]))
    raise ValueError(f"No 2-D image found in {path.name}")


def center_crop(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    y0 = max((a.shape[0] - h) // 2, 0)
    x0 = max((a.shape[1] - w) // 2, 0)
    return a[y0:y0+h, x0:x0+w]


def register(arrays: list[np.ndarray]):
    shape = (min(a.shape[0] for a in arrays), min(a.shape[1] for a in arrays))
    cropped = [center_crop(a, shape) for a in arrays]
    ref = cropped[-1] - np.median(cropped[-1])
    aligned, shifts = [], []
    for i, a in enumerate(cropped):
        if i == len(cropped) - 1:
            aligned.append(a)
            shifts.append((0.0, 0.0))
            continue
        s, _, _ = phase_cross_correlation(ref, a - np.median(a), upsample_factor=30)
        aligned.append(ndi_shift(a, s, order=1, mode="nearest"))
        shifts.append((float(s[0]), float(s[1])))
    return aligned, shifts


def stretch(a: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = a.shape
    b = max(4, int(0.10 * min(h, w)))
    edge = np.concatenate([a[:b].ravel(), a[-b:].ravel(), a[:, :b].ravel(), a[:, -b:].ravel()])
    bg = float(np.median(edge))
    x = a - bg
    lo, hi = np.percentile(x[np.isfinite(x)], [2.0, 99.8])
    hi = max(float(hi), float(lo) + 1e-12)
    z = np.clip((x - lo) / (hi - lo), 0, 1)
    z = np.arcsinh(10 * z) / np.arcsinh(10)
    return z, {"background": bg, "low": float(lo), "high": float(hi)}


def combine(channels: list[np.ndarray]) -> np.ndarray:
    rgb = np.zeros((*channels[0].shape, 3), dtype=float)
    for channel, (_, _, color) in zip(channels, FILTERS):
        rgb += channel[..., None] * color
    scale = np.percentile(rgb[np.isfinite(rgb)], 99.7)
    return np.clip(rgb / max(scale, 1e-12), 0, 1) ** 0.9


def central_zoom(a: np.ndarray, fraction: float = 0.22) -> np.ndarray:
    h, w = a.shape[:2]
    zh, zw = max(24, int(h * fraction)), max(24, int(w * fraction))
    y0, x0 = (h - zh) // 2, (w - zw) // 2
    return a[y0:y0+zh, x0:x0+zw]


def main() -> None:
    mount_drive()
    paths = [DRIVE_DIR / f"MoM-14_{name}.fits" for name, _, _ in FILTERS]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing FITS files:\n" + "\n".join(missing))

    arrays = [load_fits(p) for p in paths]
    aligned, shifts = register(arrays)
    channels, stats = [], []
    for a in aligned:
        s, st = stretch(a)
        channels.append(s)
        stats.append(st)

    rgb = combine(channels)
    zoom = central_zoom(rgb, 0.22)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for i, (name, wave, _) in enumerate(FILTERS):
        r, c = divmod(i, 2)
        axes[r, c].imshow(central_zoom(channels[i], 0.22), cmap="gray", origin="lower")
        axes[r, c].set_title(f"{name}  {wave:.2f} µm — central zoom")
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 2].imshow(rgb, origin="lower")
    axes[0, 2].set_title("Full-field four-filter composite")
    axes[1, 2].imshow(zoom, origin="lower", interpolation="nearest")
    axes[1, 2].set_title("MoM-z14 enlarged central composite")
    for ax in (axes[0, 2], axes[1, 2]):
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MoM-z14 four-filter composite — first zoom pass", fontsize=18)

    dashboard = PNG_DIR / f"{VERSION}_MOMZ14_ZOOM_DASHBOARD.png"
    composite = PNG_DIR / f"{VERSION}_MOMZ14_ZOOM_COMPOSITE.png"
    fig.savefig(dashboard, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.imsave(composite, zoom, origin="lower", dpi=360)
    plt.show(); plt.close(fig)

    rows = []
    for (name, wave, _), path, shift, st in zip(FILTERS, paths, shifts, stats):
        rows.append({"filter": name, "wavelength_um": wave, "file": str(path),
                     "shift_y_pix": shift[0], "shift_x_pix": shift[1], **st})
    manifest = CSV_DIR / f"{VERSION}_MOMZ14_ZOOM_MANIFEST.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"INPUT FOLDER    {DRIVE_DIR}")
    print("FILTERS         F115W F150W F277W F444W")
    print("ZOOM            central 22% of registered field")
    print(f"DASHBOARD PNG   {dashboard}")
    print(f"COMPOSITE PNG   {composite}")
    print(f"MANIFEST CSV    {manifest}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

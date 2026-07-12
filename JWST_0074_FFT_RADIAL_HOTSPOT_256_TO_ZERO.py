#!/usr/bin/env python3
"""
JWST_0074_FFT_RADIAL_HOTSPOT_256_TO_ZERO.py

Create a synthetic radial hotspot whose intensity falls linearly from 256 at
its center to 0 at a finite radius. Compute the raw and mean-subtracted 2-D
FFT power spectra and the azimuthally averaged radial FFT power curve.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0074"
N = 256
PEAK_LEVEL = 256.0
SPOT_RADIUS_PIX = 92.0
EPS = 1.0e-14

ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_radial_hotspot() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((N, N), dtype=float)
    center = (N - 1) / 2.0
    radius = np.hypot(x - center, y - center)
    image = PEAK_LEVEL * np.clip(1.0 - radius / SPOT_RADIUS_PIX, 0.0, 1.0)
    return image, radius


def fft_power(image: np.ndarray, subtract_mean: bool) -> np.ndarray:
    work = image - image.mean() if subtract_mean else image.copy()
    transform = np.fft.fftshift(np.fft.fft2(work))
    power = np.abs(transform) ** 2
    return power


def radial_average(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    radial_frequency = np.hypot(fx, fy)

    bin_width = 1.0 / N
    max_frequency = np.sqrt(0.5**2 + 0.5**2)
    edges = np.arange(0.0, max_frequency + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.digitize(radial_frequency.ravel(), edges) - 1

    sums = np.bincount(indices, weights=power.ravel(), minlength=len(centers))
    counts = np.bincount(indices, minlength=len(centers))
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    valid = (counts > 0) & (centers <= 0.5)
    centers = centers[valid]
    profile = profile[valid]
    profile /= max(profile.max(), EPS)
    return centers, profile


def save_profile_csv(frequency: np.ndarray, profile: np.ndarray) -> Path:
    output = CSV_DIR / f"{VERSION}_RADIAL_FFT_POWER_PROFILE.csv"
    pd.DataFrame({
        "radial_frequency_cycles_per_pixel": frequency,
        "normalized_mean_subtracted_fft_power": profile,
    }).to_csv(output, index=False)
    return output


def make_dashboard(image: np.ndarray,
                   raw_power: np.ndarray,
                   centered_power: np.ndarray,
                   radial_frequency: np.ndarray,
                   radial_profile: np.ndarray) -> Path:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)

    ax_image = fig.add_subplot(grid[0, 0])
    ax_raw = fig.add_subplot(grid[0, 1])
    ax_centered = fig.add_subplot(grid[0, 2])
    ax_curve = fig.add_subplot(grid[1, :])

    hotspot = ax_image.imshow(
        image,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=PEAK_LEVEL,
        interpolation="nearest",
    )
    ax_image.set_title("Radial hotspot\n256 at center, 0 at radius")
    ax_image.set_xlabel("x pixel")
    ax_image.set_ylabel("y pixel")
    plt.colorbar(hotspot, ax=ax_image, fraction=0.046, pad=0.04, label="intensity level")

    extent = [-0.5, 0.5, -0.5, 0.5]
    raw_view = np.log10(raw_power / max(raw_power.max(), EPS) + EPS)
    centered_view = np.log10(centered_power / max(centered_power.max(), EPS) + EPS)

    raw_img = ax_raw.imshow(
        raw_view,
        origin="lower",
        cmap="magma",
        extent=extent,
        vmin=-12,
        vmax=0,
        interpolation="nearest",
    )
    ax_raw.set_title("Raw 2-D FFT power\nlarge DC peak at center")
    ax_raw.set_xlabel("fx [cycles/pixel]")
    ax_raw.set_ylabel("fy [cycles/pixel]")
    plt.colorbar(raw_img, ax=ax_raw, fraction=0.046, pad=0.04, label="log10 normalized power")

    centered_img = ax_centered.imshow(
        centered_view,
        origin="lower",
        cmap="magma",
        extent=extent,
        vmin=-12,
        vmax=0,
        interpolation="nearest",
    )
    ax_centered.set_title("Mean-subtracted 2-D FFT power\nconcentric frequency rings")
    ax_centered.set_xlabel("fx [cycles/pixel]")
    ax_centered.set_ylabel("fy [cycles/pixel]")
    plt.colorbar(centered_img, ax=ax_centered, fraction=0.046, pad=0.04, label="log10 normalized power")

    ax_curve.plot(
        radial_frequency,
        np.maximum(radial_profile, 1.0e-12),
        linewidth=2.6,
    )
    ax_curve.set_yscale("log")
    ax_curve.set_xlim(0.0, 0.5)
    ax_curve.set_ylim(1.0e-12, 3.0)
    ax_curve.set_xlabel("radial spatial frequency [cycles/pixel]")
    ax_curve.set_ylabel("normalized azimuthal FFT power")
    ax_curve.set_title("Radially averaged mean-subtracted FFT power spectrum")
    ax_curve.grid(alpha=0.22, which="both")
    ax_curve.annotate(
        "broad low-frequency concentration\nfrom the smooth, wide hotspot",
        xy=(radial_frequency[2], radial_profile[2]),
        xytext=(0.16, 0.22),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
    )

    fig.suptitle(
        "Synthetic radial hotspot and its Fourier transform\n"
        "linear intensity falloff from level 256 to zero",
        fontsize=18,
        y=0.985,
    )

    output = PNG_DIR / f"{VERSION}_RADIAL_HOTSPOT_FFT_DASHBOARD.png"
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def main() -> None:
    image, radius = build_radial_hotspot()
    raw_power = fft_power(image, subtract_mean=False)
    centered_power = fft_power(image, subtract_mean=True)
    radial_frequency, radial_profile = radial_average(centered_power)

    png_path = make_dashboard(
        image,
        raw_power,
        centered_power,
        radial_frequency,
        radial_profile,
    )
    csv_path = save_profile_csv(radial_frequency, radial_profile)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"IMAGE SIZE      {N} x {N} pixels")
    print(f"PEAK LEVEL      {PEAK_LEVEL:.1f}")
    print(f"ZERO RADIUS     {SPOT_RADIUS_PIX:.1f} pixels")
    print(f"MEAN LEVEL      {image.mean():.6f}")
    print(f"PLOT PNG        {png_path}")
    print(f"CSV             {csv_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

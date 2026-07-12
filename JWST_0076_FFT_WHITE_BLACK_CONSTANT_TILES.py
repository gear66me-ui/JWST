#!/usr/bin/env python3
"""
JWST_0076_FFT_WHITE_BLACK_CONSTANT_TILES.py

Compare a uniform white tile and a uniform black tile. Show the tiles,
raw 2-D FFT power maps, and the radially averaged mean-subtracted FFT
power spectra. All graphics are generated numerically with NumPy and
Matplotlib only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0076"
N = 256
WHITE_LEVEL = 255.0
EPS = 1.0e-30

ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_tiles() -> tuple[np.ndarray, np.ndarray]:
    white = np.full((N, N), WHITE_LEVEL, dtype=float)
    black = np.zeros((N, N), dtype=float)
    return white, black


def fft_power(image: np.ndarray, subtract_mean: bool) -> np.ndarray:
    work = image - image.mean() if subtract_mean else image.copy()
    transform = np.fft.fftshift(np.fft.fft2(work))
    return np.abs(transform) ** 2


def radial_average(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.hypot(fx, fy)

    bin_width = 1.0 / N
    edges = np.arange(0.0, 0.5 + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    index = np.digitize(radius.ravel(), edges) - 1

    sums = np.bincount(index, weights=power.ravel(), minlength=len(centers))
    counts = np.bincount(index, minlength=len(centers))
    profile = np.divide(
        sums[:len(centers)],
        counts[:len(centers)],
        out=np.zeros(len(centers), dtype=float),
        where=counts[:len(centers)] > 0,
    )

    peak = float(profile.max())
    if peak > 0.0:
        profile = profile / peak
    else:
        profile = np.zeros_like(profile)

    return centers, profile


def make_dashboard(
    white: np.ndarray,
    black: np.ndarray,
    white_raw_power: np.ndarray,
    black_raw_power: np.ndarray,
    frequency: np.ndarray,
    white_profile: np.ndarray,
    black_profile: np.ndarray,
) -> Path:
    fig = plt.figure(figsize=(15, 15))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25], hspace=0.34, wspace=0.24)

    ax_white = fig.add_subplot(grid[0, 0])
    ax_black = fig.add_subplot(grid[0, 1])
    ax_white_fft = fig.add_subplot(grid[1, 0])
    ax_black_fft = fig.add_subplot(grid[1, 1])
    ax_curve = fig.add_subplot(grid[2, :])

    ax_white.imshow(white, cmap="gray", vmin=0.0, vmax=WHITE_LEVEL, interpolation="nearest")
    ax_black.imshow(black, cmap="gray", vmin=0.0, vmax=WHITE_LEVEL, interpolation="nearest")

    for ax, title in (
        (ax_white, "Uniform white tile\nall pixels = 255"),
        (ax_black, "Uniform black tile\nall pixels = 0"),
    ):
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])

    white_view = np.log10(white_raw_power / max(float(white_raw_power.max()), EPS) + EPS)
    black_view = np.log10(black_raw_power + EPS)

    ax_white_fft.imshow(
        white_view,
        cmap="gray",
        origin="lower",
        extent=[-0.5, 0.5, -0.5, 0.5],
        vmin=-12,
        vmax=0,
        interpolation="nearest",
    )
    ax_black_fft.imshow(
        black_view,
        cmap="gray",
        origin="lower",
        extent=[-0.5, 0.5, -0.5, 0.5],
        vmin=-30,
        vmax=0,
        interpolation="nearest",
    )

    ax_white_fft.set_title("Raw white-tile 2-D FFT power\none DC coefficient at the center", fontsize=14)
    ax_black_fft.set_title("Raw black-tile 2-D FFT power\nall coefficients are zero", fontsize=14)

    for ax in (ax_white_fft, ax_black_fft):
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")

    ax_curve.plot(
        frequency,
        white_profile,
        linewidth=2.8,
        label="white tile after mean subtraction",
    )
    ax_curve.plot(
        frequency,
        black_profile,
        linewidth=2.2,
        linestyle="--",
        label="black tile after mean subtraction",
    )
    ax_curve.set_xlim(0.0, 0.5)
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.set_xlabel("radial spatial frequency [cycles/pixel]")
    ax_curve.set_ylabel("normalized mean-subtracted FFT power")
    ax_curve.set_title(
        "Radially averaged mean-subtracted FFT power spectrum\n"
        "both constant tiles become exact zero arrays after subtracting their means",
        fontsize=15,
    )
    ax_curve.grid(alpha=0.28)
    ax_curve.legend(loc="upper right")
    ax_curve.annotate(
        "both spectra are identically zero",
        xy=(0.25, 0.0),
        xytext=(0.17, 0.42),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=12,
    )

    fig.suptitle(
        "Uniform white tile versus uniform black tile — Fourier comparison",
        fontsize=20,
        y=0.98,
    )

    output = PNG_DIR / f"{VERSION}_WHITE_BLACK_TILE_FFT_COMPARISON.png"
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output


def save_csv(
    frequency: np.ndarray,
    white_profile: np.ndarray,
    black_profile: np.ndarray,
) -> Path:
    output = CSV_DIR / f"{VERSION}_WHITE_BLACK_TILE_RADIAL_PROFILES.csv"
    pd.DataFrame({
        "radial_frequency_cycles_per_pixel": frequency,
        "white_mean_subtracted_normalized_power": white_profile,
        "black_mean_subtracted_normalized_power": black_profile,
    }).to_csv(output, index=False)
    return output


def main() -> None:
    white, black = build_tiles()

    white_raw_power = fft_power(white, subtract_mean=False)
    black_raw_power = fft_power(black, subtract_mean=False)
    white_centered_power = fft_power(white, subtract_mean=True)
    black_centered_power = fft_power(black, subtract_mean=True)

    frequency, white_profile = radial_average(white_centered_power)
    _, black_profile = radial_average(black_centered_power)

    png_path = make_dashboard(
        white,
        black,
        white_raw_power,
        black_raw_power,
        frequency,
        white_profile,
        black_profile,
    )
    csv_path = save_csv(frequency, white_profile, black_profile)

    center = N // 2
    print(f"CODE OUTPUT: {VERSION}")
    print(f"IMAGE SIZE             {N} x {N} pixels")
    print(f"WHITE RAW DC POWER     {white_raw_power[center, center]:.6f}")
    print(f"BLACK RAW DC POWER     {black_raw_power[center, center]:.6f}")
    print(f"WHITE CENTERED MAX     {white_centered_power.max():.6f}")
    print(f"BLACK CENTERED MAX     {black_centered_power.max():.6f}")
    print(f"PLOT PNG               {png_path}")
    print(f"CSV                    {csv_path}")
    print(f"Timestamp              {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

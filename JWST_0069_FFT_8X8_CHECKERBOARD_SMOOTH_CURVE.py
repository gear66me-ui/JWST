#!/usr/bin/env python3
"""
JWST_0069_FFT_8X8_CHECKERBOARD_SMOOTH_CURVE.py

Numerically generate an 8 x 8 binary checkerboard and a single smooth diagonal
FFT power-spectrum curve. The curve is densely sampled by zero-padding before
the FFT; no AI-generated imagery is used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0069"
N = 8
FFT_GRID = 4096
POWER_FLOOR = 1.0e-10
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_checkerboard() -> np.ndarray:
    yy, xx = np.indices((N, N))
    return ((xx + yy) % 2).astype(float)


def compute_curve(checker: np.ndarray):
    centered = checker - checker.mean()

    fft_dense = np.fft.fftshift(
        np.fft.fft2(centered, s=(FFT_GRID, FFT_GRID))
    )
    power_dense = np.abs(fft_dense) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(FFT_GRID, d=1.0))

    # The checkerboard's alternating mode lies on the diagonal fx = fy.
    # np.diag can return a read-only view, so copy before normalization.
    diagonal_power = np.diag(power_dense).copy()
    peak = float(diagonal_power.max())
    if peak <= 0.0:
        raise RuntimeError("Checkerboard FFT power unexpectedly has no positive peak.")
    diagonal_power = diagonal_power / peak

    # Add +0.5 explicitly because fftfreq includes -0.5 but excludes +0.5.
    frequency_plot = np.append(frequency, 0.5)
    power_plot = np.append(diagonal_power, diagonal_power[0])

    return centered, frequency_plot, power_plot


def make_figure(
    checker: np.ndarray,
    frequency: np.ndarray,
    power: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(9.4, 4.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.85, 3.4], wspace=0.30)

    ax_image = fig.add_subplot(grid[0, 0])
    ax_curve = fig.add_subplot(grid[0, 1])

    ax_image.imshow(
        checker,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="upper",
    )
    ax_image.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax_image.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax_image.grid(which="minor", linewidth=0.55, alpha=0.55)
    ax_image.tick_params(which="minor", bottom=False, left=False)
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_title("8 x 8 checkerboard", fontsize=11)

    plotted_power = np.maximum(power, POWER_FLOOR)
    ax_curve.plot(frequency, plotted_power, linewidth=1.8)
    ax_curve.set_yscale("log")
    ax_curve.set_xlim(-0.5, 0.5)
    ax_curve.set_ylim(POWER_FLOOR, 1.25)
    ax_curve.set_xlabel("Diagonal spatial frequency, fx = fy [cycles/pixel]")
    ax_curve.set_ylabel("Normalized FFT power")
    ax_curve.set_title("Smooth diagonal FFT power-spectrum curve", fontsize=12)
    ax_curve.grid(alpha=0.24, which="both")
    ax_curve.axvline(0.0, linewidth=0.8, alpha=0.45)

    ax_curve.annotate(
        "checkerboard peak\n2-pixel alternation",
        xy=(-0.5, 1.0),
        xytext=(-0.37, 0.12),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )
    ax_curve.annotate(
        "mean removed\nzero-frequency power = 0",
        xy=(0.0, POWER_FLOOR),
        xytext=(0.08, 2.0e-7),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )

    fig.suptitle(
        "8 x 8 Checkerboard — Single FFT Curve",
        fontsize=14,
        y=0.99,
    )
    output = PNG_DIR / f"{VERSION}_CHECKERBOARD_SMOOTH_FFT_CURVE.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_csv(frequency: np.ndarray, power: np.ndarray) -> Path:
    output = CSV_DIR / f"{VERSION}_CHECKERBOARD_SMOOTH_FFT_CURVE.csv"
    pd.DataFrame(
        {
            "frequency_cycles_per_pixel": frequency,
            "normalized_fft_power": power,
        }
    ).to_csv(output, index=False)
    return output


def main() -> None:
    checker = build_checkerboard()
    centered, frequency, power = compute_curve(checker)
    png_path = make_figure(checker, frequency, power)
    csv_path = save_csv(frequency, power)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"INPUT             {N} x {N} binary checkerboard")
    print(f"MEAN REMOVED      {checker.mean():.6f}")
    print(f"FFT GRID          {FFT_GRID} x {FFT_GRID}")
    print("CURVE             diagonal cut, fx = fy")
    print("PEAK              +/-0.500 cycles/pixel")
    print("NOTE              zero-padding densely samples the same finite transform")
    print(f"PLOT PNG          {png_path}")
    print(f"CSV               {csv_path}")
    print(f"Timestamp         {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

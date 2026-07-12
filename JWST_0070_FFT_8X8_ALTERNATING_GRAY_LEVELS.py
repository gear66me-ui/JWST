#!/usr/bin/env python3
"""
JWST_0070_FFT_8X8_ALTERNATING_GRAY_LEVELS.py

Build an 8 x 8 grayscale tile whose neighboring cells alternate between
low and high gray levels in both x and y, then display one smooth diagonal
FFT power-spectrum curve generated numerically with NumPy and Matplotlib.

Sequence begins 0, 64, 1, 63, 2, 62, ... and converges toward mid-gray.
Because an 8 x 8 tile has 64 cells while the inclusive labels 0..64 contain
65 values, level 32 is omitted; the final middle pair is 31 and 33.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0070"
N = 8
MAX_LEVEL = 64
FFT_GRID = 2048
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_alternating_gray_tile() -> np.ndarray:
    """Assign low values to one checker parity and high values to the other."""
    tile = np.empty((N, N), dtype=float)
    low = 0
    high = MAX_LEVEL

    for row in range(N):
        for col in range(N):
            if (row + col) % 2 == 0:
                tile[row, col] = low
                low += 1
            else:
                tile[row, col] = high
                high -= 1

    return tile


def compute_diagonal_fft_curve(tile: np.ndarray):
    """Return a dense diagonal cut through the mean-subtracted 2-D FFT power."""
    centered = tile - tile.mean()

    fft_dense = np.fft.fftshift(
        np.fft.fft2(centered, s=(FFT_GRID, FFT_GRID))
    )
    power_dense = np.abs(fft_dense) ** 2
    frequency = np.fft.fftshift(
        np.fft.fftfreq(FFT_GRID, d=1.0)
    )

    diagonal_power = np.diag(power_dense).copy()
    peak = float(diagonal_power.max())
    if peak <= 0.0:
        raise RuntimeError("FFT power maximum is zero; tile construction failed.")
    diagonal_power /= peak

    frequency_plot = np.append(frequency, 0.5)
    power_plot = np.append(diagonal_power, diagonal_power[0])
    return centered, frequency_plot, power_plot


def draw_tile(ax: plt.Axes, tile: np.ndarray) -> None:
    ax.imshow(
        tile,
        cmap="gray",
        vmin=0,
        vmax=MAX_LEVEL,
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", linewidth=0.7, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "8 x 8 alternating gray tile\n0, 64, 1, 63, 2, 62, ...",
        fontsize=11,
    )

    for row in range(N):
        for col in range(N):
            value = int(tile[row, col])
            text_color = "white" if value < 30 else "black"
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                weight="bold",
            )


def make_figure(
    tile: np.ndarray,
    frequency: np.ndarray,
    power: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11.2, 5.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 3.45], wspace=0.30)

    ax_tile = fig.add_subplot(grid[0, 0])
    ax_curve = fig.add_subplot(grid[0, 1])

    draw_tile(ax_tile, tile)

    floor = 1.0e-10
    ax_curve.plot(
        frequency,
        np.maximum(power, floor),
        linewidth=2.1,
    )
    ax_curve.set_yscale("log")
    ax_curve.set_xlim(-0.5, 0.5)
    ax_curve.set_ylim(floor, 1.25)
    ax_curve.set_xlabel(
        "Diagonal spatial frequency, fx = fy [cycles/pixel]"
    )
    ax_curve.set_ylabel("Normalized FFT power")
    ax_curve.set_title(
        "Single smooth diagonal FFT power-spectrum curve",
        fontsize=12,
    )
    ax_curve.grid(alpha=0.25)

    ax_curve.annotate(
        "dominant alternating component",
        xy=(-0.5, 1.0),
        xytext=(-0.39, 0.18),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )
    ax_curve.annotate(
        "mean removed\nzero-frequency power = 0",
        xy=(0.0, floor),
        xytext=(0.08, 2.0e-7),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )

    fig.suptitle(
        "8 x 8 Alternating Gray Levels — Single FFT Curve",
        fontsize=15,
    )

    output = PNG_DIR / f"{VERSION}_ALTERNATING_GRAY_LEVELS_FFT_CURVE.png"
    fig.savefig(
        output,
        dpi=170,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.show()
    plt.close(fig)
    return output


def save_csvs(
    tile: np.ndarray,
    frequency: np.ndarray,
    power: np.ndarray,
) -> tuple[Path, Path]:
    tile_path = CSV_DIR / f"{VERSION}_ALTERNATING_GRAY_LEVEL_TILE.csv"
    spectrum_path = CSV_DIR / f"{VERSION}_ALTERNATING_GRAY_LEVEL_FFT_CURVE.csv"

    pd.DataFrame(tile.astype(int)).to_csv(tile_path, index=False, header=False)
    pd.DataFrame(
        {
            "frequency_cycles_per_pixel": frequency,
            "normalized_fft_power": power,
        }
    ).to_csv(spectrum_path, index=False)

    return tile_path, spectrum_path


def main() -> None:
    tile = build_alternating_gray_tile()
    centered, frequency, power = compute_diagonal_fft_curve(tile)
    png_path = make_figure(tile, frequency, power)
    tile_csv, spectrum_csv = save_csvs(tile, frequency, power)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"INPUT TILE          {N} x {N} alternating gray levels")
    print(f"LEVEL RANGE         0 to {MAX_LEVEL}; level 32 omitted")
    print(f"MEAN LEVEL          {tile.mean():.6f}")
    print(f"CENTERED MEAN       {centered.mean():.6f}")
    print(f"DENSE FFT GRID      {FFT_GRID} x {FFT_GRID}")
    print("CURVE CUT           diagonal fx = fy")
    print(f"PLOT PNG            {png_path}")
    print(f"TILE CSV            {tile_csv}")
    print(f"SPECTRUM CSV        {spectrum_csv}")
    print(f"Timestamp           {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
JWST_0072_FFT_8X8_VS_16X16_GRAY_COMPARE.py

High-resolution Python/NumPy/Matplotlib comparison of two complementary
alternating gray tiles and their diagonal FFT power-spectrum curves.
No AI-generated imagery is used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0072"
LEVEL_MAX = 64.0
FFT_GRID = 2048
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_complementary_tile(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Create size*size values as complementary pairs.

    There are size*size/2 pairs.  Pair 0 is (0, 64), and the final pair is
    (64, 0).  A continuous step is required for a 16x16 grid because 128
    pairs cannot be represented by only 65 distinct integer levels without
    repeats.  The generated values contain no skipped pair positions.
    """
    cell_count = size * size
    if cell_count % 2:
        raise ValueError("Tile must contain an even number of cells")

    pair_count = cell_count // 2
    low = np.linspace(0.0, LEVEL_MAX, pair_count, dtype=float)
    high = LEVEL_MAX - low

    sequence = np.empty(cell_count, dtype=float)
    sequence[0::2] = low
    sequence[1::2] = high
    tile = sequence.reshape(size, size)
    return tile, sequence


def diagonal_fft_curve(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = tile - np.mean(tile)
    transform = np.fft.fftshift(
        np.fft.fft2(centered, s=(FFT_GRID, FFT_GRID))
    )
    power = np.abs(transform) ** 2
    diagonal = np.diagonal(power).copy()
    maximum = float(np.max(diagonal))
    if maximum <= 0.0:
        raise ValueError("FFT diagonal has no positive power")
    diagonal /= maximum

    frequency = np.fft.fftshift(np.fft.fftfreq(FFT_GRID, d=1.0))
    frequency = np.append(frequency, 0.5)
    diagonal = np.append(diagonal, diagonal[0])
    return frequency, diagonal


def draw_tile(ax: plt.Axes, tile: np.ndarray, title: str) -> None:
    ax.imshow(
        tile,
        cmap="gray",
        vmin=0.0,
        vmax=LEVEL_MAX,
        interpolation="nearest",
        origin="upper",
    )
    n = tile.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.grid(which="minor", linewidth=0.35, alpha=0.28)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)


def make_figure(
    tile8: np.ndarray,
    tile16: np.ndarray,
    frequency8: np.ndarray,
    power8: np.ndarray,
    frequency16: np.ndarray,
    power16: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 2.0],
        width_ratios=[1.0, 1.0, 1.8],
        hspace=0.30,
        wspace=0.24,
    )

    ax8 = fig.add_subplot(grid[0, 0])
    ax16 = fig.add_subplot(grid[0, 1])
    ax_note = fig.add_subplot(grid[0, 2])
    ax_curve = fig.add_subplot(grid[1, :])

    draw_tile(ax8, tile8, "8 × 8 complementary gray tile")
    draw_tile(ax16, tile16, "16 × 16 complementary gray tile")

    ax_note.axis("off")
    ax_note.text(
        0.02,
        0.84,
        "Exact endpoint rule",
        fontsize=15,
        weight="bold",
        transform=ax_note.transAxes,
    )
    ax_note.text(
        0.02,
        0.62,
        "first pair   0, 64\nlast pair   64, 0",
        fontsize=13,
        linespacing=1.6,
        transform=ax_note.transAxes,
    )
    ax_note.text(
        0.02,
        0.28,
        "Tiles are blank: no numbers are drawn\ninside the cells, so the display itself\ndoes not obscure the gray levels.",
        fontsize=11,
        linespacing=1.5,
        transform=ax_note.transAxes,
    )

    floor = 1.0e-12
    ax_curve.semilogy(
        frequency8,
        np.maximum(power8, floor),
        linewidth=1.8,
        label="8 × 8 diagonal FFT power",
    )
    ax_curve.semilogy(
        frequency16,
        np.maximum(power16, floor),
        linewidth=2.0,
        label="16 × 16 diagonal FFT power",
    )
    ax_curve.set_xlim(-0.5, 0.5)
    ax_curve.set_ylim(1.0e-12, 1.2)
    ax_curve.set_xlabel("Diagonal spatial frequency, fx = fy [cycles/pixel]", fontsize=12)
    ax_curve.set_ylabel("Normalized FFT power", fontsize=12)
    ax_curve.set_title(
        "High-resolution comparison of the two diagonal FFT spectra",
        fontsize=15,
    )
    ax_curve.grid(alpha=0.24)
    ax_curve.legend(loc="upper center", ncol=2, fontsize=11)

    fig.suptitle(
        "Complementary Alternating Gray Sequences — 8 × 8 versus 16 × 16",
        fontsize=20,
        y=0.98,
    )

    output = PNG_DIR / f"{VERSION}_8X8_VS_16X16_GRAY_FFT_COMPARE.png"
    fig.savefig(
        output,
        dpi=360,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.show()
    plt.close(fig)
    return output


def save_outputs(
    tile8: np.ndarray,
    tile16: np.ndarray,
    frequency8: np.ndarray,
    power8: np.ndarray,
    frequency16: np.ndarray,
    power16: np.ndarray,
) -> tuple[Path, Path, Path]:
    tile8_path = CSV_DIR / f"{VERSION}_8X8_TILE.csv"
    tile16_path = CSV_DIR / f"{VERSION}_16X16_TILE.csv"
    curve_path = CSV_DIR / f"{VERSION}_FFT_CURVES.csv"

    pd.DataFrame(tile8).to_csv(tile8_path, index=False, header=False)
    pd.DataFrame(tile16).to_csv(tile16_path, index=False, header=False)
    pd.DataFrame(
        {
            "frequency_cycles_per_pixel": frequency8,
            "fft_power_8x8_normalized": power8,
            "fft_power_16x16_normalized": power16,
        }
    ).to_csv(curve_path, index=False)
    return tile8_path, tile16_path, curve_path


def main() -> None:
    tile8, sequence8 = build_complementary_tile(8)
    tile16, sequence16 = build_complementary_tile(16)

    frequency8, power8 = diagonal_fft_curve(tile8)
    frequency16, power16 = diagonal_fft_curve(tile16)

    png_path = make_figure(
        tile8,
        tile16,
        frequency8,
        power8,
        frequency16,
        power16,
    )
    tile8_csv, tile16_csv, curve_csv = save_outputs(
        tile8,
        tile16,
        frequency8,
        power8,
        frequency16,
        power16,
    )

    print(f"CODE OUTPUT: {VERSION}")
    print("SEQUENCE                     FIRST PAIR       LAST PAIR       PAIRS")
    print(
        f"8 x 8                        {sequence8[0]:6.3f}, {sequence8[1]:6.3f}"
        f"    {sequence8[-2]:6.3f}, {sequence8[-1]:6.3f}       {len(sequence8)//2:3d}"
    )
    print(
        f"16 x 16                      {sequence16[0]:6.3f}, {sequence16[1]:6.3f}"
        f"    {sequence16[-2]:6.3f}, {sequence16[-1]:6.3f}       {len(sequence16)//2:3d}"
    )
    print(f"PLOT PNG                     {png_path}")
    print(f"8X8 CSV                      {tile8_csv}")
    print(f"16X16 CSV                    {tile16_csv}")
    print(f"CURVE CSV                    {curve_csv}")
    print(f"Timestamp                    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

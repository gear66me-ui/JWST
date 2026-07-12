#!/usr/bin/env python3
"""
JWST_0072_FFT_GRAY_TILE_COMPARISON_NO_LABELS.py

Compare the earlier 8 x 8 and 16 x 16 alternating gray-level tiles with
all numerical text removed from the tile displays. The FFT is computed only
from the underlying numeric arrays; display annotations never enter the FFT.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0072"
LEVEL_MAX = 64
FFT_GRID = 4096
FLOOR = 1.0e-12

ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_8x8_tile() -> np.ndarray:
    """Reproduce the 8 x 8 parity-alternating tile used in JWST_0070."""
    n = 8
    tile = np.empty((n, n), dtype=float)
    low = 0
    high = LEVEL_MAX

    for row in range(n):
        for col in range(n):
            if (row + col) % 2 == 0:
                tile[row, col] = low
                low += 1
            else:
                tile[row, col] = high
                high -= 1

    return tile


def build_16x16_tile() -> np.ndarray:
    """Reproduce the 16 x 16 complementary pair sequence used in JWST_0071."""
    n = 16
    pair_count = (n * n) // 2
    low = np.rint(np.linspace(0, LEVEL_MAX, pair_count)).astype(float)
    high = LEVEL_MAX - low

    flat = np.empty(n * n, dtype=float)
    flat[0::2] = low
    flat[1::2] = high
    tile = flat.reshape(n, n)

    assert tuple(tile.ravel()[:2].astype(int)) == (0, 64)
    assert tuple(tile.ravel()[-2:].astype(int)) == (64, 0)
    return tile


def diagonal_fft_curve(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense diagonal cut through mean-subtracted 2-D FFT power."""
    centered = tile - tile.mean()
    transform = np.fft.fftshift(
        np.fft.fft2(centered, s=(FFT_GRID, FFT_GRID))
    )
    power = np.abs(transform) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(FFT_GRID, d=1.0))

    diagonal = np.diag(power).copy()
    peak = float(diagonal.max())
    if peak <= 0.0:
        raise RuntimeError("Zero FFT peak encountered.")
    diagonal /= peak

    frequency = np.append(frequency, 0.5)
    diagonal = np.append(diagonal, diagonal[0])
    return centered, frequency, diagonal


def draw_blank_tile(ax: plt.Axes, tile: np.ndarray, title: str) -> None:
    n = tile.shape[0]
    ax.imshow(
        tile,
        cmap="gray",
        vmin=0,
        vmax=LEVEL_MAX,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.35, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def make_figure(
    tile8: np.ndarray,
    tile16: np.ndarray,
    frequency8: np.ndarray,
    power8: np.ndarray,
    frequency16: np.ndarray,
    power16: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11.2, 7.2))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.65],
        hspace=0.34,
        wspace=0.28,
    )

    ax8 = fig.add_subplot(grid[0, 0])
    ax16 = fig.add_subplot(grid[0, 1])
    ax_curve = fig.add_subplot(grid[1, :])

    draw_blank_tile(
        ax8,
        tile8,
        "8 x 8 gray tile — no numbers\nlevels 0/64 converge toward mid-gray",
    )
    draw_blank_tile(
        ax16,
        tile16,
        "16 x 16 gray tile — no numbers\nfirst pair 0,64 • last pair 64,0",
    )

    ax_curve.semilogy(
        frequency8,
        np.maximum(power8, FLOOR),
        linewidth=2.0,
        label="8 x 8 normalized diagonal FFT power",
    )
    ax_curve.semilogy(
        frequency16,
        np.maximum(power16, FLOOR),
        linewidth=2.0,
        linestyle="--",
        label="16 x 16 normalized diagonal FFT power",
    )
    ax_curve.set_xlim(-0.5, 0.5)
    ax_curve.set_ylim(FLOOR, 1.25)
    ax_curve.set_xlabel("Diagonal spatial frequency, fx = fy [cycles/pixel]")
    ax_curve.set_ylabel("Normalized FFT power")
    ax_curve.set_title("Direct curve comparison on the same axes", fontsize=12)
    ax_curve.grid(alpha=0.24)
    ax_curve.legend(loc="lower center", fontsize=9)
    ax_curve.axvline(0.0, linewidth=0.8, alpha=0.45)

    difference = float(np.max(np.abs(power8 - power16)))
    ax_curve.text(
        0.02,
        0.04,
        f"maximum normalized curve difference = {difference:.6f}",
        transform=ax_curve.transAxes,
        fontsize=9,
    )

    fig.suptitle(
        "Alternating Gray Tiles — 8 x 8 versus 16 x 16 FFT",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.945,
        "Tile numbers removed from display; FFT uses only the numeric gray arrays",
        ha="center",
        fontsize=9,
    )

    output = PNG_DIR / f"{VERSION}_GRAY_TILE_FFT_COMPARISON_NO_LABELS.png"
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_outputs(
    tile8: np.ndarray,
    tile16: np.ndarray,
    frequency: np.ndarray,
    power8: np.ndarray,
    power16: np.ndarray,
) -> tuple[Path, Path, Path]:
    tile8_path = CSV_DIR / f"{VERSION}_8X8_TILE.csv"
    tile16_path = CSV_DIR / f"{VERSION}_16X16_TILE.csv"
    curve_path = CSV_DIR / f"{VERSION}_FFT_CURVE_COMPARISON.csv"

    pd.DataFrame(tile8.astype(int)).to_csv(tile8_path, index=False, header=False)
    pd.DataFrame(tile16.astype(int)).to_csv(tile16_path, index=False, header=False)
    pd.DataFrame(
        {
            "frequency_cycles_per_pixel": frequency,
            "power_8x8_normalized": power8,
            "power_16x16_normalized": power16,
            "absolute_difference": np.abs(power8 - power16),
        }
    ).to_csv(curve_path, index=False)

    return tile8_path, tile16_path, curve_path


def main() -> None:
    tile8 = build_8x8_tile()
    tile16 = build_16x16_tile()

    centered8, frequency8, power8 = diagonal_fft_curve(tile8)
    centered16, frequency16, power16 = diagonal_fft_curve(tile16)

    if not np.allclose(frequency8, frequency16):
        raise RuntimeError("Frequency grids do not match.")

    png_path = make_figure(
        tile8,
        tile16,
        frequency8,
        power8,
        frequency16,
        power16,
    )
    tile8_path, tile16_path, curve_path = save_outputs(
        tile8,
        tile16,
        frequency8,
        power8,
        power16,
    )

    print(f"CODE OUTPUT: {VERSION}")
    print("DISPLAY LABELS     none inside either tile")
    print(f"8X8 MEAN           {tile8.mean():.6f}")
    print(f"16X16 MEAN         {tile16.mean():.6f}")
    print(f"8X8 CENTERED MEAN  {centered8.mean():.6f}")
    print(f"16X16 CENTERED MEAN {centered16.mean():.6f}")
    print(f"FFT GRID           {FFT_GRID} x {FFT_GRID}")
    print(f"MAX CURVE DELTA    {np.max(np.abs(power8 - power16)):.6f}")
    print(f"PLOT PNG           {png_path}")
    print(f"8X8 TILE CSV       {tile8_path}")
    print(f"16X16 TILE CSV     {tile16_path}")
    print(f"CURVE CSV          {curve_path}")
    print(f"Timestamp          {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
JWST_0071_FFT_16X16_ALTERNATING_GRAY_SEQUENCE.py

Build a 16 x 16 gray-level tile from 128 complementary pairs.
The first pair is (0, 64), the last pair is (64, 0), and all
intermediate pairs progress monotonically between those endpoints.
A dense zero-padded 2-D FFT diagonal power curve is plotted beside it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0071"
N = 16
LEVEL_MAX = 64
PAIR_COUNT = (N * N) // 2
FFT_GRID = 2048

ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_tile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tile and its complementary low/high pair sequences."""
    low = np.rint(np.linspace(0, LEVEL_MAX, PAIR_COUNT)).astype(int)
    high = LEVEL_MAX - low

    flat = np.empty(N * N, dtype=float)
    flat[0::2] = low
    flat[1::2] = high
    tile = flat.reshape(N, N)

    assert tuple(tile.ravel()[:2].astype(int)) == (0, 64)
    assert tuple(tile.ravel()[-2:].astype(int)) == (64, 0)
    return tile, low, high


def compute_fft_curve(tile: np.ndarray):
    centered = tile - tile.mean()

    fft_dense = np.fft.fftshift(
        np.fft.fft2(centered, s=(FFT_GRID, FFT_GRID))
    )
    power_dense = np.abs(fft_dense) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(FFT_GRID, d=1.0))

    diagonal_power = np.diag(power_dense).copy()
    peak = float(diagonal_power.max())
    if peak > 0.0:
        diagonal_power /= peak

    frequency_plot = np.append(frequency, 0.5)
    power_plot = np.append(diagonal_power, diagonal_power[0])

    fft_exact = np.fft.fftshift(np.fft.fft2(centered))
    exact_power = np.diag(np.abs(fft_exact) ** 2).copy()
    exact_peak = float(exact_power.max())
    if exact_peak > 0.0:
        exact_power /= exact_peak
    exact_frequency = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))

    return centered, frequency_plot, power_plot, exact_frequency, exact_power


def draw_tile(ax: plt.Axes, tile: np.ndarray) -> None:
    ax.imshow(
        tile,
        cmap="gray",
        vmin=0,
        vmax=LEVEL_MAX,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", linewidth=0.35, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "16 x 16 alternating gray tile\nfirst pair 0,64  •  last pair 64,0",
        fontsize=10,
    )

    for row in range(N):
        for col in range(N):
            value = int(tile[row, col])
            text_color = "white" if value < 32 else "black"
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=4.6,
                color=text_color,
                weight="bold",
            )


def make_figure(
    tile: np.ndarray,
    frequency: np.ndarray,
    power: np.ndarray,
    exact_frequency: np.ndarray,
    exact_power: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11.2, 5.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 3.2], wspace=0.28)

    ax_tile = fig.add_subplot(grid[0, 0])
    ax_curve = fig.add_subplot(grid[0, 1])

    draw_tile(ax_tile, tile)

    safe_power = np.maximum(power, 1.0e-12)
    ax_curve.semilogy(
        frequency,
        safe_power,
        linewidth=2.0,
        label="Dense diagonal FFT power curve",
    )
    ax_curve.scatter(
        exact_frequency,
        np.maximum(exact_power, 1.0e-12),
        s=20,
        zorder=3,
        label="Exact 16 x 16 FFT bins",
    )
    ax_curve.set_xlim(-0.5, 0.5)
    ax_curve.set_ylim(1.0e-10, 1.2)
    ax_curve.set_xlabel("Diagonal spatial frequency, fx = fy [cycles/pixel]")
    ax_curve.set_ylabel("Normalized FFT power")
    ax_curve.set_title(
        "16 x 16 complementary gray sequence\nSingle smooth diagonal FFT power-spectrum curve",
        fontsize=12,
    )
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="upper center", fontsize=8)
    ax_curve.axvline(0.0, linewidth=0.8, alpha=0.5)
    ax_curve.annotate(
        "mean removed\nzero-frequency power",
        xy=(0.0, safe_power[np.argmin(np.abs(frequency))]),
        xytext=(0.08, 2.0e-6),
        arrowprops=dict(arrowstyle="->", linewidth=0.9),
        fontsize=8,
    )

    fig.suptitle(
        "16 x 16 Alternating Gray Levels — FFT",
        fontsize=15,
    )

    output = PNG_DIR / f"{VERSION}_16X16_ALTERNATING_GRAY_FFT.png"
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_outputs(
    tile: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    frequency: np.ndarray,
    power: np.ndarray,
) -> tuple[Path, Path, Path]:
    tile_path = CSV_DIR / f"{VERSION}_16X16_TILE.csv"
    pair_path = CSV_DIR / f"{VERSION}_PAIR_SEQUENCE.csv"
    curve_path = CSV_DIR / f"{VERSION}_FFT_CURVE.csv"

    pd.DataFrame(tile.astype(int)).to_csv(tile_path, index=False, header=False)
    pd.DataFrame(
        {
            "pair_index": np.arange(PAIR_COUNT),
            "low_level": low,
            "high_level": high,
        }
    ).to_csv(pair_path, index=False)
    pd.DataFrame(
        {
            "frequency_cycles_per_pixel": frequency,
            "normalized_fft_power": power,
        }
    ).to_csv(curve_path, index=False)

    return tile_path, pair_path, curve_path


def main() -> None:
    tile, low, high = build_tile()
    _, frequency, power, exact_frequency, exact_power = compute_fft_curve(tile)

    png_path = make_figure(
        tile,
        frequency,
        power,
        exact_frequency,
        exact_power,
    )
    tile_path, pair_path, curve_path = save_outputs(
        tile,
        low,
        high,
        frequency,
        power,
    )

    print(f"CODE OUTPUT: {VERSION}")
    print(f"GRID               {N} x {N}")
    print(f"PAIR COUNT         {PAIR_COUNT}")
    print(f"FIRST PAIR         {int(low[0])}, {int(high[0])}")
    print(f"LAST PAIR          {int(low[-1])}, {int(high[-1])}")
    print(f"MEAN               {tile.mean():.6f}")
    print(f"FFT GRID           {FFT_GRID} x {FFT_GRID}")
    print(f"PLOT PNG           {png_path}")
    print(f"TILE CSV           {tile_path}")
    print(f"PAIR CSV           {pair_path}")
    print(f"CURVE CSV          {curve_path}")
    print(f"Timestamp          {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

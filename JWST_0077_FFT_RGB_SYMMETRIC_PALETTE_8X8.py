#!/usr/bin/env python3
"""
JWST_0077_FFT_RGB_SYMMETRIC_PALETTE_8X8.py

Build an 8 x 8, axis-symmetric tile palette using yellow, green, and blue
families with four shade levels (0 = light, 3 = deep). Compute separate
mean-subtracted FFTs for the R, G, B, and luminance channels, plus radial
power-spectrum curves.

All figures are generated numerically with NumPy and Matplotlib.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0077"
N = 8
EPS = 1.0e-14
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_NAMES = np.array(["yellow", "green", "blue"], dtype=object)
LIGHT_RGB = np.array([
    [1.00, 1.00, 0.72],
    [0.72, 1.00, 0.78],
    [0.70, 0.86, 1.00],
], dtype=float)
DEEP_RGB = np.array([
    [0.42, 0.29, 0.00],
    [0.00, 0.27, 0.07],
    [0.015, 0.045, 0.28],
], dtype=float)


def mirror_quadrant(values: np.ndarray) -> np.ndarray:
    top = np.concatenate([values, values[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


def build_palette() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices((4, 4))
    family_quadrant = (y + 2 * x) % 3
    level_quadrant = (2 * y + x) % 4

    family = mirror_quadrant(family_quadrant)
    level = mirror_quadrant(level_quadrant)

    rgb = np.zeros((N, N, 3), dtype=float)
    for iy in range(N):
        for ix in range(N):
            fam = int(family[iy, ix])
            lev = int(level[iy, ix])
            t = lev / 3.0
            rgb[iy, ix] = (1.0 - t) * LIGHT_RGB[fam] + t * DEEP_RGB[fam]

    if not np.allclose(rgb, rgb[:, ::-1, :]):
        raise RuntimeError("Horizontal symmetry check failed.")
    if not np.allclose(rgb, rgb[::-1, :, :]):
        raise RuntimeError("Vertical symmetry check failed.")

    return rgb, family, level


def fft_power(channel: np.ndarray) -> np.ndarray:
    centered = channel - np.mean(channel)
    transform = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(transform) ** 2
    return power / max(float(power.max()), EPS)


def radial_average(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    radius = np.hypot(fx, fy)

    bin_width = 1.0 / N
    edges = np.arange(0.0, np.sqrt(0.5**2 + 0.5**2) + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.digitize(radius.ravel(), edges) - 1

    sums = np.bincount(indices, weights=power.ravel(), minlength=len(centers))
    counts = np.bincount(indices, minlength=len(centers))
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    valid = (counts > 0) & (centers <= 0.5)
    centers = centers[valid]
    profile = profile[valid]
    profile /= max(float(profile.max()), EPS)
    return centers, profile


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def make_dashboard(
    rgb: np.ndarray,
    channels: dict[str, np.ndarray],
    powers: dict[str, np.ndarray],
) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    axes[0, 0].imshow(rgb, origin="upper", interpolation="nearest")
    axes[0, 0].set_title("8 x 8 symmetric RGB palette\nyellow, green, blue; levels 0-3")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    channel_order = ["R", "G", "B"]
    for ax, name in zip([axes[0, 1], axes[0, 2], axes[1, 0]], channel_order):
        image = ax.imshow(
            channels[name],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(f"{name} channel values")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax, name in zip([axes[1, 1], axes[1, 2]], ["Luminance", "B"]):
        view = np.log10(powers[name] + EPS)
        image = ax.imshow(
            view,
            cmap="gray",
            vmin=-12,
            vmax=0,
            extent=[-0.5, 0.5, -0.5, 0.5],
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(f"Mean-subtracted {name} FFT power")
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                     label="log10 normalized power")

    fig.suptitle(
        "Axis-symmetric yellow/green/blue tile palette and channel Fourier structure",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_RGB_PALETTE_FFT_DASHBOARD.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_radial_curve(profiles: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 7))

    curve_colors = {
        "R": "red",
        "G": "limegreen",
        "B": "deepskyblue",
        "Luminance": "white",
    }
    for name in ["R", "G", "B", "Luminance"]:
        frequency, profile = profiles[name]
        ax.plot(
            frequency,
            np.maximum(profile, 1.0e-12),
            linewidth=2.6,
            label=name,
            color=curve_colors[name],
        )

    ax.set_yscale("log")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(1.0e-12, 3.0)
    ax.set_xlabel("radial spatial frequency [cycles/pixel]")
    ax.set_ylabel("normalized mean-subtracted FFT power")
    ax.set_title("Radially averaged FFT power by color channel")
    ax.grid(alpha=0.22, which="both")
    ax.legend(ncol=4, loc="upper center")

    output = PNG_DIR / f"{VERSION}_RGB_PALETTE_RADIAL_FFT_CURVES.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_tables(
    rgb: np.ndarray,
    family: np.ndarray,
    level: np.ndarray,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[Path, Path]:
    tile_rows = []
    for iy in range(N):
        for ix in range(N):
            tile_rows.append({
                "row": iy,
                "column": ix,
                "family": FAMILY_NAMES[int(family[iy, ix])],
                "shade_level_0_light_3_deep": int(level[iy, ix]),
                "R": float(rgb[iy, ix, 0]),
                "G": float(rgb[iy, ix, 1]),
                "B": float(rgb[iy, ix, 2]),
            })
    tile_path = CSV_DIR / f"{VERSION}_RGB_PALETTE_TILES.csv"
    pd.DataFrame(tile_rows).to_csv(tile_path, index=False)

    common_frequency = profiles["R"][0]
    profile_path = CSV_DIR / f"{VERSION}_RGB_PALETTE_RADIAL_FFT.csv"
    pd.DataFrame({
        "radial_frequency_cycles_per_pixel": common_frequency,
        "R_normalized_power": profiles["R"][1],
        "G_normalized_power": profiles["G"][1],
        "B_normalized_power": profiles["B"][1],
        "luminance_normalized_power": profiles["Luminance"][1],
    }).to_csv(profile_path, index=False)

    return tile_path, profile_path


def main() -> None:
    rgb, family, level = build_palette()
    channels = {
        "R": rgb[:, :, 0],
        "G": rgb[:, :, 1],
        "B": rgb[:, :, 2],
        "Luminance": luminance(rgb),
    }
    powers = {name: fft_power(channel) for name, channel in channels.items()}
    profiles = {name: radial_average(power) for name, power in powers.items()}

    dashboard_path = make_dashboard(rgb, channels, powers)
    curves_path = make_radial_curve(profiles)
    tile_csv, profile_csv = save_tables(rgb, family, level, profiles)

    print(f"CODE OUTPUT: {VERSION}")
    print("GRID            8 x 8 tiles")
    print("PALETTE         yellow, green, blue")
    print("SHADE LEVELS    0=light, 1, 2, 3=deep")
    print("SYMMETRY        exact horizontal and vertical mirror symmetry")
    print("FFT             separate mean-subtracted R, G, B, luminance transforms")
    print(f"DASHBOARD PNG   {dashboard_path}")
    print(f"CURVES PNG      {curves_path}")
    print(f"TILE CSV        {tile_csv}")
    print(f"PROFILE CSV     {profile_csv}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
JWST_0059 — exact 8-pixel FFT bin comparison.

Fixes the blank second-row 2-D FFT panel from JWST_0058 and makes the
frequency-bin interpretation explicit.

Patterns:
A = 0 0 0 0 1 1 1 1
B = 0 1 0 1 0 1 0 1

Each row is repeated vertically to form an 8x8 image. The 2-D FFT is shown
as an exact discrete-bin scatter map rather than an image interpolation.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    req = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib"}
    missing = [pip for mod, pip in req.items() if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

VERSION = "JWST_0059"
ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
for d in (PNG, CSV):
    d.mkdir(parents=True, exist_ok=True)

PATTERNS = [
    {
        "name": "BLOCK_STEP",
        "label": "00001111 — one broad step",
        "row": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
    },
    {
        "name": "ALTERNATING",
        "label": "01010101 — alternating every pixel",
        "row": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float),
    },
]


def analyze(row: np.ndarray) -> dict:
    image = np.tile(row, (8, 1))
    freq = np.fft.fftshift(np.fft.fftfreq(row.size))
    fft1 = np.fft.fftshift(np.fft.fft(row))
    power1 = np.abs(fft1) ** 2
    centered = row - row.mean()
    fft1_centered = np.fft.fftshift(np.fft.fft(centered))
    power1_centered = np.abs(fft1_centered) ** 2

    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    fft2 = np.fft.fftshift(np.fft.fft2(image))
    power2 = np.abs(fft2) ** 2

    return {
        "image": image,
        "row": row,
        "freq": freq,
        "fft1": fft1,
        "power1": power1,
        "power1_centered": power1_centered,
        "fx": fx,
        "fy": fy,
        "power2": power2,
    }


def draw_2d_bins(ax, data: dict) -> None:
    fx_grid, fy_grid = np.meshgrid(data["fx"], data["fy"])
    power = data["power2"]
    positive = power > 1e-12

    ax.scatter(fx_grid.ravel(), fy_grid.ravel(), s=18, alpha=0.28)

    if np.any(positive):
        p = power[positive]
        pmin = float(np.min(p))
        pmax = float(np.max(p))
        if pmax <= pmin:
            pmin = max(pmax / 100.0, np.finfo(float).tiny)
        sizes = 120.0 + 520.0 * np.sqrt(p / pmax)
        scatter = ax.scatter(
            fx_grid[positive], fy_grid[positive],
            c=p, s=sizes, cmap="magma",
            norm=LogNorm(vmin=pmin, vmax=pmax),
            edgecolors="white", linewidths=0.6,
        )
        for x, y, val in zip(fx_grid[positive], fy_grid[positive], p):
            ax.text(x, y + 0.055, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="2-D FFT power")

    ax.set_xlim(-0.56, 0.44)
    ax.set_ylim(-0.56, 0.44)
    ax.set_xticks(data["fx"])
    ax.set_yticks(data["fy"])
    ax.set_xlabel("fx [cycles/pixel]")
    ax.set_ylabel("fy [cycles/pixel]")
    ax.set_title("Exact 2-D FFT bins", fontsize=11)
    ax.grid(alpha=0.18)


def make_dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)

    for r, rec in enumerate(records):
        d = rec["data"]

        ax = axes[r, 0]
        ax.imshow(d["image"], origin="lower", cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(rec["label"], fontsize=13)
        ax.set_xticks(np.arange(8))
        ax.set_yticks(np.arange(8))
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", linewidth=0.7, alpha=0.45)
        for y in range(8):
            for x in range(8):
                value = int(d["image"][y, x])
                ax.text(x, y, str(value), ha="center", va="center", fontsize=9,
                        color="black" if value else "white")

        ax = axes[r, 1]
        x_edges = np.arange(9)
        y_step = np.r_[d["row"], d["row"][-1]]
        ax.step(x_edges, y_step, where="post", linewidth=2.2)
        ax.scatter(np.arange(8) + 0.5, d["row"], s=35)
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xticks(np.arange(9))
        ax.set_yticks([0, 1])
        ax.set_xlabel("x position [pixels]")
        ax.set_ylabel("intensity")
        ax.set_title("Middle-row pixel profile", fontsize=11)
        ax.grid(alpha=0.22)

        ax = axes[r, 2]
        markerline, stemlines, baseline = ax.stem(d["freq"], d["power1"])
        plt.setp(stemlines, linewidth=1.8)
        plt.setp(markerline, markersize=6)
        plt.setp(baseline, linewidth=0.7, alpha=0.6)
        ax.set_xlabel("spatial frequency [cycles/pixel]")
        ax.set_ylabel("1-D FFT power")
        ax.set_title("Raw 1-D FFT power", fontsize=11)
        ax.set_xticks(d["freq"])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.20)
        for f, p in zip(d["freq"], d["power1"]):
            if p > 1e-12:
                period = "DC" if f == 0 else f"T={1/abs(f):.2f}px"
                ax.text(f, p + 0.45, period, ha="center", va="bottom", fontsize=8)

        ax = axes[r, 3]
        draw_2d_bins(ax, d)

    fig.suptitle(
        "Exact eight-pixel FFT comparison\n"
        "Frequency-bin spacing is 1/N = 1/8 = 0.125 cycles/pixel",
        fontsize=18,
    )
    out = PNG / f"{VERSION}_8PIXEL_EXACT_BIN_COMPARISON.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return out


def make_bin_table(records: list[dict]) -> tuple[Path, Path]:
    rows = []
    for rec in records:
        d = rec["data"]
        for f, p, pc in zip(d["freq"], d["power1"], d["power1_centered"]):
            rows.append({
                "pattern": rec["name"],
                "frequency_cycles_per_pixel": float(f),
                "period_pixels": np.inf if f == 0 else float(1.0 / abs(f)),
                "raw_fft_power": float(p),
                "mean_subtracted_fft_power": float(pc),
            })
    df = pd.DataFrame(rows)
    csv_path = CSV / f"{VERSION}_8PIXEL_EXACT_FFT_BINS.csv"
    df.to_csv(csv_path, index=False)

    summary = []
    for rec in records:
        d = rec["data"]
        nonzero = [(float(f), float(p)) for f, p in zip(d["freq"], d["power1"]) if p > 1e-12]
        summary.append({
            "Pattern": rec["label"],
            "Nonzero frequency bins": ", ".join(f"{f:+.3f}" for f, _ in nonzero),
            "Raw powers": ", ".join(f"{p:.3f}" for _, p in nonzero),
        })
    sdf = pd.DataFrame(summary)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 4.0), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=sdf.values, colLabels=sdf.columns,
                     loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#536879")
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")
    ax.set_title("Exact nonzero FFT bins for the two 8-pixel patterns", fontsize=15, pad=16)
    png_path = PNG / f"{VERSION}_8PIXEL_EXACT_FFT_TABLE.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    records = [{**p, "data": analyze(p["row"])} for p in PATTERNS]
    dashboard = make_dashboard(records)
    csv_path, table_path = make_bin_table(records)

    print(f"CODE OUTPUT: {VERSION}")
    print("Input A      0 0 0 0 1 1 1 1")
    print("Input B      0 1 0 1 0 1 0 1")
    print("FFT spacing  1/8 = 0.125 cycles/pixel")
    print("Bug fix      Second-row 2-D FFT now plotted as exact discrete bins")
    print(f"Plot PNG     {dashboard}")
    print(f"Table PNG    {table_path}")
    print(f"CSV          {csv_path}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

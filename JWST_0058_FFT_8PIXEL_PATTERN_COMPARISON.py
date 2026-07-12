#!/usr/bin/env python3
"""
JWST_0058 — 8-pixel spatial-frequency comparison.

Compares two exact 8-pixel patterns:
A = 0 0 0 0 1 1 1 1
B = 0 1 0 1 0 1 0 1

Each row is repeated vertically to form an 8x8 image. The script plots the
image, middle-row square profile, 1-D FFT power, and 2-D FFT power.
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

VERSION = "JWST_0058"
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

    f1 = np.fft.fftshift(np.fft.fft(row))
    p1 = np.abs(f1) ** 2
    freq1 = np.fft.fftshift(np.fft.fftfreq(row.size))

    f2 = np.fft.fftshift(np.fft.fft2(image))
    p2 = np.abs(f2) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))

    return {
        "image": image,
        "row": row,
        "freq1": freq1,
        "fft1": f1,
        "power1": p1,
        "fx": fx,
        "fy": fy,
        "power2": p2,
    }


def make_dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)

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
                ax.text(x, y, f"{int(d['image'][y, x])}", ha="center", va="center",
                        fontsize=9, color="black" if d["image"][y, x] > 0.5 else "white")

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
        ax.set_title("Middle-row square profile", fontsize=11)
        ax.grid(alpha=0.22)

        ax = axes[r, 2]
        markerline, stemlines, baseline = ax.stem(d["freq1"], d["power1"])
        plt.setp(stemlines, linewidth=1.7)
        plt.setp(markerline, markersize=6)
        plt.setp(baseline, linewidth=0.7, alpha=0.6)
        ax.set_xlabel("spatial frequency [cycles/pixel]")
        ax.set_ylabel("1-D FFT power")
        ax.set_title("1-D FFT power of the 8-pixel row", fontsize=11)
        ax.set_xticks(d["freq1"])
        ax.grid(alpha=0.20)

        positive = d["power2"][d["power2"] > 0]
        vmin = max(float(np.min(positive)), np.finfo(float).tiny)
        vmax = float(np.max(positive))
        extent = [d["fx"][0], d["fx"][-1], d["fy"][0], d["fy"][-1]]
        ax = axes[r, 3]
        ax.imshow(d["power2"], origin="lower", cmap="magma",
                  norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                  interpolation="nearest", aspect="equal")
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")
        ax.set_title("2-D FFT power of repeated rows", fontsize=11)

    fig.suptitle(
        "Eight-pixel spatial-frequency comparison\n"
        "Same number of 0 and 1 pixels; different arrangement, different FFT",
        fontsize=18,
    )
    out = PNG / f"{VERSION}_8PIXEL_PATTERN_FFT_COMPARISON.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return out


def make_table(records: list[dict]) -> tuple[Path, Path]:
    rows = []
    for rec in records:
        d = rec["data"]
        for f, p, coeff in zip(d["freq1"], d["power1"], d["fft1"]):
            rows.append({
                "pattern": rec["name"],
                "frequency_cycles_per_pixel": float(f),
                "period_pixels": np.inf if f == 0 else float(1.0 / abs(f)),
                "fft_real": float(coeff.real),
                "fft_imag": float(coeff.imag),
                "fft_power": float(p),
            })
    df = pd.DataFrame(rows)
    csv_path = CSV / f"{VERSION}_8PIXEL_FFT_BINS.csv"
    df.to_csv(csv_path, index=False)

    summary = []
    for rec in records:
        d = rec["data"]
        mask = np.abs(d["freq1"]) > 0
        idx = np.argmax(d["power1"][mask])
        f = float(np.abs(d["freq1"][mask][idx]))
        summary.append({
            "Pattern": rec["label"],
            "Strongest nonzero f [cyc/pix]": f,
            "Equivalent period [pix]": 1.0 / f,
            "Nonzero bins": int(np.count_nonzero(d["power1"] > 1e-12)),
        })
    sdf = pd.DataFrame(summary)
    shown = sdf.copy()
    shown["Strongest nonzero f [cyc/pix]"] = shown["Strongest nonzero f [cyc/pix]"].map(lambda v: f"{v:.3f}")
    shown["Equivalent period [pix]"] = shown["Equivalent period [pix]"].map(lambda v: f"{v:.2f}")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 3.7), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns,
                     loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#536879")
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")
    ax.set_title("Eight-pixel FFT comparison summary", fontsize=15, pad=16)
    png_path = PNG / f"{VERSION}_8PIXEL_FFT_SUMMARY_TABLE.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    records = [{**p, "data": analyze(p["row"])} for p in PATTERNS]
    dashboard = make_dashboard(records)
    csv_path, table_path = make_table(records)

    print(f"CODE OUTPUT: {VERSION}")
    print("Patterns     00001111 | 01010101")
    print("Image model  Each 8-pixel row repeated vertically into an 8x8 image")
    print("Result       Same intensity counts; different spatial arrangement and FFT")
    print(f"Plot PNG     {dashboard}")
    print(f"Table PNG    {table_path}")
    print(f"CSV          {csv_path}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

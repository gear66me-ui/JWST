#!/usr/bin/env python3
"""
JWST_0060 — single-strip 8-pixel spatial-frequency-power comparison.

Two exact 1x8 strips only:
A = 0 0 0 0 1 1 1 1
B = 0 1 0 1 0 1 0 1

Each row shows:
1) the literal eight-pixel strip,
2) the x-direction intensity cross-section,
3) one-sided spatial-frequency power (SFP), normalized to 100%.

No 8x8 repetition and no 2-D FFT are used.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib"}
    missing = [pip for module, pip in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

VERSION = "JWST_0060"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

PATTERNS = [
    {
        "name": "BLOCK_STEP",
        "label": "00001111 — one broad step",
        "values": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
    },
    {
        "name": "ALTERNATING",
        "label": "01010101 — alternating every pixel",
        "values": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float),
    },
]


def one_sided_sfp(values: np.ndarray) -> dict:
    n = values.size
    coeff = np.fft.rfft(values)
    freq = np.fft.rfftfreq(n, d=1.0)
    raw_power = np.abs(coeff) ** 2

    weights = np.ones_like(raw_power)
    if n % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0

    one_sided_power = raw_power * weights
    total = float(one_sided_power.sum())
    share_percent = 100.0 * one_sided_power / total if total > 0 else np.zeros_like(one_sided_power)
    period_pixels = np.full_like(freq, np.inf, dtype=float)
    period_pixels[freq > 0] = 1.0 / freq[freq > 0]

    return {
        "coeff": coeff,
        "freq": freq,
        "raw_power": raw_power,
        "one_sided_power": one_sided_power,
        "share_percent": share_percent,
        "period_pixels": period_pixels,
    }


def draw_strip(ax: plt.Axes, values: np.ndarray, label: str) -> None:
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for x, value in enumerate(values):
        face = "white" if value > 0.5 else "black"
        text = "black" if value > 0.5 else "white"
        ax.add_patch(Rectangle((x, 0), 1, 1, facecolor=face, edgecolor="0.55", linewidth=1.0))
        ax.text(x + 0.5, 0.5, f"{int(value)}", ha="center", va="center",
                fontsize=13, color=text, weight="bold")
    ax.set_xticks(np.arange(8) + 0.5, labels=[str(i) for i in range(8)])
    ax.set_yticks([])
    ax.set_xlabel("pixel index")
    ax.set_title(label, fontsize=13)


def draw_profile(ax: plt.Axes, values: np.ndarray) -> None:
    edges = np.arange(9, dtype=float)
    step_values = np.r_[values, values[-1]]
    ax.step(edges, step_values, where="post", linewidth=2.2)
    ax.scatter(np.arange(8) + 0.5, values, s=42, zorder=3)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xticks(np.arange(9))
    ax.set_yticks([0, 1])
    ax.set_xlabel("x position [pixels]")
    ax.set_ylabel("intensity")
    ax.set_title("Literal x-direction cross-section", fontsize=11)
    ax.grid(alpha=0.22)


def draw_sfp(ax: plt.Axes, result: dict) -> None:
    freq = result["freq"]
    share = result["share_percent"]
    bars = ax.bar(freq, share, width=0.072, edgecolor="white", linewidth=0.7)
    ax.set_xlim(-0.04, 0.54)
    ax.set_ylim(0, max(55.0, float(share.max()) * 1.18))
    ax.set_xticks(freq)
    ax.set_xlabel("spatial frequency [cycles/pixel]")
    ax.set_ylabel("share of total SFP power [%]")
    ax.set_title("One-sided spatial-frequency power", fontsize=11)
    ax.grid(axis="y", alpha=0.22)

    for bar, f, pct, period in zip(bars, freq, share, result["period_pixels"]):
        if pct < 0.05:
            continue
        if f == 0:
            meaning = "DC\n(mean level)"
        else:
            meaning = f"{period:.2f}-pixel\nperiod"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{pct:.1f}%\n{meaning}", ha="center", va="bottom", fontsize=8)


def make_dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), constrained_layout=True)

    for row, record in enumerate(records):
        draw_strip(axes[row, 0], record["values"], record["label"])
        draw_profile(axes[row, 1], record["values"])
        draw_sfp(axes[row, 2], record["sfp"])

    fig.suptitle(
        "Eight-pixel single-strip FFT comparison\n"
        "No repeated rows: literal strip, literal cross-section, normalized SFP power",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_8PIXEL_SINGLE_STRIP_SFP.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_data(records: list[dict]) -> tuple[Path, Path]:
    rows = []
    for record in records:
        result = record["sfp"]
        for f, period, raw, one_sided, pct in zip(
            result["freq"], result["period_pixels"], result["raw_power"],
            result["one_sided_power"], result["share_percent"]
        ):
            rows.append({
                "pattern": record["name"],
                "frequency_cycles_per_pixel": float(f),
                "period_pixels": float(period),
                "raw_fft_power": float(raw),
                "one_sided_sfp_power": float(one_sided),
                "share_of_total_sfp_percent": float(pct),
            })

    df = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{VERSION}_8PIXEL_SINGLE_STRIP_SFP.csv"
    df.to_csv(csv_path, index=False)

    summary = []
    for record in records:
        result = record["sfp"]
        pieces = []
        for f, period, pct in zip(result["freq"], result["period_pixels"], result["share_percent"]):
            if pct < 0.05:
                continue
            label = "DC" if f == 0 else f"{period:.2f}px"
            pieces.append(f"{label}: {pct:.1f}%")
        summary.append({"Pattern": record["label"], "SFP share": " | ".join(pieces)})

    shown = pd.DataFrame(summary)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 3.6), constrained_layout=True)
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
    ax.set_title("Single-strip spatial-frequency-power summary", fontsize=15, pad=16)
    table_path = PNG_DIR / f"{VERSION}_8PIXEL_SINGLE_STRIP_SFP_TABLE.png"
    fig.savefig(table_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return csv_path, table_path


def main() -> None:
    records = [{**pattern, "sfp": one_sided_sfp(pattern["values"])} for pattern in PATTERNS]
    dashboard = make_dashboard(records)
    csv_path, table_path = save_data(records)

    print(f"CODE OUTPUT: {VERSION}")
    print("Input A      0 0 0 0 1 1 1 1")
    print("Input B      0 1 0 1 0 1 0 1")
    print("Geometry     One literal 1x8 strip only; no repeated rows")
    print("SFP display  One-sided Fourier-power share normalized to 100%")
    print(f"Plot PNG     {dashboard}")
    print(f"Table PNG    {table_path}")
    print(f"CSV          {csv_path}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

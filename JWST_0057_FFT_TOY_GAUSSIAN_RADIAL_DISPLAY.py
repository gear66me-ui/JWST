#!/usr/bin/env python3
"""
JWST_0057_FFT_TOY_GAUSSIAN_RADIAL_DISPLAY.py

Verified two-row toy FFT dashboard.
Column 3 displays a radially averaged reconstruction of FFT power, computed
from the actual 2-D FFT. This intentionally removes row/column numerical
streaks and contains no guide lines, crosshairs, markers, or overlays.
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

VERSION = "JWST_0057"
ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
for d in (PNG, CSV):
    d.mkdir(parents=True, exist_ok=True)

N = 256
SIGMA_PIX = 14.0
HIGHPASS_SIGMA_PIX = 26.0


def gaussian_image(sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices((N, N), dtype=float)
    c = (N - 1) / 2.0
    x -= c
    y -= c
    return np.exp(-0.5 * (x * x + y * y) / sigma**2), x, y


def make_cases() -> list[dict]:
    base, x, y = gaussian_image(SIGMA_PIX)
    r = np.hypot(x, y)
    broad = np.exp(-0.5 * (r / HIGHPASS_SIGMA_PIX) ** 2)
    return [
        {"title": "Pure Gaussian lamp", "image": base,
         "note": "Smooth hotspot; Fourier power is concentrated at low spatial frequency."},
        {"title": "High-pass Gaussian", "image": base - 0.65 * broad / broad.max(),
         "note": "Broad structure removed; zero-frequency power is suppressed."},
    ]


def radialize(power: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = power.shape
    y, x = np.indices(power.shape, dtype=float)
    cy, cx = ny // 2, nx // 2
    r = np.hypot(x - cx, y - cy)
    ridx = np.floor(r).astype(int)
    max_r = int(ridx.max())
    profile = np.full(max_r + 1, np.nan)
    for k in range(max_r + 1):
        q = ridx == k
        if np.any(q):
            profile[k] = np.nanmedian(power[q])
    finite = np.isfinite(profile)
    if not np.all(finite):
        profile[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), profile[finite])
    radial_map = profile[np.clip(ridx, 0, max_r)]
    return radial_map, np.arange(max_r + 1, dtype=float), profile


def fft_data(image: np.ndarray) -> dict:
    f = np.fft.fftshift(np.fft.fft2(image))
    power = np.abs(f) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    cy, cx = image.shape[0] // 2, image.shape[1] // 2
    radial_map, radius, radial_profile = radialize(power)
    return {
        "power": power,
        "radial_map": radial_map,
        "radius": radius,
        "radial_profile": radial_profile,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_cut": image[cy, :],
        "x_power": power[cy, :],
    }


def make_dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)

    for row, record in enumerate(records):
        image = record["image"]
        d = record["fft"]

        ax = axes[row, 0]
        ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_title(record["title"], fontsize=13)
        ax.text(0.02, -0.12, record["note"], transform=ax.transAxes, fontsize=9, va="top")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        ax.plot(np.arange(N) - d["cx"], d["image_cut"], linewidth=1.8)
        ax.set_title("Image-space center line", fontsize=11)
        ax.set_xlabel("pixel offset")
        ax.set_ylabel("intensity")
        ax.grid(alpha=0.20)

        positive = d["radial_map"][d["radial_map"] > 0]
        vmin = max(float(np.percentile(positive, 2.0)), np.finfo(float).tiny)
        vmax = float(np.percentile(positive, 99.98))
        extent = [d["fx"][0], d["fx"][-1], d["fy"][0], d["fy"][-1]]
        ax = axes[row, 2]
        ax.imshow(d["radial_map"], origin="lower", cmap="magma",
                  norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                  interpolation="bilinear", aspect="equal")
        ax.set_title("Radially averaged FFT power", fontsize=11)
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")

        freq = np.abs(d["fx"])
        order = np.argsort(freq)
        ax = axes[row, 3]
        ax.plot(freq[order], d["x_power"][order], linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlim(0, 0.5)
        ax.set_title("Raw FFT center cross-section", fontsize=11)
        ax.set_xlabel("|f| [cycles/pixel]")
        ax.set_ylabel("power")
        ax.grid(alpha=0.20)

    fig.suptitle(
        "Toy Gaussian FFT demo — verified no-cross display\n"
        "Column 3 is radialized FFT power: no crosshairs, guide lines, markers, or axis glow",
        fontsize=18,
    )
    out = PNG / f"{VERSION}_TOY_GAUSSIAN_VERIFIED_NOCROSS.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return out


def make_table(records: list[dict]) -> tuple[Path, Path]:
    rows = []
    for record in records:
        d = record["fft"]
        freq = np.abs(d["fx"])
        order = np.argsort(freq)
        q = freq[order] > 0
        peak = int(np.argmax(d["x_power"][order][q]))
        fpeak = float(freq[order][q][peak])
        rows.append({
            "case": record["title"],
            "dominant_frequency_cycles_per_pixel": fpeak,
            "equivalent_period_pixels": 1.0 / fpeak,
            "center_power": float(d["x_power"][d["cx"]]),
        })
    df = pd.DataFrame(rows)
    csv_path = CSV / f"{VERSION}_SUMMARY.csv"
    df.to_csv(csv_path, index=False)

    shown = df.copy()
    shown["dominant_frequency_cycles_per_pixel"] = shown["dominant_frequency_cycles_per_pixel"].map(lambda v: f"{v:.5f}")
    shown["equivalent_period_pixels"] = shown["equivalent_period_pixels"].map(lambda v: f"{v:.2f}")
    shown["center_power"] = shown["center_power"].map(lambda v: f"{v:.6g}")
    shown.columns = ["Case", "Dominant f [cyc/pix]", "Equivalent period [pix]", "Center power"]

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
    ax.set_title("Toy Gaussian FFT verified no-cross summary", fontsize=15, pad=16)
    png_path = PNG / f"{VERSION}_SUMMARY_TABLE.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    records = []
    for case in make_cases():
        records.append({**case, "fft": fft_data(np.asarray(case["image"], dtype=float))})

    dashboard = make_dashboard(records)
    csv_path, table_path = make_table(records)

    print(f"CODE OUTPUT: {VERSION}")
    print("Purpose      Two-row Gaussian FFT learning dashboard")
    print("Verification Column 3 uses radialized FFT power only")
    print("Verification No axhline, axvline, scatter, or center-marker calls in Column 3")
    print(f"Plot PNG     {dashboard}")
    print(f"Table PNG    {table_path}")
    print(f"CSV          {csv_path}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""JWST_0052 — MoM-z14 FFT crosshair and axis-power analysis.

Uses calibrated NIRCam FITS mosaics. The 2-D FFT is shifted so zero spatial
frequency is at the center. Horizontal and vertical cuts quantify power along
fx and fy; radial profiles show direction-averaged power versus spatial scale.
"""
from __future__ import annotations

import importlib.util
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_0051 = "JWST_0051_MOMZ14_FFT_RASTER_ARTIFACT_ANALYSIS.py"
BASE_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_0051}"
BASE_PATH = Path("/content") / BASE_0051
VERSION = "JWST_0052"
ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
FITS = ROOT / "FITS"
for d in (PNG, CSV, FITS):
    d.mkdir(parents=True, exist_ok=True)

urllib.request.urlretrieve(BASE_URL, BASE_PATH)
spec = importlib.util.spec_from_file_location("jwst0051", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to import JWST_0051")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

FREQ_MIN = 0.025
FREQ_MAX = 0.46
PERIOD_GUIDES = [32, 16, 8, 4, 2]


def pixel_scale_arcsec(header) -> float:
    try:
        s = np.asarray(proj_plane_pixel_scales(WCS(header).celestial)) * 3600.0
        s = s[np.isfinite(s) & (s > 0) & (s < 10)]
        if s.size:
            return float(np.median(s))
    except Exception:
        pass
    pixar = header.get("PIXAR_A2")
    if pixar is not None and 0 < float(pixar) < 100:
        return float(np.sqrt(float(pixar)))
    raise RuntimeError("No plausible celestial pixel scale in FITS WCS")


def fold_axis(values: np.ndarray, center: int) -> np.ndarray:
    a = values[:center][::-1]
    b = values[center + 1:]
    n = min(len(a), len(b))
    return 0.5 * (a[:n] + b[:n])


def peak_result(freq: np.ndarray, profile: np.ndarray) -> dict:
    mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX) & np.isfinite(profile)
    f, p = freq[mask], profile[mask]
    if f.size == 0:
        return {"f": np.nan, "period": np.nan, "power": np.nan}
    logp = gaussian_filter1d(np.log10(np.maximum(p, np.finfo(float).tiny)), 1.3)
    peaks, _ = find_peaks(logp, prominence=0.08, distance=3)
    i = int(peaks[np.argmax(logp[peaks])]) if peaks.size else int(np.argmax(logp))
    return {"f": float(f[i]), "period": float(1.0 / f[i]), "power": float(p[i])}


def radial_profile(power: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, 0.5, 121)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(centers.shape, np.nan)
    for i in range(len(centers)):
        q = (radius >= edges[i]) & (radius < edges[i + 1])
        if np.any(q):
            values[i] = np.nanmedian(power[q])
    return centers, values


def analyze(windowed: np.ndarray) -> dict:
    ny, nx = windowed.shape
    fft = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(fft) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    fxx, fyy = np.meshgrid(fx, fy)
    radius = np.hypot(fxx, fyy)
    annulus = (radius >= FREQ_MIN) & (radius <= FREQ_MAX)
    norm = power / max(float(np.nanmedian(power[annulus])), np.finfo(float).tiny)
    cy, cx = ny // 2, nx // 2
    xp = fold_axis(norm[cy, :], cx)
    yp = fold_axis(norm[:, cx], cy)
    xf = fx[cx + 1:cx + 1 + len(xp)]
    yf = fy[cy + 1:cy + 1 + len(yp)]
    rf, rp = radial_profile(norm, radius)
    return {
        "power": power, "fx": fx, "fy": fy, "radius": radius,
        "xf": xf, "yf": yf, "xp": xp, "yp": yp,
        "xpeak": peak_result(xf, xp), "ypeak": peak_result(yf, yp),
        "rf": rf, "rp": rp,
    }


def period_guides(ax) -> None:
    for period in PERIOD_GUIDES:
        f = 1.0 / period
        ax.axvline(f, lw=0.55, ls="--", alpha=0.28)
        ax.text(f, 0.96, f"{period}px", transform=ax.get_xaxis_transform(),
                rotation=90, ha="right", va="top", fontsize=7, alpha=0.75)


def explainer() -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.set(xlim=(-0.52, 0.52), ylim=(-0.52, 0.52), aspect="equal",
           xlabel="fx [cycles/pixel]", ylabel="fy [cycles/pixel]")
    ax.axhline(0, lw=2, color="#56d8ff")
    ax.axvline(0, lw=2, color="#ff9b54")
    for r, label in [(0.08, "broad scales"), (0.23, "intermediate scales"),
                     (0.42, "fine scales / edges")]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=1, alpha=0.55))
        ax.text(r / np.sqrt(2), r / np.sqrt(2), label, rotation=35, fontsize=10)
    ax.scatter([0], [0], s=65, color="white")
    ax.annotate("zero spatial frequency\n(mean and very broad structure)", (0, 0),
                (0.13, -0.13), arrowprops={"arrowstyle": "->"}, fontsize=11)
    ax.text(0, -0.49, "horizontal FFT axis → vertical image variation",
            color="#56d8ff", ha="center", fontsize=11)
    ax.text(-0.49, 0, "vertical FFT axis → horizontal image variation",
            color="#ff9b54", ha="center", va="center", rotation=90, fontsize=11)
    ax.set_title("How to read a shifted 2-D FFT power map\n"
                 "radius = spatial frequency; angle = orientation; brightness = Fourier power",
                 fontsize=15)
    ax.grid(alpha=0.15)
    path = PNG / f"{VERSION}_FFT_POWER_MAP_EXPLAINER.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show(); plt.close(fig)
    return path


def dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 4, figsize=(22, 18), constrained_layout=True)
    for col, r in enumerate(records):
        e, image, a = r["entry"], r["prep"]["image"], r["analysis"]
        lo, hi = np.nanpercentile(image, [1, 99.5])
        axes[0, col].imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi,
                            interpolation="nearest")
        axes[0, col].set_title(f"{e['name']} | {e['lambda_um']:.2f} µm", fontsize=13)
        axes[0, col].set_xlabel(f"{r['scale']:.5f} arcsec/pixel", fontsize=9)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        positive = a["power"][a["power"] > 0]
        vmin = max(float(np.percentile(positive, 35)), np.finfo(float).tiny)
        vmax = float(np.percentile(positive, 99.97))
        extent = [a["fx"][0], a["fx"][-1], a["fy"][0], a["fy"][-1]]
        axes[1, col].imshow(a["power"], origin="lower", cmap="magma",
                            norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                            interpolation="nearest", aspect="equal")
        axes[1, col].axhline(0, lw=1.4, color="#56d8ff")
        axes[1, col].axvline(0, lw=1.4, color="#ff9b54")
        axes[1, col].scatter([a["xpeak"]["f"], -a["xpeak"]["f"]], [0, 0],
                             s=22, color="#56d8ff")
        axes[1, col].scatter([0, 0], [a["ypeak"]["f"], -a["ypeak"]["f"]],
                             s=22, color="#ff9b54")
        axes[1, col].set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5),
                         xlabel="fx [cycles/pixel]", ylabel="fy [cycles/pixel]")
        axes[1, col].set_title("2-D FFT power + x/y crosshairs", fontsize=11)

        axes[2, col].plot(a["xf"], a["xp"], lw=1.25, color="#56d8ff")
        period_guides(axes[2, col])
        axes[2, col].scatter([a["xpeak"]["f"]], [a["xpeak"]["power"]],
                             s=30, color="#56d8ff")
        axes[2, col].set_yscale("log"); axes[2, col].set_xlim(0, 0.5)
        axes[2, col].set(xlabel="|fx| [cycles/pixel]",
                         ylabel="power / annular median")
        axes[2, col].set_title("horizontal FFT axis → vertical structure\n"
                               f"strongest period ≈ {a['xpeak']['period']:.2f} px",
                               fontsize=10)
        axes[2, col].grid(alpha=0.18)

        axes[3, col].plot(a["yf"], a["yp"], lw=1.25, color="#ff9b54")
        period_guides(axes[3, col])
        axes[3, col].scatter([a["ypeak"]["f"]], [a["ypeak"]["power"]],
                             s=30, color="#ff9b54")
        axes[3, col].set_yscale("log"); axes[3, col].set_xlim(0, 0.5)
        axes[3, col].set(xlabel="|fy| [cycles/pixel]",
                         ylabel="power / annular median")
        axes[3, col].set_title("vertical FFT axis → horizontal structure\n"
                               f"strongest period ≈ {a['ypeak']['period']:.2f} px",
                               fontsize=10)
        axes[3, col].grid(alpha=0.18)

    fig.suptitle("MoM-z14 FFT crosshair analysis\n"
                 "center = broad scales; farther out = finer repetition; intensity changes power, not radius",
                 fontsize=18)
    path = PNG / f"{VERSION}_MOMZ14_FFT_CROSSHAIR_AXIS_PROFILES.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show(); plt.close(fig)
    return path


def radial_plot(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    for r in records:
        a = r["analysis"]
        q = (a["rf"] >= 0.01) & np.isfinite(a["rp"])
        ax.plot(a["rf"][q], a["rp"][q], lw=1.8, label=r["entry"]["name"])
    period_guides(ax)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.01, 0.5)
    ax.set(xlabel="radial spatial frequency [cycles/pixel]",
           ylabel="median FFT power / annular median")
    ax.set_title("Radial FFT power — direction-averaged structure by spatial scale\n"
                 "low frequency = broad objects/background; high frequency = fine edges/noise",
                 fontsize=15)
    ax.grid(alpha=0.2, which="both"); ax.legend(ncol=4)
    path = PNG / f"{VERSION}_MOMZ14_FFT_RADIAL_POWER.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show(); plt.close(fig)
    return path


def summary_table(df: pd.DataFrame) -> Path:
    shown = df[["filter", "scale", "x_f", "x_period_pix", "x_period_arcsec",
                "y_f", "y_period_pix", "y_period_arcsec"]].copy()
    for c in shown.columns[1:]:
        shown[c] = shown[c].map(lambda x: f"{x:.4f}")
    shown.columns = ["Filter", "Scale [arcsec/pix]", "X f [cyc/pix]", "X period [pix]",
                     "X period [arcsec]", "Y f [cyc/pix]", "Y period [pix]",
                     "Y period [arcsec]"]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(17.5, 4.8), constrained_layout=True)
    ax.axis("off")
    t = ax.table(cellText=shown.values, colLabels=shown.columns,
                 cellLoc="center", colLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.85)
    for (row, col), cell in t.get_celld().items():
        cell.set_edgecolor("#536879"); cell.set_linewidth(0.7)
        cell.set_facecolor("#18344f" if row == 0 else ("#111923" if row % 2 else "#17212c"))
        cell.set_text_props(color="white", weight="bold" if row == 0 else "normal")
    ax.set_title("MoM-z14 FFT x/y-axis candidate periods\n"
                 "A peak is a repeated scale candidate, not automatic proof of rastering",
                 fontsize=15, pad=18)
    path = PNG / f"{VERSION}_MOMZ14_FFT_AXIS_SUMMARY_TABLE.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show(); plt.close(fig)
    return path


def main() -> None:
    base = m.load_base_module()
    base.VERSION = VERSION
    base.FITS_DIR = FITS
    session = base.build_session()
    records, summary_rows, profile_rows = [], [], []
    for entry in base.FILTERS:
        path, url, nbytes = base.download_channel(session, entry)
        data, header = base.load_image(path)
        scale = pixel_scale_arcsec(header)
        prep = m.preprocess(data)
        a = analyze(prep["windowed"])
        records.append({"entry": entry, "path": path, "url": url,
                        "bytes": nbytes, "scale": scale, "prep": prep, "analysis": a})
        summary_rows.append({
            "filter": entry["name"], "lambda_um": entry["lambda_um"], "scale": scale,
            "x_f": a["xpeak"]["f"], "x_period_pix": a["xpeak"]["period"],
            "x_period_arcsec": a["xpeak"]["period"] * scale,
            "y_f": a["ypeak"]["f"], "y_period_pix": a["ypeak"]["period"],
            "y_period_arcsec": a["ypeak"]["period"] * scale,
            "fits_path": str(path), "source_url": url,
        })
        for name, f, p in [("X_AXIS", a["xf"], a["xp"]),
                           ("Y_AXIS", a["yf"], a["yp"]),
                           ("RADIAL", a["rf"], a["rp"])]:
            for freq, power in zip(f, p):
                profile_rows.append({"filter": entry["name"], "profile": name,
                                     "frequency_cycles_per_pix": float(freq),
                                     "period_pix": float(1 / freq) if freq > 0 else np.inf,
                                     "power_over_annular_median": float(power)})

    summary = pd.DataFrame(summary_rows)
    profiles = pd.DataFrame(profile_rows)
    p0 = explainer(); p1 = dashboard(records); p2 = radial_plot(records); p3 = summary_table(summary)
    c0 = CSV / f"{VERSION}_MOMZ14_FFT_AXIS_SUMMARY.csv"
    c1 = CSV / f"{VERSION}_MOMZ14_FFT_POWER_PROFILES.csv"
    summary.to_csv(c0, index=False); profiles.to_csv(c1, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print("Meaning      FFT radius = spatial frequency; brightness = Fourier power")
    print("Crosshair    X FFT axis diagnoses vertical image structure")
    print("Crosshair    Y FFT axis diagnoses horizontal image structure")
    print("Filter   Scale arcsec/pix   X period pix   Y period pix")
    for r in summary.itertuples(index=False):
        print(f"{r.filter:<7} {r.scale:>16.5f}   {r.x_period_pix:>12.4f}   {r.y_period_pix:>12.4f}")
    print("Caution      candidate periods are not automatic proof of rastering")
    print(f"Explainer PNG {p0}")
    print(f"Crosshair PNG {p1}")
    print(f"Radial PNG    {p2}")
    print(f"Table PNG     {p3}")
    print(f"Summary CSV   {c0}")
    print(f"Profiles CSV  {c1}")
    print(f"Timestamp     {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
